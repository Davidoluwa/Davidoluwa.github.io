#include "sera/categorical_engine.hpp"
#include "sera/shape_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct GeneratedShape {
    sera::CategoricalShape graph;
    std::vector<std::uint32_t> planted_main;
    std::uint32_t main_nodes{};
    std::uint32_t main_factors{};
    std::uint32_t query_node{};
};

std::vector<double> modular_cost(
    std::uint32_t states,
    std::uint32_t delta,
    double penalty
) {
    std::vector<double> matrix(
        static_cast<std::size_t>(states) * states, penalty
    );
    for (std::uint32_t left = 0; left < states; ++left) {
        const std::uint32_t right = (left + delta) % states;
        matrix[static_cast<std::size_t>(left) * states + right] = 0.0;
    }
    return matrix;
}

void add_component(
    GeneratedShape& generated,
    std::uint32_t nodes,
    std::uint32_t states,
    std::mt19937_64& random,
    bool anchored,
    bool randomized_offsets,
    double constraint_penalty,
    double anchor_penalty
) {
    const std::uint32_t base = static_cast<std::uint32_t>(
        generated.graph.nodes.size()
    );
    std::vector<std::uint32_t> planted(nodes, 0);
    if (randomized_offsets) {
        for (auto& state : planted) {
            state = static_cast<std::uint32_t>(random() % states);
        }
    }
    for (std::uint32_t node = 0; node < nodes; ++node) {
        std::vector<double> unary(states, 0.0);
        if (anchored && node == 0) {
            std::fill(unary.begin(), unary.end(), anchor_penalty);
            unary[planted[node]] = 0.0;
        }
        generated.graph.add_node("categorical-state", unary);
    }

    std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
    const auto add_relation = [&](std::uint32_t left, std::uint32_t right) {
        const auto identity = std::minmax(left, right);
        if (!edges.insert(identity).second) {
            return;
        }
        const std::uint32_t delta =
            (planted[right] + states - planted[left]) % states;
        generated.graph.add_factor(
            base + left,
            base + right,
            "modular-offset",
            modular_cost(states, delta, constraint_penalty)
        );
    };
    for (std::uint32_t node = 0; node + 1 < nodes; ++node) {
        add_relation(node, node + 1);
    }
    if (nodes > 2) {
        add_relation(nodes - 1, 0);
    }
    for (std::uint32_t node = 0; node + 3 < nodes; node += 3) {
        add_relation(node, node + 3);
    }

    if (base == 0) {
        generated.planted_main = planted;
        generated.main_nodes = nodes;
        generated.main_factors = static_cast<std::uint32_t>(
            generated.graph.factors.size()
        );
        generated.query_node = nodes - 1;
    }
}

GeneratedShape generate_anchored(
    std::uint32_t nodes,
    std::uint32_t states,
    std::uint32_t distractor_components,
    std::uint64_t seed
) {
    GeneratedShape generated;
    std::mt19937_64 random(seed);
    add_component(
        generated, nodes, states, random, true, true, 25.0, 100.0
    );
    for (std::uint32_t component = 0;
         component < distractor_components;
         ++component) {
        add_component(
            generated, nodes, states, random, false, false, 25.0, 100.0
        );
    }
    generated.graph.validate();
    return generated;
}

GeneratedShape generate_ambiguous(
    std::uint32_t nodes,
    std::uint32_t states,
    std::uint64_t seed
) {
    GeneratedShape generated;
    std::mt19937_64 random(seed);
    add_component(
        generated, nodes, states, random, false, true, 25.0, 100.0
    );
    generated.graph.validate();
    return generated;
}

sera::CategoricalShape shuffled_copy(
    const sera::CategoricalShape& input,
    std::uint64_t seed
) {
    sera::CategoricalShape shuffled = input;
    std::mt19937_64 random(seed);
    std::shuffle(
        shuffled.factors.begin(), shuffled.factors.end(), random
    );
    for (std::size_t i = 0; i < shuffled.factors.size(); ++i) {
        shuffled.factors[i].id = static_cast<std::uint32_t>(i);
    }
    return shuffled;
}

sera::CategoricalShape contradictory_copy(
    const GeneratedShape& generated,
    std::uint32_t states
) {
    sera::CategoricalShape contradictory = generated.graph;
    if (generated.main_factors == 0) {
        throw std::runtime_error("No main factor to corrupt");
    }
    auto& factor = contradictory.factors[generated.main_factors - 1];
    const std::uint32_t planted_left =
        generated.planted_main[factor.left];
    const std::uint32_t planted_right =
        generated.planted_main[factor.right];
    const std::uint32_t planted_delta =
        (planted_right + states - planted_left) % states;
    factor.cost = modular_cost(
        states, (planted_delta + 1) % states, 25.0
    );
    contradictory.validate();
    return contradictory;
}

double main_accuracy(
    const GeneratedShape& generated,
    std::span<const std::uint32_t> assignment
) {
    std::uint32_t correct = 0;
    for (std::uint32_t node = 0; node < generated.main_nodes; ++node) {
        if (assignment[node] == generated.planted_main[node]) {
            ++correct;
        }
    }
    return static_cast<double>(correct)
        / static_cast<double>(generated.main_nodes);
}

void write_header(std::ostream& out) {
    out << "case_id,role,nodes,factors,states,distractors,seed,converged,"
           "sweeps,energy,violations,main_accuracy,wall_us,workspace_bytes,"
           "message_proposals,message_commits,message_skips,"
           "candidate_evaluations,scalar_add_subtracts,scalar_comparisons,"
           "scalar_multiplies,exponentials,max_residual,mean_entropy,"
           "assignment_hash\n";
}

void write_row(
    std::ostream& out,
    const std::string& case_id,
    std::string_view role,
    const sera::CategoricalShape& graph,
    const sera::CategoricalSolveResult& result,
    std::uint32_t states,
    std::uint32_t distractors,
    std::uint64_t seed,
    double accuracy
) {
    const auto& telemetry = result.telemetry;
    const auto& operations = telemetry.operations;
    out << std::setprecision(12)
        << case_id << ',' << role << ','
        << graph.nodes.size() << ',' << graph.factors.size() << ','
        << states << ',' << distractors << ',' << seed << ','
        << (telemetry.converged ? 1 : 0) << ','
        << telemetry.sweeps << ',' << telemetry.final_energy << ','
        << telemetry.violated_factors << ',' << accuracy << ','
        << static_cast<double>(telemetry.wall_nanoseconds) / 1000.0 << ','
        << telemetry.workspace_bytes << ','
        << operations.message_proposals << ','
        << operations.message_commits << ','
        << operations.message_skips << ','
        << operations.candidate_evaluations << ','
        << operations.scalar_add_subtracts << ','
        << operations.scalar_comparisons << ','
        << operations.scalar_multiplies << ','
        << operations.exponentials << ','
        << telemetry.max_message_residual << ','
        << telemetry.mean_entropy << ','
        << telemetry.assignment_hash << '\n';
}

bool throws_invalid(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

std::uint32_t run_validation_tests(std::ostream& out) {
    std::uint32_t passed = 0;
    std::uint32_t total = 0;
    const auto record = [&](std::string_view name, bool success) {
        out << name << ',' << (success ? "PASS" : "FAIL") << '\n';
        ++total;
        if (success) {
            ++passed;
        }
    };

    {
        sera::ShapeGraph graph;
        graph.dimensions = 1;
        graph.add_node("x");
        const sera::FactorTerm term{0, 1.0F};
        const float target = std::numeric_limits<float>::quiet_NaN();
        graph.add_factor(
            std::span<const sera::FactorTerm>(&term, 1),
            std::span<const float>(&target, 1)
        );
        record("s0_nonfinite_target", throws_invalid([&] { graph.validate(); }));
    }
    {
        sera::ShapeGraph graph;
        graph.dimensions = 1;
        graph.add_node("x");
        const sera::FactorTerm terms[2] = {{0, 1.0F}, {0, -1.0F}};
        const float target = 0.0F;
        graph.add_factor(
            terms, std::span<const float>(&target, 1)
        );
        record("s0_duplicate_factor_node", throws_invalid([&] {
            graph.validate();
        }));
    }
    {
        sera::CategoricalShape graph;
        const std::vector<double> unary{0.0, 0.0};
        graph.add_node("a", unary);
        graph.add_node("b", unary);
        const std::vector<double> cost{0.0, 1.0, 1.0, 0.0};
        graph.add_factor(0, 1, "eq", cost);
        graph.add_factor(0, 1, "duplicate", cost);
        record("s1_parallel_factor_rejected", throws_invalid([&] {
            graph.validate();
        }));
    }
    {
        sera::CategoricalShape graph;
        const std::vector<double> unary{
            0.0, std::numeric_limits<double>::infinity()
        };
        graph.add_node("bad", unary);
        record("s1_nonfinite_unary", throws_invalid([&] {
            graph.validate();
        }));
    }
    record("s1_invalid_damping", throws_invalid([] {
        sera::CategoricalSolveOptions options;
        options.damping = 0.0;
        const sera::CategoricalShapeSolver solver(options);
        static_cast<void>(solver);
    }));
    out << "summary," << passed << '/' << total << '\n';
    return passed;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(values.size())) - 1.0
    );
    return values[std::min(index, values.size() - 1)];
}

sera::CategoricalShape generate_oracle_shape(
    std::uint32_t nodes,
    std::uint32_t states,
    std::uint64_t seed,
    bool loopy
) {
    sera::CategoricalShape graph;
    std::mt19937_64 random(seed);
    for (std::uint32_t node = 0; node < nodes; ++node) {
        std::vector<double> unary(states, 0.0);
        for (std::uint32_t state = 0; state < states; ++state) {
            unary[state] =
                static_cast<double>(random() % 1100) / 100.0
                + static_cast<double>(node * states + state) * 1.0e-7;
        }
        graph.add_node("oracle-state", unary);
    }
    const auto add_random_factor =
        [&](std::uint32_t left, std::uint32_t right) {
            std::vector<double> cost(
                static_cast<std::size_t>(states) * states, 0.0
            );
            for (std::size_t index = 0; index < cost.size(); ++index) {
                cost[index] =
                    static_cast<double>(random() % 1300) / 100.0
                    + static_cast<double>(index) * 1.0e-8;
            }
            graph.add_factor(left, right, "random-cost", cost);
        };
    for (std::uint32_t node = 0; node + 1 < nodes; ++node) {
        add_random_factor(node, node + 1);
    }
    if (loopy) {
        for (std::uint32_t node = 0; node + 2 < nodes; node += 2) {
            add_random_factor(node, node + 2);
        }
        if (nodes > 3) {
            add_random_factor(nodes - 1, 0);
        }
    }
    graph.validate();
    return graph;
}

std::pair<double, std::vector<std::uint32_t>> exact_enumeration(
    const sera::CategoricalShape& graph
) {
    std::vector<std::uint32_t> assignment(graph.nodes.size(), 0);
    std::vector<std::uint32_t> best = assignment;
    double best_energy = std::numeric_limits<double>::infinity();
    bool done = false;
    while (!done) {
        const double energy = sera::categorical_energy(graph, assignment);
        if (energy < best_energy) {
            best_energy = energy;
            best = assignment;
        }
        for (std::size_t node = 0; node < assignment.size(); ++node) {
            ++assignment[node];
            if (assignment[node] < graph.nodes[node].unary_cost.size()) {
                break;
            }
            assignment[node] = 0;
            if (node + 1 == assignment.size()) {
                done = true;
            }
        }
    }
    return {best_energy, best};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output =
            argc > 1 ? argv[1] : "results_s1";
        std::filesystem::create_directories(output);
        std::ofstream cases(output / "cases.csv");
        std::ofstream relations(output / "relational_checks.csv");
        std::ofstream particles_csv(output / "particles.csv");
        std::ofstream validation(output / "validation.csv");
        std::ofstream oracle_csv(output / "exact_oracle.csv");
        if (
            !cases || !relations || !particles_csv || !validation
            || !oracle_csv
        ) {
            throw std::runtime_error("Unable to create result files");
        }
        write_header(cases);
        relations
            << "case_id,order_assignment_drift,order_energy_drift,"
               "slice_query_drift,slice_factor_reduction,"
               "contradiction_energy_ratio,ops_accounting_ok\n";
        particles_csv
            << "case_id,nodes,states,base_entropy,split_node,"
               "particles_spawned,unique_modes,mode_coverage,"
               "all_modes_zero_energy,aggregate_candidate_evaluations,"
               "aggregate_message_proposals,wall_us\n";
        oracle_csv
            << "case_id,topology,nodes,states,converged,sera_energy,"
               "exact_energy,optimality_gap,assignment_exact,sweeps,"
               "candidate_evaluations,wall_us,particle_energy,"
               "particle_gap,particle_exact,particle_modes\n";

        sera::CategoricalSolveOptions options;
        options.max_sweeps = 640;
        options.damping = 0.72;
        options.residual_gate = 1.0e-8;
        options.output_temperature = 0.25;
        options.trace_stride = 32;
        const sera::CategoricalShapeSolver solver(options);

        std::uint32_t clean_cases = 0;
        std::uint32_t clean_successes = 0;
        std::uint32_t order_failures = 0;
        std::uint32_t slice_failures = 0;
        std::uint32_t contradiction_separations = 0;
        std::uint32_t ops_failures = 0;
        std::uint64_t total_actual_candidates = 0;
        std::uint64_t total_dense_candidates = 0;
        double worst_order_energy_drift = 0.0;
        double minimum_contradiction_ratio =
            std::numeric_limits<double>::infinity();
        double maximum_reduction = 0.0;
        std::vector<double> clean_wall_us;
        bool sample_written = false;

        const std::uint32_t sizes[] = {16, 64, 192};
        const std::uint32_t state_counts[] = {2, 3, 5};
        const std::uint32_t distractor_counts[] = {0, 4};
        const std::uint64_t seeds[] = {17, 991};
        for (std::uint32_t nodes : sizes) {
            for (std::uint32_t states : state_counts) {
                for (std::uint32_t distractors : distractor_counts) {
                    for (std::uint64_t seed : seeds) {
                        const std::string case_id =
                            "n" + std::to_string(nodes)
                            + "_k" + std::to_string(states)
                            + "_d" + std::to_string(distractors)
                            + "_s" + std::to_string(seed);
                        const GeneratedShape generated = generate_anchored(
                            nodes, states, distractors, seed
                        );
                        const auto clean = solver.solve(generated.graph);
                        const auto shuffled_graph = shuffled_copy(
                            generated.graph, seed ^ 0xA51CE55ULL
                        );
                        const auto shuffled = solver.solve(shuffled_graph);
                        const auto slice = sera::extract_goal_shape(
                            generated.graph, generated.query_node
                        );
                        const auto sliced = solver.solve(slice.graph);
                        const auto contradiction_graph = contradictory_copy(
                            generated, states
                        );
                        const auto contradiction = solver.solve(
                            contradiction_graph
                        );
                        const double accuracy = main_accuracy(
                            generated, clean.assignment
                        );
                        const double shuffled_accuracy = main_accuracy(
                            generated, shuffled.assignment
                        );
                        const bool clean_ok =
                            clean.telemetry.converged
                            && clean.telemetry.violated_factors == 0
                            && accuracy == 1.0;
                        ++clean_cases;
                        clean_successes += clean_ok ? 1U : 0U;
                        clean_wall_us.push_back(
                            static_cast<double>(
                                clean.telemetry.wall_nanoseconds
                            ) / 1000.0
                        );
                        write_row(
                            cases, case_id, "clean", generated.graph, clean,
                            states, distractors, seed, accuracy
                        );
                        write_row(
                            cases, case_id, "shuffled", shuffled_graph,
                            shuffled, states, distractors, seed,
                            shuffled_accuracy
                        );
                        write_row(
                            cases, case_id, "goal_slice", slice.graph, sliced,
                            states, distractors, seed,
                            sliced.assignment[slice.local_query_node]
                                    == generated.planted_main[
                                        generated.query_node
                                    ]
                                ? 1.0 : 0.0
                        );
                        write_row(
                            cases, case_id, "contradiction",
                            contradiction_graph, contradiction, states,
                            distractors, seed,
                            main_accuracy(generated, contradiction.assignment)
                        );

                        const bool assignment_drift =
                            clean.assignment != shuffled.assignment;
                        const double order_energy_drift = std::abs(
                            clean.telemetry.final_energy
                            - shuffled.telemetry.final_energy
                        );
                        const bool query_drift =
                            clean.assignment[generated.query_node]
                            != sliced.assignment[slice.local_query_node];
                        const double reduction =
                            generated.graph.factors.empty() ? 0.0
                            : 1.0 - static_cast<double>(
                                slice.graph.factors.size()
                            ) / static_cast<double>(
                                generated.graph.factors.size()
                            );
                        const double contradiction_ratio =
                            (contradiction.telemetry.final_energy + 1.0e-12)
                            / (clean.telemetry.final_energy + 1.0e-12);
                        const std::uint64_t dense_proposals =
                            2ULL * generated.graph.factors.size()
                            * clean.telemetry.sweeps;
                        const std::uint64_t expected_candidates =
                            clean.telemetry.operations.message_proposals
                            * static_cast<std::uint64_t>(states) * states;
                        const std::uint64_t dense_candidates =
                            dense_proposals
                            * static_cast<std::uint64_t>(states) * states;
                        const bool ops_ok =
                            clean.telemetry.operations.candidate_evaluations
                                == expected_candidates
                            && clean.telemetry.operations.message_commits
                                + clean.telemetry.operations.message_skips
                                == clean.telemetry.operations.message_proposals
                            && clean.telemetry.operations.message_proposals
                                <= dense_proposals;
                        order_failures += assignment_drift ? 1U : 0U;
                        slice_failures += query_drift ? 1U : 0U;
                        contradiction_separations +=
                            contradiction_ratio >= 10.0 ? 1U : 0U;
                        ops_failures += ops_ok ? 0U : 1U;
                        total_actual_candidates +=
                            clean.telemetry.operations.candidate_evaluations;
                        total_dense_candidates += dense_candidates;
                        worst_order_energy_drift = std::max(
                            worst_order_energy_drift, order_energy_drift
                        );
                        minimum_contradiction_ratio = std::min(
                            minimum_contradiction_ratio, contradiction_ratio
                        );
                        maximum_reduction = std::max(
                            maximum_reduction, reduction
                        );
                        relations << std::setprecision(12)
                            << case_id << ',' << (assignment_drift ? 1 : 0)
                            << ',' << order_energy_drift << ','
                            << (query_drift ? 1 : 0) << ',' << reduction
                            << ',' << contradiction_ratio << ','
                            << (ops_ok ? 1 : 0) << '\n';

                        if (
                            !sample_written && nodes == 16 && states == 3
                            && distractors == 4 && seed == 17
                        ) {
                            std::ofstream clean_json(
                                output / "sample_clean_output.json"
                            );
                            sera::write_categorical_result_json(
                                clean_json, case_id, generated.graph, clean
                            );
                            std::ofstream contradiction_json(
                                output / "sample_contradiction_output.json"
                            );
                            sera::write_categorical_result_json(
                                contradiction_json, case_id,
                                contradiction_graph, contradiction
                            );
                            std::ofstream trace(
                                output / "sample_settlement_trace.csv"
                            );
                            trace
                                << "sweep,max_message_residual,"
                                   "total_message_residual,"
                                   "provisional_energy,committed_messages,"
                                   "skipped_messages,elapsed_ns\n";
                            for (const auto& item : clean.telemetry.trace) {
                                trace << item.sweep << ','
                                    << item.max_message_residual << ','
                                    << item.total_message_residual << ','
                                    << item.provisional_energy << ','
                                    << item.committed_messages << ','
                                    << item.skipped_messages << ','
                                    << item.elapsed_nanoseconds << '\n';
                            }
                            sample_written = true;
                        }
                    }
                }
            }
        }

        std::uint32_t particle_cases = 0;
        std::uint32_t particle_successes = 0;
        for (std::uint32_t states : state_counts) {
            for (std::uint64_t seed : seeds) {
                const std::uint32_t nodes = 24;
                const std::string case_id =
                    "ambiguous_n24_k" + std::to_string(states)
                    + "_s" + std::to_string(seed);
                const auto generated = generate_ambiguous(
                    nodes, states, seed
                );
                sera::ParticleOptions particle_options;
                particle_options.budget = 8;
                particle_options.entropy_split_threshold = 0.5;
                const sera::ShapeParticleSolver particle_solver(
                    options, particle_options
                );
                const auto result = particle_solver.solve(generated.graph);
                const double coverage =
                    static_cast<double>(result.telemetry.unique_modes)
                    / static_cast<double>(states);
                const bool all_zero = std::all_of(
                    result.particles.begin(), result.particles.end(),
                    [](const sera::ShapeParticle& particle) {
                        return particle.original_energy <= 1.0e-9
                            && particle.result.telemetry.violated_factors == 0;
                    }
                );
                const bool success = coverage >= 1.0 && all_zero;
                ++particle_cases;
                particle_successes += success ? 1U : 0U;
                particles_csv << std::setprecision(12)
                    << case_id << ',' << nodes << ',' << states << ','
                    << result.base.telemetry.max_entropy << ','
                    << result.telemetry.split_node << ','
                    << result.telemetry.particles_spawned << ','
                    << result.telemetry.unique_modes << ',' << coverage
                    << ',' << (all_zero ? 1 : 0) << ','
                    << result.telemetry.aggregate_candidate_evaluations
                    << ','
                    << result.telemetry.aggregate_message_proposals << ','
                    << static_cast<double>(
                        result.telemetry.wall_nanoseconds
                    ) / 1000.0 << '\n';
                if (states == 5 && seed == 17) {
                    std::ofstream modes(output / "sample_particle_modes.json");
                    modes << "{\n  \"case_id\": \"" << case_id
                          << "\",\n  \"split_node\": "
                          << result.telemetry.split_node
                          << ",\n  \"modes\": [\n";
                    for (std::size_t index = 0;
                         index < result.particles.size(); ++index) {
                        const auto& particle = result.particles[index];
                        modes << "    {\"clamped_state\": "
                              << particle.clamped_state
                              << ", \"energy\": "
                              << particle.original_energy
                              << ", \"assignment\": [";
                        for (std::size_t node = 0;
                             node < particle.result.assignment.size();
                             ++node) {
                            if (node != 0) {
                                modes << ", ";
                            }
                            modes << particle.result.assignment[node];
                        }
                        modes << "]}";
                        if (index + 1 != result.particles.size()) {
                            modes << ',';
                        }
                        modes << '\n';
                    }
                    modes << "  ]\n}\n";
                }
            }
        }

        std::uint32_t tree_oracle_cases = 0;
        std::uint32_t tree_oracle_matches = 0;
        std::uint32_t loop_oracle_cases = 0;
        std::uint32_t loop_oracle_matches = 0;
        std::uint32_t loop_particle_matches = 0;
        double tree_gap_total = 0.0;
        double loop_gap_total = 0.0;
        double tree_gap_max = 0.0;
        double loop_gap_max = 0.0;
        for (bool loopy : {false, true}) {
            for (std::uint32_t states : {2U, 3U}) {
                for (std::uint64_t seed = 1; seed <= 12; ++seed) {
                    const std::uint32_t nodes = states == 2 ? 8 : 7;
                    const auto graph = generate_oracle_shape(
                        nodes, states, 1009 * seed + states, loopy
                    );
                    const auto inferred = solver.solve(graph);
                    const auto [exact_energy, exact_assignment] =
                        exact_enumeration(graph);
                    const double gap = std::max(
                        0.0,
                        inferred.telemetry.final_energy - exact_energy
                    );
                    const bool exact =
                        gap <= 1.0e-8
                        && inferred.assignment == exact_assignment;
                    double particle_energy =
                        inferred.telemetry.final_energy;
                    std::uint32_t particle_modes = 0;
                    if (loopy) {
                        sera::ParticleOptions oracle_particle_options;
                        oracle_particle_options.budget = states * states;
                        oracle_particle_options.entropy_split_threshold = 0.5;
                        const sera::ShapeParticleSolver oracle_particle_solver(
                            options, oracle_particle_options
                        );
                        const auto particle_result =
                            oracle_particle_solver.solve(graph);
                        particle_modes =
                            particle_result.telemetry.unique_modes;
                        for (const auto& particle :
                             particle_result.particles) {
                            particle_energy = std::min(
                                particle_energy, particle.original_energy
                            );
                        }
                    }
                    const double particle_gap = std::max(
                        0.0, particle_energy - exact_energy
                    );
                    const bool particle_exact = particle_gap <= 1.0e-8;
                    if (loopy) {
                        ++loop_oracle_cases;
                        loop_oracle_matches += exact ? 1U : 0U;
                        loop_particle_matches += particle_exact ? 1U : 0U;
                        loop_gap_total += gap;
                        loop_gap_max = std::max(loop_gap_max, gap);
                    } else {
                        ++tree_oracle_cases;
                        tree_oracle_matches += exact ? 1U : 0U;
                        tree_gap_total += gap;
                        tree_gap_max = std::max(tree_gap_max, gap);
                    }
                    oracle_csv << std::setprecision(12)
                        << (loopy ? "loop_" : "tree_")
                        << "k" << states << "_s" << seed << ','
                        << (loopy ? "loopy" : "tree") << ','
                        << nodes << ',' << states << ','
                        << (inferred.telemetry.converged ? 1 : 0) << ','
                        << inferred.telemetry.final_energy << ','
                        << exact_energy << ',' << gap << ','
                        << (exact ? 1 : 0) << ','
                        << inferred.telemetry.sweeps << ','
                        << inferred.telemetry.operations.candidate_evaluations
                        << ','
                        << static_cast<double>(
                            inferred.telemetry.wall_nanoseconds
                        ) / 1000.0 << ','
                        << particle_energy << ',' << particle_gap << ','
                        << (particle_exact ? 1 : 0) << ','
                        << particle_modes << '\n';
                }
            }
        }

        const std::uint32_t validation_passed =
            run_validation_tests(validation);
        const double mean_wall = clean_wall_us.empty() ? 0.0
            : std::accumulate(
                clean_wall_us.begin(), clean_wall_us.end(), 0.0
            ) / static_cast<double>(clean_wall_us.size());
        std::ofstream summary(output / "summary.json");
        summary << std::setprecision(12)
            << "{\n"
            << "  \"anchored_cases\": " << clean_cases << ",\n"
            << "  \"anchored_successes\": " << clean_successes << ",\n"
            << "  \"anchored_success_rate\": "
            << static_cast<double>(clean_successes) / clean_cases << ",\n"
            << "  \"factor_order_assignment_failures\": "
            << order_failures << ",\n"
            << "  \"worst_factor_order_energy_drift\": "
            << worst_order_energy_drift << ",\n"
            << "  \"goal_slice_query_failures\": "
            << slice_failures << ",\n"
            << "  \"maximum_factor_reduction\": "
            << maximum_reduction << ",\n"
            << "  \"contradiction_separations\": "
            << contradiction_separations << ",\n"
            << "  \"minimum_contradiction_energy_ratio\": "
            << minimum_contradiction_ratio << ",\n"
            << "  \"operation_accounting_failures\": "
            << ops_failures << ",\n"
            << "  \"active_frontier_candidate_reduction\": "
            << (
                total_dense_candidates == 0 ? 0.0
                : 1.0 - static_cast<double>(total_actual_candidates)
                    / static_cast<double>(total_dense_candidates)
            ) << ",\n"
            << "  \"particle_cases\": " << particle_cases << ",\n"
            << "  \"particle_successes\": " << particle_successes << ",\n"
            << "  \"tree_oracle_cases\": " << tree_oracle_cases << ",\n"
            << "  \"tree_oracle_exact_matches\": "
            << tree_oracle_matches << ",\n"
            << "  \"tree_oracle_mean_gap\": "
            << tree_gap_total / tree_oracle_cases << ",\n"
            << "  \"tree_oracle_max_gap\": " << tree_gap_max << ",\n"
            << "  \"loopy_oracle_cases\": " << loop_oracle_cases << ",\n"
            << "  \"loopy_oracle_exact_matches\": "
            << loop_oracle_matches << ",\n"
            << "  \"loopy_oracle_particle_exact_matches\": "
            << loop_particle_matches << ",\n"
            << "  \"loopy_oracle_mean_gap\": "
            << loop_gap_total / loop_oracle_cases << ",\n"
            << "  \"loopy_oracle_max_gap\": " << loop_gap_max << ",\n"
            << "  \"validation_tests_passed\": "
            << validation_passed << ",\n"
            << "  \"clean_wall_us_mean\": " << mean_wall << ",\n"
            << "  \"clean_wall_us_p50\": "
            << percentile(clean_wall_us, 0.50) << ",\n"
            << "  \"clean_wall_us_p95\": "
            << percentile(clean_wall_us, 0.95) << ",\n"
            << "  \"clean_wall_us_p99\": "
            << percentile(clean_wall_us, 0.99) << "\n"
            << "}\n";
        std::cout << "S1 stress suite complete: " << clean_successes
                  << '/' << clean_cases << " anchored, "
                  << particle_successes << '/' << particle_cases
                  << " particle, " << validation_passed
                  << "/5 validation tests\n";
        return (
            clean_successes == clean_cases
            && order_failures == 0
            && slice_failures == 0
            && contradiction_separations == clean_cases
            && ops_failures == 0
            && particle_successes == particle_cases
            && validation_passed == 5
        ) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

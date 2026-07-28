#include "sera/shape_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct GeneratedCase {
    sera::ShapeGraph graph;
    std::vector<float> truth;
    std::uint32_t query_node{};
    std::size_t relevant_nodes{};
    std::size_t relevant_factors{};
    bool contradictory{};
};

struct CaseMetrics {
    std::string case_id;
    std::size_t nodes{};
    std::size_t factors{};
    std::size_t sliced_nodes{};
    std::size_t sliced_factors{};
    std::uint32_t dimensions{};
    double rmse{};
    double query_drift{};
    double order_drift{};
    double final_energy{};
    double wall_microseconds{};
    std::uint32_t iterations{};
    bool converged{};
    bool contradictory{};
};

std::vector<float> node_value(
    std::span<const float> values,
    std::uint32_t node,
    std::uint32_t dimensions
) {
    const std::size_t begin = static_cast<std::size_t>(node) * dimensions;
    return {
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() + static_cast<std::ptrdiff_t>(begin + dimensions)
    };
}

void add_anchor(
    sera::ShapeGraph& graph,
    std::uint32_t node,
    std::span<const float> value,
    float weight = 10.0F
) {
    const std::array terms{sera::FactorTerm{node, 1.0F}};
    graph.add_factor(terms, value, weight);
}

void add_difference(
    sera::ShapeGraph& graph,
    std::uint32_t source,
    std::uint32_t destination,
    std::span<const float> delta,
    float weight = 1.0F
) {
    const std::array terms{
        sera::FactorTerm{destination, 1.0F},
        sera::FactorTerm{source, -1.0F}
    };
    graph.add_factor(terms, delta, weight);
}

GeneratedCase generate_case(
    std::uint64_t seed,
    std::size_t relevant_nodes,
    std::size_t distractor_multiplier,
    std::uint32_t dimensions,
    bool contradictory
) {
    if (relevant_nodes < 2) {
        throw std::invalid_argument("At least two relevant nodes are required");
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> value_dist(-3.0F, 3.0F);
    GeneratedCase generated;
    generated.graph.dimensions = dimensions;
    generated.relevant_nodes = relevant_nodes;
    generated.contradictory = contradictory;

    const std::size_t distractor_components = distractor_multiplier;
    const std::size_t distractor_nodes_per_component = relevant_nodes;
    const std::size_t total_nodes =
        relevant_nodes + distractor_components * distractor_nodes_per_component;
    generated.truth.resize(total_nodes * dimensions);
    for (std::size_t node = 0; node < total_nodes; ++node) {
        generated.graph.add_node(
            node < relevant_nodes ? "reasoning_slot" : "distractor_slot"
        );
        for (std::size_t d = 0; d < dimensions; ++d) {
            generated.truth[node * dimensions + d] = value_dist(rng);
        }
    }

    const auto connect_component = [&](std::size_t begin, std::size_t count) {
        const auto anchor = node_value(
            generated.truth, static_cast<std::uint32_t>(begin), dimensions
        );
        add_anchor(
            generated.graph, static_cast<std::uint32_t>(begin), anchor
        );
        for (std::size_t local = 1; local < count; ++local) {
            std::uniform_int_distribution<std::size_t> parent_dist(0, local - 1);
            const std::size_t parent_local = parent_dist(rng);
            const std::uint32_t parent =
                static_cast<std::uint32_t>(begin + parent_local);
            const std::uint32_t child =
                static_cast<std::uint32_t>(begin + local);
            std::vector<float> delta(dimensions);
            for (std::size_t d = 0; d < dimensions; ++d) {
                delta[d] =
                    generated.truth[static_cast<std::size_t>(child) * dimensions + d]
                    - generated.truth[static_cast<std::size_t>(parent) * dimensions + d];
            }
            add_difference(generated.graph, parent, child, delta);
        }
        // Redundant chords create branch/merge constraints without changing truth.
        const std::size_t chords = count / 3;
        for (std::size_t chord = 0; chord < chords; ++chord) {
            std::uniform_int_distribution<std::size_t> node_dist(0, count - 1);
            std::size_t a = node_dist(rng);
            std::size_t b = node_dist(rng);
            if (a == b) {
                b = (b + 1) % count;
            }
            const std::uint32_t source = static_cast<std::uint32_t>(begin + a);
            const std::uint32_t destination =
                static_cast<std::uint32_t>(begin + b);
            std::vector<float> delta(dimensions);
            for (std::size_t d = 0; d < dimensions; ++d) {
                delta[d] =
                    generated.truth[
                        static_cast<std::size_t>(destination) * dimensions + d
                    ] - generated.truth[
                        static_cast<std::size_t>(source) * dimensions + d
                    ];
            }
            add_difference(generated.graph, source, destination, delta, 0.5F);
        }
    };

    connect_component(0, relevant_nodes);
    generated.relevant_factors = generated.graph.factors.size();
    for (std::size_t component = 0;
         component < distractor_components;
         ++component) {
        connect_component(
            relevant_nodes + component * distractor_nodes_per_component,
            distractor_nodes_per_component
        );
    }
    generated.query_node = static_cast<std::uint32_t>(relevant_nodes - 1);

    if (contradictory) {
        // Corrupt one redundant or tree relation in the relevant component.
        std::uniform_int_distribution<std::size_t> factor_dist(
            1, generated.relevant_factors - 1
        );
        auto& factor = generated.graph.factors[factor_dist(rng)];
        for (std::size_t d = 0; d < dimensions; ++d) {
            factor.target[d] += (d % 2 == 0 ? 1.75F : -1.25F);
        }
    }
    generated.graph.validate();
    return generated;
}

double rmse_relevant(
    const GeneratedCase& generated,
    const sera::SolveResult& result
) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t node = 0; node < generated.relevant_nodes; ++node) {
        for (std::size_t d = 0; d < generated.graph.dimensions; ++d) {
            const std::size_t index =
                node * generated.graph.dimensions + d;
            const double error =
                static_cast<double>(result.state[index])
                - static_cast<double>(generated.truth[index]);
            sum += error * error;
            ++count;
        }
    }
    return std::sqrt(sum / static_cast<double>(count));
}

double max_node_drift(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t lhs_node,
    std::uint32_t rhs_node,
    std::uint32_t dimensions
) {
    double drift = 0.0;
    for (std::size_t d = 0; d < dimensions; ++d) {
        drift = std::max(
            drift,
            std::abs(
                static_cast<double>(
                    lhs[static_cast<std::size_t>(lhs_node) * dimensions + d]
                ) - static_cast<double>(
                    rhs[static_cast<std::size_t>(rhs_node) * dimensions + d]
                )
            )
        );
    }
    return drift;
}

void shuffle_factors(sera::ShapeGraph& graph, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(graph.factors.begin(), graph.factors.end(), rng);
    for (std::size_t i = 0; i < graph.factors.size(); ++i) {
        graph.factors[i].id = static_cast<std::uint32_t>(i);
    }
}

void write_summary_header(std::ostream& out) {
    out << "case_id,nodes,factors,sliced_nodes,sliced_factors,dimensions,"
           "rmse,query_drift,order_drift,final_energy,wall_microseconds,"
           "iterations,converged,contradictory\n";
}

void write_summary(std::ostream& out, const CaseMetrics& m) {
    out << m.case_id << ',' << m.nodes << ',' << m.factors << ','
        << m.sliced_nodes << ',' << m.sliced_factors << ',' << m.dimensions
        << ',' << std::setprecision(12) << m.rmse << ',' << m.query_drift
        << ',' << m.order_drift << ',' << m.final_energy << ','
        << m.wall_microseconds << ',' << m.iterations << ','
        << (m.converged ? 1 : 0) << ',' << (m.contradictory ? 1 : 0)
        << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output =
            argc > 1 ? argv[1] : "results";
        std::filesystem::create_directories(output);
        std::ofstream summary(output / "stress_summary.csv");
        std::ofstream trace(output / "iteration_trace.csv");
        std::ofstream skeleton_file(output / "sample_skeleton.json");
        if (!summary || !trace || !skeleton_file) {
            throw std::runtime_error("Could not open output files");
        }
        write_summary_header(summary);
        sera::write_trace_csv_header(trace);

        sera::SolveOptions options;
        options.max_iterations = 4096;
        options.relative_tolerance = 1.0e-10;
        options.absolute_tolerance = 1.0e-11;
        options.l2_regularization = 1.0e-9;
        options.trace_stride = 8;
        sera::EquilibriumSolver solver(options);

        const std::array<std::size_t, 4> sizes{8, 32, 128, 256};
        const std::array<std::size_t, 3> distractors{0, 1, 4};
        const std::array<std::uint32_t, 2> dimensions{1, 4};
        std::vector<CaseMetrics> all_metrics;
        std::size_t case_index = 0;
        bool wrote_skeleton = false;

        for (const auto dims : dimensions) {
            for (const auto size : sizes) {
                for (const auto distractor : distractors) {
                    for (const bool contradictory : {false, true}) {
                        for (std::uint64_t repeat = 0; repeat < 6; ++repeat) {
                            const std::uint64_t seed =
                                0xC0FFEEULL + repeat * 7919ULL
                                + size * 101ULL + distractor * 1009ULL
                                + dims * 65537ULL
                                + (contradictory ? 99991ULL : 0ULL);
                            auto generated = generate_case(
                                seed, size, distractor, dims, contradictory
                            );
                            const auto full = solver.solve(generated.graph);
                            auto slice = sera::extract_goal_shape(
                                generated.graph, generated.query_node
                            );
                            const auto sliced = solver.solve(slice.graph);

                            auto shuffled_graph = generated.graph;
                            shuffle_factors(shuffled_graph, seed ^ 0xABCDEFULL);
                            const auto shuffled = solver.solve(shuffled_graph);
                            const double query_drift = max_node_drift(
                                full.state,
                                sliced.state,
                                generated.query_node,
                                slice.local_query_node,
                                dims
                            );
                            const double order_drift = max_node_drift(
                                full.state,
                                shuffled.state,
                                generated.query_node,
                                generated.query_node,
                                dims
                            );
                            std::ostringstream name;
                            name << "d" << dims << "_n" << size
                                 << "_x" << distractor
                                 << (contradictory ? "_conflict_" : "_clean_")
                                 << repeat;
                            CaseMetrics metrics{
                                name.str(),
                                generated.graph.node_count(),
                                generated.graph.factors.size(),
                                slice.graph.node_count(),
                                slice.graph.factors.size(),
                                dims,
                                rmse_relevant(generated, full),
                                query_drift,
                                order_drift,
                                full.telemetry.final_energy,
                                static_cast<double>(full.telemetry.wall_nanoseconds)
                                    / 1000.0,
                                full.telemetry.iterations,
                                full.telemetry.converged,
                                contradictory
                            };
                            write_summary(summary, metrics);
                            all_metrics.push_back(metrics);
                            if (
                                !wrote_skeleton && !contradictory
                                && size == 32 && distractor == 4 && dims == 4
                            ) {
                                std::vector<std::uint32_t> support(
                                    slice.local_to_global_factor.begin(),
                                    slice.local_to_global_factor.end()
                                );
                                const auto skeleton = sera::compile_skeleton(
                                    slice.graph,
                                    sliced,
                                    slice.local_query_node,
                                    support
                                );
                                sera::write_skeleton_json(
                                    skeleton_file, skeleton
                                );
                                sera::write_trace_csv(
                                    trace, metrics.case_id, generated.graph, full
                                );
                                wrote_skeleton = true;
                            }
                            ++case_index;
                        }
                    }
                }
            }
        }

        std::size_t clean_count = 0;
        std::size_t clean_success = 0;
        double clean_energy_sum = 0.0;
        double conflict_energy_sum = 0.0;
        std::size_t conflict_count = 0;
        double max_order_drift = 0.0;
        double max_slice_drift = 0.0;
        double max_clean_rmse = 0.0;
        double reduction_sum_4x = 0.0;
        std::size_t reduction_count_4x = 0;
        for (const auto& m : all_metrics) {
            max_order_drift = std::max(max_order_drift, m.order_drift);
            max_slice_drift = std::max(max_slice_drift, m.query_drift);
            if (!m.contradictory) {
                ++clean_count;
                clean_energy_sum += m.final_energy;
                max_clean_rmse = std::max(max_clean_rmse, m.rmse);
                if (m.converged && m.rmse <= 1.0e-3) {
                    ++clean_success;
                }
            } else {
                ++conflict_count;
                conflict_energy_sum += m.final_energy;
            }
            if (m.nodes >= m.sliced_nodes * 5 && m.factors != 0) {
                reduction_sum_4x +=
                    1.0 - static_cast<double>(m.sliced_factors)
                        / static_cast<double>(m.factors);
                ++reduction_count_4x;
            }
        }
        const double clean_mean_energy =
            clean_energy_sum / static_cast<double>(clean_count);
        const double conflict_mean_energy =
            conflict_energy_sum / static_cast<double>(conflict_count);
        const double energy_ratio = conflict_mean_energy
            / std::max(clean_mean_energy, 1.0e-18);
        const double clean_success_rate =
            static_cast<double>(clean_success) / static_cast<double>(clean_count);
        const double mean_4x_reduction =
            reduction_sum_4x / static_cast<double>(reduction_count_4x);
        const bool passed =
            clean_success_rate >= 0.99
            && max_clean_rmse <= 1.0e-3
            && max_order_drift <= 1.0e-4
            && max_slice_drift <= 1.0e-4
            && energy_ratio >= 5.0
            && mean_4x_reduction >= 0.70;

        std::ofstream aggregate(output / "aggregate.json");
        aggregate << "{\n"
            << "  \"status\": \"" << (passed ? "success" : "failed") << "\",\n"
            << "  \"cases\": " << all_metrics.size() << ",\n"
            << "  \"clean_success_rate\": " << std::setprecision(12)
            << clean_success_rate << ",\n"
            << "  \"max_clean_rmse\": " << max_clean_rmse << ",\n"
            << "  \"max_factor_order_drift\": " << max_order_drift << ",\n"
            << "  \"max_goal_slice_query_drift\": " << max_slice_drift << ",\n"
            << "  \"mean_clean_energy\": " << clean_mean_energy << ",\n"
            << "  \"mean_conflict_energy\": " << conflict_mean_energy << ",\n"
            << "  \"contradiction_energy_ratio\": " << energy_ratio << ",\n"
            << "  \"mean_4x_distractor_factor_reduction\": "
            << mean_4x_reduction << "\n"
            << "}\n";

        std::cout << "SERA S0 stress complete\n"
            << "status=" << (passed ? "success" : "failed") << '\n'
            << "cases=" << all_metrics.size() << '\n'
            << "clean_success_rate=" << clean_success_rate << '\n'
            << "max_clean_rmse=" << max_clean_rmse << '\n'
            << "max_order_drift=" << max_order_drift << '\n'
            << "max_goal_slice_query_drift=" << max_slice_drift << '\n'
            << "contradiction_energy_ratio=" << energy_ratio << '\n'
            << "mean_4x_factor_reduction=" << mean_4x_reduction << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

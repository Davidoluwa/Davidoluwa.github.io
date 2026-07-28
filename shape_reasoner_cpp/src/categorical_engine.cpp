#include "sera/categorical_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace sera {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t bytes) {
    const auto* ptr = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(ptr[i]);
        hash *= kFnvPrime;
    }
}

struct Layout {
    std::vector<std::size_t> node_offset;
    std::vector<std::size_t> left_message_offset;
    std::vector<std::size_t> right_message_offset;
    std::size_t states{};
    std::size_t message_states{};
};

Layout make_layout(const CategoricalShape& graph) {
    Layout layout;
    layout.node_offset.resize(graph.nodes.size() + 1, 0);
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        layout.node_offset[i + 1] =
            layout.node_offset[i] + graph.nodes[i].unary_cost.size();
    }
    layout.states = layout.node_offset.back();
    layout.left_message_offset.resize(graph.factors.size(), 0);
    layout.right_message_offset.resize(graph.factors.size(), 0);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < graph.factors.size(); ++i) {
        const auto& factor = graph.factors[i];
        layout.left_message_offset[i] = cursor;
        cursor += graph.nodes[factor.left].unary_cost.size();
        layout.right_message_offset[i] = cursor;
        cursor += graph.nodes[factor.right].unary_cost.size();
    }
    layout.message_states = cursor;
    return layout;
}

void decode_assignment(
    const CategoricalShape& graph,
    const Layout& layout,
    std::span<const double> beliefs,
    std::span<std::uint32_t> assignment,
    PrimitiveOperations* operations
) {
    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        const std::size_t begin = layout.node_offset[node];
        const std::size_t count = graph.nodes[node].unary_cost.size();
        std::size_t best = 0;
        for (std::size_t state = 1; state < count; ++state) {
            if (operations != nullptr) {
                ++operations->scalar_comparisons;
            }
            if (beliefs[begin + state] < beliefs[begin + best]) {
                best = state;
            }
        }
        assignment[node] = static_cast<std::uint32_t>(best);
    }
}

double entropy_of_node(
    std::span<const double> belief,
    double temperature,
    double& confidence,
    PrimitiveOperations& operations
) {
    const double minimum = *std::min_element(belief.begin(), belief.end());
    std::vector<double> probabilities(belief.size(), 0.0);
    double total = 0.0;
    for (std::size_t state = 0; state < belief.size(); ++state) {
        probabilities[state] = std::exp(
            -(belief[state] - minimum) / temperature
        );
        ++operations.exponentials;
        operations.scalar_add_subtracts += 2;
        ++operations.scalar_multiplies;
        total += probabilities[state];
        ++operations.scalar_add_subtracts;
    }
    double entropy = 0.0;
    confidence = 0.0;
    for (double probability : probabilities) {
        const double normalized = probability / total;
        ++operations.scalar_multiplies;
        confidence = std::max(confidence, normalized);
        ++operations.scalar_comparisons;
        if (normalized > 0.0) {
            entropy -= normalized * std::log(normalized);
            ++operations.scalar_multiplies;
            ++operations.scalar_add_subtracts;
        }
    }
    return entropy;
}

}  // namespace

std::uint32_t CategoricalShape::add_node(
    std::string type,
    std::span<const double> unary_cost
) {
    CategoricalNode node;
    node.id = static_cast<std::uint32_t>(nodes.size());
    node.type = std::move(type);
    node.unary_cost.assign(unary_cost.begin(), unary_cost.end());
    nodes.push_back(std::move(node));
    return static_cast<std::uint32_t>(nodes.size() - 1);
}

std::uint32_t CategoricalShape::add_factor(
    std::uint32_t left,
    std::uint32_t right,
    std::string relation,
    std::span<const double> row_major_cost
) {
    PairwiseFactor factor;
    factor.id = static_cast<std::uint32_t>(factors.size());
    factor.left = left;
    factor.right = right;
    factor.relation = std::move(relation);
    factor.cost.assign(row_major_cost.begin(), row_major_cost.end());
    factors.push_back(std::move(factor));
    return static_cast<std::uint32_t>(factors.size() - 1);
}

void CategoricalShape::validate() const {
    if (nodes.empty()) {
        throw std::invalid_argument("CategoricalShape requires a node");
    }
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        if (node.id != i) {
            throw std::invalid_argument("Node ids must be dense and stable");
        }
        if (node.type.empty()) {
            throw std::invalid_argument("Node type must not be empty");
        }
        if (node.unary_cost.size() < 2) {
            throw std::invalid_argument("Categorical nodes require at least two states");
        }
        for (double cost : node.unary_cost) {
            if (!std::isfinite(cost)) {
                throw std::invalid_argument("Unary costs must be finite");
            }
        }
    }
    std::unordered_set<std::uint64_t> identities;
    identities.reserve(factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        const auto& factor = factors[i];
        if (factor.id != i) {
            throw std::invalid_argument("Factor ids must be dense and stable");
        }
        if (
            factor.left >= nodes.size() || factor.right >= nodes.size()
            || factor.left == factor.right
        ) {
            throw std::invalid_argument("Pairwise factor endpoints are invalid");
        }
        if (factor.relation.empty()) {
            throw std::invalid_argument("Factor relation must not be empty");
        }
        const std::size_t expected =
            nodes[factor.left].unary_cost.size()
            * nodes[factor.right].unary_cost.size();
        if (factor.cost.size() != expected) {
            throw std::invalid_argument("Pairwise cost matrix size mismatch");
        }
        for (double cost : factor.cost) {
            if (!std::isfinite(cost)) {
                throw std::invalid_argument("Pairwise costs must be finite");
            }
        }
        const std::uint64_t identity =
            (static_cast<std::uint64_t>(factor.left) << 32U)
            | static_cast<std::uint64_t>(factor.right);
        const std::uint64_t reverse =
            (static_cast<std::uint64_t>(factor.right) << 32U)
            | static_cast<std::uint64_t>(factor.left);
        if (
            identities.contains(identity) || identities.contains(reverse)
        ) {
            throw std::invalid_argument(
                "Parallel pairwise factors must be fused into one cost matrix"
            );
        }
        identities.insert(identity);
    }
}

CategoricalShapeSolver::CategoricalShapeSolver(
    CategoricalSolveOptions options
) : options_(options) {
    if (options_.max_sweeps == 0) {
        throw std::invalid_argument("max_sweeps must be positive");
    }
    if (
        !(options_.damping > 0.0 && options_.damping <= 1.0)
        || !std::isfinite(options_.damping)
    ) {
        throw std::invalid_argument("damping must be finite in (0, 1]");
    }
    if (
        !(options_.residual_gate >= 0.0)
        || !std::isfinite(options_.residual_gate)
    ) {
        throw std::invalid_argument("residual_gate must be finite and non-negative");
    }
    if (
        !(options_.output_temperature > 0.0)
        || !std::isfinite(options_.output_temperature)
    ) {
        throw std::invalid_argument("output_temperature must be finite and positive");
    }
    if (
        !(options_.violation_tolerance >= 0.0)
        || !std::isfinite(options_.violation_tolerance)
    ) {
        throw std::invalid_argument(
            "violation_tolerance must be finite and non-negative"
        );
    }
}

CategoricalSolveResult CategoricalShapeSolver::solve(
    const CategoricalShape& graph
) const {
    graph.validate();
    const auto wall_start = Clock::now();
    const Layout layout = make_layout(graph);
    std::vector<double> messages(layout.message_states, 0.0);
    std::vector<double> proposals(layout.message_states, 0.0);
    std::vector<double> beliefs(layout.states, 0.0);
    std::vector<std::uint32_t> provisional(graph.nodes.size(), 0);
    std::vector<unsigned char> active(graph.factors.size() * 2, 1);
    std::vector<unsigned char> next_active(graph.factors.size() * 2, 0);
    std::vector<std::vector<std::size_t>> outgoing(graph.nodes.size());
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        outgoing[graph.factors[index].left].push_back(index * 2);
        outgoing[graph.factors[index].right].push_back(index * 2 + 1);
    }
    CategoricalSolveResult result;
    auto& operations = result.telemetry.operations;
    result.telemetry.graph_hash = hash_categorical_graph(graph);
    result.telemetry.workspace_bytes =
        layout.message_states * sizeof(double) * 2
        + layout.states * sizeof(double)
        + graph.nodes.size() * sizeof(std::uint32_t)
        + active.size() * sizeof(unsigned char) * 2;
    result.telemetry.trace.reserve(
        options_.max_sweeps / std::max(1U, options_.trace_stride) + 1
    );

    bool converged = graph.factors.empty();
    double last_max_residual = 0.0;
    double last_total_residual = 0.0;
    std::uint64_t last_commits = 0;
    std::uint64_t last_skips = 0;
    std::uint32_t completed_sweeps = 0;

    for (
        std::uint32_t sweep = 0;
        sweep < options_.max_sweeps && !converged;
        ++sweep
    ) {
        const auto sweep_start = Clock::now();
        for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
            const auto& unary = graph.nodes[node].unary_cost;
            std::copy(
                unary.begin(), unary.end(),
                beliefs.begin() + static_cast<std::ptrdiff_t>(
                    layout.node_offset[node]
                )
            );
        }
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            const auto& factor = graph.factors[index];
            const std::size_t left_states =
                graph.nodes[factor.left].unary_cost.size();
            const std::size_t right_states =
                graph.nodes[factor.right].unary_cost.size();
            const std::size_t left_belief = layout.node_offset[factor.left];
            const std::size_t right_belief = layout.node_offset[factor.right];
            const std::size_t left_message = layout.left_message_offset[index];
            const std::size_t right_message = layout.right_message_offset[index];
            for (std::size_t state = 0; state < left_states; ++state) {
                beliefs[left_belief + state] += messages[left_message + state];
                ++operations.scalar_add_subtracts;
            }
            for (std::size_t state = 0; state < right_states; ++state) {
                beliefs[right_belief + state] += messages[right_message + state];
                ++operations.scalar_add_subtracts;
            }
        }

        double sweep_max = 0.0;
        double sweep_total = 0.0;
        std::uint64_t sweep_commits = 0;
        std::uint64_t sweep_skips = 0;
        std::fill(next_active.begin(), next_active.end(), 0);
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            const auto& factor = graph.factors[index];
            const std::size_t left_states =
                graph.nodes[factor.left].unary_cost.size();
            const std::size_t right_states =
                graph.nodes[factor.right].unary_cost.size();
            const std::size_t left_belief = layout.node_offset[factor.left];
            const std::size_t right_belief = layout.node_offset[factor.right];
            const std::size_t left_message = layout.left_message_offset[index];
            const std::size_t right_message = layout.right_message_offset[index];

            if (active[index * 2] != 0) {
                for (std::size_t destination = 0; destination < right_states;
                     ++destination) {
                    double best = std::numeric_limits<double>::infinity();
                    for (std::size_t source = 0; source < left_states; ++source) {
                        const double cavity =
                            beliefs[left_belief + source]
                            - messages[left_message + source];
                        const double candidate =
                            factor.cost[source * right_states + destination]
                            + cavity;
                        operations.scalar_add_subtracts += 2;
                        ++operations.candidate_evaluations;
                        ++operations.scalar_comparisons;
                        best = std::min(best, candidate);
                    }
                    proposals[right_message + destination] = best;
                }
                double minimum = proposals[right_message];
                for (std::size_t state = 1; state < right_states; ++state) {
                    ++operations.scalar_comparisons;
                    minimum = std::min(
                        minimum, proposals[right_message + state]
                    );
                }
                double message_residual = 0.0;
                for (std::size_t state = 0; state < right_states; ++state) {
                    const std::size_t at = right_message + state;
                    const double normalized = proposals[at] - minimum;
                    const double next =
                        messages[at]
                        + options_.damping * (normalized - messages[at]);
                    operations.scalar_add_subtracts += 3;
                    ++operations.scalar_multiplies;
                    message_residual = std::max(
                        message_residual, std::abs(next - messages[at])
                    );
                    proposals[at] = next;
                }
                ++operations.message_proposals;
                sweep_max = std::max(sweep_max, message_residual);
                sweep_total += message_residual;
            }

            if (active[index * 2 + 1] != 0) {
                for (std::size_t destination = 0; destination < left_states;
                     ++destination) {
                    double best = std::numeric_limits<double>::infinity();
                    for (std::size_t source = 0; source < right_states; ++source) {
                        const double cavity =
                            beliefs[right_belief + source]
                            - messages[right_message + source];
                        const double candidate =
                            factor.cost[destination * right_states + source]
                            + cavity;
                        operations.scalar_add_subtracts += 2;
                        ++operations.candidate_evaluations;
                        ++operations.scalar_comparisons;
                        best = std::min(best, candidate);
                    }
                    proposals[left_message + destination] = best;
                }
                double minimum = proposals[left_message];
                for (std::size_t state = 1; state < left_states; ++state) {
                    ++operations.scalar_comparisons;
                    minimum = std::min(
                        minimum, proposals[left_message + state]
                    );
                }
                double message_residual = 0.0;
                for (std::size_t state = 0; state < left_states; ++state) {
                    const std::size_t at = left_message + state;
                    const double normalized = proposals[at] - minimum;
                    const double next =
                        messages[at]
                        + options_.damping * (normalized - messages[at]);
                    operations.scalar_add_subtracts += 3;
                    ++operations.scalar_multiplies;
                    message_residual = std::max(
                        message_residual, std::abs(next - messages[at])
                    );
                    proposals[at] = next;
                }
                ++operations.message_proposals;
                sweep_max = std::max(sweep_max, message_residual);
                sweep_total += message_residual;
            }
        }

        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            const auto& factor = graph.factors[index];
            const std::size_t left_states =
                graph.nodes[factor.left].unary_cost.size();
            const std::size_t right_states =
                graph.nodes[factor.right].unary_cost.size();
            const std::size_t offsets[2] = {
                layout.right_message_offset[index],
                layout.left_message_offset[index]
            };
            const std::size_t counts[2] = {right_states, left_states};
            for (std::size_t direction = 0; direction < 2; ++direction) {
                const std::size_t directed = index * 2 + direction;
                if (active[directed] == 0) {
                    continue;
                }
                double residual = 0.0;
                for (std::size_t state = 0; state < counts[direction]; ++state) {
                    const std::size_t at = offsets[direction] + state;
                    residual = std::max(
                        residual, std::abs(proposals[at] - messages[at])
                    );
                }
                if (residual > options_.residual_gate) {
                    std::copy_n(
                        proposals.begin()
                            + static_cast<std::ptrdiff_t>(offsets[direction]),
                        counts[direction],
                        messages.begin()
                            + static_cast<std::ptrdiff_t>(offsets[direction])
                    );
                    ++operations.message_commits;
                    ++sweep_commits;
                    next_active[directed] = 1;
                    const std::uint32_t destination =
                        direction == 0 ? factor.right : factor.left;
                    for (std::size_t downstream : outgoing[destination]) {
                        if (downstream / 2 != index) {
                            next_active[downstream] = 1;
                        }
                    }
                } else {
                    ++operations.message_skips;
                    ++sweep_skips;
                }
            }
        }
        ++completed_sweeps;
        last_max_residual = sweep_max;
        last_total_residual = sweep_total;
        last_commits = sweep_commits;
        last_skips = sweep_skips;
        converged = std::none_of(
            next_active.begin(), next_active.end(),
            [](unsigned char item) { return item != 0; }
        );
        active.swap(next_active);

        if (
            sweep % std::max(1U, options_.trace_stride) == 0 || converged
        ) {
            decode_assignment(
                graph, layout, beliefs, provisional, &operations
            );
            result.telemetry.trace.push_back({
                sweep + 1,
                sweep_max,
                sweep_total,
                categorical_energy(graph, provisional),
                sweep_commits,
                sweep_skips,
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - sweep_start
                    ).count()
                )
            });
        }
    }

    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        const auto& unary = graph.nodes[node].unary_cost;
        std::copy(
            unary.begin(), unary.end(),
            beliefs.begin() + static_cast<std::ptrdiff_t>(
                layout.node_offset[node]
            )
        );
    }
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        const auto& factor = graph.factors[index];
        const std::size_t left_states =
            graph.nodes[factor.left].unary_cost.size();
        const std::size_t right_states =
            graph.nodes[factor.right].unary_cost.size();
        const std::size_t left_belief = layout.node_offset[factor.left];
        const std::size_t right_belief = layout.node_offset[factor.right];
        for (std::size_t state = 0; state < left_states; ++state) {
            beliefs[left_belief + state] +=
                messages[layout.left_message_offset[index] + state];
            ++operations.scalar_add_subtracts;
        }
        for (std::size_t state = 0; state < right_states; ++state) {
            beliefs[right_belief + state] +=
                messages[layout.right_message_offset[index] + state];
            ++operations.scalar_add_subtracts;
        }
    }
    result.assignment.resize(graph.nodes.size());
    decode_assignment(
        graph, layout, beliefs, result.assignment, &operations
    );
    result.belief_costs = beliefs;
    result.confidence.resize(graph.nodes.size(), 0.0);
    result.entropy.resize(graph.nodes.size(), 0.0);
    double entropy_total = 0.0;
    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        const std::size_t begin = layout.node_offset[node];
        const std::size_t count = graph.nodes[node].unary_cost.size();
        result.entropy[node] = entropy_of_node(
            std::span<const double>(beliefs.data() + begin, count),
            options_.output_temperature,
            result.confidence[node],
            operations
        );
        entropy_total += result.entropy[node];
        if (
            node == 0
            || result.entropy[node]
                > result.telemetry.max_entropy
        ) {
            result.telemetry.max_entropy = result.entropy[node];
            result.telemetry.max_entropy_node =
                static_cast<std::uint32_t>(node);
        }
    }
    result.telemetry.mean_entropy =
        entropy_total / static_cast<double>(graph.nodes.size());
    result.telemetry.converged = converged;
    result.telemetry.sweeps = completed_sweeps;
    result.telemetry.final_energy =
        categorical_energy(graph, result.assignment);
    result.telemetry.violated_factors = categorical_violations(
        graph, result.assignment, options_.violation_tolerance
    );
    operations.factor_energy_lookups += graph.factors.size();
    result.telemetry.max_message_residual = last_max_residual;
    result.telemetry.total_message_residual = last_total_residual;
    const std::uint64_t last_messages = last_commits + last_skips;
    result.telemetry.active_message_fraction = last_messages == 0
        ? 0.0
        : static_cast<double>(last_commits)
            / static_cast<double>(last_messages);
    result.telemetry.assignment_hash = hash_assignment(result.assignment);
    result.telemetry.wall_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - wall_start
        ).count()
    );
    return result;
}

ShapeParticleSolver::ShapeParticleSolver(
    CategoricalSolveOptions solve_options,
    ParticleOptions particle_options
) : solve_options_(solve_options), particle_options_(particle_options) {
    if (particle_options_.budget == 0) {
        throw std::invalid_argument("Particle budget must be positive");
    }
    if (particle_options_.max_split_depth == 0) {
        throw std::invalid_argument("Particle split depth must be positive");
    }
    if (
        !(particle_options_.entropy_split_threshold >= 0.0)
        || !(particle_options_.clamp_penalty > 0.0)
        || !std::isfinite(particle_options_.entropy_split_threshold)
        || !std::isfinite(particle_options_.clamp_penalty)
    ) {
        throw std::invalid_argument("Invalid particle options");
    }
}

ParticleResult ShapeParticleSolver::solve(
    const CategoricalShape& graph
) const {
    graph.validate();
    const auto wall_start = Clock::now();
    CategoricalShapeSolver solver(solve_options_);
    ParticleResult output;
    output.base = solver.solve(graph);
    output.telemetry.split_node = output.base.telemetry.max_entropy_node;
    output.telemetry.aggregate_candidate_evaluations =
        output.base.telemetry.operations.candidate_evaluations;
    output.telemetry.aggregate_message_proposals =
        output.base.telemetry.operations.message_proposals;
    const std::uint32_t split_node = output.telemetry.split_node;
    const std::size_t states = graph.nodes[split_node].unary_cost.size();
    const bool should_split =
        !output.base.telemetry.converged
        || output.base.entropy[split_node]
            >= particle_options_.entropy_split_threshold;
    if (!should_split) {
        ShapeParticle particle;
        particle.clamped_state = output.base.assignment[split_node];
        particle.original_energy = output.base.telemetry.final_energy;
        particle.free_energy_score = particle.original_energy;
        particle.result = output.base;
        output.particles.push_back(std::move(particle));
    } else {
        struct Candidate {
            CategoricalShape graph;
            CategoricalSolveResult result;
            std::vector<unsigned char> clamped;
            std::uint32_t first_state{};
            std::uint32_t depth{};
            double score{};
        };
        const auto score_candidate = [&](Candidate& candidate) {
            const double entropy_sum = std::accumulate(
                candidate.result.entropy.begin(),
                candidate.result.entropy.end(),
                0.0
            );
            candidate.score =
                categorical_energy(graph, candidate.result.assignment)
                + particle_options_.entropy_weight * entropy_sum
                + particle_options_.operation_weight
                    * static_cast<double>(
                        candidate.result.telemetry.operations
                            .candidate_evaluations
                    );
        };
        const auto select_split = [&](const Candidate& candidate) {
            std::uint32_t selected = 0;
            double selected_entropy = -1.0;
            for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
                if (
                    candidate.clamped[node] == 0
                    && candidate.result.entropy[node] > selected_entropy
                ) {
                    selected = static_cast<std::uint32_t>(node);
                    selected_entropy = candidate.result.entropy[node];
                }
            }
            return std::pair<std::uint32_t, double>{
                selected, selected_entropy
            };
        };
        std::vector<Candidate> frontier;
        const std::size_t initial_spawn = std::min<std::size_t>(
            states, particle_options_.budget
        );
        frontier.reserve(initial_spawn);
        for (std::size_t state = 0; state < initial_spawn; ++state) {
            Candidate candidate;
            candidate.graph = graph;
            candidate.clamped.assign(graph.nodes.size(), 0);
            candidate.clamped[split_node] = 1;
            candidate.first_state = static_cast<std::uint32_t>(state);
            candidate.depth = 1;
            for (std::size_t alternative = 0; alternative < states;
                 ++alternative) {
                if (alternative != state) {
                    candidate.graph.nodes[split_node]
                        .unary_cost[alternative] +=
                        particle_options_.clamp_penalty;
                }
            }
            candidate.result = solver.solve(candidate.graph);
            score_candidate(candidate);
            output.telemetry.aggregate_candidate_evaluations +=
                candidate.result.telemetry.operations.candidate_evaluations;
            output.telemetry.aggregate_message_proposals +=
                candidate.result.telemetry.operations.message_proposals;
            ++output.telemetry.particles_spawned;
            frontier.push_back(std::move(candidate));
        }
        std::vector<Candidate> finalized;
        for (
            std::uint32_t depth = 1;
            depth <= particle_options_.max_split_depth && !frontier.empty();
            ++depth
        ) {
            output.telemetry.max_depth_reached = std::max(
                output.telemetry.max_depth_reached, depth
            );
            std::vector<Candidate> expanded;
            for (auto& candidate : frontier) {
                const auto [next_node, next_entropy] =
                    select_split(candidate);
                const bool expand =
                    depth < particle_options_.max_split_depth
                    && (
                        !candidate.result.telemetry.converged
                        || next_entropy
                            >= particle_options_.entropy_split_threshold
                    );
                if (!expand) {
                    finalized.push_back(std::move(candidate));
                    continue;
                }
                ++output.telemetry.particles_expanded;
                const std::size_t next_states =
                    graph.nodes[next_node].unary_cost.size();
                for (std::size_t state = 0; state < next_states; ++state) {
                    Candidate child;
                    child.graph = candidate.graph;
                    child.clamped = candidate.clamped;
                    child.clamped[next_node] = 1;
                    child.first_state = candidate.first_state;
                    child.depth = depth + 1;
                    for (std::size_t alternative = 0;
                         alternative < next_states; ++alternative) {
                        if (alternative != state) {
                            child.graph.nodes[next_node]
                                .unary_cost[alternative] +=
                                particle_options_.clamp_penalty;
                        }
                    }
                    child.result = solver.solve(child.graph);
                    score_candidate(child);
                    output.telemetry.aggregate_candidate_evaluations +=
                        child.result.telemetry.operations
                            .candidate_evaluations;
                    output.telemetry.aggregate_message_proposals +=
                        child.result.telemetry.operations.message_proposals;
                    ++output.telemetry.particles_spawned;
                    expanded.push_back(std::move(child));
                }
            }
            if (expanded.size() > particle_options_.budget) {
                std::partial_sort(
                    expanded.begin(),
                    expanded.begin()
                        + static_cast<std::ptrdiff_t>(
                            particle_options_.budget
                        ),
                    expanded.end(),
                    [](const Candidate& lhs, const Candidate& rhs) {
                        return lhs.score < rhs.score;
                    }
                );
                expanded.resize(particle_options_.budget);
            }
            frontier = std::move(expanded);
        }
        for (auto& candidate : frontier) {
            finalized.push_back(std::move(candidate));
        }
        std::unordered_set<std::uint64_t> seen;
        for (auto& candidate : finalized) {
            ShapeParticle particle;
            particle.clamped_state = candidate.first_state;
            particle.result = std::move(candidate.result);
            particle.original_energy = categorical_energy(
                graph, particle.result.assignment
            );
            particle.free_energy_score = candidate.score;
            if (
                seen.insert(
                    particle.result.telemetry.assignment_hash
                ).second
            ) {
                output.particles.push_back(std::move(particle));
            }
        }
    }
    std::sort(
        output.particles.begin(), output.particles.end(),
        [](const ShapeParticle& lhs, const ShapeParticle& rhs) {
            if (lhs.free_energy_score != rhs.free_energy_score) {
                return lhs.free_energy_score < rhs.free_energy_score;
            }
            return lhs.clamped_state < rhs.clamped_state;
        }
    );
    output.telemetry.unique_modes =
        static_cast<std::uint32_t>(output.particles.size());
    output.telemetry.wall_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - wall_start
        ).count()
    );
    return output;
}

CategoricalGoalSlice extract_goal_shape(
    const CategoricalShape& source,
    std::uint32_t query_node
) {
    source.validate();
    if (query_node >= source.nodes.size()) {
        throw std::out_of_range("Categorical query node is invalid");
    }
    std::vector<std::vector<std::uint32_t>> incident(source.nodes.size());
    for (const auto& factor : source.factors) {
        incident[factor.left].push_back(factor.id);
        incident[factor.right].push_back(factor.id);
    }
    std::vector<unsigned char> node_seen(source.nodes.size(), 0);
    std::vector<unsigned char> factor_seen(source.factors.size(), 0);
    std::queue<std::uint32_t> frontier;
    frontier.push(query_node);
    node_seen[query_node] = 1;
    while (!frontier.empty()) {
        const std::uint32_t node = frontier.front();
        frontier.pop();
        for (std::uint32_t factor_id : incident[node]) {
            factor_seen[factor_id] = 1;
            const auto& factor = source.factors[factor_id];
            const std::uint32_t other =
                factor.left == node ? factor.right : factor.left;
            if (node_seen[other] == 0) {
                node_seen[other] = 1;
                frontier.push(other);
            }
        }
    }

    CategoricalGoalSlice slice;
    std::vector<std::uint32_t> global_to_local(
        source.nodes.size(), std::numeric_limits<std::uint32_t>::max()
    );
    for (std::size_t global = 0; global < source.nodes.size(); ++global) {
        if (node_seen[global] != 0) {
            global_to_local[global] = slice.graph.add_node(
                source.nodes[global].type,
                source.nodes[global].unary_cost
            );
            slice.local_to_global_node.push_back(
                static_cast<std::uint32_t>(global)
            );
        }
    }
    slice.local_query_node = global_to_local[query_node];
    for (std::size_t global = 0; global < source.factors.size(); ++global) {
        if (factor_seen[global] != 0) {
            const auto& factor = source.factors[global];
            slice.graph.add_factor(
                global_to_local[factor.left],
                global_to_local[factor.right],
                factor.relation,
                factor.cost
            );
            slice.local_to_global_factor.push_back(
                static_cast<std::uint32_t>(global)
            );
        }
    }
    return slice;
}

double categorical_energy(
    const CategoricalShape& graph,
    std::span<const std::uint32_t> assignment
) {
    if (assignment.size() != graph.nodes.size()) {
        throw std::invalid_argument("Assignment size mismatch");
    }
    double energy = 0.0;
    for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
        if (assignment[node] >= graph.nodes[node].unary_cost.size()) {
            throw std::invalid_argument("Assignment state is invalid");
        }
        energy += graph.nodes[node].unary_cost[assignment[node]];
    }
    for (const auto& factor : graph.factors) {
        const std::size_t right_states =
            graph.nodes[factor.right].unary_cost.size();
        energy += factor.cost[
            static_cast<std::size_t>(assignment[factor.left]) * right_states
            + assignment[factor.right]
        ];
    }
    return energy;
}

std::uint32_t categorical_violations(
    const CategoricalShape& graph,
    std::span<const std::uint32_t> assignment,
    double tolerance
) {
    std::uint32_t violations = 0;
    for (const auto& factor : graph.factors) {
        const std::size_t right_states =
            graph.nodes[factor.right].unary_cost.size();
        const double selected = factor.cost[
            static_cast<std::size_t>(assignment[factor.left]) * right_states
            + assignment[factor.right]
        ];
        const double minimum = *std::min_element(
            factor.cost.begin(), factor.cost.end()
        );
        if (selected > minimum + tolerance) {
            ++violations;
        }
    }
    return violations;
}

std::uint64_t hash_categorical_graph(const CategoricalShape& graph) {
    std::uint64_t hash = kFnvOffset;
    for (const auto& node : graph.nodes) {
        hash_bytes(hash, node.type.data(), node.type.size());
        for (double cost : node.unary_cost) {
            hash_bytes(hash, &cost, sizeof(cost));
        }
    }
    for (const auto& factor : graph.factors) {
        hash_bytes(hash, &factor.left, sizeof(factor.left));
        hash_bytes(hash, &factor.right, sizeof(factor.right));
        hash_bytes(hash, factor.relation.data(), factor.relation.size());
        for (double cost : factor.cost) {
            hash_bytes(hash, &cost, sizeof(cost));
        }
    }
    return hash;
}

std::uint64_t hash_assignment(
    std::span<const std::uint32_t> assignment
) {
    std::uint64_t hash = kFnvOffset;
    hash_bytes(
        hash, assignment.data(), assignment.size_bytes()
    );
    return hash;
}

void write_categorical_result_json(
    std::ostream& out,
    std::string_view case_id,
    const CategoricalShape& graph,
    const CategoricalSolveResult& result
) {
    const auto& telemetry = result.telemetry;
    const auto& operations = telemetry.operations;
    out << std::setprecision(12);
    out << "{\n  \"case_id\": \"" << case_id << "\",\n"
        << "  \"nodes\": " << graph.nodes.size() << ",\n"
        << "  \"factors\": " << graph.factors.size() << ",\n"
        << "  \"converged\": "
        << (telemetry.converged ? "true" : "false") << ",\n"
        << "  \"sweeps\": " << telemetry.sweeps << ",\n"
        << "  \"energy\": " << telemetry.final_energy << ",\n"
        << "  \"violated_factors\": " << telemetry.violated_factors << ",\n"
        << "  \"mean_entropy\": " << telemetry.mean_entropy << ",\n"
        << "  \"assignment\": [";
    for (std::size_t i = 0; i < result.assignment.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << result.assignment[i];
    }
    out << "],\n  \"confidence\": [";
    for (std::size_t i = 0; i < result.confidence.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << result.confidence[i];
    }
    out << "],\n  \"operations\": {\n"
        << "    \"message_proposals\": " << operations.message_proposals << ",\n"
        << "    \"message_commits\": " << operations.message_commits << ",\n"
        << "    \"message_skips\": " << operations.message_skips << ",\n"
        << "    \"candidate_evaluations\": "
        << operations.candidate_evaluations << ",\n"
        << "    \"scalar_add_subtracts\": "
        << operations.scalar_add_subtracts << ",\n"
        << "    \"scalar_comparisons\": "
        << operations.scalar_comparisons << ",\n"
        << "    \"scalar_multiplies\": "
        << operations.scalar_multiplies << ",\n"
        << "    \"exponentials\": " << operations.exponentials << ",\n"
        << "    \"factor_energy_lookups\": "
        << operations.factor_energy_lookups << "\n"
        << "  }\n}\n";
}

}  // namespace sera

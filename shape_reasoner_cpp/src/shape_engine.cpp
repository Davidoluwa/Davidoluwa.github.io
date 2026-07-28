#include "sera/shape_engine.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <unordered_map>
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

double dot(std::span<const double> a, std::span<const double> b) {
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

double norm(std::span<const double> x) {
    return std::sqrt(std::max(0.0, dot(x, x)));
}

void apply_hessian(
    const ShapeGraph& graph,
    std::span<const double> input,
    std::span<double> output,
    double regularization
) {
    std::fill(output.begin(), output.end(), 0.0);
    const std::size_t dims = graph.dimensions;
    for (const auto& factor : graph.factors) {
        for (std::size_t d = 0; d < dims; ++d) {
            double projection = 0.0;
            for (const auto& term : factor.terms) {
                projection += static_cast<double>(term.coefficient)
                    * input[static_cast<std::size_t>(term.node) * dims + d];
            }
            for (const auto& term : factor.terms) {
                output[static_cast<std::size_t>(term.node) * dims + d] +=
                    static_cast<double>(factor.weight)
                    * static_cast<double>(term.coefficient) * projection;
            }
        }
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] += regularization * input[i];
    }
}

void build_rhs_and_diagonal(
    const ShapeGraph& graph,
    std::span<double> rhs,
    std::span<double> diagonal,
    double regularization
) {
    std::fill(rhs.begin(), rhs.end(), 0.0);
    std::fill(diagonal.begin(), diagonal.end(), regularization);
    const std::size_t dims = graph.dimensions;
    for (const auto& factor : graph.factors) {
        for (const auto& term : factor.terms) {
            for (std::size_t d = 0; d < dims; ++d) {
                const std::size_t index =
                    static_cast<std::size_t>(term.node) * dims + d;
                rhs[index] += static_cast<double>(factor.weight)
                    * static_cast<double>(term.coefficient)
                    * static_cast<double>(factor.target[d]);
                diagonal[index] += static_cast<double>(factor.weight)
                    * static_cast<double>(term.coefficient)
                    * static_cast<double>(term.coefficient);
            }
        }
    }
}

std::vector<std::pair<double, std::uint32_t>> factor_residuals(
    const ShapeGraph& graph,
    std::span<const float> state
) {
    std::vector<std::pair<double, std::uint32_t>> residuals;
    residuals.reserve(graph.factors.size());
    const std::size_t dims = graph.dimensions;
    for (const auto& factor : graph.factors) {
        double squared = 0.0;
        for (std::size_t d = 0; d < dims; ++d) {
            double residual = -static_cast<double>(factor.target[d]);
            for (const auto& term : factor.terms) {
                residual += static_cast<double>(term.coefficient)
                    * static_cast<double>(
                        state[static_cast<std::size_t>(term.node) * dims + d]
                    );
            }
            squared += residual * residual;
        }
        residuals.emplace_back(std::sqrt(squared), factor.id);
    }
    return residuals;
}

}  // namespace

std::uint32_t ShapeGraph::add_node(std::string type) {
    node_types.push_back(std::move(type));
    return static_cast<std::uint32_t>(node_types.size() - 1);
}

std::uint32_t ShapeGraph::add_factor(
    std::span<const FactorTerm> terms,
    std::span<const float> target,
    float weight
) {
    LinearFactor factor;
    factor.id = static_cast<std::uint32_t>(factors.size());
    factor.weight = weight;
    factor.terms.assign(terms.begin(), terms.end());
    factor.target.assign(target.begin(), target.end());
    factors.push_back(std::move(factor));
    return static_cast<std::uint32_t>(factors.size() - 1);
}

void ShapeGraph::validate() const {
    if (dimensions == 0) {
        throw std::invalid_argument("ShapeGraph dimensions must be positive");
    }
    if (node_types.empty()) {
        throw std::invalid_argument("ShapeGraph requires at least one node");
    }
    for (const auto& type : node_types) {
        if (type.empty()) {
            throw std::invalid_argument("Node types must not be empty");
        }
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        const auto& factor = factors[i];
        if (factor.id != i) {
            throw std::invalid_argument("Factor ids must be dense and stable");
        }
        if (!(factor.weight > 0.0F) || !std::isfinite(factor.weight)) {
            throw std::invalid_argument("Factor weights must be finite and positive");
        }
        if (factor.terms.empty()) {
            throw std::invalid_argument("Factors require at least one term");
        }
        if (factor.target.size() != dimensions) {
            throw std::invalid_argument("Factor target dimension mismatch");
        }
        for (float target : factor.target) {
            if (!std::isfinite(target)) {
                throw std::invalid_argument("Non-finite factor target");
            }
        }
        std::unordered_set<std::uint32_t> nodes_in_factor;
        nodes_in_factor.reserve(factor.terms.size());
        for (const auto& term : factor.terms) {
            if (term.node >= node_types.size()) {
                throw std::invalid_argument("Factor references an invalid node");
            }
            if (!std::isfinite(term.coefficient) || term.coefficient == 0.0F) {
                throw std::invalid_argument(
                    "Factor coefficients must be finite and non-zero"
                );
            }
            if (!nodes_in_factor.insert(term.node).second) {
                throw std::invalid_argument(
                    "A factor may reference each node only once"
                );
            }
        }
    }
}

EquilibriumSolver::EquilibriumSolver(SolveOptions options)
    : options_(options) {
    if (options_.max_iterations == 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }
    if (
        !(options_.relative_tolerance >= 0.0)
        || !std::isfinite(options_.relative_tolerance)
        || !(options_.absolute_tolerance >= 0.0)
        || !std::isfinite(options_.absolute_tolerance)
    ) {
        throw std::invalid_argument("Solver tolerances must be finite and non-negative");
    }
    if (
        !(options_.l2_regularization > 0.0)
        || !std::isfinite(options_.l2_regularization)
    ) {
        throw std::invalid_argument("l2_regularization must be finite and positive");
    }
    if (
        !(options_.active_residual_threshold >= 0.0)
        || !std::isfinite(options_.active_residual_threshold)
    ) {
        throw std::invalid_argument(
            "active_residual_threshold must be finite and non-negative"
        );
    }
}

std::string_view termination_reason_name(TerminationReason reason) noexcept {
    switch (reason) {
    case TerminationReason::converged:
        return "converged";
    case TerminationReason::precision_limit:
        return "precision_limit";
    case TerminationReason::maximum_iterations:
        return "maximum_iterations";
    case TerminationReason::numerical_breakdown:
        return "numerical_breakdown";
    }
    return "unknown";
}

SolveResult EquilibriumSolver::solve(const ShapeGraph& graph) const {
    graph.validate();
    const auto wall_start = Clock::now();
    const std::size_t variables = graph.variable_count();
    std::vector<double> x(variables, 0.0);
    std::vector<double> rhs(variables, 0.0);
    std::vector<double> diagonal(variables, 0.0);
    std::vector<double> residual(variables, 0.0);
    std::vector<double> preconditioned(variables, 0.0);
    std::vector<double> direction(variables, 0.0);
    std::vector<double> product(variables, 0.0);
    std::vector<double> previous_x(variables, 0.0);
    std::vector<float> snapshot(variables, 0.0F);

    build_rhs_and_diagonal(
        graph, rhs, diagonal, options_.l2_regularization
    );
    residual = rhs;
    for (std::size_t i = 0; i < variables; ++i) {
        preconditioned[i] = residual[i] / diagonal[i];
    }
    direction = preconditioned;
    double rz = dot(residual, preconditioned);
    const double initial_norm = norm(residual);
    const double requested_stop_norm = std::max(
        options_.absolute_tolerance,
        options_.relative_tolerance * initial_norm
    );
    // The public state and factor targets are float32. Demanding a Krylov
    // residual far beneath their representational floor causes correct solves
    // to be labelled non-converged after the search direction has numerically
    // vanished. Keep solving toward the requested tolerance; use the
    // precision-aware floor only to classify numerical exhaustion.
    const double precision_floor =
        32.0 * static_cast<double>(std::numeric_limits<float>::epsilon())
        * std::max(1.0, norm(rhs));
    const double stop_norm = requested_stop_norm;

    SolveResult result;
    result.telemetry.initial_energy = 0.0;
    for (const auto& factor : graph.factors) {
        for (float target : factor.target) {
            result.telemetry.initial_energy +=
                0.5 * static_cast<double>(factor.weight)
                * static_cast<double>(target) * static_cast<double>(target);
        }
    }
    result.telemetry.graph_hash = hash_graph(graph);
    result.telemetry.precision_limited = false;
    result.telemetry.termination_reason = TerminationReason::maximum_iterations;
    result.telemetry.requested_stop_norm = requested_stop_norm;
    result.telemetry.effective_stop_norm = precision_floor;
    result.telemetry.workspace_bytes =
        variables * sizeof(double) * 8 + variables * sizeof(float) * 2;
    result.telemetry.trace.reserve(
        options_.max_iterations / std::max(1U, options_.trace_stride) + 2
    );

    std::uint32_t completed = 0;
    double current_residual_norm = initial_norm;
    bool converged = initial_norm <= stop_norm;
    for (std::uint32_t iteration = 0;
         iteration < options_.max_iterations && !converged;
         ++iteration) {
        const auto iteration_start = Clock::now();
        previous_x = x;
        apply_hessian(
            graph, direction, product, options_.l2_regularization
        );
        ++result.telemetry.matrix_vector_products;
        result.telemetry.factor_evaluations += graph.factors.size();
        const double denominator = dot(direction, product);
        if (
            !(denominator > std::numeric_limits<double>::epsilon())
            || !std::isfinite(denominator)
        ) {
            if (current_residual_norm <= precision_floor) {
                converged = true;
                result.telemetry.precision_limited = true;
                result.telemetry.termination_reason =
                    TerminationReason::precision_limit;
            } else {
                result.telemetry.termination_reason =
                    TerminationReason::numerical_breakdown;
            }
            break;
        }
        const double alpha = rz / denominator;
        if (!std::isfinite(alpha)) {
            result.telemetry.termination_reason =
                TerminationReason::numerical_breakdown;
            break;
        }
        for (std::size_t i = 0; i < variables; ++i) {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * product[i];
        }
        const double residual_norm = norm(residual);
        current_residual_norm = residual_norm;
        for (std::size_t i = 0; i < variables; ++i) {
            preconditioned[i] = residual[i] / diagonal[i];
        }
        const double next_rz = dot(residual, preconditioned);
        const double beta = rz == 0.0 ? 0.0 : next_rz / rz;
        for (std::size_t i = 0; i < variables; ++i) {
            direction[i] = preconditioned[i] + beta * direction[i];
        }
        rz = next_rz;
        ++completed;
        converged = residual_norm <= stop_norm;

        if (
            iteration % std::max(1U, options_.trace_stride) == 0
            || converged
        ) {
            double max_delta = 0.0;
            for (std::size_t i = 0; i < variables; ++i) {
                max_delta = std::max(max_delta, std::abs(x[i] - previous_x[i]));
            }
            std::transform(x.begin(), x.end(), snapshot.begin(),
                           [](double value) { return static_cast<float>(value); });
            result.telemetry.trace.push_back({
                iteration + 1,
                factor_energy(graph, snapshot),
                residual_norm,
                max_delta,
                alpha,
                beta,
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - iteration_start
                    ).count()
                )
            });
        }
    }

    result.state.resize(variables);
    std::transform(x.begin(), x.end(), result.state.begin(),
                   [](double value) { return static_cast<float>(value); });
    result.telemetry.converged = converged;
    if (
        converged
        && result.telemetry.termination_reason
            == TerminationReason::maximum_iterations
    ) {
        result.telemetry.termination_reason = TerminationReason::converged;
    }
    result.telemetry.iterations = completed;
    result.telemetry.final_energy = factor_energy(graph, result.state);
    result.telemetry.final_linear_residual_l2 = norm(residual);
    result.telemetry.wall_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - wall_start
        ).count()
    );
    result.telemetry.output_hash = hash_state(result.state);

    auto residuals = factor_residuals(graph, result.state);
    std::sort(
        residuals.begin(), residuals.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; }
    );
    result.telemetry.max_factor_residual =
        residuals.empty() ? 0.0 : residuals.front().first;
    const auto active = std::count_if(
        residuals.begin(), residuals.end(),
        [&](const auto& item) {
            return item.first > options_.active_residual_threshold;
        }
    );
    result.telemetry.active_factor_fraction = residuals.empty()
        ? 0.0
        : static_cast<double>(active) / static_cast<double>(residuals.size());
    const std::size_t keep = std::min<std::size_t>(
        options_.top_residual_factors, residuals.size()
    );
    for (std::size_t i = 0; i < keep; ++i) {
        if (residuals[i].first <= options_.active_residual_threshold) {
            break;
        }
        result.telemetry.highest_residual_factor_ids.push_back(
            residuals[i].second
        );
    }
    return result;
}

GoalSlice extract_goal_shape(
    const ShapeGraph& source,
    std::uint32_t query_node
) {
    source.validate();
    if (query_node >= source.node_count()) {
        throw std::out_of_range("Query node is invalid");
    }
    std::vector<std::vector<std::uint32_t>> incident(source.node_count());
    for (const auto& factor : source.factors) {
        for (const auto& term : factor.terms) {
            incident[term.node].push_back(factor.id);
        }
    }
    std::vector<unsigned char> node_seen(source.node_count(), 0);
    std::vector<unsigned char> factor_seen(source.factors.size(), 0);
    std::queue<std::uint32_t> frontier;
    node_seen[query_node] = 1;
    frontier.push(query_node);
    while (!frontier.empty()) {
        const auto node = frontier.front();
        frontier.pop();
        for (const auto factor_id : incident[node]) {
            if (factor_seen[factor_id] != 0) {
                continue;
            }
            factor_seen[factor_id] = 1;
            for (const auto& term : source.factors[factor_id].terms) {
                if (node_seen[term.node] == 0) {
                    node_seen[term.node] = 1;
                    frontier.push(term.node);
                }
            }
        }
    }

    GoalSlice slice;
    slice.graph.dimensions = source.dimensions;
    std::vector<std::uint32_t> global_to_local(
        source.node_count(), std::numeric_limits<std::uint32_t>::max()
    );
    for (std::uint32_t node = 0; node < source.node_count(); ++node) {
        if (node_seen[node] == 0) {
            continue;
        }
        global_to_local[node] = slice.graph.add_node(source.node_types[node]);
        slice.local_to_global_node.push_back(node);
    }
    slice.local_query_node = global_to_local[query_node];
    for (const auto& factor : source.factors) {
        if (factor_seen[factor.id] == 0) {
            continue;
        }
        std::vector<FactorTerm> terms;
        terms.reserve(factor.terms.size());
        for (const auto& term : factor.terms) {
            terms.push_back({
                global_to_local[term.node],
                term.coefficient
            });
        }
        slice.graph.add_factor(terms, factor.target, factor.weight);
        slice.local_to_global_factor.push_back(factor.id);
    }
    return slice;
}

SemanticSkeleton compile_skeleton(
    const ShapeGraph& graph,
    const SolveResult& result,
    std::uint32_t query_node,
    std::span<const std::uint32_t> support_factor_ids
) {
    if (query_node >= graph.node_count()) {
        throw std::out_of_range("Query node is invalid");
    }
    SemanticSkeleton skeleton;
    skeleton.converged = result.telemetry.converged;
    skeleton.contradiction_score = result.telemetry.final_energy;
    skeleton.query_node = query_node;
    const std::size_t begin =
        static_cast<std::size_t>(query_node) * graph.dimensions;
    skeleton.query_value.assign(
        result.state.begin() + static_cast<std::ptrdiff_t>(begin),
        result.state.begin()
            + static_cast<std::ptrdiff_t>(begin + graph.dimensions)
    );
    skeleton.support_factor_ids.assign(
        support_factor_ids.begin(), support_factor_ids.end()
    );
    skeleton.contradiction_factor_ids =
        result.telemetry.highest_residual_factor_ids;
    skeleton.shape_hash = result.telemetry.graph_hash;
    skeleton.answer_hash = result.telemetry.output_hash;
    return skeleton;
}

void write_skeleton_json(std::ostream& out, const SemanticSkeleton& skeleton) {
    out << "{\n"
        << "  \"converged\": " << (skeleton.converged ? "true" : "false") << ",\n"
        << "  \"contradiction_score\": " << std::setprecision(12)
        << skeleton.contradiction_score << ",\n"
        << "  \"query_node\": " << skeleton.query_node << ",\n"
        << "  \"query_value\": [";
    for (std::size_t i = 0; i < skeleton.query_value.size(); ++i) {
        if (i != 0) out << ", ";
        out << skeleton.query_value[i];
    }
    out << "],\n  \"support_factor_ids\": [";
    for (std::size_t i = 0; i < skeleton.support_factor_ids.size(); ++i) {
        if (i != 0) out << ", ";
        out << skeleton.support_factor_ids[i];
    }
    out << "],\n  \"contradiction_factor_ids\": [";
    for (std::size_t i = 0; i < skeleton.contradiction_factor_ids.size(); ++i) {
        if (i != 0) out << ", ";
        out << skeleton.contradiction_factor_ids[i];
    }
    out << "],\n  \"shape_hash\": " << skeleton.shape_hash
        << ",\n  \"answer_hash\": " << skeleton.answer_hash << "\n}\n";
}

void write_trace_csv_header(std::ostream& out) {
    out << "case_id,nodes,dimensions,factors,iteration,energy,"
           "linear_residual_l2,max_state_delta,alpha,beta,"
           "elapsed_nanoseconds\n";
}

void write_trace_csv(
    std::ostream& out,
    std::string_view case_id,
    const ShapeGraph& graph,
    const SolveResult& result
) {
    for (const auto& item : result.telemetry.trace) {
        out << case_id << ','
            << graph.node_count() << ','
            << graph.dimensions << ','
            << graph.factors.size() << ','
            << item.iteration << ','
            << std::setprecision(12) << item.energy << ','
            << item.linear_residual_l2 << ','
            << item.max_state_delta << ','
            << item.alpha << ','
            << item.beta << ','
            << item.elapsed_nanoseconds << '\n';
    }
}

double factor_energy(
    const ShapeGraph& graph,
    std::span<const float> state
) {
    if (state.size() != graph.variable_count()) {
        throw std::invalid_argument("State size does not match graph");
    }
    double energy = 0.0;
    const std::size_t dims = graph.dimensions;
    for (const auto& factor : graph.factors) {
        for (std::size_t d = 0; d < dims; ++d) {
            double residual = -static_cast<double>(factor.target[d]);
            for (const auto& term : factor.terms) {
                residual += static_cast<double>(term.coefficient)
                    * static_cast<double>(
                        state[static_cast<std::size_t>(term.node) * dims + d]
                    );
            }
            energy += 0.5 * static_cast<double>(factor.weight)
                * residual * residual;
        }
    }
    return energy;
}

std::uint64_t hash_graph(const ShapeGraph& graph) {
    std::uint64_t hash = kFnvOffset;
    hash_bytes(hash, &graph.dimensions, sizeof(graph.dimensions));
    for (const auto& type : graph.node_types) {
        hash_bytes(hash, type.data(), type.size());
        const unsigned char separator = 0xFF;
        hash_bytes(hash, &separator, sizeof(separator));
    }
    for (const auto& factor : graph.factors) {
        hash_bytes(hash, &factor.id, sizeof(factor.id));
        hash_bytes(hash, &factor.weight, sizeof(factor.weight));
        for (const auto& term : factor.terms) {
            hash_bytes(hash, &term.node, sizeof(term.node));
            hash_bytes(hash, &term.coefficient, sizeof(term.coefficient));
        }
        for (const float target : factor.target) {
            hash_bytes(hash, &target, sizeof(target));
        }
    }
    return hash;
}

std::uint64_t hash_state(std::span<const float> state) {
    std::uint64_t hash = kFnvOffset;
    for (const float value : state) {
        const float rounded =
            std::round(value * 1'000'000.0F) / 1'000'000.0F;
        hash_bytes(hash, &rounded, sizeof(rounded));
    }
    return hash;
}

}  // namespace sera

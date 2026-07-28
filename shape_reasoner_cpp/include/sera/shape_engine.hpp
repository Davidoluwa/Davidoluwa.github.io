#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sera {

struct FactorTerm {
    std::uint32_t node{};
    float coefficient{};
};

struct LinearFactor {
    std::uint32_t id{};
    float weight{1.0F};
    std::vector<FactorTerm> terms;
    std::vector<float> target;
};

struct ShapeGraph {
    std::uint32_t dimensions{1};
    std::vector<std::string> node_types;
    std::vector<LinearFactor> factors;

    [[nodiscard]] std::size_t node_count() const noexcept {
        return node_types.size();
    }
    [[nodiscard]] std::size_t variable_count() const noexcept {
        return node_count() * dimensions;
    }

    std::uint32_t add_node(std::string type);
    std::uint32_t add_factor(
        std::span<const FactorTerm> terms,
        std::span<const float> target,
        float weight = 1.0F
    );
    void validate() const;
};

struct IterationTelemetry {
    std::uint32_t iteration{};
    double energy{};
    double linear_residual_l2{};
    double max_state_delta{};
    double alpha{};
    double beta{};
    std::uint64_t elapsed_nanoseconds{};
};

enum class TerminationReason : std::uint8_t {
    converged,
    precision_limit,
    maximum_iterations,
    numerical_breakdown
};

[[nodiscard]] std::string_view termination_reason_name(
    TerminationReason reason
) noexcept;

struct SolveTelemetry {
    bool converged{};
    bool precision_limited{};
    TerminationReason termination_reason{TerminationReason::maximum_iterations};
    std::uint32_t iterations{};
    std::uint64_t matrix_vector_products{};
    std::uint64_t factor_evaluations{};
    double initial_energy{};
    double final_energy{};
    double final_linear_residual_l2{};
    double requested_stop_norm{};
    double effective_stop_norm{};
    double max_factor_residual{};
    double active_factor_fraction{};
    std::uint64_t wall_nanoseconds{};
    std::size_t workspace_bytes{};
    std::uint64_t graph_hash{};
    std::uint64_t output_hash{};
    std::vector<IterationTelemetry> trace;
    std::vector<std::uint32_t> highest_residual_factor_ids;
};

struct SolveOptions {
    std::uint32_t max_iterations{2048};
    double relative_tolerance{1.0e-9};
    double absolute_tolerance{1.0e-10};
    double l2_regularization{1.0e-8};
    double active_residual_threshold{1.0e-4};
    std::uint32_t trace_stride{1};
    std::uint32_t top_residual_factors{8};
};

struct SolveResult {
    std::vector<float> state;
    SolveTelemetry telemetry;
};

class EquilibriumSolver {
public:
    explicit EquilibriumSolver(SolveOptions options = {});
    [[nodiscard]] SolveResult solve(const ShapeGraph& graph) const;

private:
    SolveOptions options_;
};

struct GoalSlice {
    ShapeGraph graph;
    std::vector<std::uint32_t> local_to_global_node;
    std::vector<std::uint32_t> local_to_global_factor;
    std::uint32_t local_query_node{};
};

[[nodiscard]] GoalSlice extract_goal_shape(
    const ShapeGraph& source,
    std::uint32_t query_node
);

struct SemanticSkeleton {
    bool converged{};
    double contradiction_score{};
    std::uint32_t query_node{};
    std::vector<float> query_value;
    std::vector<std::uint32_t> support_factor_ids;
    std::vector<std::uint32_t> contradiction_factor_ids;
    std::uint64_t shape_hash{};
    std::uint64_t answer_hash{};
};

[[nodiscard]] SemanticSkeleton compile_skeleton(
    const ShapeGraph& graph,
    const SolveResult& result,
    std::uint32_t query_node,
    std::span<const std::uint32_t> support_factor_ids = {}
);

void write_skeleton_json(std::ostream& out, const SemanticSkeleton& skeleton);
void write_trace_csv_header(std::ostream& out);
void write_trace_csv(
    std::ostream& out,
    std::string_view case_id,
    const ShapeGraph& graph,
    const SolveResult& result
);

[[nodiscard]] double factor_energy(
    const ShapeGraph& graph,
    std::span<const float> state
);

[[nodiscard]] std::uint64_t hash_graph(const ShapeGraph& graph);
[[nodiscard]] std::uint64_t hash_state(std::span<const float> state);

}  // namespace sera

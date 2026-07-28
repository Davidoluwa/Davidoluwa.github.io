#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sera {

struct CategoricalNode {
    std::uint32_t id{};
    std::string type;
    std::vector<double> unary_cost;
};

struct PairwiseFactor {
    std::uint32_t id{};
    std::uint32_t left{};
    std::uint32_t right{};
    std::string relation;
    std::vector<double> cost;
};

struct CategoricalShape {
    std::vector<CategoricalNode> nodes;
    std::vector<PairwiseFactor> factors;

    std::uint32_t add_node(
        std::string type,
        std::span<const double> unary_cost
    );
    std::uint32_t add_factor(
        std::uint32_t left,
        std::uint32_t right,
        std::string relation,
        std::span<const double> row_major_cost
    );
    void validate() const;
};

struct CategoricalIterationTelemetry {
    std::uint32_t sweep{};
    double max_message_residual{};
    double total_message_residual{};
    double provisional_energy{};
    std::uint64_t committed_messages{};
    std::uint64_t skipped_messages{};
    std::uint64_t elapsed_nanoseconds{};
};

struct PrimitiveOperations {
    std::uint64_t message_proposals{};
    std::uint64_t message_commits{};
    std::uint64_t message_skips{};
    std::uint64_t candidate_evaluations{};
    std::uint64_t scalar_add_subtracts{};
    std::uint64_t scalar_comparisons{};
    std::uint64_t scalar_multiplies{};
    std::uint64_t exponentials{};
    std::uint64_t factor_energy_lookups{};
};

struct CategoricalSolveTelemetry {
    bool converged{};
    std::uint32_t sweeps{};
    double final_energy{};
    std::uint32_t violated_factors{};
    double max_message_residual{};
    double total_message_residual{};
    double mean_entropy{};
    double max_entropy{};
    std::uint32_t max_entropy_node{};
    double active_message_fraction{};
    std::uint64_t wall_nanoseconds{};
    std::size_t workspace_bytes{};
    std::uint64_t graph_hash{};
    std::uint64_t assignment_hash{};
    PrimitiveOperations operations;
    std::vector<CategoricalIterationTelemetry> trace;
};

struct CategoricalSolveOptions {
    std::uint32_t max_sweeps{512};
    double damping{0.72};
    double residual_gate{1.0e-7};
    double output_temperature{0.25};
    double violation_tolerance{1.0e-9};
    std::uint32_t trace_stride{1};
};

struct CategoricalSolveResult {
    std::vector<std::uint32_t> assignment;
    std::vector<double> confidence;
    std::vector<double> entropy;
    std::vector<double> belief_costs;
    CategoricalSolveTelemetry telemetry;
};

class CategoricalShapeSolver {
public:
    explicit CategoricalShapeSolver(CategoricalSolveOptions options = {});
    [[nodiscard]] CategoricalSolveResult solve(
        const CategoricalShape& graph
    ) const;

private:
    CategoricalSolveOptions options_;
};

struct ShapeParticle {
    std::uint32_t clamped_state{};
    double original_energy{};
    double free_energy_score{};
    CategoricalSolveResult result;
};

struct ParticleTelemetry {
    std::uint32_t split_node{};
    std::uint32_t particles_spawned{};
    std::uint32_t particles_expanded{};
    std::uint32_t unique_modes{};
    std::uint32_t max_depth_reached{};
    std::uint64_t aggregate_candidate_evaluations{};
    std::uint64_t aggregate_message_proposals{};
    std::uint64_t wall_nanoseconds{};
};

struct ParticleResult {
    CategoricalSolveResult base;
    std::vector<ShapeParticle> particles;
    ParticleTelemetry telemetry;
};

struct ParticleOptions {
    std::uint32_t budget{8};
    std::uint32_t max_split_depth{2};
    double entropy_split_threshold{0.5};
    double clamp_penalty{1000.0};
    double entropy_weight{0.05};
    double operation_weight{1.0e-12};
};

class ShapeParticleSolver {
public:
    ShapeParticleSolver(
        CategoricalSolveOptions solve_options = {},
        ParticleOptions particle_options = {}
    );
    [[nodiscard]] ParticleResult solve(const CategoricalShape& graph) const;

private:
    CategoricalSolveOptions solve_options_;
    ParticleOptions particle_options_;
};

struct CategoricalGoalSlice {
    CategoricalShape graph;
    std::vector<std::uint32_t> local_to_global_node;
    std::vector<std::uint32_t> local_to_global_factor;
    std::uint32_t local_query_node{};
};

[[nodiscard]] CategoricalGoalSlice extract_goal_shape(
    const CategoricalShape& source,
    std::uint32_t query_node
);

[[nodiscard]] double categorical_energy(
    const CategoricalShape& graph,
    std::span<const std::uint32_t> assignment
);

[[nodiscard]] std::uint32_t categorical_violations(
    const CategoricalShape& graph,
    std::span<const std::uint32_t> assignment,
    double tolerance = 1.0e-9
);

[[nodiscard]] std::uint64_t hash_categorical_graph(
    const CategoricalShape& graph
);

[[nodiscard]] std::uint64_t hash_assignment(
    std::span<const std::uint32_t> assignment
);

void write_categorical_result_json(
    std::ostream& out,
    std::string_view case_id,
    const CategoricalShape& graph,
    const CategoricalSolveResult& result
);

}  // namespace sera

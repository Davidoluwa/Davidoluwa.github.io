#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

#include "sera/neural_proof_kernel.hpp"

namespace sera {

struct WorkspaceKernelConfig {
    std::uint32_t bit_width{6};
    std::uint32_t channels{16};
    std::uint32_t shared_rank{6};
    std::uint32_t reasoning_steps{14};
};

struct WorkspaceTelemetry {
    std::uint64_t optimizer_steps{};
    std::uint64_t training_cases{};
    std::uint64_t parameters{};
    std::uint64_t cp_equivalent_parameters{};
    std::uint64_t dense_equivalent_parameters{};
    double most_recent_loss{};
    double mean_recent_loss{};
    std::uint64_t inference_nanoseconds{};
};

// S5: a non-transformer neural kernel whose operand interaction lives on an
// explicit two-dimensional bit lattice.
//
// Motivation. The S4 kernel exposed operand interaction only through a rank-R
// CP contraction, which is algebraically a rank-R factorization of a tensor
// T[p][i][j]. Unsigned multiplication requires the polynomial-multiplication
// tensor T[p][i][j] = 1[i + j == p], whose CP rank grows like 2W - 1, so any
// configuration with shared_rank < 2W - 1 cannot represent multiplication at
// all. See research/cp_rank_bound.py for the measured rank curve.
//
// S5 removes the factorization bottleneck instead of paying Theta(W^2)
// parameters to widen it. Cell (i, j) of the workspace observes the exact
// bilinear term a_i * b_j, and a single translation-equivariant stencil, tied
// across every cell and every reasoning step, routes and accumulates that
// field. Translation equivariance is a generic lattice prior, not an
// arithmetic cue: no carry rule, shift amount, anti-diagonal sum, or
// operation identity is written into the model. The raw operator byte enters
// only as literal bits, exactly as in S4.
class InteractionWorkspaceKernel {
public:
    explicit InteractionWorkspaceKernel(
        WorkspaceKernelConfig config = {},
        std::uint64_t seed = 0x534552413553354DULL
    );

    struct Parameters;
    struct AdamState;

    double train_batch(
        std::span<const ArithmeticTrainingCase> cases,
        float learning_rate,
        float group_penalty = 0.0F
    );

    [[nodiscard]] ArithmeticInference infer(
        std::uint32_t left,
        std::uint32_t right,
        std::uint8_t raw_operator
    ) const;

    // Analytic gradient of the training loss for one batch, flattened in the
    // canonical tensor order. Exposed so the test suite can verify it against
    // central finite differences.
    [[nodiscard]] std::vector<double> flat_gradient(
        std::span<const ArithmeticTrainingCase> cases
    ) const;
    [[nodiscard]] std::vector<double> flat_parameters() const;
    void set_flat_parameters(std::span<const double> values);
    [[nodiscard]] double batch_loss(
        std::span<const ArithmeticTrainingCase> cases
    ) const;

    // Group norm of each shared rank channel. A channel spans the projection
    // and both stencil lifts, so driving one to zero removes
    // channels * (1 + 2 * taps) real parameters at once. The group penalty in
    // train_batch is what makes these norms separate.
    [[nodiscard]] std::vector<double> rank_channel_norms() const;

    // Hard-zero the `count` lowest-norm rank channels and report how many
    // parameters that removes. Used to measure, rather than assert,
    // parameter efficiency: accuracy is re-evaluated after pruning.
    std::uint64_t prune_rank_channels(std::uint32_t count);

    [[nodiscard]] const WorkspaceKernelConfig& config() const noexcept;
    [[nodiscard]] const WorkspaceTelemetry& telemetry() const noexcept;
    [[nodiscard]] std::uint64_t parameter_checksum() const noexcept;

    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);
    void write_json(std::ostream& out) const;

    InteractionWorkspaceKernel(const InteractionWorkspaceKernel&) = delete;
    InteractionWorkspaceKernel& operator=(const InteractionWorkspaceKernel&)
        = delete;
    InteractionWorkspaceKernel(InteractionWorkspaceKernel&&) noexcept;
    InteractionWorkspaceKernel& operator=(InteractionWorkspaceKernel&&)
        noexcept;
    ~InteractionWorkspaceKernel();

private:
    WorkspaceKernelConfig config_;
    Parameters* parameters_{};
    AdamState* adam_{};
    mutable WorkspaceTelemetry telemetry_;
};

}  // namespace sera

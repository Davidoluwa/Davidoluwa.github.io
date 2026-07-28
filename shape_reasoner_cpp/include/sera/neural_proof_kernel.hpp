#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace sera {

struct ArithmeticTrainingCase {
    std::uint32_t left{};
    std::uint32_t right{};
    std::uint32_t target{};
    std::uint8_t raw_operator{};
};

struct ProofKernelConfig {
    std::uint32_t bit_width{12};
    std::uint32_t hidden_units{32};
    std::uint32_t shared_rank{8};
    std::uint32_t reasoning_steps{20};
    float residual_rate{0.4F};
};

struct ArithmeticInference {
    std::uint32_t value{};
    std::vector<float> logits;
    std::uint32_t stable_after_step{};
    float mean_confidence{};
};

struct ProofKernelTelemetry {
    std::uint64_t optimizer_steps{};
    std::uint64_t training_cases{};
    std::uint64_t parameters{};
    std::uint64_t dense_equivalent_parameters{};
    double most_recent_loss{};
    double mean_recent_loss{};
    std::uint64_t inference_nanoseconds{};
};

// A non-transformer CP-relational, low-rank gated neural microkernel over a
// bit lattice. The same parameters are tied across reasoning time, positions,
// operators, and proof-tree nodes. The raw operator byte is part of the
// problem representation; no operation ID or arithmetic primitive is
// dispatched inside this class.
class NeuralProofKernel {
public:
    explicit NeuralProofKernel(
        ProofKernelConfig config = {},
        std::uint64_t seed = 0x534552413450524FULL
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

    [[nodiscard]] const ProofKernelConfig& config() const noexcept;
    [[nodiscard]] const ProofKernelTelemetry& telemetry() const noexcept;
    [[nodiscard]] std::uint64_t parameter_checksum() const noexcept;
    [[nodiscard]] std::vector<double> rank_channel_norms() const;

    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);
    void write_json(std::ostream& out) const;

    NeuralProofKernel(const NeuralProofKernel&) = delete;
    NeuralProofKernel& operator=(const NeuralProofKernel&) = delete;
    NeuralProofKernel(NeuralProofKernel&& other) noexcept;
    NeuralProofKernel& operator=(NeuralProofKernel&& other) noexcept;
    ~NeuralProofKernel();

private:
    ProofKernelConfig config_;
    Parameters* parameters_{};
    AdamState* adam_{};
    mutable ProofKernelTelemetry telemetry_;
};

}  // namespace sera

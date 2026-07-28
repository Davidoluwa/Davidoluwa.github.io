#pragma once

#include "sera/boolean_program_learner.hpp"

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace sera {

struct NeuralShapeConfig {
    std::uint32_t input_bits{6};
    std::uint32_t output_bits{6};
    std::uint32_t hidden_units{24};
    std::uint32_t reasoning_steps{6};
    float residual_rate{0.5F};
};

struct NeuralShapeTelemetry {
    std::uint64_t optimizer_steps{};
    std::uint64_t fields_trained{};
    std::uint64_t parameters{};
    double most_recent_loss{};
    double mean_recent_loss{};
    std::uint64_t compile_nanoseconds{};
};

// A non-transformer neural process over a Boolean hypercube.
//
// Shared recurrent message-passing parameters are meta-trained across
// anonymous relations. At deployment, compile_shape() converts an unordered
// support set into one frozen whole-relation field. predict() only reads that
// field; it performs no gradient update and consumes no operation identifier.
class NeuralShapeField {
public:
    // Exposed only so the translation unit can define compact tensor storage;
    // callers should treat both types as implementation details.
    struct Parameters;
    struct AdamState;

    explicit NeuralShapeField(
        NeuralShapeConfig config = {},
        std::uint64_t initialization_seed = 0x5345524133ULL
    );

    // One meta-training update. support is the learner-visible partial
    // relation. complete_relation supplies training targets only and must
    // contain every Boolean input exactly once.
    double train_episode(
        std::span<const BooleanExample> support,
        std::span<const BooleanExample> complete_relation,
        float learning_rate
    );

    // Build and freeze a complete latent relation shape from demonstrations.
    void compile_shape(std::span<const BooleanExample> support);

    [[nodiscard]] std::vector<std::uint8_t> predict(
        std::span<const std::uint8_t> input
    ) const;

    [[nodiscard]] const std::vector<float>& frozen_shape() const noexcept;
    [[nodiscard]] const std::vector<float>& frozen_logits() const noexcept;
    [[nodiscard]] const NeuralShapeConfig& config() const noexcept;
    [[nodiscard]] const NeuralShapeTelemetry& telemetry() const noexcept;
    [[nodiscard]] std::uint64_t parameter_checksum() const noexcept;

    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);
    void write_json(std::ostream& out) const;

private:
    NeuralShapeConfig config_;
    Parameters* parameters_{};
    AdamState* adam_{};
    NeuralShapeTelemetry telemetry_;
    std::vector<float> frozen_shape_;
    std::vector<float> frozen_logits_;
    std::vector<std::uint8_t> observed_mask_;
    std::vector<std::uint8_t> observed_outputs_;

public:
    NeuralShapeField(const NeuralShapeField&) = delete;
    NeuralShapeField& operator=(const NeuralShapeField&) = delete;
    NeuralShapeField(NeuralShapeField&& other) noexcept;
    NeuralShapeField& operator=(NeuralShapeField&& other) noexcept;
    ~NeuralShapeField();
};

}  // namespace sera

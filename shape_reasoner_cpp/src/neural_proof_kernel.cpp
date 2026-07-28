#include "sera/neural_proof_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <utility>

namespace sera {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kOperatorBits = 8;
constexpr std::uint32_t kBoundaryFeatures = 2;
constexpr std::uint32_t kOperandFeatures = 2;
constexpr std::uint32_t kBaseFeatureCount =
    kOperatorBits + kBoundaryFeatures + kOperandFeatures;
constexpr float kAdamBeta1 = 0.9F;
constexpr float kAdamBeta2 = 0.999F;
constexpr float kAdamEpsilon = 1.0e-8F;
constexpr std::uint64_t kCheckpointMagic = 0x5345524134504B52ULL;
constexpr std::uint32_t kCheckpointVersion = 4;

std::uint32_t feature_count(const ProofKernelConfig& config) {
    return kBaseFeatureCount + config.shared_rank;
}

void validate_config(const ProofKernelConfig& config) {
    if (
        config.bit_width == 0
        || config.bit_width > 30
        || config.hidden_units == 0
        || config.shared_rank == 0
        || config.reasoning_steps == 0
        || !(config.residual_rate > 0.0F)
        || !(config.residual_rate < 1.0F)
    ) {
        throw std::invalid_argument("Invalid neural proof configuration");
    }
}

std::uint32_t value_limit(const ProofKernelConfig& config) {
    return (std::uint32_t{1} << config.bit_width) - 1U;
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float exponential = std::exp(-value);
        return 1.0F / (1.0F + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float binary_cross_entropy(float logit, float target) {
    const float magnitude = std::abs(logit);
    return std::max(logit, 0.0F) - logit * target
        + std::log1p(std::exp(-magnitude));
}

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed while writing proof checkpoint");
    }
}

template <typename T>
void read_scalar(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Truncated proof checkpoint");
    }
}

void write_vector(std::ofstream& out, const std::vector<float>& values) {
    const std::uint64_t size = values.size();
    write_scalar(out, size);
    out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!out) {
        throw std::runtime_error("Failed while writing proof tensor");
    }
}

void read_vector(
    std::ifstream& in,
    std::vector<float>& values,
    std::size_t expected
) {
    std::uint64_t size = 0;
    read_scalar(in, size);
    if (size != expected) {
        throw std::runtime_error("Proof checkpoint tensor shape mismatch");
    }
    values.resize(expected);
    in.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!in) {
        throw std::runtime_error("Truncated proof checkpoint tensor");
    }
}

}  // namespace

struct NeuralProofKernel::Parameters {
    std::vector<float> input;
    std::vector<float> input_bias;
    std::vector<float> projection;
    std::vector<float> self_lift;
    std::vector<float> left_lift;
    std::vector<float> right_lift;
    std::vector<float> anchor_lift;
    std::vector<float> recurrent_bias;
    std::vector<float> update_self_lift;
    std::vector<float> update_left_lift;
    std::vector<float> update_right_lift;
    std::vector<float> update_anchor_lift;
    std::vector<float> update_bias;
    std::vector<float> relation_left;
    std::vector<float> relation_right;
    std::vector<float> relation_position;
    std::vector<float> output;
    float output_bias{};

    [[nodiscard]] std::vector<std::vector<float>*> tensors() {
        return {
            &input, &input_bias, &projection, &self_lift, &left_lift,
            &right_lift, &anchor_lift, &recurrent_bias,
            &update_self_lift, &update_left_lift, &update_right_lift,
            &update_anchor_lift, &update_bias, &relation_left,
            &relation_right, &relation_position, &output
        };
    }

    [[nodiscard]] std::vector<const std::vector<float>*> tensors() const {
        return {
            &input, &input_bias, &projection, &self_lift, &left_lift,
            &right_lift, &anchor_lift, &recurrent_bias,
            &update_self_lift, &update_left_lift, &update_right_lift,
            &update_anchor_lift, &update_bias, &relation_left,
            &relation_right, &relation_position, &output
        };
    }
};

struct NeuralProofKernel::AdamState {
    Parameters first;
    Parameters second;
    std::uint64_t step{};
    double recent_loss{};
    std::uint32_t recent_count{};
};

namespace {

NeuralProofKernel::Parameters make_like(
    const ProofKernelConfig& config,
    float fill
) {
    const std::size_t hidden = config.hidden_units;
    const std::size_t rank = config.shared_rank;
    NeuralProofKernel::Parameters result;
    result.input.assign(hidden * feature_count(config), fill);
    result.input_bias.assign(hidden, fill);
    result.projection.assign(rank * hidden, fill);
    result.self_lift.assign(hidden * rank, fill);
    result.left_lift.assign(hidden * rank, fill);
    result.right_lift.assign(hidden * rank, fill);
    result.anchor_lift.assign(hidden * rank, fill);
    result.recurrent_bias.assign(hidden, fill);
    result.update_self_lift.assign(hidden * rank, fill);
    result.update_left_lift.assign(hidden * rank, fill);
    result.update_right_lift.assign(hidden * rank, fill);
    result.update_anchor_lift.assign(hidden * rank, fill);
    result.update_bias.assign(hidden, fill);
    result.relation_left.assign(config.bit_width * rank, fill);
    result.relation_right.assign(config.bit_width * rank, fill);
    result.relation_position.assign(config.bit_width * rank, fill);
    result.output.assign(hidden, fill);
    result.output_bias = fill;
    return result;
}

std::uint64_t parameter_count(
    const NeuralProofKernel::Parameters& parameters
) {
    std::uint64_t total = 1;
    for (const auto* tensor : parameters.tensors()) {
        total += tensor->size();
    }
    return total;
}

std::uint64_t dense_equivalent_count(const ProofKernelConfig& config) {
    const std::uint64_t hidden = config.hidden_units;
    return hidden * feature_count(config)
        + hidden
        + 8 * hidden * hidden
        + 2 * hidden
        + config.bit_width * config.bit_width * config.bit_width
        + hidden
        + 1;
}

void initialize(
    NeuralProofKernel::Parameters& parameters,
    const ProofKernelConfig& config,
    std::uint64_t seed
) {
    std::mt19937_64 random(seed);
    auto xavier = [&random](
        std::vector<float>& values,
        std::size_t fan_in,
        std::size_t fan_out
    ) {
        const float limit = std::sqrt(
            6.0F / static_cast<float>(fan_in + fan_out)
        );
        std::uniform_real_distribution<float> distribution(-limit, limit);
        for (float& value : values) {
            value = distribution(random);
        }
    };
    xavier(
        parameters.input, feature_count(config), config.hidden_units
    );
    xavier(
        parameters.projection, config.hidden_units, config.shared_rank
    );
    xavier(
        parameters.self_lift, config.shared_rank, config.hidden_units
    );
    xavier(
        parameters.left_lift, config.shared_rank, config.hidden_units
    );
    xavier(
        parameters.right_lift, config.shared_rank, config.hidden_units
    );
    xavier(
        parameters.anchor_lift, config.shared_rank, config.hidden_units
    );
    xavier(
        parameters.update_self_lift,
        config.shared_rank,
        config.hidden_units
    );
    xavier(
        parameters.update_left_lift,
        config.shared_rank,
        config.hidden_units
    );
    xavier(
        parameters.update_right_lift,
        config.shared_rank,
        config.hidden_units
    );
    xavier(
        parameters.update_anchor_lift,
        config.shared_rank,
        config.hidden_units
    );
    const float initial_gate_bias = std::log(
        config.residual_rate / (1.0F - config.residual_rate)
    );
    std::fill(
        parameters.update_bias.begin(),
        parameters.update_bias.end(),
        initial_gate_bias
    );
    xavier(
        parameters.relation_left,
        config.bit_width,
        config.shared_rank
    );
    xavier(
        parameters.relation_right,
        config.bit_width,
        config.shared_rank
    );
    xavier(
        parameters.relation_position,
        config.bit_width,
        config.shared_rank
    );
    xavier(parameters.output, config.hidden_units, 1);
}

struct FieldCache {
    std::vector<float> features;
    std::vector<float> hidden;
    std::vector<float> candidates;
    std::vector<float> update_gates;
    std::vector<float> logits_by_step;
};

std::vector<float> project_states(
    const ProofKernelConfig& config,
    const NeuralProofKernel::Parameters& parameters,
    std::span<const float> hidden
) {
    const std::size_t width = config.bit_width;
    const std::size_t units = config.hidden_units;
    const std::size_t rank = config.shared_rank;
    std::vector<float> projected(width * rank, 0.0F);
    for (std::size_t position = 0; position < width; ++position) {
        for (std::size_t channel = 0; channel < rank; ++channel) {
            float value = 0.0F;
            for (std::size_t unit = 0; unit < units; ++unit) {
                value += parameters.projection[channel * units + unit]
                    * hidden[position * units + unit];
            }
            projected[position * rank + channel] = value;
        }
    }
    return projected;
}

FieldCache forward_case(
    const ProofKernelConfig& config,
    const NeuralProofKernel::Parameters& parameters,
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t raw_operator
) {
    const std::size_t width = config.bit_width;
    const std::size_t hidden = config.hidden_units;
    const std::size_t rank = config.shared_rank;
    const std::size_t steps = config.reasoning_steps;
    const std::size_t features = feature_count(config);
    FieldCache cache;
    cache.features.assign(width * features, 0.0F);
    cache.hidden.assign((steps + 1) * width * hidden, 0.0F);
    cache.candidates.assign(steps * width * hidden, 0.0F);
    cache.update_gates.assign(steps * width * hidden, 0.0F);
    cache.logits_by_step.assign((steps + 1) * width, 0.0F);

    std::vector<float> left_relations(rank, 0.0F);
    std::vector<float> right_relations(rank, 0.0F);
    for (std::size_t source = 0; source < width; ++source) {
        const float left_bit =
            ((left >> source) & 1U) != 0 ? 1.0F : 0.0F;
        const float right_bit =
            ((right >> source) & 1U) != 0 ? 1.0F : 0.0F;
        for (std::size_t channel = 0; channel < rank; ++channel) {
            left_relations[channel] += left_bit
                * parameters.relation_left[source * rank + channel];
            right_relations[channel] += right_bit
                * parameters.relation_right[source * rank + channel];
        }
    }
    for (std::size_t position = 0; position < width; ++position) {
        const std::size_t feature_offset = position * features;
        cache.features[feature_offset] =
            ((left >> position) & 1U) != 0 ? 1.0F : -1.0F;
        cache.features[feature_offset + 1] =
            ((right >> position) & 1U) != 0 ? 1.0F : -1.0F;
        for (std::uint32_t bit = 0; bit < kOperatorBits; ++bit) {
            cache.features[feature_offset + 2 + bit] =
                ((raw_operator >> bit) & 1U) != 0 ? 1.0F : -1.0F;
        }
        cache.features[feature_offset + 10] =
            position == 0 ? 1.0F : -1.0F;
        cache.features[feature_offset + 11] =
            position + 1 == width ? 1.0F : -1.0F;
        for (std::size_t channel = 0; channel < rank; ++channel) {
            cache.features[
                feature_offset + kBaseFeatureCount + channel
            ] = parameters.relation_position[
                position * rank + channel
            ] * left_relations[channel] * right_relations[channel];
        }
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            float value = parameters.input_bias[unit];
            for (std::size_t feature = 0;
                 feature < features;
                 ++feature) {
                value += parameters.input[
                    unit * features + feature
                ] * cache.features[feature_offset + feature];
            }
            cache.hidden[position * hidden + unit] = std::tanh(value);
        }
    }

    const auto initial_projection = project_states(
        config,
        parameters,
        std::span<const float>(
            cache.hidden.data(), width * hidden
        )
    );
    for (std::size_t step = 0; step < steps; ++step) {
        const std::size_t current_offset = step * width * hidden;
        const std::size_t next_offset = (step + 1) * width * hidden;
        const std::size_t candidate_offset = step * width * hidden;
        const auto current_projection = project_states(
            config,
            parameters,
            std::span<const float>(
                cache.hidden.data() + current_offset,
                width * hidden
            )
        );
        for (std::size_t position = 0; position < width; ++position) {
            for (std::size_t unit = 0; unit < hidden; ++unit) {
                float value = parameters.recurrent_bias[unit];
                float update_value = parameters.update_bias[unit];
                for (std::size_t channel = 0;
                     channel < rank;
                     ++channel) {
                    value += parameters.self_lift[
                        unit * rank + channel
                    ] * current_projection[position * rank + channel];
                    value += parameters.anchor_lift[
                        unit * rank + channel
                    ] * initial_projection[position * rank + channel];
                    update_value += parameters.update_self_lift[
                        unit * rank + channel
                    ] * current_projection[position * rank + channel];
                    update_value += parameters.update_anchor_lift[
                        unit * rank + channel
                    ] * initial_projection[position * rank + channel];
                    if (position > 0) {
                        value += parameters.left_lift[
                            unit * rank + channel
                        ] * current_projection[
                            (position - 1) * rank + channel
                        ];
                        update_value += parameters.update_left_lift[
                            unit * rank + channel
                        ] * current_projection[
                            (position - 1) * rank + channel
                        ];
                    }
                    if (position + 1 < width) {
                        value += parameters.right_lift[
                            unit * rank + channel
                        ] * current_projection[
                            (position + 1) * rank + channel
                        ];
                        update_value += parameters.update_right_lift[
                            unit * rank + channel
                        ] * current_projection[
                            (position + 1) * rank + channel
                        ];
                    }
                }
                const float candidate = std::tanh(value);
                const float update_gate = sigmoid(update_value);
                cache.candidates[
                    candidate_offset + position * hidden + unit
                ] = candidate;
                cache.update_gates[
                    candidate_offset + position * hidden + unit
                ] = update_gate;
                cache.hidden[
                    next_offset + position * hidden + unit
                ] =
                    (1.0F - update_gate) * cache.hidden[
                        current_offset + position * hidden + unit
                    ]
                    + update_gate * candidate;
            }
        }
    }
    for (std::size_t step = 0; step <= steps; ++step) {
        const std::size_t hidden_offset = step * width * hidden;
        for (std::size_t position = 0; position < width; ++position) {
            float logit = parameters.output_bias;
            for (std::size_t unit = 0; unit < hidden; ++unit) {
                logit += parameters.output[unit]
                    * cache.hidden[
                        hidden_offset + position * hidden + unit
                    ];
            }
            cache.logits_by_step[step * width + position] = logit;
        }
    }
    return cache;
}

double backward_case(
    const ProofKernelConfig& config,
    const NeuralProofKernel::Parameters& parameters,
    const FieldCache& cache,
    std::uint32_t target,
    NeuralProofKernel::Parameters& gradient
) {
    const std::size_t width = config.bit_width;
    const std::size_t hidden = config.hidden_units;
    const std::size_t rank = config.shared_rank;
    const std::size_t steps = config.reasoning_steps;
    const std::size_t features = feature_count(config);
    std::vector<float> hidden_gradient(cache.hidden.size(), 0.0F);
    double loss = 0.0;
    const std::size_t supervision_start = steps / 2;
    const std::size_t supervised_steps =
        steps - supervision_start + 1;
    const float weight_normalizer = static_cast<float>(
        supervised_steps * (supervised_steps + 1) / 2
    );
    for (std::size_t supervised = supervision_start;
         supervised <= steps;
         ++supervised) {
        const float temporal_weight = static_cast<float>(
            supervised - supervision_start + 1
        ) / weight_normalizer;
        const std::size_t hidden_offset =
            supervised * width * hidden;
        for (std::size_t position = 0; position < width; ++position) {
            const float truth =
                ((target >> position) & 1U) != 0 ? 1.0F : 0.0F;
            const float logit =
                cache.logits_by_step[supervised * width + position];
            loss += temporal_weight
                * binary_cross_entropy(logit, truth);
            const float logit_gradient =
                temporal_weight * (sigmoid(logit) - truth);
            gradient.output_bias += logit_gradient;
            for (std::size_t unit = 0; unit < hidden; ++unit) {
                gradient.output[unit] += logit_gradient
                    * cache.hidden[
                        hidden_offset + position * hidden + unit
                    ];
                hidden_gradient[
                    hidden_offset + position * hidden + unit
                ] += logit_gradient * parameters.output[unit];
            }
        }
    }

    for (std::size_t reverse = steps; reverse > 0; --reverse) {
        const std::size_t step = reverse - 1;
        const std::size_t current_offset = step * width * hidden;
        const std::size_t next_offset = (step + 1) * width * hidden;
        const std::size_t candidate_offset = step * width * hidden;
        const auto current_projection = project_states(
            config,
            parameters,
            std::span<const float>(
                cache.hidden.data() + current_offset,
                width * hidden
            )
        );
        const auto initial_projection = project_states(
            config,
            parameters,
            std::span<const float>(
                cache.hidden.data(), width * hidden
            )
        );
        std::vector<float> projected_gradient(width * rank, 0.0F);
        std::vector<float> initial_projected_gradient(
            width * rank, 0.0F
        );
        for (std::size_t position = 0; position < width; ++position) {
            for (std::size_t unit = 0; unit < hidden; ++unit) {
                const float upstream = hidden_gradient[
                    next_offset + position * hidden + unit
                ];
                const float current = cache.hidden[
                    current_offset + position * hidden + unit
                ];
                const float update_gate = cache.update_gates[
                    candidate_offset + position * hidden + unit
                ];
                hidden_gradient[
                    current_offset + position * hidden + unit
                ] += (1.0F - update_gate) * upstream;
                const float candidate = cache.candidates[
                    candidate_offset + position * hidden + unit
                ];
                const float candidate_local =
                    update_gate
                    * upstream
                    * (1.0F - candidate * candidate);
                const float update_local =
                    upstream
                    * (candidate - current)
                    * update_gate
                    * (1.0F - update_gate);
                gradient.recurrent_bias[unit] += candidate_local;
                gradient.update_bias[unit] += update_local;
                for (std::size_t channel = 0;
                     channel < rank;
                     ++channel) {
                    const std::size_t lift_index =
                        unit * rank + channel;
                    gradient.self_lift[lift_index] += candidate_local
                        * current_projection[position * rank + channel];
                    gradient.update_self_lift[lift_index] += update_local
                        * current_projection[position * rank + channel];
                    projected_gradient[position * rank + channel] +=
                        candidate_local
                            * parameters.self_lift[lift_index]
                        + update_local
                            * parameters.update_self_lift[lift_index];
                    gradient.anchor_lift[lift_index] += candidate_local
                        * initial_projection[position * rank + channel];
                    gradient.update_anchor_lift[lift_index] += update_local
                        * initial_projection[position * rank + channel];
                    initial_projected_gradient[
                        position * rank + channel
                    ] += candidate_local
                            * parameters.anchor_lift[lift_index]
                        + update_local
                            * parameters.update_anchor_lift[lift_index];
                    if (position > 0) {
                        gradient.left_lift[lift_index] += candidate_local
                            * current_projection[
                                (position - 1) * rank + channel
                            ];
                        gradient.update_left_lift[
                            lift_index
                        ] += update_local * current_projection[
                            (position - 1) * rank + channel
                        ];
                        projected_gradient[
                            (position - 1) * rank + channel
                        ] += candidate_local
                                * parameters.left_lift[lift_index]
                            + update_local
                                * parameters.update_left_lift[lift_index];
                    }
                    if (position + 1 < width) {
                        gradient.right_lift[lift_index] += candidate_local
                            * current_projection[
                                (position + 1) * rank + channel
                            ];
                        gradient.update_right_lift[
                            lift_index
                        ] += update_local * current_projection[
                            (position + 1) * rank + channel
                        ];
                        projected_gradient[
                            (position + 1) * rank + channel
                        ] += candidate_local
                                * parameters.right_lift[lift_index]
                            + update_local
                                * parameters.update_right_lift[lift_index];
                    }
                }
            }
        }
        for (std::size_t position = 0; position < width; ++position) {
            for (std::size_t channel = 0; channel < rank; ++channel) {
                const float current_local =
                    projected_gradient[position * rank + channel];
                const float initial_local =
                    initial_projected_gradient[
                        position * rank + channel
                    ];
                for (std::size_t unit = 0; unit < hidden; ++unit) {
                    const std::size_t projection_index =
                        channel * hidden + unit;
                    gradient.projection[projection_index] +=
                        current_local * cache.hidden[
                            current_offset + position * hidden + unit
                        ];
                    gradient.projection[projection_index] +=
                        initial_local
                        * cache.hidden[position * hidden + unit];
                    hidden_gradient[
                        current_offset + position * hidden + unit
                    ] += current_local
                        * parameters.projection[projection_index];
                    hidden_gradient[position * hidden + unit] +=
                        initial_local
                        * parameters.projection[projection_index];
                }
            }
        }
    }

    std::vector<float> relation_feature_gradient(width * rank, 0.0F);
    for (std::size_t position = 0; position < width; ++position) {
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            const float activation =
                cache.hidden[position * hidden + unit];
            const float local =
                hidden_gradient[position * hidden + unit]
                * (1.0F - activation * activation);
            gradient.input_bias[unit] += local;
            for (std::size_t feature = 0;
                 feature < features;
                 ++feature) {
                gradient.input[unit * features + feature] +=
                    local
                    * cache.features[position * features + feature];
            }
            for (std::size_t channel = 0; channel < rank; ++channel) {
                relation_feature_gradient[position * rank + channel] +=
                    local * parameters.input[
                        unit * features
                        + kBaseFeatureCount
                        + channel
                    ];
            }
        }
    }
    std::vector<float> left_relations(rank, 0.0F);
    std::vector<float> right_relations(rank, 0.0F);
    for (std::size_t source = 0; source < width; ++source) {
        const float left_bit =
            cache.features[source * features] > 0.0F ? 1.0F : 0.0F;
        const float right_bit =
            cache.features[source * features + 1] > 0.0F
                ? 1.0F
                : 0.0F;
        for (std::size_t channel = 0; channel < rank; ++channel) {
            left_relations[channel] += left_bit
                * parameters.relation_left[source * rank + channel];
            right_relations[channel] += right_bit
                * parameters.relation_right[source * rank + channel];
        }
    }
    std::vector<float> left_relation_gradient(rank, 0.0F);
    std::vector<float> right_relation_gradient(rank, 0.0F);
    for (std::size_t position = 0; position < width; ++position) {
        for (std::size_t channel = 0; channel < rank; ++channel) {
            const std::size_t index = position * rank + channel;
            const float local = relation_feature_gradient[index];
            gradient.relation_position[index] += local
                * left_relations[channel]
                * right_relations[channel];
            left_relation_gradient[channel] += local
                * parameters.relation_position[index]
                * right_relations[channel];
            right_relation_gradient[channel] += local
                * parameters.relation_position[index]
                * left_relations[channel];
        }
    }
    for (std::size_t source = 0; source < width; ++source) {
        const float left_bit =
            cache.features[source * features] > 0.0F ? 1.0F : 0.0F;
        const float right_bit =
            cache.features[source * features + 1] > 0.0F
                ? 1.0F
                : 0.0F;
        for (std::size_t channel = 0; channel < rank; ++channel) {
            gradient.relation_left[source * rank + channel] +=
                left_relation_gradient[channel] * left_bit;
            gradient.relation_right[source * rank + channel] +=
                right_relation_gradient[channel] * right_bit;
        }
    }
    const float normalization = 1.0F / static_cast<float>(width);
    for (auto* tensor : gradient.tensors()) {
        for (float& value : *tensor) {
            value *= normalization;
        }
    }
    gradient.output_bias *= normalization;
    return loss / static_cast<double>(width);
}

void add(
    NeuralProofKernel::Parameters& destination,
    const NeuralProofKernel::Parameters& source
) {
    auto destination_tensors = destination.tensors();
    const auto source_tensors = source.tensors();
    for (std::size_t tensor = 0;
         tensor < destination_tensors.size();
         ++tensor) {
        for (std::size_t index = 0;
             index < destination_tensors[tensor]->size();
             ++index) {
            (*destination_tensors[tensor])[index] +=
                (*source_tensors[tensor])[index];
        }
    }
    destination.output_bias += source.output_bias;
}

void add_group_penalty(
    const ProofKernelConfig& config,
    const NeuralProofKernel::Parameters& parameters,
    NeuralProofKernel::Parameters& gradient,
    float strength
) {
    if (!(strength > 0.0F)) {
        return;
    }
    const std::size_t hidden = config.hidden_units;
    const std::size_t rank = config.shared_rank;
    for (std::size_t channel = 0; channel < rank; ++channel) {
        double squared = 1.0e-12;
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            const float projected =
                parameters.projection[channel * hidden + unit];
            squared += static_cast<double>(projected) * projected;
            for (const auto* lift : {
                &parameters.self_lift,
                &parameters.left_lift,
                &parameters.right_lift,
                &parameters.anchor_lift,
                &parameters.update_self_lift,
                &parameters.update_left_lift,
                &parameters.update_right_lift,
                &parameters.update_anchor_lift
            }) {
                const float value = (*lift)[unit * rank + channel];
                squared += static_cast<double>(value) * value;
            }
        }
        for (std::size_t position = 0;
             position < config.bit_width;
             ++position) {
            for (const auto* factor : {
                &parameters.relation_left,
                &parameters.relation_right,
                &parameters.relation_position
            }) {
                const float value =
                    (*factor)[position * rank + channel];
                squared += static_cast<double>(value) * value;
            }
        }
        const float inverse_norm =
            1.0F / static_cast<float>(std::sqrt(squared));
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            gradient.projection[channel * hidden + unit] +=
                strength
                * parameters.projection[channel * hidden + unit]
                * inverse_norm;
            for (auto pair : {
                std::pair{
                    &parameters.self_lift, &gradient.self_lift
                },
                std::pair{
                    &parameters.left_lift, &gradient.left_lift
                },
                std::pair{
                    &parameters.right_lift, &gradient.right_lift
                },
                std::pair{
                    &parameters.anchor_lift, &gradient.anchor_lift
                },
                std::pair{
                    &parameters.update_self_lift,
                    &gradient.update_self_lift
                },
                std::pair{
                    &parameters.update_left_lift,
                    &gradient.update_left_lift
                },
                std::pair{
                    &parameters.update_right_lift,
                    &gradient.update_right_lift
                },
                std::pair{
                    &parameters.update_anchor_lift,
                    &gradient.update_anchor_lift
                }
            }) {
                (*pair.second)[unit * rank + channel] +=
                    strength
                    * (*pair.first)[unit * rank + channel]
                    * inverse_norm;
            }
        }
        for (std::size_t position = 0;
             position < config.bit_width;
             ++position) {
            for (auto pair : {
                std::pair{
                    &parameters.relation_left,
                    &gradient.relation_left
                },
                std::pair{
                    &parameters.relation_right,
                    &gradient.relation_right
                },
                std::pair{
                    &parameters.relation_position,
                    &gradient.relation_position
                }
            }) {
                const std::size_t index =
                    position * rank + channel;
                (*pair.second)[index] += strength
                    * (*pair.first)[index] * inverse_norm;
            }
        }
    }
}

void adam_update(
    NeuralProofKernel::Parameters& parameters,
    NeuralProofKernel::AdamState& adam,
    const NeuralProofKernel::Parameters& gradient,
    float learning_rate,
    float scale
) {
    ++adam.step;
    const float correction1 =
        1.0F - std::pow(kAdamBeta1, static_cast<float>(adam.step));
    const float correction2 =
        1.0F - std::pow(kAdamBeta2, static_cast<float>(adam.step));
    auto values = parameters.tensors();
    auto first = adam.first.tensors();
    auto second = adam.second.tensors();
    const auto gradients = gradient.tensors();
    for (std::size_t tensor = 0; tensor < values.size(); ++tensor) {
        for (std::size_t index = 0;
             index < values[tensor]->size();
             ++index) {
            const float local = (*gradients[tensor])[index] * scale;
            (*first[tensor])[index] =
                kAdamBeta1 * (*first[tensor])[index]
                + (1.0F - kAdamBeta1) * local;
            (*second[tensor])[index] =
                kAdamBeta2 * (*second[tensor])[index]
                + (1.0F - kAdamBeta2) * local * local;
            (*values[tensor])[index] -= learning_rate
                * ((*first[tensor])[index] / correction1)
                / (
                    std::sqrt((*second[tensor])[index] / correction2)
                    + kAdamEpsilon
                );
        }
    }
    const float bias_gradient = gradient.output_bias * scale;
    adam.first.output_bias =
        kAdamBeta1 * adam.first.output_bias
        + (1.0F - kAdamBeta1) * bias_gradient;
    adam.second.output_bias =
        kAdamBeta2 * adam.second.output_bias
        + (1.0F - kAdamBeta2) * bias_gradient * bias_gradient;
    parameters.output_bias -= learning_rate
        * (adam.first.output_bias / correction1)
        / (
            std::sqrt(adam.second.output_bias / correction2)
            + kAdamEpsilon
        );
}

}  // namespace

NeuralProofKernel::NeuralProofKernel(
    ProofKernelConfig config,
    std::uint64_t seed
) : config_(config),
    parameters_(new Parameters(make_like(config, 0.0F))),
    adam_(new AdamState{
        make_like(config, 0.0F), make_like(config, 0.0F)
    }) {
    validate_config(config_);
    initialize(*parameters_, config_, seed);
    telemetry_.parameters = parameter_count(*parameters_);
    telemetry_.dense_equivalent_parameters =
        dense_equivalent_count(config_);
}

NeuralProofKernel::NeuralProofKernel(NeuralProofKernel&& other) noexcept
    : config_(other.config_),
      parameters_(std::exchange(other.parameters_, nullptr)),
      adam_(std::exchange(other.adam_, nullptr)),
      telemetry_(other.telemetry_) {}

NeuralProofKernel& NeuralProofKernel::operator=(
    NeuralProofKernel&& other
) noexcept {
    if (this == &other) {
        return *this;
    }
    delete parameters_;
    delete adam_;
    config_ = other.config_;
    parameters_ = std::exchange(other.parameters_, nullptr);
    adam_ = std::exchange(other.adam_, nullptr);
    telemetry_ = other.telemetry_;
    return *this;
}

NeuralProofKernel::~NeuralProofKernel() {
    delete parameters_;
    delete adam_;
}

double NeuralProofKernel::train_batch(
    std::span<const ArithmeticTrainingCase> cases,
    float learning_rate,
    float group_penalty
) {
    if (cases.empty()) {
        throw std::invalid_argument("Proof training batch is empty");
    }
    if (!(learning_rate > 0.0F) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Invalid proof learning rate");
    }
    const std::uint32_t limit = value_limit(config_);
    Parameters batch_gradient = make_like(config_, 0.0F);
    double loss = 0.0;
    for (const auto& item : cases) {
        if (
            item.left > limit
            || item.right > limit
            || item.target > limit
        ) {
            throw std::invalid_argument(
                "Proof training value exceeds configured bit width"
            );
        }
        const FieldCache cache = forward_case(
            config_, *parameters_, item.left, item.right,
            item.raw_operator
        );
        Parameters local_gradient = make_like(config_, 0.0F);
        loss += backward_case(
            config_, *parameters_, cache, item.target, local_gradient
        );
        add(batch_gradient, local_gradient);
    }
    add_group_penalty(
        config_, *parameters_, batch_gradient, group_penalty
    );
    const float scale = 1.0F / static_cast<float>(cases.size());
    adam_update(
        *parameters_, *adam_, batch_gradient, learning_rate, scale
    );
    ++telemetry_.optimizer_steps;
    telemetry_.training_cases += cases.size();
    telemetry_.most_recent_loss =
        loss / static_cast<double>(cases.size());
    adam_->recent_loss += telemetry_.most_recent_loss;
    ++adam_->recent_count;
    if (adam_->recent_count >= 128) {
        telemetry_.mean_recent_loss =
            adam_->recent_loss / static_cast<double>(adam_->recent_count);
        adam_->recent_loss = 0.0;
        adam_->recent_count = 0;
    }
    return telemetry_.most_recent_loss;
}

ArithmeticInference NeuralProofKernel::infer(
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t raw_operator
) const {
    const std::uint32_t limit = value_limit(config_);
    if (left > limit || right > limit) {
        throw std::invalid_argument(
            "Proof inference operand exceeds configured bit width"
        );
    }
    const auto start = Clock::now();
    const FieldCache cache = forward_case(
        config_, *parameters_, left, right, raw_operator
    );
    ArithmeticInference result;
    result.logits.resize(config_.bit_width);
    const std::size_t final_offset =
        config_.reasoning_steps * config_.bit_width;
    float confidence = 0.0F;
    for (std::size_t bit = 0; bit < config_.bit_width; ++bit) {
        const float logit = cache.logits_by_step[final_offset + bit];
        result.logits[bit] = logit;
        if (logit >= 0.0F) {
            result.value |= std::uint32_t{1} << bit;
        }
        confidence += std::abs(sigmoid(logit) - 0.5F) * 2.0F;
    }
    result.mean_confidence =
        confidence / static_cast<float>(config_.bit_width);
    result.stable_after_step = config_.reasoning_steps;
    std::uint32_t previous = 0;
    std::uint32_t stable_count = 0;
    for (std::uint32_t step = 0;
         step <= config_.reasoning_steps;
         ++step) {
        std::uint32_t value = 0;
        for (std::size_t bit = 0; bit < config_.bit_width; ++bit) {
            if (
                cache.logits_by_step[
                    step * config_.bit_width + bit
                ] >= 0.0F
            ) {
                value |= std::uint32_t{1} << bit;
            }
        }
        if (step != 0 && value == previous) {
            ++stable_count;
            if (stable_count >= 3) {
                result.stable_after_step = step - 2;
                break;
            }
        } else {
            stable_count = 0;
        }
        previous = value;
    }
    telemetry_.inference_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start
        ).count()
    );
    return result;
}

const ProofKernelConfig& NeuralProofKernel::config() const noexcept {
    return config_;
}

const ProofKernelTelemetry&
NeuralProofKernel::telemetry() const noexcept {
    return telemetry_;
}

std::uint64_t NeuralProofKernel::parameter_checksum() const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    };
    for (const auto* tensor : parameters_->tensors()) {
        mix(tensor->data(), tensor->size() * sizeof(float));
    }
    mix(&parameters_->output_bias, sizeof(parameters_->output_bias));
    return hash;
}

std::vector<double> NeuralProofKernel::rank_channel_norms() const {
    const std::size_t hidden = config_.hidden_units;
    const std::size_t rank = config_.shared_rank;
    std::vector<double> norms(rank, 0.0);
    for (std::size_t channel = 0; channel < rank; ++channel) {
        double squared = 0.0;
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            const float projected =
                parameters_->projection[channel * hidden + unit];
            squared += static_cast<double>(projected) * projected;
            for (const auto* lift : {
                &parameters_->self_lift,
                &parameters_->left_lift,
                &parameters_->right_lift,
                &parameters_->anchor_lift,
                &parameters_->update_self_lift,
                &parameters_->update_left_lift,
                &parameters_->update_right_lift,
                &parameters_->update_anchor_lift
            }) {
                const float value = (*lift)[unit * rank + channel];
                squared += static_cast<double>(value) * value;
            }
        }
        for (std::size_t position = 0;
             position < config_.bit_width;
             ++position) {
            for (const auto* factor : {
                &parameters_->relation_left,
                &parameters_->relation_right,
                &parameters_->relation_position
            }) {
                const float value =
                    (*factor)[position * rank + channel];
                squared += static_cast<double>(value) * value;
            }
        }
        norms[channel] = std::sqrt(squared);
    }
    return norms;
}

void NeuralProofKernel::save_checkpoint(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to create proof checkpoint");
    }
    write_scalar(out, kCheckpointMagic);
    write_scalar(out, kCheckpointVersion);
    write_scalar(out, config_.bit_width);
    write_scalar(out, config_.hidden_units);
    write_scalar(out, config_.shared_rank);
    write_scalar(out, config_.reasoning_steps);
    write_scalar(out, config_.residual_rate);
    for (const auto* tensor : parameters_->tensors()) {
        write_vector(out, *tensor);
    }
    write_scalar(out, parameters_->output_bias);
    write_scalar(out, adam_->step);
    write_scalar(out, telemetry_.training_cases);
}

void NeuralProofKernel::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open proof checkpoint");
    }
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    ProofKernelConfig stored;
    read_scalar(in, magic);
    read_scalar(in, version);
    read_scalar(in, stored.bit_width);
    read_scalar(in, stored.hidden_units);
    read_scalar(in, stored.shared_rank);
    read_scalar(in, stored.reasoning_steps);
    read_scalar(in, stored.residual_rate);
    if (
        magic != kCheckpointMagic
        || version != kCheckpointVersion
        || stored.bit_width != config_.bit_width
        || stored.hidden_units != config_.hidden_units
        || stored.shared_rank != config_.shared_rank
        || stored.reasoning_steps != config_.reasoning_steps
        || stored.residual_rate != config_.residual_rate
    ) {
        throw std::runtime_error(
            "Proof checkpoint version or configuration mismatch"
        );
    }
    for (auto* tensor : parameters_->tensors()) {
        read_vector(in, *tensor, tensor->size());
    }
    read_scalar(in, parameters_->output_bias);
    read_scalar(in, adam_->step);
    read_scalar(in, telemetry_.training_cases);
    telemetry_.optimizer_steps = adam_->step;
    telemetry_.parameters = parameter_count(*parameters_);
    telemetry_.dense_equivalent_parameters =
        dense_equivalent_count(config_);
}

void NeuralProofKernel::write_json(std::ostream& out) const {
    out << std::setprecision(12)
        << "{\n"
        << "  \"architecture\": "
           "\"cp_relational_gated_neural_proof_kernel\",\n"
        << "  \"transformer_attention\": false,\n"
        << "  \"token_autoregression\": false,\n"
        << "  \"bit_width\": " << config_.bit_width << ",\n"
        << "  \"hidden_units\": " << config_.hidden_units << ",\n"
        << "  \"shared_rank\": " << config_.shared_rank << ",\n"
        << "  \"reasoning_steps\": " << config_.reasoning_steps
        << ",\n"
        << "  \"parameters\": " << telemetry_.parameters << ",\n"
        << "  \"dense_equivalent_parameters\": "
        << telemetry_.dense_equivalent_parameters << ",\n"
        << "  \"parameter_reduction\": "
        << 1.0
            - static_cast<double>(telemetry_.parameters)
                / static_cast<double>(
                    telemetry_.dense_equivalent_parameters
                )
        << ",\n"
        << "  \"optimizer_steps\": " << telemetry_.optimizer_steps
        << ",\n"
        << "  \"training_cases\": " << telemetry_.training_cases
        << ",\n"
        << "  \"parameter_checksum\": " << parameter_checksum()
        << ",\n"
        << "  \"rank_channel_norms\": [";
    const auto norms = rank_channel_norms();
    for (std::size_t index = 0; index < norms.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << norms[index];
    }
    out << "]\n}\n";
}

}  // namespace sera

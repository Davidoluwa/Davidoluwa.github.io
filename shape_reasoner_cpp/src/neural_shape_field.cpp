#include "sera/neural_shape_field.hpp"

#include <algorithm>
#include <bit>
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

constexpr std::uint32_t kBaseFeatureCount = 2;
constexpr float kObservedLossWeight = 0.10F;
constexpr float kAdamBeta1 = 0.9F;
constexpr float kAdamBeta2 = 0.999F;
constexpr float kAdamEpsilon = 1.0e-8F;
constexpr std::uint64_t kCheckpointMagic = 0x534552414E534633ULL;
constexpr std::uint32_t kCheckpointVersion = 1;

std::size_t checked_states(std::uint32_t input_bits) {
    if (input_bits == 0 || input_bits > 12) {
        throw std::invalid_argument(
            "Neural shape field supports 1..12 input bits"
        );
    }
    return std::size_t{1} << input_bits;
}

std::size_t feature_count(const NeuralShapeConfig& config) {
    return kBaseFeatureCount + config.input_bits;
}

std::uint64_t encode_input(std::span<const std::uint8_t> input) {
    std::uint64_t encoded = 0;
    for (std::size_t bit = 0; bit < input.size(); ++bit) {
        if (input[bit] > 1) {
            throw std::invalid_argument("Boolean input must contain 0 or 1");
        }
        encoded |= static_cast<std::uint64_t>(input[bit]) << bit;
    }
    return encoded;
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
        throw std::runtime_error("Failed while writing neural checkpoint");
    }
}

template <typename T>
void read_scalar(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Truncated neural checkpoint");
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
        throw std::runtime_error("Failed while writing checkpoint vector");
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
        throw std::runtime_error("Neural checkpoint tensor shape mismatch");
    }
    values.resize(expected);
    in.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!in) {
        throw std::runtime_error("Truncated checkpoint vector");
    }
}

struct ObservationTable {
    std::vector<std::uint8_t> mask;
    std::vector<std::uint8_t> outputs;
};

ObservationTable make_observations(
    std::span<const BooleanExample> examples,
    const NeuralShapeConfig& config
) {
    const std::size_t states = checked_states(config.input_bits);
    ObservationTable table{
        std::vector<std::uint8_t>(states, 0),
        std::vector<std::uint8_t>(states * config.output_bits, 0)
    };
    for (const auto& example : examples) {
        if (
            example.input.size() != config.input_bits
            || example.output.size() != config.output_bits
        ) {
            throw std::invalid_argument(
                "Neural shape example dimension mismatch"
            );
        }
        const std::size_t state = encode_input(example.input);
        for (std::uint32_t output = 0;
             output < config.output_bits;
             ++output) {
            if (example.output[output] > 1) {
                throw std::invalid_argument(
                    "Boolean output must contain 0 or 1"
                );
            }
            const std::size_t index =
                state * config.output_bits + output;
            if (
                table.mask[state] != 0
                && table.outputs[index] != example.output[output]
            ) {
                throw std::invalid_argument(
                    "Conflicting outputs for one neural shape input"
                );
            }
            table.outputs[index] = example.output[output];
        }
        table.mask[state] = 1;
    }
    return table;
}

}  // namespace

struct NeuralShapeField::Parameters {
    std::vector<float> input;
    std::vector<float> input_bias;
    std::vector<float> self;
    std::vector<float> neighbor;
    std::vector<float> anchor;
    std::vector<float> recurrent_bias;
    std::vector<float> output;
    float output_bias{};

    [[nodiscard]] std::vector<std::vector<float>*> tensors() {
        return {
            &input, &input_bias, &self, &neighbor, &anchor,
            &recurrent_bias, &output
        };
    }

    [[nodiscard]] std::vector<const std::vector<float>*> tensors() const {
        return {
            &input, &input_bias, &self, &neighbor, &anchor,
            &recurrent_bias, &output
        };
    }
};

struct NeuralShapeField::AdamState {
    Parameters first;
    Parameters second;
    std::uint64_t step{};
    double recent_loss_sum{};
    std::uint32_t recent_loss_count{};
};

namespace {

NeuralShapeField::Parameters make_like(
    const NeuralShapeConfig& config,
    float fill
) {
    const std::size_t hidden = config.hidden_units;
    const std::size_t features = feature_count(config);
    NeuralShapeField::Parameters result;
    result.input.assign(hidden * features, fill);
    result.input_bias.assign(hidden, fill);
    result.self.assign(hidden * hidden, fill);
    result.neighbor.assign(hidden * hidden, fill);
    result.anchor.assign(hidden * hidden, fill);
    result.recurrent_bias.assign(hidden, fill);
    result.output.assign(hidden, fill);
    result.output_bias = fill;
    return result;
}

std::uint64_t count_parameters(
    const NeuralShapeField::Parameters& parameters
) {
    std::uint64_t count = 1;
    for (const auto* tensor : parameters.tensors()) {
        count += tensor->size();
    }
    return count;
}

void initialize_parameters(
    NeuralShapeField::Parameters& parameters,
    const NeuralShapeConfig& config,
    std::uint64_t seed
) {
    std::mt19937_64 random(seed);
    auto fill_xavier = [&random](
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
    fill_xavier(
        parameters.input, feature_count(config), config.hidden_units
    );
    fill_xavier(
        parameters.self, config.hidden_units, config.hidden_units
    );
    fill_xavier(
        parameters.neighbor, config.hidden_units, config.hidden_units
    );
    fill_xavier(
        parameters.anchor, config.hidden_units, config.hidden_units
    );
    fill_xavier(parameters.output, config.hidden_units, 1);
}

struct FieldCache {
    std::vector<float> features;
    std::vector<float> hidden;
    std::vector<float> candidates;
    std::vector<float> logits;
};

FieldCache forward_field(
    const NeuralShapeConfig& config,
    const NeuralShapeField::Parameters& parameters,
    std::span<const std::uint8_t> mask,
    std::span<const std::uint8_t> observed
) {
    const std::size_t states = checked_states(config.input_bits);
    const std::size_t hidden = config.hidden_units;
    const std::size_t steps = config.reasoning_steps;
    const std::size_t features = feature_count(config);
    FieldCache cache;
    cache.features.assign(states * features, 0.0F);
    cache.hidden.assign((steps + 1) * states * hidden, 0.0F);
    cache.candidates.assign(steps * states * hidden, 0.0F);
    cache.logits.assign(states, 0.0F);

    for (std::size_t state = 0; state < states; ++state) {
        cache.features[state * features] =
            mask[state] != 0 ? 1.0F : -1.0F;
        cache.features[state * features + 1] =
            mask[state] != 0
                ? (observed[state] != 0 ? 1.0F : -1.0F)
                : 0.0F;
        for (std::uint32_t bit = 0; bit < config.input_bits; ++bit) {
            cache.features[state * features + 2 + bit] =
                ((state >> bit) & 1U) != 0 ? 1.0F : -1.0F;
        }
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            float value = parameters.input_bias[unit];
            for (std::size_t feature = 0;
                 feature < features;
                 ++feature) {
                value += parameters.input[
                    unit * features + feature
                ] * cache.features[state * features + feature];
            }
            cache.hidden[state * hidden + unit] = std::tanh(value);
        }
    }

    const float residual = config.residual_rate;
    for (std::size_t step = 0; step < steps; ++step) {
        const std::size_t current_offset = step * states * hidden;
        const std::size_t next_offset = (step + 1) * states * hidden;
        const std::size_t candidate_offset = step * states * hidden;
        // Aggregate each neighborhood once. The original research prototype
        // recomputed this value for every destination unit, multiplying the
        // graph work by hidden_units without changing the mathematics.
        std::vector<float> neighbor_means(states * hidden, 0.0F);
        for (std::size_t state = 0; state < states; ++state) {
            for (std::size_t unit = 0; unit < hidden; ++unit) {
                float mean = 0.0F;
                for (std::uint32_t bit = 0;
                     bit < config.input_bits;
                     ++bit) {
                    const std::size_t neighbor_state =
                        state ^ (std::size_t{1} << bit);
                    mean += cache.hidden[
                        current_offset + neighbor_state * hidden + unit
                    ];
                }
                neighbor_means[state * hidden + unit] =
                    mean / static_cast<float>(config.input_bits);
            }
        }
        for (std::size_t state = 0; state < states; ++state) {
            for (std::size_t output_unit = 0;
                 output_unit < hidden;
                 ++output_unit) {
                float value = parameters.recurrent_bias[output_unit];
                for (std::size_t input_unit = 0;
                     input_unit < hidden;
                     ++input_unit) {
                    value += parameters.self[
                        output_unit * hidden + input_unit
                    ] * cache.hidden[
                        current_offset + state * hidden + input_unit
                    ];
                    value += parameters.neighbor[
                        output_unit * hidden + input_unit
                    ] * neighbor_means[state * hidden + input_unit];
                    value += parameters.anchor[
                        output_unit * hidden + input_unit
                    ] * cache.hidden[state * hidden + input_unit];
                }
                const float candidate = std::tanh(value);
                cache.candidates[
                    candidate_offset + state * hidden + output_unit
                ] = candidate;
                cache.hidden[
                    next_offset + state * hidden + output_unit
                ] =
                    (1.0F - residual) * cache.hidden[
                        current_offset + state * hidden + output_unit
                    ]
                    + residual * candidate;
            }
        }
    }

    const std::size_t final_offset = steps * states * hidden;
    for (std::size_t state = 0; state < states; ++state) {
        float logit = parameters.output_bias;
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            logit += parameters.output[unit]
                * cache.hidden[final_offset + state * hidden + unit];
        }
        cache.logits[state] = logit;
    }
    return cache;
}

double backward_field(
    const NeuralShapeConfig& config,
    const NeuralShapeField::Parameters& parameters,
    const FieldCache& cache,
    std::span<const std::uint8_t> mask,
    std::span<const std::uint8_t> target,
    NeuralShapeField::Parameters& gradient
) {
    const std::size_t states = checked_states(config.input_bits);
    const std::size_t hidden = config.hidden_units;
    const std::size_t steps = config.reasoning_steps;
    const std::size_t features = feature_count(config);
    std::vector<float> hidden_gradient(cache.hidden.size(), 0.0F);
    double total_weight = 0.0;
    double loss = 0.0;
    const std::size_t final_offset = steps * states * hidden;
    for (std::size_t state = 0; state < states; ++state) {
        const float weight =
            mask[state] != 0 ? kObservedLossWeight : 1.0F;
        const float truth = target[state] != 0 ? 1.0F : 0.0F;
        loss += weight * binary_cross_entropy(cache.logits[state], truth);
        total_weight += weight;
        const float logit_gradient =
            weight * (sigmoid(cache.logits[state]) - truth);
        gradient.output_bias += logit_gradient;
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            gradient.output[unit] += logit_gradient
                * cache.hidden[final_offset + state * hidden + unit];
            hidden_gradient[
                final_offset + state * hidden + unit
            ] += logit_gradient * parameters.output[unit];
        }
    }

    const float residual = config.residual_rate;
    for (std::size_t reverse = steps; reverse > 0; --reverse) {
        const std::size_t step = reverse - 1;
        const std::size_t current_offset = step * states * hidden;
        const std::size_t next_offset = (step + 1) * states * hidden;
        const std::size_t candidate_offset = step * states * hidden;
        for (std::size_t state = 0; state < states; ++state) {
            for (std::size_t output_unit = 0;
                 output_unit < hidden;
                 ++output_unit) {
                const float upstream = hidden_gradient[
                    next_offset + state * hidden + output_unit
                ];
                hidden_gradient[
                    current_offset + state * hidden + output_unit
                ] += (1.0F - residual) * upstream;
                const float candidate = cache.candidates[
                    candidate_offset + state * hidden + output_unit
                ];
                const float local =
                    residual * upstream * (1.0F - candidate * candidate);
                gradient.recurrent_bias[output_unit] += local;
                for (std::size_t input_unit = 0;
                     input_unit < hidden;
                     ++input_unit) {
                    float neighbor_mean = 0.0F;
                    for (std::uint32_t bit = 0;
                         bit < config.input_bits;
                         ++bit) {
                        const std::size_t neighbor_state =
                            state ^ (std::size_t{1} << bit);
                        neighbor_mean += cache.hidden[
                            current_offset
                            + neighbor_state * hidden + input_unit
                        ];
                    }
                    neighbor_mean /= static_cast<float>(config.input_bits);
                    const float current = cache.hidden[
                        current_offset + state * hidden + input_unit
                    ];
                    const float anchor = cache.hidden[
                        state * hidden + input_unit
                    ];
                    const std::size_t matrix_index =
                        output_unit * hidden + input_unit;
                    gradient.self[matrix_index] += local * current;
                    gradient.neighbor[matrix_index] +=
                        local * neighbor_mean;
                    gradient.anchor[matrix_index] += local * anchor;
                    hidden_gradient[
                        current_offset + state * hidden + input_unit
                    ] += local * parameters.self[matrix_index];
                    hidden_gradient[state * hidden + input_unit] +=
                        local * parameters.anchor[matrix_index];
                    const float distributed =
                        local * parameters.neighbor[matrix_index]
                        / static_cast<float>(config.input_bits);
                    for (std::uint32_t bit = 0;
                         bit < config.input_bits;
                         ++bit) {
                        const std::size_t neighbor_state =
                            state ^ (std::size_t{1} << bit);
                        hidden_gradient[
                            current_offset
                            + neighbor_state * hidden + input_unit
                        ] += distributed;
                    }
                }
            }
        }
    }

    for (std::size_t state = 0; state < states; ++state) {
        for (std::size_t unit = 0; unit < hidden; ++unit) {
            const float activation =
                cache.hidden[state * hidden + unit];
            const float local =
                hidden_gradient[state * hidden + unit]
                * (1.0F - activation * activation);
            gradient.input_bias[unit] += local;
            for (std::size_t feature = 0;
                 feature < features;
                 ++feature) {
                gradient.input[unit * features + feature] +=
                    local
                    * cache.features[state * features + feature];
            }
        }
    }
    const float normalization =
        total_weight > 0.0 ? static_cast<float>(1.0 / total_weight) : 1.0F;
    for (auto* tensor : gradient.tensors()) {
        for (float& value : *tensor) {
            value *= normalization;
        }
    }
    gradient.output_bias *= normalization;
    return total_weight > 0.0 ? loss / total_weight : 0.0;
}

void add_parameters(
    NeuralShapeField::Parameters& destination,
    const NeuralShapeField::Parameters& source
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

void adam_update(
    NeuralShapeField::Parameters& parameters,
    NeuralShapeField::AdamState& adam,
    const NeuralShapeField::Parameters& gradient,
    float learning_rate,
    float scale
) {
    ++adam.step;
    const float bias1 =
        1.0F - std::pow(kAdamBeta1, static_cast<float>(adam.step));
    const float bias2 =
        1.0F - std::pow(kAdamBeta2, static_cast<float>(adam.step));
    auto parameter_tensors = parameters.tensors();
    auto first_tensors = adam.first.tensors();
    auto second_tensors = adam.second.tensors();
    const auto gradient_tensors = gradient.tensors();
    for (std::size_t tensor = 0;
         tensor < parameter_tensors.size();
         ++tensor) {
        for (std::size_t index = 0;
             index < parameter_tensors[tensor]->size();
             ++index) {
            const float value =
                (*gradient_tensors[tensor])[index] * scale;
            float& first = (*first_tensors[tensor])[index];
            float& second = (*second_tensors[tensor])[index];
            first = kAdamBeta1 * first + (1.0F - kAdamBeta1) * value;
            second =
                kAdamBeta2 * second
                + (1.0F - kAdamBeta2) * value * value;
            (*parameter_tensors[tensor])[index] -= learning_rate
                * (first / bias1)
                / (std::sqrt(second / bias2) + kAdamEpsilon);
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
        * (adam.first.output_bias / bias1)
        / (
            std::sqrt(adam.second.output_bias / bias2)
            + kAdamEpsilon
        );
}

}  // namespace

NeuralShapeField::NeuralShapeField(
    NeuralShapeConfig config,
    std::uint64_t initialization_seed
) : config_(config),
    parameters_(new Parameters(make_like(config, 0.0F))),
    adam_(new AdamState{
        make_like(config, 0.0F), make_like(config, 0.0F)
    }) {
    if (
        config_.output_bits == 0
        || config_.hidden_units == 0
        || config_.reasoning_steps == 0
        || !(config_.residual_rate > 0.0F)
        || !(config_.residual_rate <= 1.0F)
    ) {
        throw std::invalid_argument("Invalid neural shape configuration");
    }
    checked_states(config_.input_bits);
    initialize_parameters(*parameters_, config_, initialization_seed);
    telemetry_.parameters = count_parameters(*parameters_);
}

NeuralShapeField::NeuralShapeField(NeuralShapeField&& other) noexcept
    : config_(other.config_),
      parameters_(std::exchange(other.parameters_, nullptr)),
      adam_(std::exchange(other.adam_, nullptr)),
      telemetry_(other.telemetry_),
      frozen_shape_(std::move(other.frozen_shape_)),
      frozen_logits_(std::move(other.frozen_logits_)),
      observed_mask_(std::move(other.observed_mask_)),
      observed_outputs_(std::move(other.observed_outputs_)) {}

NeuralShapeField& NeuralShapeField::operator=(
    NeuralShapeField&& other
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
    frozen_shape_ = std::move(other.frozen_shape_);
    frozen_logits_ = std::move(other.frozen_logits_);
    observed_mask_ = std::move(other.observed_mask_);
    observed_outputs_ = std::move(other.observed_outputs_);
    return *this;
}

NeuralShapeField::~NeuralShapeField() {
    delete parameters_;
    delete adam_;
}

double NeuralShapeField::train_episode(
    std::span<const BooleanExample> support,
    std::span<const BooleanExample> complete_relation,
    float learning_rate
) {
    if (!(learning_rate > 0.0F) || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    }
    const std::size_t states = checked_states(config_.input_bits);
    const auto observed = make_observations(support, config_);
    const auto complete = make_observations(complete_relation, config_);
    if (
        std::count(complete.mask.begin(), complete.mask.end(), 1)
        != static_cast<std::ptrdiff_t>(states)
    ) {
        throw std::invalid_argument(
            "Meta-training target must contain the complete relation"
        );
    }

    Parameters episode_gradient = make_like(config_, 0.0F);
    double episode_loss = 0.0;
    for (std::uint32_t output = 0;
         output < config_.output_bits;
         ++output) {
        std::vector<std::uint8_t> observed_field(states, 0);
        std::vector<std::uint8_t> target_field(states, 0);
        for (std::size_t state = 0; state < states; ++state) {
            observed_field[state] = observed.outputs[
                state * config_.output_bits + output
            ];
            target_field[state] = complete.outputs[
                state * config_.output_bits + output
            ];
        }
        const FieldCache cache = forward_field(
            config_, *parameters_, observed.mask, observed_field
        );
        Parameters field_gradient = make_like(config_, 0.0F);
        episode_loss += backward_field(
            config_, *parameters_, cache, observed.mask, target_field,
            field_gradient
        );
        add_parameters(episode_gradient, field_gradient);
    }
    const float scale = 1.0F / static_cast<float>(config_.output_bits);
    adam_update(
        *parameters_, *adam_, episode_gradient, learning_rate, scale
    );
    ++telemetry_.optimizer_steps;
    telemetry_.fields_trained += config_.output_bits;
    telemetry_.most_recent_loss =
        episode_loss / static_cast<double>(config_.output_bits);
    adam_->recent_loss_sum += telemetry_.most_recent_loss;
    ++adam_->recent_loss_count;
    if (adam_->recent_loss_count >= 128) {
        telemetry_.mean_recent_loss =
            adam_->recent_loss_sum
            / static_cast<double>(adam_->recent_loss_count);
        adam_->recent_loss_sum = 0.0;
        adam_->recent_loss_count = 0;
    }
    return telemetry_.most_recent_loss;
}

void NeuralShapeField::compile_shape(
    std::span<const BooleanExample> support
) {
    const auto start = Clock::now();
    const std::size_t states = checked_states(config_.input_bits);
    const auto observed = make_observations(support, config_);
    observed_mask_ = observed.mask;
    observed_outputs_ = observed.outputs;
    frozen_shape_.assign(
        config_.output_bits * states * config_.hidden_units, 0.0F
    );
    frozen_logits_.assign(config_.output_bits * states, 0.0F);
    for (std::uint32_t output = 0;
         output < config_.output_bits;
         ++output) {
        std::vector<std::uint8_t> observed_field(states, 0);
        for (std::size_t state = 0; state < states; ++state) {
            observed_field[state] = observed.outputs[
                state * config_.output_bits + output
            ];
        }
        const FieldCache cache = forward_field(
            config_, *parameters_, observed.mask, observed_field
        );
        const std::size_t final_offset =
            config_.reasoning_steps * states * config_.hidden_units;
        std::copy(
            cache.hidden.begin()
                + static_cast<std::ptrdiff_t>(final_offset),
            cache.hidden.begin()
                + static_cast<std::ptrdiff_t>(
                    final_offset + states * config_.hidden_units
                ),
            frozen_shape_.begin()
                + static_cast<std::ptrdiff_t>(
                    output * states * config_.hidden_units
                )
        );
        std::copy(
            cache.logits.begin(), cache.logits.end(),
            frozen_logits_.begin()
                + static_cast<std::ptrdiff_t>(output * states)
        );
    }
    telemetry_.compile_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start
        ).count()
    );
}

std::vector<std::uint8_t> NeuralShapeField::predict(
    std::span<const std::uint8_t> input
) const {
    if (
        input.size() != config_.input_bits
        || frozen_logits_.empty()
    ) {
        throw std::invalid_argument(
            "Neural shape is uncompiled or input dimension is wrong"
        );
    }
    const std::size_t states = checked_states(config_.input_bits);
    const std::size_t state = encode_input(input);
    std::vector<std::uint8_t> output(config_.output_bits, 0);
    for (std::uint32_t wire = 0;
         wire < config_.output_bits;
         ++wire) {
        if (observed_mask_[state] != 0) {
            output[wire] =
                observed_outputs_[state * config_.output_bits + wire];
        } else {
            output[wire] =
                frozen_logits_[wire * states + state] >= 0.0F ? 1U : 0U;
        }
    }
    return output;
}

const std::vector<float>&
NeuralShapeField::frozen_shape() const noexcept {
    return frozen_shape_;
}

const std::vector<float>&
NeuralShapeField::frozen_logits() const noexcept {
    return frozen_logits_;
}

const NeuralShapeConfig&
NeuralShapeField::config() const noexcept {
    return config_;
}

const NeuralShapeTelemetry&
NeuralShapeField::telemetry() const noexcept {
    return telemetry_;
}

std::uint64_t NeuralShapeField::parameter_checksum() const noexcept {
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

void NeuralShapeField::save_checkpoint(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open neural checkpoint for writing");
    }
    write_scalar(out, kCheckpointMagic);
    write_scalar(out, kCheckpointVersion);
    write_scalar(out, config_.input_bits);
    write_scalar(out, config_.output_bits);
    write_scalar(out, config_.hidden_units);
    write_scalar(out, config_.reasoning_steps);
    write_scalar(out, config_.residual_rate);
    for (const auto* tensor : parameters_->tensors()) {
        write_vector(out, *tensor);
    }
    write_scalar(out, parameters_->output_bias);
    write_scalar(out, adam_->step);
}

void NeuralShapeField::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open neural checkpoint");
    }
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    NeuralShapeConfig stored;
    read_scalar(in, magic);
    read_scalar(in, version);
    read_scalar(in, stored.input_bits);
    read_scalar(in, stored.output_bits);
    read_scalar(in, stored.hidden_units);
    read_scalar(in, stored.reasoning_steps);
    read_scalar(in, stored.residual_rate);
    if (
        magic != kCheckpointMagic
        || version != kCheckpointVersion
        || stored.input_bits != config_.input_bits
        || stored.output_bits != config_.output_bits
        || stored.hidden_units != config_.hidden_units
        || stored.reasoning_steps != config_.reasoning_steps
        || stored.residual_rate != config_.residual_rate
    ) {
        throw std::runtime_error(
            "Neural checkpoint version or configuration mismatch"
        );
    }
    for (auto* tensor : parameters_->tensors()) {
        read_vector(in, *tensor, tensor->size());
    }
    read_scalar(in, parameters_->output_bias);
    read_scalar(in, adam_->step);
    telemetry_.optimizer_steps = adam_->step;
    telemetry_.parameters = count_parameters(*parameters_);
}

void NeuralShapeField::write_json(std::ostream& out) const {
    const std::size_t states = checked_states(config_.input_bits);
    out << std::setprecision(12)
        << "{\n"
        << "  \"architecture\": \"recurrent_neural_shape_field\",\n"
        << "  \"transformer_attention\": false,\n"
        << "  \"input_bits\": " << config_.input_bits << ",\n"
        << "  \"output_bits\": " << config_.output_bits << ",\n"
        << "  \"state_nodes\": " << states << ",\n"
        << "  \"hidden_units\": " << config_.hidden_units << ",\n"
        << "  \"reasoning_steps\": " << config_.reasoning_steps << ",\n"
        << "  \"parameters\": " << telemetry_.parameters << ",\n"
        << "  \"optimizer_steps\": " << telemetry_.optimizer_steps << ",\n"
        << "  \"parameter_checksum\": " << parameter_checksum() << ",\n"
        << "  \"frozen_shape_values\": " << frozen_shape_.size() << ",\n"
        << "  \"frozen_field_logits\": " << frozen_logits_.size() << "\n"
        << "}\n";
}

}  // namespace sera

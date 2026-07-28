#include "sera/interaction_workspace_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <random>
#include <stdexcept>
#include <utility>

namespace sera {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kOperatorBits = 8;
// [a_i, b_j, a_i*b_j, 8 operator bits, 4 lattice boundary flags]
constexpr std::uint32_t kFeatureCount = 3 + kOperatorBits + 4;
// Nine 3x3 neighbourhood taps plus one anchor tap onto the initial state.
constexpr std::uint32_t kNeighbourTaps = 9;
constexpr std::uint32_t kTaps = kNeighbourTaps + 1;
constexpr std::uint32_t kAnchorTap = kNeighbourTaps;

constexpr float kAdamBeta1 = 0.9F;
constexpr float kAdamBeta2 = 0.999F;
constexpr float kAdamEpsilon = 1.0e-8F;
constexpr std::uint64_t kCheckpointMagic = 0x5345524135574B52ULL;
constexpr std::uint32_t kCheckpointVersion = 3;

void validate_config(const WorkspaceKernelConfig& config) {
    if (
        config.bit_width == 0
        || config.bit_width > 16
        || config.channels == 0
        || config.shared_rank == 0
        || config.reasoning_steps == 0
    ) {
        throw std::invalid_argument("Invalid workspace configuration");
    }
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

double binary_cross_entropy(float logit, float target) {
    const double magnitude = std::abs(static_cast<double>(logit));
    return std::max(static_cast<double>(logit), 0.0)
        - static_cast<double>(logit) * static_cast<double>(target)
        + std::log1p(std::exp(-magnitude));
}

// Linearly increasing supervision weights over the latter half of reasoning
// time. Every weight supervises the same final target, so no intermediate
// arithmetic answer is ever revealed.
std::vector<float> temporal_weights(std::uint32_t steps) {
    const std::uint32_t start = steps / 2U;
    const std::uint32_t count = steps - start + 1U;
    std::vector<float> weights(steps + 1U, 0.0F);
    const float total =
        static_cast<float>(count) * static_cast<float>(count + 1U) / 2.0F;
    for (std::uint32_t step = start; step <= steps; ++step) {
        weights[step] = static_cast<float>(step - start + 1U) / total;
    }
    return weights;
}

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed while writing workspace checkpoint");
    }
}

template <typename T>
void read_scalar(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Truncated workspace checkpoint");
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
        throw std::runtime_error("Failed while writing workspace tensor");
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
        throw std::runtime_error("Workspace checkpoint tensor shape mismatch");
    }
    values.resize(expected);
    in.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!in) {
        throw std::runtime_error("Truncated workspace checkpoint tensor");
    }
}

}  // namespace

struct InteractionWorkspaceKernel::Parameters {
    std::vector<float> input;          // channels x kFeatureCount
    std::vector<float> input_bias;     // channels
    std::vector<float> projection;     // shared_rank x channels
    std::vector<float> candidate;      // kTaps x channels x shared_rank
    std::vector<float> candidate_bias; // channels
    std::vector<float> update;         // kTaps x channels x shared_rank
    std::vector<float> update_bias;    // channels
    std::vector<float> output;         // channels, corner site (p, 0)
    std::vector<float> output_diagonal;// channels, diagonal site (p, p)
    std::vector<float> output_bias;    // 1

    [[nodiscard]] std::vector<std::vector<float>*> tensors() {
        return {
            &input, &input_bias, &projection, &candidate, &candidate_bias,
            &update, &update_bias, &output, &output_diagonal, &output_bias
        };
    }

    [[nodiscard]] std::vector<const std::vector<float>*> tensors() const {
        return {
            &input, &input_bias, &projection, &candidate, &candidate_bias,
            &update, &update_bias, &output, &output_diagonal, &output_bias
        };
    }
};

struct InteractionWorkspaceKernel::AdamState {
    std::vector<std::vector<float>> first;
    std::vector<std::vector<float>> second;
    std::uint64_t steps{};
};

namespace {

using Parameters = InteractionWorkspaceKernel::Parameters;

void shape_parameters(
    Parameters& parameters,
    const WorkspaceKernelConfig& config,
    float fill
) {
    const std::size_t channels = config.channels;
    const std::size_t rank = config.shared_rank;
    parameters.input.assign(channels * kFeatureCount, fill);
    parameters.input_bias.assign(channels, fill);
    parameters.projection.assign(rank * channels, fill);
    parameters.candidate.assign(kTaps * channels * rank, fill);
    parameters.candidate_bias.assign(channels, fill);
    parameters.update.assign(kTaps * channels * rank, fill);
    parameters.update_bias.assign(channels, fill);
    parameters.output.assign(channels, fill);
    parameters.output_diagonal.assign(channels, fill);
    parameters.output_bias.assign(1U, fill);
}

// Forward activations for a single expression node.
struct Cache {
    std::vector<float> features;  // cells x kFeatureCount
    std::vector<float> states;    // (steps + 1) x cells x channels
    std::vector<float> projected; // (steps + 1) x cells x rank
    std::vector<float> candidates;// steps x cells x channels
    std::vector<float> gates;     // steps x cells x channels
    std::vector<float> logits;    // (steps + 1) x bit_width
};

void build_features(
    const WorkspaceKernelConfig& config,
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t raw_operator,
    std::vector<float>& features
) {
    const std::uint32_t width = config.bit_width;
    features.assign(
        static_cast<std::size_t>(width) * width * kFeatureCount, 0.0F
    );
    for (std::uint32_t i = 0; i < width; ++i) {
        const bool left_bit = ((left >> i) & 1U) != 0U;
        for (std::uint32_t j = 0; j < width; ++j) {
            const bool right_bit = ((right >> j) & 1U) != 0U;
            const std::size_t offset =
                (static_cast<std::size_t>(i) * width + j) * kFeatureCount;
            features[offset] = left_bit ? 1.0F : -1.0F;
            features[offset + 1] = right_bit ? 1.0F : -1.0F;
            // The exact bilinear term. This is the complete second-order
            // interaction of the two operand fields, which the S4 CP relation
            // could only approximate at rank R.
            features[offset + 2] = (left_bit && right_bit) ? 1.0F : 0.0F;
            for (std::uint32_t bit = 0; bit < kOperatorBits; ++bit) {
                features[offset + 3 + bit] =
                    ((raw_operator >> bit) & 1U) != 0U ? 1.0F : -1.0F;
            }
            features[offset + 11] = i == 0U ? 1.0F : -1.0F;
            features[offset + 12] = j == 0U ? 1.0F : -1.0F;
            features[offset + 13] = i + 1U == width ? 1.0F : -1.0F;
            features[offset + 14] = j + 1U == width ? 1.0F : -1.0F;
        }
    }
}

void project_states(
    const WorkspaceKernelConfig& config,
    const Parameters& parameters,
    const float* states,
    float* projected
) {
    const std::size_t cells =
        static_cast<std::size_t>(config.bit_width) * config.bit_width;
    const std::size_t channels = config.channels;
    const std::size_t rank = config.shared_rank;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        for (std::size_t r = 0; r < rank; ++r) {
            float value = 0.0F;
            for (std::size_t c = 0; c < channels; ++c) {
                value += parameters.projection[r * channels + c]
                    * states[cell * channels + c];
            }
            projected[cell * rank + r] = value;
        }
    }
}

void forward(
    const WorkspaceKernelConfig& config,
    const Parameters& parameters,
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t raw_operator,
    Cache& cache
) {
    const std::uint32_t width = config.bit_width;
    const std::size_t cells = static_cast<std::size_t>(width) * width;
    const std::size_t channels = config.channels;
    const std::size_t rank = config.shared_rank;
    const std::size_t steps = config.reasoning_steps;

    build_features(config, left, right, raw_operator, cache.features);
    cache.states.assign((steps + 1) * cells * channels, 0.0F);
    cache.projected.assign((steps + 1) * cells * rank, 0.0F);
    cache.candidates.assign(steps * cells * channels, 0.0F);
    cache.gates.assign(steps * cells * channels, 0.0F);
    cache.logits.assign((steps + 1) * width, 0.0F);

    for (std::size_t cell = 0; cell < cells; ++cell) {
        for (std::size_t c = 0; c < channels; ++c) {
            float value = parameters.input_bias[c];
            for (std::size_t f = 0; f < kFeatureCount; ++f) {
                value += parameters.input[c * kFeatureCount + f]
                    * cache.features[cell * kFeatureCount + f];
            }
            cache.states[cell * channels + c] = std::tanh(value);
        }
    }

    const auto read_out = [&](std::size_t step) {
        const float* states = &cache.states[step * cells * channels];
        for (std::uint32_t p = 0; p < width; ++p) {
            // Which cells carry which answer bit is a readout convention, not
            // an arithmetic rule.
            const std::size_t row = static_cast<std::size_t>(p) * width;
            const std::size_t diagonal = row + p;
            float value = parameters.output_bias[0];
            switch (config.readout) {
            case WorkspaceKernelConfig::Readout::kRowPool:
                for (std::size_t c = 0; c < channels; ++c) {
                    float pooled = 0.0F;
                    for (std::size_t j = 0; j < width; ++j) {
                        pooled += states[(row + j) * channels + c];
                    }
                    value += parameters.output[c] * pooled;
                }
                break;
            case WorkspaceKernelConfig::Readout::kDual:
                for (std::size_t c = 0; c < channels; ++c) {
                    value += parameters.output[c] * states[row * channels + c]
                        + parameters.output_diagonal[c]
                            * states[diagonal * channels + c];
                }
                break;
            case WorkspaceKernelConfig::Readout::kCorner:
            default:
                for (std::size_t c = 0; c < channels; ++c) {
                    value += parameters.output[c] * states[row * channels + c];
                }
                break;
            }
            cache.logits[step * width + p] = value;
        }
    };

    project_states(
        config, parameters, cache.states.data(), cache.projected.data()
    );
    read_out(0);

    const float* anchor = cache.projected.data();

    for (std::size_t step = 0; step < steps; ++step) {
        const float* states = &cache.states[step * cells * channels];
        const float* projected = &cache.projected[step * cells * rank];
        float* next = &cache.states[(step + 1) * cells * channels];
        float* candidates = &cache.candidates[step * cells * channels];
        float* gates = &cache.gates[step * cells * channels];

        for (std::uint32_t i = 0; i < width; ++i) {
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::size_t cell =
                    static_cast<std::size_t>(i) * width + j;
                for (std::size_t c = 0; c < channels; ++c) {
                    float candidate = parameters.candidate_bias[c];
                    float update = parameters.update_bias[c];
                    for (std::uint32_t tap = 0; tap < kNeighbourTaps; ++tap) {
                        const int di = static_cast<int>(tap / 3U) - 1;
                        const int dj = static_cast<int>(tap % 3U) - 1;
                        const int si = static_cast<int>(i) + di;
                        const int sj = static_cast<int>(j) + dj;
                        if (
                            si < 0 || sj < 0
                            || si >= static_cast<int>(width)
                            || sj >= static_cast<int>(width)
                        ) {
                            continue;  // Outside the lattice reads as zero.
                        }
                        const std::size_t source =
                            static_cast<std::size_t>(si) * width
                            + static_cast<std::size_t>(sj);
                        const std::size_t base =
                            (tap * channels + c) * rank;
                        for (std::size_t r = 0; r < rank; ++r) {
                            const float z = projected[source * rank + r];
                            candidate += parameters.candidate[base + r] * z;
                            update += parameters.update[base + r] * z;
                        }
                    }
                    const std::size_t anchor_base =
                        (kAnchorTap * channels + c) * rank;
                    for (std::size_t r = 0; r < rank; ++r) {
                        const float z = anchor[cell * rank + r];
                        candidate +=
                            parameters.candidate[anchor_base + r] * z;
                        update += parameters.update[anchor_base + r] * z;
                    }
                    const float candidate_value = std::tanh(candidate);
                    const float gate_value = sigmoid(update);
                    candidates[cell * channels + c] = candidate_value;
                    gates[cell * channels + c] = gate_value;
                    next[cell * channels + c] =
                        (1.0F - gate_value) * states[cell * channels + c]
                        + gate_value * candidate_value;
                }
            }
        }
        project_states(
            config,
            parameters,
            next,
            &cache.projected[(step + 1) * cells * rank]
        );
        read_out(step + 1);
    }
}

// Accumulates the analytic gradient of the temporally weighted loss for one
// case into `gradient`, and returns that case's loss.
double backward(
    const WorkspaceKernelConfig& config,
    const Parameters& parameters,
    const ArithmeticTrainingCase& example,
    const std::vector<float>& weights,
    Cache& cache,
    Parameters& gradient
) {
    const std::uint32_t width = config.bit_width;
    const std::size_t cells = static_cast<std::size_t>(width) * width;
    const std::size_t channels = config.channels;
    const std::size_t rank = config.shared_rank;
    const std::size_t steps = config.reasoning_steps;

    forward(
        config, parameters, example.left, example.right,
        example.raw_operator, cache
    );

    double loss = 0.0;
    std::vector<float> state_grad(cells * channels, 0.0F);
    std::vector<float> anchor_grad(cells * rank, 0.0F);
    std::vector<float> projected_grad(cells * rank, 0.0F);

    for (std::size_t step = steps; ; --step) {
        const float weight = weights[step];
        if (weight > 0.0F) {
            const float* states = &cache.states[step * cells * channels];
            float* grads = state_grad.data();
            for (std::uint32_t p = 0; p < width; ++p) {
                const float logit = cache.logits[step * width + p];
                const float target =
                    ((example.target >> p) & 1U) != 0U ? 1.0F : 0.0F;
                loss += static_cast<double>(weight)
                    * binary_cross_entropy(logit, target);
                const float delta = weight * (sigmoid(logit) - target);
                const std::size_t row = static_cast<std::size_t>(p) * width;
                const std::size_t diagonal = row + p;
                gradient.output_bias[0] += delta;
                switch (config.readout) {
                case WorkspaceKernelConfig::Readout::kRowPool:
                    for (std::size_t c = 0; c < channels; ++c) {
                        float pooled = 0.0F;
                        for (std::size_t j = 0; j < width; ++j) {
                            pooled += states[(row + j) * channels + c];
                            grads[(row + j) * channels + c] +=
                                delta * parameters.output[c];
                        }
                        gradient.output[c] += delta * pooled;
                    }
                    break;
                case WorkspaceKernelConfig::Readout::kDual:
                    for (std::size_t c = 0; c < channels; ++c) {
                        gradient.output[c] +=
                            delta * states[row * channels + c];
                        gradient.output_diagonal[c] +=
                            delta * states[diagonal * channels + c];
                        grads[row * channels + c] +=
                            delta * parameters.output[c];
                        grads[diagonal * channels + c] +=
                            delta * parameters.output_diagonal[c];
                    }
                    break;
                case WorkspaceKernelConfig::Readout::kCorner:
                default:
                    for (std::size_t c = 0; c < channels; ++c) {
                        gradient.output[c] +=
                            delta * states[row * channels + c];
                        grads[row * channels + c] +=
                            delta * parameters.output[c];
                    }
                    break;
                }
            }
        }
        if (step == 0) {
            break;
        }

        const std::size_t previous = step - 1;
        const float* states = &cache.states[previous * cells * channels];
        const float* projected = &cache.projected[previous * cells * rank];
        const float* candidates =
            &cache.candidates[previous * cells * channels];
        const float* gates = &cache.gates[previous * cells * channels];
        const float* anchor = cache.projected.data();

        std::fill(projected_grad.begin(), projected_grad.end(), 0.0F);

        for (std::uint32_t i = 0; i < width; ++i) {
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::size_t cell =
                    static_cast<std::size_t>(i) * width + j;
                for (std::size_t c = 0; c < channels; ++c) {
                    const std::size_t index = cell * channels + c;
                    const float upstream = state_grad[index];
                    if (upstream == 0.0F) {
                        continue;
                    }
                    const float gate = gates[index];
                    const float candidate = candidates[index];
                    const float previous_state = states[index];

                    const float candidate_pre = upstream * gate
                        * (1.0F - candidate * candidate);
                    const float update_pre =
                        upstream * (candidate - previous_state)
                        * gate * (1.0F - gate);
                    // Carry the residual path back to the previous state.
                    state_grad[index] = upstream * (1.0F - gate);

                    gradient.candidate_bias[c] += candidate_pre;
                    gradient.update_bias[c] += update_pre;

                    for (std::uint32_t tap = 0; tap < kNeighbourTaps; ++tap) {
                        const int di = static_cast<int>(tap / 3U) - 1;
                        const int dj = static_cast<int>(tap % 3U) - 1;
                        const int si = static_cast<int>(i) + di;
                        const int sj = static_cast<int>(j) + dj;
                        if (
                            si < 0 || sj < 0
                            || si >= static_cast<int>(width)
                            || sj >= static_cast<int>(width)
                        ) {
                            continue;
                        }
                        const std::size_t source =
                            static_cast<std::size_t>(si) * width
                            + static_cast<std::size_t>(sj);
                        const std::size_t base = (tap * channels + c) * rank;
                        for (std::size_t r = 0; r < rank; ++r) {
                            const float z = projected[source * rank + r];
                            gradient.candidate[base + r] += candidate_pre * z;
                            gradient.update[base + r] += update_pre * z;
                            projected_grad[source * rank + r] +=
                                candidate_pre * parameters.candidate[base + r]
                                + update_pre * parameters.update[base + r];
                        }
                    }

                    const std::size_t anchor_base =
                        (kAnchorTap * channels + c) * rank;
                    for (std::size_t r = 0; r < rank; ++r) {
                        const float z = anchor[cell * rank + r];
                        gradient.candidate[anchor_base + r] +=
                            candidate_pre * z;
                        gradient.update[anchor_base + r] += update_pre * z;
                        anchor_grad[cell * rank + r] +=
                            candidate_pre
                                * parameters.candidate[anchor_base + r]
                            + update_pre * parameters.update[anchor_base + r];
                    }
                }
            }
        }

        // Push the projection gradient of step `previous` into its states.
        for (std::size_t cell = 0; cell < cells; ++cell) {
            for (std::size_t r = 0; r < rank; ++r) {
                const float delta = projected_grad[cell * rank + r];
                if (delta == 0.0F) {
                    continue;
                }
                for (std::size_t c = 0; c < channels; ++c) {
                    gradient.projection[r * channels + c] +=
                        delta * states[cell * channels + c];
                    state_grad[cell * channels + c] +=
                        delta * parameters.projection[r * channels + c];
                }
            }
        }
    }

    // The anchor tap reads the projection of the initial state at every step.
    for (std::size_t cell = 0; cell < cells; ++cell) {
        for (std::size_t r = 0; r < rank; ++r) {
            const float delta = anchor_grad[cell * rank + r];
            if (delta == 0.0F) {
                continue;
            }
            for (std::size_t c = 0; c < channels; ++c) {
                gradient.projection[r * channels + c] +=
                    delta * cache.states[cell * channels + c];
                state_grad[cell * channels + c] +=
                    delta * parameters.projection[r * channels + c];
            }
        }
    }

    for (std::size_t cell = 0; cell < cells; ++cell) {
        for (std::size_t c = 0; c < channels; ++c) {
            const float state = cache.states[cell * channels + c];
            const float pre =
                state_grad[cell * channels + c] * (1.0F - state * state);
            gradient.input_bias[c] += pre;
            for (std::size_t f = 0; f < kFeatureCount; ++f) {
                gradient.input[c * kFeatureCount + f] +=
                    pre * cache.features[cell * kFeatureCount + f];
            }
        }
    }

    return loss;
}

}  // namespace

InteractionWorkspaceKernel::InteractionWorkspaceKernel(
    WorkspaceKernelConfig config,
    std::uint64_t seed
)
    : config_(config) {
    validate_config(config_);
    parameters_ = new Parameters();
    adam_ = new AdamState();
    shape_parameters(*parameters_, config_, 0.0F);

    std::mt19937_64 engine(seed);
    const auto tensors = parameters_->tensors();
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        auto* tensor = tensors[index];
        const auto fan_in = static_cast<float>(
            std::max<std::size_t>(tensor->size() / config_.channels, 1U)
        );
        const float scale = 1.0F / std::sqrt(fan_in);
        std::uniform_real_distribution<float> distribution(-scale, scale);
        for (auto& value : *tensor) {
            value = distribution(engine);
        }
    }
    std::fill(
        parameters_->input_bias.begin(), parameters_->input_bias.end(), 0.0F
    );
    std::fill(
        parameters_->candidate_bias.begin(),
        parameters_->candidate_bias.end(),
        0.0F
    );
    // A mildly open initial gate keeps early reasoning steps informative.
    std::fill(
        parameters_->update_bias.begin(),
        parameters_->update_bias.end(),
        -1.0F
    );
    parameters_->output_bias[0] = 0.0F;

    for (const auto* tensor : parameters_->tensors()) {
        adam_->first.emplace_back(tensor->size(), 0.0F);
        adam_->second.emplace_back(tensor->size(), 0.0F);
    }

    std::uint64_t count = 0;
    for (const auto* tensor : parameters_->tensors()) {
        count += tensor->size();
    }
    telemetry_.parameters = count;
    // The S4 CP relation needs shared_rank >= 2W - 1 to represent W-bit
    // multiplication; this is what that relation alone would then cost.
    const std::uint64_t width = config_.bit_width;
    telemetry_.cp_equivalent_parameters = 3ULL * width * (2ULL * width - 1ULL);
    // A dense analogue replaces the tied stencil with an untied per-cell map
    // and the bilinear workspace with a full W^3 interaction tensor.
    telemetry_.dense_equivalent_parameters =
        width * width * width
        + static_cast<std::uint64_t>(config_.channels) * config_.channels
            * kTaps * 2ULL;
}

InteractionWorkspaceKernel::~InteractionWorkspaceKernel() {
    delete parameters_;
    delete adam_;
}

InteractionWorkspaceKernel::InteractionWorkspaceKernel(
    InteractionWorkspaceKernel&& other
) noexcept
    : config_(other.config_),
      parameters_(std::exchange(other.parameters_, nullptr)),
      adam_(std::exchange(other.adam_, nullptr)),
      telemetry_(other.telemetry_) {}

InteractionWorkspaceKernel& InteractionWorkspaceKernel::operator=(
    InteractionWorkspaceKernel&& other
) noexcept {
    if (this != &other) {
        delete parameters_;
        delete adam_;
        config_ = other.config_;
        parameters_ = std::exchange(other.parameters_, nullptr);
        adam_ = std::exchange(other.adam_, nullptr);
        telemetry_ = other.telemetry_;
    }
    return *this;
}

const WorkspaceKernelConfig& InteractionWorkspaceKernel::config()
    const noexcept {
    return config_;
}

const WorkspaceTelemetry& InteractionWorkspaceKernel::telemetry()
    const noexcept {
    return telemetry_;
}

double InteractionWorkspaceKernel::batch_loss(
    std::span<const ArithmeticTrainingCase> cases
) const {
    if (cases.empty()) {
        return 0.0;
    }
    const auto weights = temporal_weights(config_.reasoning_steps);
    Cache cache;
    double total = 0.0;
    for (const auto& example : cases) {
        forward(
            config_, *parameters_, example.left, example.right,
            example.raw_operator, cache
        );
        for (std::size_t step = 0; step <= config_.reasoning_steps; ++step) {
            const float weight = weights[step];
            if (weight <= 0.0F) {
                continue;
            }
            for (std::uint32_t p = 0; p < config_.bit_width; ++p) {
                const float target =
                    ((example.target >> p) & 1U) != 0U ? 1.0F : 0.0F;
                total += static_cast<double>(weight)
                    * binary_cross_entropy(
                        cache.logits[step * config_.bit_width + p], target
                    );
            }
        }
    }
    return total / static_cast<double>(cases.size());
}

std::vector<double> InteractionWorkspaceKernel::flat_parameters() const {
    std::vector<double> flat;
    for (const auto* tensor : parameters_->tensors()) {
        for (float value : *tensor) {
            flat.push_back(static_cast<double>(value));
        }
    }
    return flat;
}

void InteractionWorkspaceKernel::set_flat_parameters(
    std::span<const double> values
) {
    std::size_t cursor = 0;
    for (auto* tensor : parameters_->tensors()) {
        for (auto& value : *tensor) {
            if (cursor >= values.size()) {
                throw std::invalid_argument("Parameter vector too short");
            }
            value = static_cast<float>(values[cursor++]);
        }
    }
}

std::vector<double> InteractionWorkspaceKernel::flat_gradient(
    std::span<const ArithmeticTrainingCase> cases
) const {
    Parameters gradient;
    shape_parameters(gradient, config_, 0.0F);
    const auto weights = temporal_weights(config_.reasoning_steps);
    Cache cache;
    for (const auto& example : cases) {
        backward(config_, *parameters_, example, weights, cache, gradient);
    }
    const double scale =
        cases.empty() ? 1.0 : 1.0 / static_cast<double>(cases.size());
    std::vector<double> flat;
    for (const auto* tensor : gradient.tensors()) {
        for (float value : *tensor) {
            flat.push_back(static_cast<double>(value) * scale);
        }
    }
    return flat;
}

double InteractionWorkspaceKernel::train_batch(
    std::span<const ArithmeticTrainingCase> cases,
    float learning_rate,
    float group_penalty
) {
    if (cases.empty()) {
        return 0.0;
    }
    const std::uint32_t limit =
        (config_.bit_width >= 32U)
            ? 0xFFFFFFFFU
            : ((std::uint32_t{1} << config_.bit_width) - 1U);
    for (const auto& example : cases) {
        if (
            example.left > limit || example.right > limit
            || example.target > limit
        ) {
            throw std::invalid_argument("Training case exceeds bit width");
        }
    }

    Parameters gradient;
    shape_parameters(gradient, config_, 0.0F);
    const auto weights = temporal_weights(config_.reasoning_steps);
    Cache cache;
    double loss = 0.0;
    for (const auto& example : cases) {
        loss += backward(
            config_, *parameters_, example, weights, cache, gradient
        );
    }
    const float scale = 1.0F / static_cast<float>(cases.size());
    loss /= static_cast<double>(cases.size());

    // Group penalty over shared rank channels: every channel is shared by the
    // projection and by both stencil lifts, so pruning a channel removes
    // parameters from all three at once.
    if (group_penalty > 0.0F) {
        const std::size_t channels = config_.channels;
        const std::size_t rank = config_.shared_rank;
        for (std::size_t r = 0; r < rank; ++r) {
            double squared = 0.0;
            for (std::size_t c = 0; c < channels; ++c) {
                const double value = parameters_->projection[r * channels + c];
                squared += value * value;
            }
            for (std::uint32_t tap = 0; tap < kTaps; ++tap) {
                for (std::size_t c = 0; c < channels; ++c) {
                    const std::size_t index = (tap * channels + c) * rank + r;
                    squared += static_cast<double>(parameters_->candidate[index])
                        * parameters_->candidate[index];
                    squared += static_cast<double>(parameters_->update[index])
                        * parameters_->update[index];
                }
            }
            const float norm = std::sqrt(static_cast<float>(squared)) + 1.0e-8F;
            const float coefficient = group_penalty / norm;
            for (std::size_t c = 0; c < channels; ++c) {
                gradient.projection[r * channels + c] +=
                    coefficient * parameters_->projection[r * channels + c]
                    / scale;
            }
            for (std::uint32_t tap = 0; tap < kTaps; ++tap) {
                for (std::size_t c = 0; c < channels; ++c) {
                    const std::size_t index = (tap * channels + c) * rank + r;
                    gradient.candidate[index] +=
                        coefficient * parameters_->candidate[index] / scale;
                    gradient.update[index] +=
                        coefficient * parameters_->update[index] / scale;
                }
            }
        }
    }

    adam_->steps += 1;
    const double bias1 =
        1.0 - std::pow(static_cast<double>(kAdamBeta1), adam_->steps);
    const double bias2 =
        1.0 - std::pow(static_cast<double>(kAdamBeta2), adam_->steps);

    auto parameter_tensors = parameters_->tensors();
    auto gradient_tensors = gradient.tensors();
    for (std::size_t index = 0; index < parameter_tensors.size(); ++index) {
        auto& values = *parameter_tensors[index];
        auto& gradients = *gradient_tensors[index];
        auto& first = adam_->first[index];
        auto& second = adam_->second[index];
        for (std::size_t slot = 0; slot < values.size(); ++slot) {
            const float derivative = gradients[slot] * scale;
            first[slot] =
                kAdamBeta1 * first[slot] + (1.0F - kAdamBeta1) * derivative;
            second[slot] = kAdamBeta2 * second[slot]
                + (1.0F - kAdamBeta2) * derivative * derivative;
            const float corrected_first =
                first[slot] / static_cast<float>(bias1);
            const float corrected_second =
                second[slot] / static_cast<float>(bias2);
            values[slot] -= learning_rate * corrected_first
                / (std::sqrt(corrected_second) + kAdamEpsilon);
        }
    }

    telemetry_.optimizer_steps += 1;
    telemetry_.training_cases += cases.size();
    telemetry_.most_recent_loss = loss;
    telemetry_.mean_recent_loss =
        0.98 * telemetry_.mean_recent_loss + 0.02 * loss;
    return loss;
}

ArithmeticInference InteractionWorkspaceKernel::infer(
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t raw_operator
) const {
    const auto started = Clock::now();
    Cache cache;
    forward(config_, *parameters_, left, right, raw_operator, cache);

    const std::uint32_t width = config_.bit_width;
    const std::size_t steps = config_.reasoning_steps;
    ArithmeticInference inference;
    inference.logits.assign(width, 0.0F);
    std::uint32_t value = 0;
    double confidence = 0.0;
    for (std::uint32_t p = 0; p < width; ++p) {
        const float logit = cache.logits[steps * width + p];
        inference.logits[p] = logit;
        if (logit > 0.0F) {
            value |= (1U << p);
        }
        confidence += std::abs(static_cast<double>(sigmoid(logit)) - 0.5) * 2.0;
    }
    inference.value = value;
    inference.mean_confidence =
        static_cast<float>(confidence / static_cast<double>(width));

    // The earliest step from which the decoded answer never changes again.
    std::uint32_t stable = static_cast<std::uint32_t>(steps);
    for (std::size_t step = steps + 1; step-- > 0;) {
        std::uint32_t decoded = 0;
        for (std::uint32_t p = 0; p < width; ++p) {
            if (cache.logits[step * width + p] > 0.0F) {
                decoded |= (1U << p);
            }
        }
        if (decoded != value) {
            break;
        }
        stable = static_cast<std::uint32_t>(step);
    }
    inference.stable_after_step = stable;
    telemetry_.inference_nanoseconds =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - started
            ).count()
        );
    return inference;
}

std::vector<double> InteractionWorkspaceKernel::rank_channel_norms() const {
    const std::size_t channels = config_.channels;
    const std::size_t rank = config_.shared_rank;
    std::vector<double> norms(rank, 0.0);
    for (std::size_t r = 0; r < rank; ++r) {
        double squared = 0.0;
        for (std::size_t c = 0; c < channels; ++c) {
            const double value = parameters_->projection[r * channels + c];
            squared += value * value;
        }
        for (std::uint32_t tap = 0; tap < kTaps; ++tap) {
            for (std::size_t c = 0; c < channels; ++c) {
                const std::size_t index = (tap * channels + c) * rank + r;
                const double candidate = parameters_->candidate[index];
                const double update = parameters_->update[index];
                squared += candidate * candidate + update * update;
            }
        }
        norms[r] = std::sqrt(squared);
    }
    return norms;
}

std::uint64_t InteractionWorkspaceKernel::prune_rank_channels(
    std::uint32_t count
) {
    const std::size_t channels = config_.channels;
    const std::size_t rank = config_.shared_rank;
    if (count > rank) {
        throw std::invalid_argument("Cannot prune more channels than exist");
    }
    const auto norms = rank_channel_norms();
    std::vector<std::size_t> order(rank);
    for (std::size_t index = 0; index < rank; ++index) {
        order[index] = index;
    }
    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t a, std::size_t b) { return norms[a] < norms[b]; }
    );

    std::uint64_t removed = 0;
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        const std::size_t r = order[slot];
        for (std::size_t c = 0; c < channels; ++c) {
            parameters_->projection[r * channels + c] = 0.0F;
            ++removed;
        }
        for (std::uint32_t tap = 0; tap < kTaps; ++tap) {
            for (std::size_t c = 0; c < channels; ++c) {
                const std::size_t index = (tap * channels + c) * rank + r;
                parameters_->candidate[index] = 0.0F;
                parameters_->update[index] = 0.0F;
                removed += 2;
            }
        }
    }
    return removed;
}

std::uint64_t InteractionWorkspaceKernel::parameter_checksum() const noexcept {
    std::uint64_t checksum = 0xCBF29CE484222325ULL;
    for (const auto* tensor : parameters_->tensors()) {
        for (float value : *tensor) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            checksum ^= bits;
            checksum *= 0x100000001B3ULL;
        }
    }
    return checksum;
}

void InteractionWorkspaceKernel::save_checkpoint(
    const std::string& path
) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Unable to open workspace checkpoint");
    }
    write_scalar(out, kCheckpointMagic);
    write_scalar(out, kCheckpointVersion);
    write_scalar(out, config_.bit_width);
    write_scalar(out, config_.channels);
    write_scalar(out, config_.shared_rank);
    write_scalar(out, config_.reasoning_steps);
    write_scalar(out, static_cast<std::uint32_t>(config_.readout));
    write_scalar(out, telemetry_.optimizer_steps);
    write_scalar(out, telemetry_.training_cases);
    for (const auto* tensor : parameters_->tensors()) {
        write_vector(out, *tensor);
    }
}

void InteractionWorkspaceKernel::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to read workspace checkpoint");
    }
    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    read_scalar(in, magic);
    read_scalar(in, version);
    if (magic != kCheckpointMagic || version != kCheckpointVersion) {
        throw std::runtime_error("Unrecognised workspace checkpoint");
    }
    WorkspaceKernelConfig config = config_;
    read_scalar(in, config.bit_width);
    read_scalar(in, config.channels);
    read_scalar(in, config.shared_rank);
    read_scalar(in, config.reasoning_steps);
    std::uint32_t readout = 0;
    read_scalar(in, readout);
    if (readout > 2U) {
        throw std::runtime_error("Unrecognised workspace readout mode");
    }
    config.readout = static_cast<WorkspaceKernelConfig::Readout>(readout);
    validate_config(config);
    if (
        config.bit_width != config_.bit_width
        || config.channels != config_.channels
        || config.shared_rank != config_.shared_rank
        || config.reasoning_steps != config_.reasoning_steps
        || config.readout != config_.readout
    ) {
        throw std::runtime_error("Workspace checkpoint shape mismatch");
    }
    read_scalar(in, telemetry_.optimizer_steps);
    read_scalar(in, telemetry_.training_cases);
    Parameters reference;
    shape_parameters(reference, config_, 0.0F);
    auto expected = reference.tensors();
    auto tensors = parameters_->tensors();
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        read_vector(in, *tensors[index], expected[index]->size());
    }
}

void InteractionWorkspaceKernel::write_json(std::ostream& out) const {
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"architecture\": "
        << "\"shift_equivariant_interaction_workspace\",\n";
    out << "  \"bit_width\": " << config_.bit_width << ",\n";
    out << "  \"channels\": " << config_.channels << ",\n";
    out << "  \"shared_rank\": " << config_.shared_rank << ",\n";
    out << "  \"reasoning_steps\": " << config_.reasoning_steps << ",\n";
    out << "  \"readout\": "
        << static_cast<std::uint32_t>(config_.readout) << ",\n";
    out << "  \"parameters\": " << telemetry_.parameters << ",\n";
    out << "  \"cp_equivalent_parameters\": "
        << telemetry_.cp_equivalent_parameters << ",\n";
    out << "  \"dense_equivalent_parameters\": "
        << telemetry_.dense_equivalent_parameters << ",\n";
    out << "  \"optimizer_steps\": " << telemetry_.optimizer_steps << ",\n";
    out << "  \"training_cases\": " << telemetry_.training_cases << ",\n";
    out << "  \"parameter_checksum\": " << parameter_checksum() << "\n";
    out << "}\n";
}

}  // namespace sera

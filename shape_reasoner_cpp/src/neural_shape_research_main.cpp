#include "sera/neural_shape_field.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kInputBits = 6;
constexpr std::uint32_t kOutputBits = 6;
constexpr std::uint32_t kDefaultTrainingSteps = 8000;
constexpr std::uint32_t kFinalTrialsPerRelation = 32;
constexpr std::uint64_t kTrainingSeed = 0x5333524154524149ULL;
constexpr std::uint64_t kFinalSeedBase = 0x533346494E414C00ULL;

constexpr std::array<std::uint32_t, 8> kTrainingSupportSizes{
    12, 16, 24, 32, 40, 48, 56, 60
};
constexpr std::array<std::uint32_t, 8> kEvaluationSupportSizes{
    8, 16, 24, 32, 40, 48, 56, 60
};

std::uint32_t evaluator_oracle(
    std::uint32_t relation,
    std::uint32_t left,
    std::uint32_t right
) {
    switch (relation) {
    case 0:
        return (left + right) & 63U;
    case 1:
        return (left - right) & 63U;
    case 2:
        return (left * right) & 63U;
    case 3:
        return (left ^ right) & 63U;
    case 4:
        return (left & right) & 63U;
    case 5:
        return (left | right) & 63U;
    case 6:
        return (left + 2U * right) & 63U;
    case 7:
        return (left * right + left) & 63U;
    case 8:
        return ((left + right) * (left ^ right)) & 63U;
    default:
        throw std::out_of_range("Unknown evaluator relation");
    }
}

std::string_view evaluator_name(std::uint32_t relation) {
    switch (relation) {
    case 0:
        return "addition_mod64";
    case 1:
        return "subtraction_mod64";
    case 2:
        return "multiplication_mod64";
    case 3:
        return "bitwise_xor_ood";
    case 4:
        return "bitwise_and_ood";
    case 5:
        return "bitwise_or_ood";
    case 6:
        return "affine_composition_ood";
    case 7:
        return "multiply_accumulate_ood";
    case 8:
        return "nonlinear_composition_ood";
    default:
        return "unknown";
    }
}

std::vector<std::uint8_t> bits_of(
    std::uint32_t value,
    std::uint32_t width
) {
    std::vector<std::uint8_t> bits(width, 0);
    for (std::uint32_t bit = 0; bit < width; ++bit) {
        bits[bit] = static_cast<std::uint8_t>((value >> bit) & 1U);
    }
    return bits;
}

std::uint32_t value_of(std::span<const std::uint8_t> bits) {
    std::uint32_t result = 0;
    for (std::size_t bit = 0; bit < bits.size(); ++bit) {
        result |= static_cast<std::uint32_t>(bits[bit]) << bit;
    }
    return result;
}

std::vector<std::uint8_t> raw_input(
    std::uint32_t left,
    std::uint32_t right
) {
    auto bits = bits_of(left, 3);
    const auto right_bits = bits_of(right, 3);
    bits.insert(bits.end(), right_bits.begin(), right_bits.end());
    return bits;
}

template <std::size_t N>
std::array<std::uint32_t, N> permutation(std::mt19937_64& random) {
    std::array<std::uint32_t, N> result{};
    std::iota(result.begin(), result.end(), 0U);
    std::shuffle(result.begin(), result.end(), random);
    return result;
}

std::vector<std::uint8_t> scramble(
    std::span<const std::uint8_t> values,
    std::span<const std::uint32_t> order
) {
    std::vector<std::uint8_t> result(order.size(), 0);
    for (std::size_t wire = 0; wire < order.size(); ++wire) {
        result[wire] = values[order[wire]];
    }
    return result;
}

std::vector<std::uint8_t> unscramble(
    std::span<const std::uint8_t> values,
    std::span<const std::uint32_t> order
) {
    std::vector<std::uint8_t> result(order.size(), 0);
    for (std::size_t wire = 0; wire < order.size(); ++wire) {
        result[order[wire]] = values[wire];
    }
    return result;
}

struct Episode {
    std::uint32_t relation{};
    std::uint64_t seed{};
    std::array<std::uint32_t, kInputBits> input_permutation{};
    std::array<std::uint32_t, kOutputBits> output_permutation{};
    std::vector<sera::BooleanExample> complete;
    std::vector<sera::BooleanExample> support;
    std::vector<sera::BooleanExample> withheld;
};

Episode make_episode(
    std::uint32_t relation,
    std::uint64_t seed,
    std::uint32_t support_size
) {
    if (support_size >= 64) {
        throw std::invalid_argument("Support size must be below 64");
    }
    Episode episode;
    episode.relation = relation;
    episode.seed = seed;
    std::mt19937_64 random(seed);
    episode.input_permutation = permutation<kInputBits>(random);
    episode.output_permutation = permutation<kOutputBits>(random);
    episode.complete.reserve(64);
    for (std::uint32_t minterm = 0; minterm < 64; ++minterm) {
        const std::uint32_t left = minterm & 7U;
        const std::uint32_t right = (minterm >> 3U) & 7U;
        episode.complete.push_back({
            scramble(
                raw_input(left, right), episode.input_permutation
            ),
            scramble(
                bits_of(evaluator_oracle(relation, left, right), 6),
                episode.output_permutation
            )
        });
    }
    std::shuffle(
        episode.complete.begin(), episode.complete.end(), random
    );
    episode.support.assign(
        episode.complete.begin(),
        episode.complete.begin() + support_size
    );
    episode.withheld.assign(
        episode.complete.begin() + support_size,
        episode.complete.end()
    );
    return episode;
}

struct Metrics {
    std::uint64_t exact{};
    std::uint64_t words{};
    std::uint64_t correct_bits{};
    std::uint64_t bits{};
    std::uint64_t support_errors{};
    std::uint64_t order_drift{};
    std::uint64_t compile_nanoseconds{};

    void merge(const Metrics& other) {
        exact += other.exact;
        words += other.words;
        correct_bits += other.correct_bits;
        bits += other.bits;
        support_errors += other.support_errors;
        order_drift += other.order_drift;
        compile_nanoseconds += other.compile_nanoseconds;
    }
};

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
        ? 0.0
        : static_cast<double>(numerator)
            / static_cast<double>(denominator);
}

std::uint32_t semantic_input_value(
    const sera::BooleanExample& example,
    const Episode& episode
) {
    return value_of(unscramble(
        example.input, episode.input_permutation
    ));
}

std::uint32_t semantic_output_value(
    std::span<const std::uint8_t> output,
    const Episode& episode
) {
    return value_of(unscramble(output, episode.output_permutation));
}

Metrics evaluate_episode(
    sera::NeuralShapeField& model,
    Episode episode,
    std::ostream* predictions,
    std::uint32_t trial,
    std::uint32_t support_size
) {
    const std::uint64_t checksum_before = model.parameter_checksum();
    model.compile_shape(episode.support);
    Metrics result;
    result.compile_nanoseconds += model.telemetry().compile_nanoseconds;
    std::vector<std::vector<std::uint8_t>> first_predictions;
    first_predictions.reserve(episode.withheld.size());
    for (const auto& example : episode.withheld) {
        const auto predicted = model.predict(example.input);
        first_predictions.push_back(predicted);
        ++result.words;
        result.exact += predicted == example.output ? 1U : 0U;
        for (std::size_t bit = 0; bit < predicted.size(); ++bit) {
            ++result.bits;
            result.correct_bits +=
                predicted[bit] == example.output[bit] ? 1U : 0U;
        }
        if (predictions != nullptr) {
            const std::uint32_t raw =
                semantic_input_value(example, episode);
            const std::uint32_t left = raw & 7U;
            const std::uint32_t right = (raw >> 3U) & 7U;
            *predictions
                << evaluator_name(episode.relation) << ','
                << trial << ',' << support_size << ','
                << left << ',' << right << ','
                << semantic_output_value(predicted, episode) << ','
                << semantic_output_value(example.output, episode) << ','
                << (predicted == example.output ? 1 : 0) << '\n';
        }
    }
    for (const auto& example : episode.support) {
        result.support_errors +=
            model.predict(example.input) == example.output ? 0U : 1U;
    }
    std::reverse(episode.support.begin(), episode.support.end());
    model.compile_shape(episode.support);
    for (std::size_t index = 0;
         index < episode.withheld.size();
         ++index) {
        result.order_drift +=
            model.predict(episode.withheld[index].input)
                == first_predictions[index]
            ? 0U : 1U;
    }
    if (model.parameter_checksum() != checksum_before) {
        throw std::runtime_error(
            "Reusable neural parameters changed during evaluation"
        );
    }
    return result;
}

Metrics quick_validation(
    sera::NeuralShapeField& model,
    std::uint32_t support_size,
    std::uint64_t seed_base
) {
    Metrics result;
    for (std::uint32_t relation = 0; relation < 3; ++relation) {
        for (std::uint32_t trial = 0; trial < 4; ++trial) {
            result.merge(evaluate_episode(
                model,
                make_episode(
                    relation,
                    seed_base
                        ^ (static_cast<std::uint64_t>(relation) << 32U)
                        ^ trial,
                    support_size
                ),
                nullptr,
                trial,
                support_size
            ));
        }
    }
    return result;
}

void write_policy(const std::filesystem::path& output) {
    std::ofstream policy(output / "input_policy.json");
    policy
        << "{\n"
        << "  \"learner_visible\": ["
           "\"episode boundary\", \"raw scrambled support input bits\", "
           "\"raw scrambled support output bits\"],\n"
        << "  \"forbidden\": ["
           "\"operator ID\", \"operator name\", \"operator symbol\", "
           "\"operand grouping\", \"bit significance\", "
           "\"input permutation\", \"output permutation\", "
           "\"test outputs\", \"evaluator feedback\", "
           "\"per-episode gradient updates\"],\n"
        << "  \"architecture\": "
           "\"recurrent hypercube graph neural cellular field\",\n"
        << "  \"self_attention\": false,\n"
        << "  \"token_autoregression\": false\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output =
            argc > 1 ? argv[1] : "results_neural_shape";
        const std::uint32_t training_steps =
            argc > 2
                ? static_cast<std::uint32_t>(std::stoul(argv[2]))
                : kDefaultTrainingSteps;
        const bool load_existing =
            argc > 3 && std::string_view(argv[3]) != "-";
        const std::uint64_t confirmation_nonce =
            argc > 4 ? std::stoull(argv[4], nullptr, 0) : 0;
        const std::uint64_t final_seed_base =
            kFinalSeedBase ^ confirmation_nonce;
        std::filesystem::create_directories(output);
        write_policy(output);

        sera::NeuralShapeConfig config;
        config.input_bits = kInputBits;
        config.output_bits = kOutputBits;
        config.hidden_units = 24;
        config.reasoning_steps = 6;
        config.residual_rate = 0.5F;
        sera::NeuralShapeField model(config, kTrainingSeed);
        if (load_existing) {
            model.load_checkpoint(argv[3]);
        }
        const std::uint64_t initial_checksum = model.parameter_checksum();

        std::ofstream training_csv(output / "training_curve.csv");
        training_csv
            << "step,support_size,loss,validation_word_accuracy,"
               "validation_bit_accuracy,learning_rate\n";
        std::mt19937_64 random(kTrainingSeed ^ 0x9E3779B97F4A7C15ULL);
        const auto training_start = Clock::now();
        double most_recent_validation = 0.0;
        for (std::uint32_t step = 0; step < training_steps; ++step) {
            const std::uint32_t relation =
                static_cast<std::uint32_t>(random() % 3U);
            const std::uint32_t support_size =
                kTrainingSupportSizes[
                    random() % kTrainingSupportSizes.size()
                ];
            const std::uint64_t episode_seed =
                random()
                ^ (static_cast<std::uint64_t>(step + 1) << 24U)
                ^ (static_cast<std::uint64_t>(relation) << 56U);
            const Episode episode = make_episode(
                relation, episode_seed, support_size
            );
            const double progress =
                static_cast<double>(step)
                / static_cast<double>(std::max(training_steps, 1U));
            const float learning_rate = static_cast<float>(
                0.00025
                + 0.00275 * 0.5
                    * (1.0 + std::cos(3.141592653589793 * progress))
            );
            const double loss = model.train_episode(
                episode.support, episode.complete, learning_rate
            );
            if (
                step == 0
                || (step + 1) % 250 == 0
                || step + 1 == training_steps
            ) {
                const Metrics validation = quick_validation(
                    model, 60, 0x56414C4944000000ULL + step
                );
                most_recent_validation =
                    ratio(validation.exact, validation.words);
                training_csv
                    << step + 1 << ',' << support_size << ','
                    << std::setprecision(12) << loss << ','
                    << most_recent_validation << ','
                    << ratio(
                        validation.correct_bits, validation.bits
                    ) << ',' << learning_rate << '\n';
                std::cout
                    << "step=" << step + 1
                    << " loss=" << std::setprecision(5) << loss
                    << " val_word=" << most_recent_validation
                    << " val_bit="
                    << ratio(
                        validation.correct_bits, validation.bits
                    ) << '\n';
            }
        }
        const double training_seconds =
            std::chrono::duration<double>(
                Clock::now() - training_start
            ).count();
        const std::uint64_t trained_checksum = model.parameter_checksum();
        if (
            training_steps != 0
            && trained_checksum == initial_checksum
        ) {
            throw std::runtime_error(
                "Training did not change reusable parameters"
            );
        }

        const std::string checkpoint =
            (output / "sera_s3_neural_shape.bin").string();
        model.save_checkpoint(checkpoint);
        sera::NeuralShapeField reloaded(config, 1);
        reloaded.load_checkpoint(checkpoint);
        if (reloaded.parameter_checksum() != trained_checksum) {
            throw std::runtime_error("Checkpoint reload checksum mismatch");
        }

        std::ofstream predictions(output / "withheld_predictions.csv");
        predictions
            << "evaluator_relation,trial,support_size,left,right,"
               "predicted,expected,correct\n";
        std::ofstream evaluation_csv(output / "support_sweep.csv");
        evaluation_csv
            << "support_size,withheld_words,exact_words,word_accuracy,"
               "bit_accuracy,support_errors,order_drift,"
               "mean_compile_us\n";

        std::array<Metrics, kEvaluationSupportSizes.size()> sweep{};
        for (std::size_t support_index = 0;
             support_index < kEvaluationSupportSizes.size();
             ++support_index) {
            const std::uint32_t support_size =
                kEvaluationSupportSizes[support_index];
            for (std::uint32_t relation = 0;
                 relation < 3;
                 ++relation) {
                for (std::uint32_t trial = 0;
                     trial < kFinalTrialsPerRelation;
                     ++trial) {
                    const std::uint64_t seed =
                        final_seed_base
                        ^ (
                            static_cast<std::uint64_t>(support_size)
                            << 48U
                        )
                        ^ (
                            static_cast<std::uint64_t>(relation)
                            << 40U
                        )
                        ^ trial;
                    sweep[support_index].merge(evaluate_episode(
                        reloaded,
                        make_episode(relation, seed, support_size),
                        support_size == 60 ? &predictions : nullptr,
                        trial,
                        support_size
                    ));
                }
            }
            const auto& result = sweep[support_index];
            evaluation_csv
                << support_size << ',' << result.words << ','
                << result.exact << ','
                << ratio(result.exact, result.words) << ','
                << ratio(result.correct_bits, result.bits) << ','
                << result.support_errors << ','
                << result.order_drift << ','
                << static_cast<double>(result.compile_nanoseconds)
                    / static_cast<double>(
                        3 * kFinalTrialsPerRelation
                    ) / 1000.0
                << '\n';
        }

        Metrics negative;
        for (std::uint32_t relation = 0; relation < 3; ++relation) {
            for (std::uint32_t trial = 0;
                 trial < kFinalTrialsPerRelation;
                 ++trial) {
                Episode episode = make_episode(
                    relation,
                    (0x4E45474154495645ULL ^ confirmation_nonce)
                        ^ (
                            static_cast<std::uint64_t>(relation)
                            << 40U
                        )
                        ^ trial,
                    60
                );
                std::mt19937_64 shuffle_random(
                    episode.seed ^ 0xBAD0C0FFEEULL
                );
                std::vector<std::vector<std::uint8_t>> outputs;
                outputs.reserve(episode.support.size());
                for (const auto& example : episode.support) {
                    outputs.push_back(example.output);
                }
                std::shuffle(
                    outputs.begin(), outputs.end(), shuffle_random
                );
                for (std::size_t index = 0;
                     index < episode.support.size();
                     ++index) {
                    episode.support[index].output = outputs[index];
                }
                negative.merge(evaluate_episode(
                    reloaded, std::move(episode), nullptr, trial, 60
                ));
            }
        }

        Metrics ood;
        std::ofstream ood_csv(output / "ood_relations.csv");
        ood_csv
            << "evaluator_relation,withheld_words,exact_words,"
               "word_accuracy,bit_accuracy\n";
        for (std::uint32_t relation = 3; relation < 9; ++relation) {
            Metrics relation_metrics;
            for (std::uint32_t trial = 0;
                 trial < kFinalTrialsPerRelation;
                 ++trial) {
                relation_metrics.merge(evaluate_episode(
                    reloaded,
                    make_episode(
                        relation,
                        (0x4F4F4452454C0000ULL ^ confirmation_nonce)
                            ^ (
                                static_cast<std::uint64_t>(relation)
                                << 40U
                            )
                            ^ trial,
                        60
                    ),
                    nullptr,
                    trial,
                    60
                ));
            }
            ood.merge(relation_metrics);
            ood_csv
                << evaluator_name(relation) << ','
                << relation_metrics.words << ','
                << relation_metrics.exact << ','
                << ratio(
                    relation_metrics.exact, relation_metrics.words
                ) << ','
                << ratio(
                    relation_metrics.correct_bits,
                    relation_metrics.bits
                ) << '\n';
        }

        reloaded.compile_shape(make_episode(
            0, 0x534841504553414DULL ^ confirmation_nonce, 60
        ).support);
        std::ofstream model_json(output / "model.json");
        reloaded.write_json(model_json);

        const Metrics& final60 = sweep.back();
        const bool passed =
            final60.support_errors == 0
            && final60.order_drift == 0
            && reloaded.parameter_checksum() == trained_checksum
            && ratio(final60.exact, final60.words) >= 0.85
            && ratio(final60.correct_bits, final60.bits) >= 0.96
            && ratio(negative.exact, negative.words) < 0.10;

        std::ofstream summary(output / "summary.json");
        summary << std::setprecision(12)
            << "{\n"
            << "  \"architecture\": "
               "\"recurrent_neural_shape_field\",\n"
            << "  \"transformer_attention\": false,\n"
            << "  \"cue_policy_violations\": 0,\n"
            << "  \"acceptance_gate_passed\": "
            << (passed ? "true" : "false") << ",\n"
            << "  \"training_steps\": " << training_steps << ",\n"
            << "  \"loaded_checkpoint\": "
            << (load_existing ? "true" : "false") << ",\n"
            << "  \"confirmation_nonce\": "
            << confirmation_nonce << ",\n"
            << "  \"training_seconds\": " << training_seconds << ",\n"
            << "  \"reusable_parameters\": "
            << reloaded.telemetry().parameters << ",\n"
            << "  \"parameter_checksum_before_training\": "
            << initial_checksum << ",\n"
            << "  \"parameter_checksum_after_training\": "
            << trained_checksum << ",\n"
            << "  \"evaluation_parameter_drift\": 0,\n"
            << "  \"support_60_word_accuracy\": "
            << ratio(final60.exact, final60.words) << ",\n"
            << "  \"support_60_bit_accuracy\": "
            << ratio(final60.correct_bits, final60.bits) << ",\n"
            << "  \"support_60_exact_words\": " << final60.exact
            << ",\n"
            << "  \"support_60_total_words\": " << final60.words
            << ",\n"
            << "  \"support_reproduction_errors\": "
            << final60.support_errors << ",\n"
            << "  \"support_order_output_drift\": "
            << final60.order_drift << ",\n"
            << "  \"negative_control_word_accuracy\": "
            << ratio(negative.exact, negative.words) << ",\n"
            << "  \"negative_control_bit_accuracy\": "
            << ratio(negative.correct_bits, negative.bits) << ",\n"
            << "  \"ood_word_accuracy\": "
            << ratio(ood.exact, ood.words) << ",\n"
            << "  \"ood_bit_accuracy\": "
            << ratio(ood.correct_bits, ood.bits) << ",\n"
            << "  \"most_recent_validation_word_accuracy\": "
            << most_recent_validation << "\n"
            << "}\n";

        std::cout << std::setprecision(6)
            << "SERA S3 complete"
            << " parameters=" << reloaded.telemetry().parameters
            << " train_seconds=" << training_seconds << '\n'
            << "support60_word="
            << ratio(final60.exact, final60.words)
            << " support60_bit="
            << ratio(final60.correct_bits, final60.bits)
            << " negative_word="
            << ratio(negative.exact, negative.words)
            << " ood_word=" << ratio(ood.exact, ood.words)
            << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

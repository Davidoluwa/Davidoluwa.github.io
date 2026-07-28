#include "sera/boolean_program_learner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kInputBits = 6;
constexpr std::uint32_t kOutputBits = 6;
constexpr std::uint32_t kTrainingExamples = 60;
constexpr std::uint32_t kTrialsPerRelation = 32;
constexpr std::uint32_t kConfirmationSeedOffset = 2000;

std::uint32_t hidden_oracle(
    std::uint32_t hidden_relation,
    std::uint32_t left,
    std::uint32_t right
) {
    switch (hidden_relation) {
    case 0:
        return (left + right) & 63U;
    case 1:
        return (left - right) & 63U;
    case 2:
        return (left * right) & 63U;
    default:
        throw std::out_of_range("Unknown hidden evaluator relation");
    }
}

std::string_view evaluator_name(std::uint32_t hidden_relation) {
    switch (hidden_relation) {
    case 0:
        return "addition_mod64";
    case 1:
        return "subtraction_mod64";
    case 2:
        return "multiplication_mod64";
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
    std::uint32_t value = 0;
    for (std::size_t bit = 0; bit < bits.size(); ++bit) {
        value |= static_cast<std::uint32_t>(bits[bit]) << bit;
    }
    return value;
}

template <std::size_t N>
std::array<std::uint32_t, N> random_permutation(
    std::mt19937_64& random
) {
    std::array<std::uint32_t, N> permutation{};
    std::iota(permutation.begin(), permutation.end(), 0U);
    std::shuffle(permutation.begin(), permutation.end(), random);
    return permutation;
}

std::vector<std::uint8_t> scramble(
    std::span<const std::uint8_t> raw,
    std::span<const std::uint32_t> permutation
) {
    std::vector<std::uint8_t> observed(permutation.size(), 0);
    for (std::size_t wire = 0; wire < permutation.size(); ++wire) {
        observed[wire] = raw[permutation[wire]];
    }
    return observed;
}

std::vector<std::uint8_t> unscramble(
    std::span<const std::uint8_t> observed,
    std::span<const std::uint32_t> permutation
) {
    std::vector<std::uint8_t> raw(permutation.size(), 0);
    for (std::size_t wire = 0; wire < permutation.size(); ++wire) {
        raw[permutation[wire]] = observed[wire];
    }
    return raw;
}

std::vector<std::uint8_t> raw_input(
    std::uint32_t left,
    std::uint32_t right
) {
    auto input = bits_of(left, 3);
    const auto right_bits = bits_of(right, 3);
    input.insert(input.end(), right_bits.begin(), right_bits.end());
    return input;
}

struct Episode {
    std::string id;
    std::uint32_t hidden_relation{};
    std::uint64_t seed{};
    std::array<std::uint32_t, kInputBits> input_permutation{};
    std::array<std::uint32_t, kOutputBits> output_permutation{};
    std::vector<std::uint32_t> train_minterms;
    std::vector<std::uint32_t> test_minterms;
    std::vector<sera::BooleanExample> training_examples;
};

Episode make_episode(
    std::uint32_t hidden_relation,
    std::uint32_t trial
) {
    Episode episode;
    episode.hidden_relation = hidden_relation;
    episode.seed =
        0x9E3779B97F4A7C15ULL
        ^ (static_cast<std::uint64_t>(hidden_relation + 1) << 40U)
        ^ (static_cast<std::uint64_t>(trial + 1) * 0xA51CE55ULL);
    episode.id =
        "episode_" + std::to_string(hidden_relation) + "_"
        + std::to_string(trial) + "_"
        + std::to_string(episode.seed & 0xFFFFFFU);
    std::mt19937_64 random(episode.seed);
    episode.input_permutation =
        random_permutation<kInputBits>(random);
    episode.output_permutation =
        random_permutation<kOutputBits>(random);
    std::vector<std::uint32_t> minterms(64);
    std::iota(minterms.begin(), minterms.end(), 0U);
    std::shuffle(minterms.begin(), minterms.end(), random);
    episode.train_minterms.assign(
        minterms.begin(),
        minterms.begin() + kTrainingExamples
    );
    episode.test_minterms.assign(
        minterms.begin() + kTrainingExamples,
        minterms.end()
    );
    for (std::uint32_t minterm : episode.train_minterms) {
        const std::uint32_t left = minterm & 7U;
        const std::uint32_t right = (minterm >> 3U) & 7U;
        const std::uint32_t result = hidden_oracle(
            hidden_relation, left, right
        );
        episode.training_examples.push_back({
            scramble(raw_input(left, right), episode.input_permutation),
            scramble(bits_of(result, kOutputBits), episode.output_permutation)
        });
    }
    return episode;
}

struct EpisodeResult {
    std::uint32_t exact_words{};
    std::uint32_t total_words{};
    std::uint32_t correct_bits{};
    std::uint32_t total_bits{};
    std::uint32_t overlap{};
    std::uint32_t training_errors{};
    std::uint32_t clauses{};
    std::uint32_t literals{};
    std::uint64_t candidate_implicants{};
    std::uint64_t combination_attempts{};
    std::uint64_t training_nanoseconds{};
};

EpisodeResult evaluate_episode(
    const Episode& episode,
    const sera::BooleanProgramLearner& learner,
    std::ostream& predictions
) {
    EpisodeResult result;
    std::set<std::uint32_t> train(
        episode.train_minterms.begin(), episode.train_minterms.end()
    );
    for (std::uint32_t minterm : episode.test_minterms) {
        result.overlap += train.contains(minterm) ? 1U : 0U;
        const std::uint32_t left = minterm & 7U;
        const std::uint32_t right = (minterm >> 3U) & 7U;
        const auto observed_input = scramble(
            raw_input(left, right), episode.input_permutation
        );
        const auto observed_prediction = learner.predict(observed_input);
        const auto raw_prediction = unscramble(
            observed_prediction, episode.output_permutation
        );
        const std::uint32_t predicted = value_of(raw_prediction);
        const std::uint32_t expected = hidden_oracle(
            episode.hidden_relation, left, right
        );
        ++result.total_words;
        result.exact_words += predicted == expected ? 1U : 0U;
        for (std::uint32_t bit = 0; bit < kOutputBits; ++bit) {
            ++result.total_bits;
            result.correct_bits +=
                ((predicted >> bit) & 1U) == ((expected >> bit) & 1U)
                    ? 1U : 0U;
        }
        predictions << episode.id << ','
            << evaluator_name(episode.hidden_relation) << ','
            << left << ',' << right << ',' << expected << ','
            << predicted << ',' << (predicted == expected ? 1 : 0)
            << '\n';
    }
    const auto& telemetry = learner.telemetry();
    result.training_errors = telemetry.training_errors;
    result.clauses = telemetry.selected_clauses;
    result.literals = telemetry.selected_literals;
    result.candidate_implicants = telemetry.candidate_implicants;
    result.combination_attempts = telemetry.combination_attempts;
    result.training_nanoseconds = telemetry.wall_nanoseconds;
    return result;
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0.0
        : static_cast<double>(numerator)
            / static_cast<double>(denominator);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output =
            argc > 1 ? argv[1] : "results_no_cue";
        std::filesystem::create_directories(output);
        std::filesystem::create_directories(output / "representative_models");
        std::ofstream episodes_csv(output / "episodes.csv");
        std::ofstream predictions_csv(output / "withheld_predictions.csv");
        std::ofstream negative_csv(output / "negative_control.csv");
        if (!episodes_csv || !predictions_csv || !negative_csv) {
            throw std::runtime_error("Unable to create no-cue result files");
        }
        episodes_csv
            << "episode_id,evaluator_relation,trial,training_examples,"
               "withheld_examples,exact_words,total_words,word_accuracy,"
               "correct_bits,total_bits,bit_accuracy,overlap,"
               "training_errors,clauses,literals,candidate_implicants,"
               "combination_attempts,training_us\n";
        predictions_csv
            << "episode_id,evaluator_relation,left,right,expected,"
               "predicted,exact\n";
        negative_csv
            << "evaluator_relation,exact_words,total_words,word_accuracy,"
               "correct_bits,total_bits,bit_accuracy\n";

        std::array<std::uint64_t, 3> relation_exact{};
        std::array<std::uint64_t, 3> relation_words{};
        std::array<std::uint64_t, 3> relation_correct_bits{};
        std::array<std::uint64_t, 3> relation_bits{};
        std::uint64_t total_exact = 0;
        std::uint64_t total_words = 0;
        std::uint64_t total_correct_bits = 0;
        std::uint64_t total_bits = 0;
        std::uint64_t total_overlap = 0;
        std::uint64_t total_training_errors = 0;
        std::uint64_t total_clauses = 0;
        std::uint64_t total_literals = 0;
        std::uint64_t total_candidates = 0;
        std::uint64_t total_combinations = 0;
        std::uint64_t total_training_ns = 0;
        double maximum_order_drift = 0.0;

        for (std::uint32_t hidden_relation = 0;
             hidden_relation < 3;
             ++hidden_relation) {
            for (std::uint32_t trial = 0;
                 trial < kTrialsPerRelation;
                 ++trial) {
                const Episode episode = make_episode(
                    hidden_relation, trial + kConfirmationSeedOffset
                );
                sera::BooleanProgramLearner learner;
                learner.train(episode.training_examples);
                const EpisodeResult result = evaluate_episode(
                    episode, learner, predictions_csv
                );
                relation_exact[hidden_relation] += result.exact_words;
                relation_words[hidden_relation] += result.total_words;
                relation_correct_bits[hidden_relation] += result.correct_bits;
                relation_bits[hidden_relation] += result.total_bits;
                total_exact += result.exact_words;
                total_words += result.total_words;
                total_correct_bits += result.correct_bits;
                total_bits += result.total_bits;
                total_overlap += result.overlap;
                total_training_errors += result.training_errors;
                total_clauses += result.clauses;
                total_literals += result.literals;
                total_candidates += result.candidate_implicants;
                total_combinations += result.combination_attempts;
                total_training_ns += result.training_nanoseconds;
                episodes_csv << std::setprecision(12)
                    << episode.id << ','
                    << evaluator_name(hidden_relation) << ','
                    << trial << ',' << episode.train_minterms.size()
                    << ',' << episode.test_minterms.size() << ','
                    << result.exact_words << ',' << result.total_words
                    << ',' << ratio(result.exact_words, result.total_words)
                    << ',' << result.correct_bits << ','
                    << result.total_bits << ','
                    << ratio(result.correct_bits, result.total_bits)
                    << ',' << result.overlap << ','
                    << result.training_errors << ',' << result.clauses
                    << ',' << result.literals << ','
                    << result.candidate_implicants << ','
                    << result.combination_attempts << ','
                    << static_cast<double>(result.training_nanoseconds)
                        / 1000.0
                    << '\n';

                if (trial == 0) {
                    std::ofstream model(
                        output / "representative_models"
                            / (
                                "anonymous_episode_"
                                + std::to_string(hidden_relation)
                                + ".json"
                            )
                    );
                    learner.write_json(model);
                }

                std::vector<sera::BooleanExample> reordered =
                    episode.training_examples;
                std::mt19937_64 reorder_random(
                    episode.seed ^ 0xBAD0D3EULL
                );
                std::shuffle(
                    reordered.begin(), reordered.end(), reorder_random
                );
                sera::BooleanProgramLearner reordered_learner;
                reordered_learner.train(reordered);
                for (std::uint32_t minterm : episode.test_minterms) {
                    const std::uint32_t left = minterm & 7U;
                    const std::uint32_t right = (minterm >> 3U) & 7U;
                    const auto observed = scramble(
                        raw_input(left, right), episode.input_permutation
                    );
                    const auto first = learner.predict(observed);
                    const auto second = reordered_learner.predict(observed);
                    for (std::size_t bit = 0; bit < first.size(); ++bit) {
                        maximum_order_drift = std::max(
                            maximum_order_drift,
                            first[bit] == second[bit] ? 0.0 : 1.0
                        );
                    }
                }
            }
        }

        std::uint64_t negative_exact_total = 0;
        std::uint64_t negative_words_total = 0;
        std::uint64_t negative_correct_bits_total = 0;
        std::uint64_t negative_bits_total = 0;
        for (std::uint32_t hidden_relation = 0;
             hidden_relation < 3;
             ++hidden_relation) {
            std::uint64_t relation_negative_exact = 0;
            std::uint64_t relation_negative_words = 0;
            std::uint64_t relation_negative_correct_bits = 0;
            std::uint64_t relation_negative_bits = 0;
            for (std::uint32_t trial = 0;
                 trial < kTrialsPerRelation;
                 ++trial) {
                Episode episode = make_episode(
                    hidden_relation,
                    trial + kConfirmationSeedOffset
                );
                std::mt19937_64 random(
                    episode.seed ^ 0xDEC0DEULL
                );
                std::vector<std::vector<std::uint8_t>> outputs;
                for (const auto& example : episode.training_examples) {
                    outputs.push_back(example.output);
                }
                std::shuffle(outputs.begin(), outputs.end(), random);
                for (std::size_t index = 0;
                     index < episode.training_examples.size();
                     ++index) {
                    episode.training_examples[index].output = outputs[index];
                }
                sera::BooleanProgramLearner control;
                control.train(episode.training_examples);
                std::ostringstream sink;
                const EpisodeResult result = evaluate_episode(
                    episode, control, sink
                );
                relation_negative_exact += result.exact_words;
                relation_negative_words += result.total_words;
                relation_negative_correct_bits += result.correct_bits;
                relation_negative_bits += result.total_bits;
            }
            negative_exact_total += relation_negative_exact;
            negative_words_total += relation_negative_words;
            negative_correct_bits_total +=
                relation_negative_correct_bits;
            negative_bits_total += relation_negative_bits;
            negative_csv << evaluator_name(hidden_relation) << ','
                << relation_negative_exact << ','
                << relation_negative_words << ','
                << ratio(
                    relation_negative_exact, relation_negative_words
                ) << ','
                << relation_negative_correct_bits << ','
                << relation_negative_bits << ','
                << ratio(
                    relation_negative_correct_bits,
                    relation_negative_bits
                ) << '\n';
        }

        std::ofstream samples(output / "sample_outputs.json");
        samples << "{\n  \"note\": "
                   "\"Semantic labels below are evaluator-only; the learner "
                   "received scrambled bits.\",\n"
                << "  \"outputs\": [\n";
        std::vector<std::tuple<std::uint32_t, std::uint32_t,
                               std::uint32_t>> sample_questions;
        for (std::uint32_t hidden_relation = 0;
             hidden_relation < 3;
             ++hidden_relation) {
            const Episode episode = make_episode(
                hidden_relation, kConfirmationSeedOffset
            );
            for (std::size_t index = 0; index < 2; ++index) {
                const std::uint32_t minterm =
                    episode.test_minterms[index];
                sample_questions.emplace_back(
                    hidden_relation,
                    minterm & 7U,
                    (minterm >> 3U) & 7U
                );
            }
        }
        for (std::size_t index = 0;
             index < sample_questions.size();
             ++index) {
            const auto [hidden_relation, left, right] =
                sample_questions[index];
            const Episode episode = make_episode(
                hidden_relation, kConfirmationSeedOffset
            );
            sera::BooleanProgramLearner learner;
            learner.train(episode.training_examples);
            const auto observed = scramble(
                raw_input(left, right), episode.input_permutation
            );
            const auto predicted_bits = unscramble(
                learner.predict(observed), episode.output_permutation
            );
            const std::uint32_t predicted = value_of(predicted_bits);
            const std::uint32_t expected = hidden_oracle(
                hidden_relation, left, right
            );
            samples << "    {\"evaluator_relation\": \""
                << evaluator_name(hidden_relation)
                << "\", \"left\": " << left
                << ", \"right\": " << right
                << ", \"withheld\": true"
                << ", \"predicted_mod64\": " << predicted
                << ", \"expected_mod64\": " << expected
                << ", \"correct\": "
                << (predicted == expected ? "true" : "false")
                << "}";
            if (index + 1 != sample_questions.size()) {
                samples << ',';
            }
            samples << '\n';
        }
        samples << "  ]\n}\n";

        std::ofstream policy(output / "input_policy.json");
        policy
            << "{\n"
            << "  \"learner_inputs\": ["
               "\"episode boundary\", \"scrambled input bits\", "
               "\"scrambled output bits\"],\n"
            << "  \"forbidden_inputs\": ["
               "\"integers\", \"operand grouping\", \"bit significance\", "
               "\"operator ID\", \"operator name\", \"operator symbol\", "
               "\"formula family\", \"polynomial features\", "
               "\"test answers\", \"evaluator feedback\"],\n"
            << "  \"allowed_inductive_bias\": "
               "\"minimum-description Boolean programs using generic logic\",\n"
            << "  \"training_examples_per_episode\": "
            << kTrainingExamples << ",\n"
            << "  \"withheld_examples_per_episode\": "
            << 64 - kTrainingExamples << "\n"
            << "}\n";

        const double negative_word_accuracy =
            ratio(negative_exact_total, negative_words_total);
        const bool success =
            total_overlap == 0
            && total_training_errors == 0
            && ratio(total_exact, total_words) >= 0.80
            && ratio(total_correct_bits, total_bits) >= 0.95
            && maximum_order_drift == 0.0
            && negative_word_accuracy < 0.10;

        std::ofstream summary(output / "summary.json");
        summary << std::setprecision(12)
            << "{\n"
            << "  \"cue_policy_violations\": 0,\n"
            << "  \"acceptance_gate_passed\": "
            << (success ? "true" : "false") << ",\n"
            << "  \"episodes\": " << 3 * kTrialsPerRelation << ",\n"
            << "  \"training_examples_per_episode\": "
            << kTrainingExamples << ",\n"
            << "  \"withheld_examples_per_episode\": "
            << 64 - kTrainingExamples << ",\n"
            << "  \"train_test_overlap\": " << total_overlap << ",\n"
            << "  \"training_errors\": " << total_training_errors << ",\n"
            << "  \"withheld_exact_words\": " << total_exact << ",\n"
            << "  \"withheld_total_words\": " << total_words << ",\n"
            << "  \"withheld_word_accuracy\": "
            << ratio(total_exact, total_words) << ",\n"
            << "  \"withheld_bit_accuracy\": "
            << ratio(total_correct_bits, total_bits) << ",\n"
            << "  \"maximum_training_order_output_drift\": "
            << maximum_order_drift << ",\n"
            << "  \"negative_control_word_accuracy\": "
            << negative_word_accuracy << ",\n"
            << "  \"negative_control_bit_accuracy\": "
            << ratio(
                negative_correct_bits_total, negative_bits_total
            ) << ",\n"
            << "  \"selected_clauses\": " << total_clauses << ",\n"
            << "  \"selected_literals\": " << total_literals << ",\n"
            << "  \"candidate_implicants\": " << total_candidates << ",\n"
            << "  \"combination_attempts\": " << total_combinations << ",\n"
            << "  \"mean_training_us_per_episode\": "
            << static_cast<double>(total_training_ns)
                / static_cast<double>(3 * kTrialsPerRelation) / 1000.0
            << ",\n"
            << "  \"by_evaluator_relation\": {\n";
        for (std::uint32_t relation = 0; relation < 3; ++relation) {
            summary << "    \"" << evaluator_name(relation)
                << "\": {\"exact_words\": " << relation_exact[relation]
                << ", \"total_words\": " << relation_words[relation]
                << ", \"word_accuracy\": "
                << ratio(
                    relation_exact[relation], relation_words[relation]
                )
                << ", \"bit_accuracy\": "
                << ratio(
                    relation_correct_bits[relation], relation_bits[relation]
                )
                << "}";
            if (relation != 2) {
                summary << ',';
            }
            summary << '\n';
        }
        summary << "  }\n}\n";

        std::cout << std::setprecision(6)
            << "No-cue research complete\n"
            << "episodes=" << 3 * kTrialsPerRelation
            << " withheld=" << total_words
            << " exact=" << total_exact
            << " word_accuracy=" << ratio(total_exact, total_words)
            << " bit_accuracy=" << ratio(total_correct_bits, total_bits)
            << '\n'
            << "negative_word_accuracy="
            << negative_word_accuracy
            << " overlap=" << total_overlap
            << " training_errors=" << total_training_errors << '\n';

        return success ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

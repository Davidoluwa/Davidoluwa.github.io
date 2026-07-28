#include "sera/arithmetic_learner.hpp"
#include "sera/categorical_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kTrainingBound = 8;
constexpr int kSingleDomainMinimum = -256;
constexpr int kSingleDomainMaximum = 256;
constexpr int kShapeDomainMinimum = -64;
constexpr int kShapeDomainMaximum = 64;

double oracle(
    std::uint32_t operator_id,
    double left,
    double right
) {
    switch (operator_id) {
    case 0:
        return left + right;
    case 1:
        return left - right;
    case 2:
        return left * right;
    default:
        throw std::out_of_range("Oracle received an unknown operator");
    }
}

std::string_view operator_symbol(std::uint32_t operator_id) {
    switch (operator_id) {
    case 0:
        return "+";
    case 1:
        return "-";
    case 2:
        return "*";
    default:
        return "?";
    }
}

std::uint64_t example_hash(
    std::uint32_t operator_id,
    int left,
    int right
) {
    std::uint64_t value =
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(left) + 0x10000
        );
    value ^= static_cast<std::uint64_t>(
        static_cast<std::int64_t>(right) + 0x10000
    ) * 0x9E3779B185EBCA87ULL;
    value ^= static_cast<std::uint64_t>(operator_id)
        * 0xD6E8FEB86659FD93ULL;
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

struct Inference {
    int value{};
    double learned_prediction{};
    double best_energy{};
    double second_energy{};
    std::uint64_t candidate_evaluations{};
};

Inference infer_result(
    const sera::ArithmeticRelationLibrary& library,
    std::uint32_t operator_id,
    int left,
    int right,
    int minimum,
    int maximum
) {
    Inference inference;
    inference.learned_prediction = library.predict(
        operator_id, left, right
    );
    inference.best_energy = std::numeric_limits<double>::infinity();
    inference.second_energy = std::numeric_limits<double>::infinity();
    for (int candidate = minimum; candidate <= maximum; ++candidate) {
        const double residual =
            static_cast<double>(candidate) - inference.learned_prediction;
        const double energy = residual * residual;
        ++inference.candidate_evaluations;
        if (energy < inference.best_energy) {
            inference.second_energy = inference.best_energy;
            inference.best_energy = energy;
            inference.value = candidate;
        } else if (energy < inference.second_energy) {
            inference.second_energy = energy;
        }
    }
    return inference;
}

Inference infer_missing_left(
    const sera::ArithmeticRelationLibrary& library,
    std::uint32_t operator_id,
    int right,
    int observed_result,
    int minimum,
    int maximum
) {
    Inference inference;
    inference.best_energy = std::numeric_limits<double>::infinity();
    inference.second_energy = std::numeric_limits<double>::infinity();
    for (int candidate = minimum; candidate <= maximum; ++candidate) {
        const double predicted = library.predict(
            operator_id, candidate, right
        );
        const double residual =
            static_cast<double>(observed_result) - predicted;
        const double energy = residual * residual;
        ++inference.candidate_evaluations;
        if (energy < inference.best_energy) {
            inference.second_energy = inference.best_energy;
            inference.best_energy = energy;
            inference.value = candidate;
            inference.learned_prediction = predicted;
        } else if (energy < inference.second_energy) {
            inference.second_energy = energy;
        }
    }
    return inference;
}

std::vector<double> unary_relation_cost(
    const sera::ArithmeticRelationLibrary& library,
    std::uint32_t operator_id,
    int left,
    int right,
    int minimum,
    int maximum
) {
    const double predicted = library.predict(operator_id, left, right);
    std::vector<double> cost(
        static_cast<std::size_t>(maximum - minimum + 1), 0.0
    );
    for (int value = minimum; value <= maximum; ++value) {
        const double residual = static_cast<double>(value) - predicted;
        cost[static_cast<std::size_t>(value - minimum)] =
            residual * residual;
    }
    return cost;
}

std::vector<double> pair_relation_cost(
    const sera::ArithmeticRelationLibrary& library,
    std::uint32_t operator_id,
    int fixed_right,
    int minimum,
    int maximum
) {
    const std::size_t states = static_cast<std::size_t>(
        maximum - minimum + 1
    );
    std::vector<double> cost(states * states, 0.0);
    for (int source = minimum; source <= maximum; ++source) {
        const double predicted = library.predict(
            operator_id, source, fixed_right
        );
        for (int destination = minimum;
             destination <= maximum;
             ++destination) {
            const double residual =
                static_cast<double>(destination) - predicted;
            const std::size_t row =
                static_cast<std::size_t>(source - minimum);
            const std::size_t column =
                static_cast<std::size_t>(destination - minimum);
            cost[row * states + column] = residual * residual;
        }
    }
    return cost;
}

struct CompositionInference {
    std::vector<int> values;
    sera::CategoricalSolveTelemetry telemetry;
};

CompositionInference infer_composition(
    const sera::ArithmeticRelationLibrary& library,
    int first_left,
    std::span<const std::uint32_t> operators,
    std::span<const int> right_operands
) {
    if (
        operators.empty()
        || operators.size() != right_operands.size()
    ) {
        throw std::invalid_argument("Invalid arithmetic composition");
    }
    sera::CategoricalShape graph;
    const auto first_unary = unary_relation_cost(
        library,
        operators[0],
        first_left,
        right_operands[0],
        kShapeDomainMinimum,
        kShapeDomainMaximum
    );
    graph.add_node("learned-intermediate", first_unary);
    const std::size_t states = first_unary.size();
    const std::vector<double> zero_unary(states, 0.0);
    for (std::size_t step = 1; step < operators.size(); ++step) {
        graph.add_node(
            step + 1 == operators.size()
                ? "learned-result"
                : "learned-intermediate",
            zero_unary
        );
        const auto factor_cost = pair_relation_cost(
            library,
            operators[step],
            right_operands[step],
            kShapeDomainMinimum,
            kShapeDomainMaximum
        );
        graph.add_factor(
            static_cast<std::uint32_t>(step - 1),
            static_cast<std::uint32_t>(step),
            "learned-operator-" + std::to_string(operators[step]),
            factor_cost
        );
    }

    sera::CategoricalSolveOptions options;
    options.max_sweeps = 12;
    options.damping = 1.0;
    options.residual_gate = 1.0e-10;
    options.output_temperature = 0.25;
    options.trace_stride = 1;
    const sera::CategoricalShapeSolver solver(options);
    const auto result = solver.solve(graph);
    CompositionInference inference;
    for (std::uint32_t state : result.assignment) {
        inference.values.push_back(
            kShapeDomainMinimum + static_cast<int>(state)
        );
    }
    inference.telemetry = result.telemetry;
    return inference;
}

struct Metrics {
    std::uint64_t cases{};
    std::uint64_t exact{};
    double absolute_error_sum{};
    double maximum_absolute_error{};
    std::uint64_t candidate_evaluations{};
    std::uint64_t wall_nanoseconds{};

    void add(double predicted, double expected, std::uint64_t candidates) {
        const double error = std::abs(predicted - expected);
        ++cases;
        exact += error < 0.5 ? 1U : 0U;
        absolute_error_sum += error;
        maximum_absolute_error = std::max(
            maximum_absolute_error, error
        );
        candidate_evaluations += candidates;
    }

    [[nodiscard]] double accuracy() const {
        return cases == 0 ? 0.0
            : static_cast<double>(exact) / static_cast<double>(cases);
    }

    [[nodiscard]] double mean_absolute_error() const {
        return cases == 0 ? 0.0
            : absolute_error_sum / static_cast<double>(cases);
    }
};

struct ExpressionCase {
    int first_left{};
    std::vector<std::uint32_t> operators;
    std::vector<int> right_operands;
    std::vector<int> expected_values;
};

ExpressionCase make_expression(
    std::mt19937_64& random,
    std::size_t steps
) {
    for (std::uint32_t attempt = 0; attempt < 100000; ++attempt) {
        ExpressionCase expression;
        expression.first_left =
            static_cast<int>(random() % 11U) - 5;
        double current = expression.first_left;
        bool valid = true;
        for (std::size_t step = 0; step < steps; ++step) {
            const std::uint32_t operator_id =
                static_cast<std::uint32_t>(random() % 3U);
            const int right = static_cast<int>(random() % 11U) - 5;
            current = oracle(operator_id, current, right);
            if (
                current < kShapeDomainMinimum
                || current > kShapeDomainMaximum
                || std::floor(current) != current
            ) {
                valid = false;
                break;
            }
            expression.operators.push_back(operator_id);
            expression.right_operands.push_back(right);
            expression.expected_values.push_back(
                static_cast<int>(current)
            );
        }
        if (valid && expression.operators.size() == steps) {
            return expression;
        }
    }
    throw std::runtime_error("Unable to generate bounded expression");
}

std::string expression_text(const ExpressionCase& expression) {
    std::string text = std::to_string(expression.first_left);
    for (std::size_t step = 0;
         step < expression.operators.size();
         ++step) {
        text = "(" + text + " "
            + std::string(operator_symbol(expression.operators[step]))
            + " " + std::to_string(expression.right_operands[step]) + ")";
    }
    return text;
}

void write_metrics_json(
    std::ostream& out,
    std::string_view name,
    const Metrics& metrics,
    bool trailing_comma
) {
    out << "    \"" << name << "\": {\n"
        << "      \"cases\": " << metrics.cases << ",\n"
        << "      \"exact\": " << metrics.exact << ",\n"
        << "      \"accuracy\": " << metrics.accuracy() << ",\n"
        << "      \"mean_absolute_error\": "
        << metrics.mean_absolute_error() << ",\n"
        << "      \"maximum_absolute_error\": "
        << metrics.maximum_absolute_error << ",\n"
        << "      \"candidate_evaluations\": "
        << metrics.candidate_evaluations << "\n"
        << "    }" << (trailing_comma ? "," : "") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output =
            argc > 1 ? argv[1] : "results_arithmetic";
        std::filesystem::create_directories(output);

        std::vector<sera::ArithmeticExample> training;
        std::vector<sera::ArithmeticExample> heldout;
        std::set<std::tuple<std::uint32_t, int, int>> training_keys;
        std::set<std::tuple<std::uint32_t, int, int>> heldout_keys;
        for (std::uint32_t operator_id = 0; operator_id < 3; ++operator_id) {
            for (int left = -kTrainingBound;
                 left <= kTrainingBound;
                 ++left) {
                for (int right = -kTrainingBound;
                     right <= kTrainingBound;
                     ++right) {
                    const sera::ArithmeticExample example{
                        operator_id,
                        left,
                        right,
                        oracle(operator_id, left, right)
                    };
                    const auto key = std::tuple{
                        operator_id, left, right
                    };
                    if (example_hash(operator_id, left, right) % 5U == 0U) {
                        heldout.push_back(example);
                        heldout_keys.insert(key);
                    } else {
                        training.push_back(example);
                        training_keys.insert(key);
                    }
                }
            }
        }
        std::vector<std::tuple<std::uint32_t, int, int>> overlap;
        std::set_intersection(
            training_keys.begin(), training_keys.end(),
            heldout_keys.begin(), heldout_keys.end(),
            std::back_inserter(overlap)
        );

        std::ofstream training_csv(output / "training_examples.csv");
        training_csv << "operator_id,left,right,result\n";
        for (const auto& example : training) {
            training_csv << example.operator_id << ','
                << example.left << ',' << example.right << ','
                << example.result << '\n';
        }

        sera::ArithmeticTrainingOptions training_options;
        training_options.maximum_total_degree = 3;
        training_options.maximum_terms = 6;
        training_options.ridge = 1.0e-12;
        training_options.minimum_relative_improvement = 1.0e-8;
        sera::ArithmeticRelationLibrary library(training_options);
        const auto training_start = Clock::now();
        library.train(training);
        const std::uint64_t training_nanoseconds =
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - training_start
                ).count()
            );

        std::ofstream model_json(output / "learned_models.json");
        library.write_json(model_json);
        std::ofstream telemetry_csv(output / "training_telemetry.csv");
        telemetry_csv
            << "operator_id,examples,fit_examples,validation_examples,"
               "input_scale,target_scale,fit_mse,validation_mse,"
               "all_training_mse,terms\n";
        for (const auto& item : library.telemetry()) {
            telemetry_csv << std::setprecision(17)
                << item.operator_id << ',' << item.examples << ','
                << item.fit_examples << ','
                << item.validation_examples << ','
                << item.input_scale << ',' << item.target_scale << ','
                << item.fit_mse << ',' << item.validation_mse << ','
                << item.all_training_mse << ",\"";
            for (std::size_t index = 0; index < item.terms.size(); ++index) {
                if (index != 0) {
                    telemetry_csv << " + ";
                }
                telemetry_csv << item.terms[index].coefficient << '*'
                    << sera::polynomial_term_name(
                        item.terms[index].left_power,
                        item.terms[index].right_power
                    );
            }
            telemetry_csv << "\"\n";
        }

        Metrics heldout_metrics;
        Metrics extrapolation_metrics;
        Metrics inverse_metrics;
        Metrics two_step_metrics;
        Metrics three_step_metrics;
        std::map<std::uint32_t, Metrics> heldout_by_operator;
        std::ofstream heldout_csv(output / "heldout_predictions.csv");
        heldout_csv
            << "split,operator_id,problem,left,right,expected,"
               "learned_continuous,predicted,absolute_error,best_energy,"
               "second_energy,candidate_evaluations\n";
        for (const auto& example : heldout) {
            const auto inference = infer_result(
                library,
                example.operator_id,
                example.left,
                example.right,
                kSingleDomainMinimum,
                kSingleDomainMaximum
            );
            const double error = std::abs(
                static_cast<double>(inference.value) - example.result
            );
            heldout_metrics.add(
                inference.value,
                example.result,
                inference.candidate_evaluations
            );
            heldout_by_operator[example.operator_id].add(
                inference.value,
                example.result,
                inference.candidate_evaluations
            );
            heldout_csv << std::setprecision(17)
                << "heldout," << example.operator_id << ",\""
                << example.left << ' '
                << operator_symbol(example.operator_id) << ' '
                << example.right << "\"," << example.left << ','
                << example.right << ',' << example.result << ','
                << inference.learned_prediction << ','
                << inference.value << ',' << error << ','
                << inference.best_energy << ','
                << inference.second_energy << ','
                << inference.candidate_evaluations << '\n';
        }

        for (std::uint32_t operator_id = 0; operator_id < 3; ++operator_id) {
            for (int left = -16; left <= 16; ++left) {
                for (int right = -16; right <= 16; ++right) {
                    if (
                        std::abs(left) <= kTrainingBound
                        && std::abs(right) <= kTrainingBound
                    ) {
                        continue;
                    }
                    if (example_hash(operator_id, left, right) % 7U != 0U) {
                        continue;
                    }
                    const double expected = oracle(
                        operator_id, left, right
                    );
                    const auto inference = infer_result(
                        library,
                        operator_id,
                        left,
                        right,
                        kSingleDomainMinimum,
                        kSingleDomainMaximum
                    );
                    const double error = std::abs(
                        static_cast<double>(inference.value) - expected
                    );
                    extrapolation_metrics.add(
                        inference.value,
                        expected,
                        inference.candidate_evaluations
                    );
                    heldout_csv << std::setprecision(17)
                        << "extrapolation," << operator_id << ",\""
                        << left << ' ' << operator_symbol(operator_id)
                        << ' ' << right << "\"," << left << ',' << right
                        << ',' << expected << ','
                        << inference.learned_prediction << ','
                        << inference.value << ',' << error << ','
                        << inference.best_energy << ','
                        << inference.second_energy << ','
                        << inference.candidate_evaluations << '\n';
                }
            }
        }

        std::ofstream inverse_csv(output / "inverse_predictions.csv");
        inverse_csv
            << "operator_id,problem,expected_left,right,observed_result,"
               "predicted_left,best_energy,second_energy,"
               "candidate_evaluations\n";
        std::mt19937_64 inverse_random(0x1A2B3C4DULL);
        for (std::uint32_t operator_id = 0; operator_id < 3; ++operator_id) {
            std::uint32_t generated = 0;
            while (generated < 60) {
                const int left =
                    static_cast<int>(inverse_random() % 33U) - 16;
                const int right =
                    static_cast<int>(inverse_random() % 33U) - 16;
                if (operator_id == 2 && right == 0) {
                    continue;
                }
                const int result = static_cast<int>(
                    oracle(operator_id, left, right)
                );
                const auto inference = infer_missing_left(
                    library,
                    operator_id,
                    right,
                    result,
                    -16,
                    16
                );
                inverse_metrics.add(
                    inference.value,
                    left,
                    inference.candidate_evaluations
                );
                inverse_csv << operator_id << ",\"? "
                    << operator_symbol(operator_id) << ' ' << right
                    << " = " << result << "\"," << left << ',' << right
                    << ',' << result << ',' << inference.value << ','
                    << inference.best_energy << ','
                    << inference.second_energy << ','
                    << inference.candidate_evaluations << '\n';
                ++generated;
            }
        }

        std::ofstream composition_csv(
            output / "composition_predictions.csv"
        );
        composition_csv
            << "depth,problem,expected_intermediates,"
               "predicted_intermediates,expected_result,predicted_result,"
               "exact,converged,sweeps,energy,message_proposals,"
               "candidate_evaluations,wall_us\n";
        std::mt19937_64 composition_random(0x5EED1234ULL);
        std::set<std::string> expression_seen;
        for (std::size_t depth : {2U, 3U}) {
            std::uint32_t generated = 0;
            while (generated < 60) {
                const auto expression = make_expression(
                    composition_random, depth
                );
                const std::string text = expression_text(expression);
                if (!expression_seen.insert(text).second) {
                    continue;
                }
                const auto inference_start = Clock::now();
                const auto inference = infer_composition(
                    library,
                    expression.first_left,
                    expression.operators,
                    expression.right_operands
                );
                const std::uint64_t wall_nanoseconds =
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - inference_start
                        ).count()
                    );
                const int expected = expression.expected_values.back();
                const int predicted = inference.values.back();
                Metrics& metrics =
                    depth == 2 ? two_step_metrics : three_step_metrics;
                metrics.add(
                    predicted,
                    expected,
                    inference.telemetry.operations.candidate_evaluations
                );
                metrics.wall_nanoseconds += wall_nanoseconds;
                composition_csv << depth << ",\"" << text << "\",\"";
                for (std::size_t i = 0;
                     i < expression.expected_values.size();
                     ++i) {
                    if (i != 0) {
                        composition_csv << ' ';
                    }
                    composition_csv << expression.expected_values[i];
                }
                composition_csv << "\",\"";
                for (std::size_t i = 0;
                     i < inference.values.size();
                     ++i) {
                    if (i != 0) {
                        composition_csv << ' ';
                    }
                    composition_csv << inference.values[i];
                }
                composition_csv << "\"," << expected << ',' << predicted
                    << ',' << (expected == predicted ? 1 : 0) << ','
                    << (inference.telemetry.converged ? 1 : 0) << ','
                    << inference.telemetry.sweeps << ','
                    << inference.telemetry.final_energy << ','
                    << inference.telemetry.operations.message_proposals
                    << ','
                    << inference.telemetry.operations.candidate_evaluations
                    << ','
                    << static_cast<double>(wall_nanoseconds) / 1000.0
                    << '\n';
                ++generated;
            }
        }

        std::vector<sera::ArithmeticExample> shuffled_training = training;
        std::mt19937_64 shuffle_random(0xBAD5EEDULL);
        std::shuffle(
            shuffled_training.begin(), shuffled_training.end(),
            shuffle_random
        );
        sera::ArithmeticRelationLibrary reordered(training_options);
        reordered.train(shuffled_training);
        double maximum_order_prediction_drift = 0.0;
        for (const auto& example : heldout) {
            maximum_order_prediction_drift = std::max(
                maximum_order_prediction_drift,
                std::abs(
                    library.predict(
                        example.operator_id, example.left, example.right
                    )
                    - reordered.predict(
                        example.operator_id, example.left, example.right
                    )
                )
            );
        }

        std::vector<sera::ArithmeticExample> permuted_labels = training;
        for (std::uint32_t operator_id = 0; operator_id < 3; ++operator_id) {
            std::vector<double> labels;
            std::vector<std::size_t> indices;
            for (std::size_t index = 0;
                 index < permuted_labels.size();
                 ++index) {
                if (permuted_labels[index].operator_id == operator_id) {
                    labels.push_back(permuted_labels[index].result);
                    indices.push_back(index);
                }
            }
            std::shuffle(labels.begin(), labels.end(), shuffle_random);
            for (std::size_t i = 0; i < indices.size(); ++i) {
                permuted_labels[indices[i]].result = labels[i];
            }
        }
        sera::ArithmeticRelationLibrary negative_control(training_options);
        negative_control.train(permuted_labels);
        std::uint64_t negative_exact = 0;
        for (const auto& example : heldout) {
            const int predicted = static_cast<int>(std::llround(
                negative_control.predict(
                    example.operator_id, example.left, example.right
                )
            ));
            negative_exact += predicted == example.result ? 1U : 0U;
        }
        const double negative_control_accuracy =
            static_cast<double>(negative_exact)
            / static_cast<double>(heldout.size());

        struct Sample {
            std::string problem;
            std::string kind;
            int predicted{};
            int expected{};
        };
        std::vector<Sample> samples;
        const auto add_single_sample = [&](std::uint32_t operator_id,
                                           int left, int right,
                                           std::string kind) {
            const auto result = infer_result(
                library, operator_id, left, right,
                kSingleDomainMinimum, kSingleDomainMaximum
            );
            samples.push_back({
                std::to_string(left) + " "
                    + std::string(operator_symbol(operator_id)) + " "
                    + std::to_string(right),
                std::move(kind),
                result.value,
                static_cast<int>(oracle(operator_id, left, right))
            });
        };
        add_single_sample(0, -7, 4, "heldout_single");
        add_single_sample(1, 9, 14, "extrapolated_single");
        add_single_sample(2, -8, 7, "heldout_single");
        {
            const std::uint32_t operators[] = {0, 2};
            const int rights[] = {7, 4};
            const auto result = infer_composition(
                library, 3, operators, rights
            );
            const int expected = static_cast<int>(
                oracle(2, oracle(0, 3, 7), 4)
            );
            samples.push_back({
                "(3 + 7) * 4",
                "two_step_shape",
                result.values.back(),
                expected
            });
        }
        {
            const std::uint32_t operators[] = {1, 2, 0};
            const int rights[] = {3, 4, 7};
            const auto result = infer_composition(
                library, 8, operators, rights
            );
            const int expected = static_cast<int>(
                oracle(0, oracle(2, oracle(1, 8, 3), 4), 7)
            );
            samples.push_back({
                "((8 - 3) * 4) + 7",
                "three_step_shape",
                result.values.back(),
                expected
            });
        }
        {
            const int expected_left = -6;
            const int fixed_right = -4;
            const int observed_result = static_cast<int>(
                oracle(2, expected_left, fixed_right)
            );
            const auto result = infer_missing_left(
                library, 2, fixed_right, observed_result, -16, 16
            );
            samples.push_back({
                "? * " + std::to_string(fixed_right)
                    + " = " + std::to_string(observed_result),
                "inverse_shape",
                result.value,
                expected_left
            });
        }
        std::ofstream samples_json(output / "sample_outputs.json");
        samples_json << "{\n  \"outputs\": [\n";
        for (std::size_t index = 0; index < samples.size(); ++index) {
            samples_json << "    {\"problem\": \""
                << samples[index].problem << "\", \"kind\": \""
                << samples[index].kind << "\", \"predicted\": "
                << samples[index].predicted << ", \"expected\": "
                << samples[index].expected << ", \"correct\": "
                << (
                    samples[index].predicted == samples[index].expected
                        ? "true" : "false"
                ) << "}";
            if (index + 1 != samples.size()) {
                samples_json << ',';
            }
            samples_json << '\n';
        }
        samples_json << "  ]\n}\n";

        std::ofstream summary(output / "summary.json");
        summary << std::setprecision(17)
            << "{\n"
            << "  \"training_examples\": " << training.size() << ",\n"
            << "  \"heldout_examples\": " << heldout.size() << ",\n"
            << "  \"train_test_overlap\": " << overlap.size() << ",\n"
            << "  \"training_wall_us\": "
            << static_cast<double>(training_nanoseconds) / 1000.0 << ",\n"
            << "  \"maximum_training_order_prediction_drift\": "
            << maximum_order_prediction_drift << ",\n"
            << "  \"permuted_label_negative_control_accuracy\": "
            << negative_control_accuracy << ",\n"
            << "  \"tests\": {\n";
        write_metrics_json(
            summary, "heldout_interpolation", heldout_metrics, true
        );
        write_metrics_json(
            summary, "unseen_extrapolation", extrapolation_metrics, true
        );
        write_metrics_json(
            summary, "inverse_unknown_operand", inverse_metrics, true
        );
        write_metrics_json(
            summary, "two_step_composition", two_step_metrics, true
        );
        write_metrics_json(
            summary, "three_step_composition", three_step_metrics, false
        );
        summary << "  },\n  \"heldout_by_operator\": {\n";
        for (std::uint32_t operator_id = 0;
             operator_id < 3;
             ++operator_id) {
            const auto& metrics = heldout_by_operator[operator_id];
            summary << "    \"" << operator_symbol(operator_id)
                << "\": {\"cases\": " << metrics.cases
                << ", \"exact\": " << metrics.exact
                << ", \"accuracy\": " << metrics.accuracy() << "}";
            if (operator_id != 2) {
                summary << ',';
            }
            summary << '\n';
        }
        summary << "  }\n}\n";

        const bool success =
            overlap.empty()
            && heldout_metrics.accuracy() >= 0.99
            && extrapolation_metrics.accuracy() >= 0.99
            && inverse_metrics.accuracy() >= 0.99
            && two_step_metrics.accuracy() >= 0.99
            && three_step_metrics.accuracy() >= 0.99
            && negative_control_accuracy < 0.25
            && maximum_order_prediction_drift < 1.0e-8;
        std::cout
            << "Arithmetic research complete\n"
            << "training=" << training.size()
            << " heldout=" << heldout.size()
            << " overlap=" << overlap.size() << '\n'
            << "heldout_accuracy=" << heldout_metrics.accuracy()
            << " extrapolation_accuracy="
            << extrapolation_metrics.accuracy() << '\n'
            << "inverse_accuracy=" << inverse_metrics.accuracy()
            << " two_step_accuracy=" << two_step_metrics.accuracy()
            << " three_step_accuracy=" << three_step_metrics.accuracy()
            << '\n'
            << "negative_control_accuracy="
            << negative_control_accuracy << '\n';
        return success ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

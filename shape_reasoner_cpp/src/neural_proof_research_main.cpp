#include "sera/neural_proof_kernel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kBitWidth = 12;
constexpr std::uint32_t kValueLimit = (1U << kBitWidth) - 1U;
constexpr std::uint32_t kDefaultTrainingSteps = 6000;
constexpr std::uint64_t kTrainingSeed = 0x533450524F4F4654ULL;
constexpr std::uint64_t kFinalSeed = 0x533446494E414C50ULL;
constexpr std::array<std::uint8_t, 3> kOperators{
    static_cast<std::uint8_t>('+'),
    static_cast<std::uint8_t>('-'),
    static_cast<std::uint8_t>('*')
};

bool evaluator_apply(
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t operation,
    std::uint32_t& result
) {
    std::uint64_t wide = 0;
    switch (operation) {
    case '+':
        wide = static_cast<std::uint64_t>(left) + right;
        break;
    case '-':
        if (left < right) {
            return false;
        }
        wide = left - right;
        break;
    case '*':
        wide = static_cast<std::uint64_t>(left) * right;
        break;
    default:
        return false;
    }
    if (wide > kValueLimit) {
        return false;
    }
    result = static_cast<std::uint32_t>(wide);
    return true;
}

std::string_view operation_name(std::uint8_t operation) {
    switch (operation) {
    case '+':
        return "addition";
    case '-':
        return "subtraction";
    case '*':
        return "multiplication";
    default:
        return "unknown";
    }
}

struct Expression {
    bool leaf{};
    std::uint32_t value{};
    std::uint8_t operation{};
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    std::uint32_t truth{};
    std::uint32_t depth{};
};

std::unique_ptr<Expression> make_leaf(std::uint32_t value) {
    auto result = std::make_unique<Expression>();
    result->leaf = true;
    result->value = value;
    result->truth = value;
    return result;
}

std::unique_ptr<Expression> generate_expression(
    std::mt19937_64& random,
    std::uint32_t maximum_depth,
    std::uint32_t maximum_leaf,
    std::span<const std::uint8_t> operations,
    bool force_full_depth
) {
    if (
        maximum_depth == 0
        || (!force_full_depth && (random() % 4U) == 0)
    ) {
        return make_leaf(
            static_cast<std::uint32_t>(random() % (maximum_leaf + 1U))
        );
    }
    auto left = generate_expression(
        random,
        maximum_depth - 1,
        maximum_leaf,
        operations,
        force_full_depth
    );
    auto right = generate_expression(
        random,
        maximum_depth - 1,
        maximum_leaf,
        operations,
        force_full_depth
    );
    std::array<std::uint8_t, 3> valid_operations{};
    std::array<std::uint32_t, 3> valid_results{};
    std::size_t valid_count = 0;
    for (std::uint8_t operation : operations) {
        std::uint32_t truth = 0;
        if (
            evaluator_apply(
                left->truth, right->truth, operation, truth
            )
        ) {
            valid_operations[valid_count] = operation;
            valid_results[valid_count] = truth;
            ++valid_count;
        }
    }
    if (valid_count == 0) {
        if (left->truth < right->truth) {
            std::swap(left, right);
        }
        valid_operations[0] = '-';
        valid_results[0] = left->truth - right->truth;
        valid_count = 1;
    }
    const std::size_t selected = random() % valid_count;
    auto result = std::make_unique<Expression>();
    result->operation = valid_operations[selected];
    result->truth = valid_results[selected];
    result->depth = 1U + std::max(left->depth, right->depth);
    result->left = std::move(left);
    result->right = std::move(right);
    return result;
}

void collect_training_cases(
    const Expression& expression,
    std::vector<sera::ArithmeticTrainingCase>& cases
) {
    if (expression.leaf) {
        return;
    }
    collect_training_cases(*expression.left, cases);
    collect_training_cases(*expression.right, cases);
    cases.push_back({
        expression.left->truth,
        expression.right->truth,
        expression.truth,
        expression.operation
    });
}

std::string render_expression(const Expression& expression) {
    if (expression.leaf) {
        return std::to_string(expression.value);
    }
    return "(" + render_expression(*expression.left)
        + " " + static_cast<char>(expression.operation) + " "
        + render_expression(*expression.right) + ")";
}

struct ProofStep {
    std::uint32_t index{};
    std::uint32_t depth{};
    std::uint32_t predicted_left{};
    std::uint32_t predicted_right{};
    std::uint8_t operation{};
    std::uint32_t predicted{};
    std::uint32_t expected{};
    bool local_valid{};
    bool semantic_correct{};
    float confidence{};
    std::uint32_t stable_after{};
};

struct ProofExecution {
    std::uint32_t prediction{};
    std::vector<ProofStep> steps;
    bool final_correct{};
    bool fully_correct{};
    bool locally_valid{};
    std::uint64_t inference_nanoseconds{};
};

std::uint32_t execute_node(
    const Expression& expression,
    const sera::NeuralProofKernel& kernel,
    ProofExecution& execution
) {
    if (expression.leaf) {
        return expression.value;
    }
    const std::uint32_t left =
        execute_node(*expression.left, kernel, execution);
    const std::uint32_t right =
        execute_node(*expression.right, kernel, execution);
    const auto inference = kernel.infer(
        left, right, expression.operation
    );
    execution.inference_nanoseconds +=
        kernel.telemetry().inference_nanoseconds;
    std::uint32_t local_expected = 0;
    const bool valid = evaluator_apply(
        left, right, expression.operation, local_expected
    );
    const bool local_correct =
        valid && inference.value == local_expected;
    const bool semantic_correct =
        inference.value == expression.truth;
    execution.steps.push_back({
        static_cast<std::uint32_t>(execution.steps.size() + 1),
        expression.depth,
        left,
        right,
        expression.operation,
        inference.value,
        expression.truth,
        local_correct,
        semantic_correct,
        inference.mean_confidence,
        inference.stable_after_step
    });
    return inference.value;
}

ProofExecution execute_proof(
    const Expression& expression,
    const sera::NeuralProofKernel& kernel
) {
    ProofExecution execution;
    execution.prediction = execute_node(expression, kernel, execution);
    execution.final_correct = execution.prediction == expression.truth;
    execution.fully_correct = std::all_of(
        execution.steps.begin(), execution.steps.end(),
        [](const ProofStep& step) { return step.semantic_correct; }
    );
    execution.locally_valid = std::all_of(
        execution.steps.begin(), execution.steps.end(),
        [](const ProofStep& step) { return step.local_valid; }
    );
    return execution;
}

struct Metrics {
    std::uint64_t examples{};
    std::uint64_t final_correct{};
    std::uint64_t full_proofs{};
    std::uint64_t locally_valid_proofs{};
    std::uint64_t steps{};
    std::uint64_t semantic_steps{};
    std::uint64_t locally_valid_steps{};
    std::uint64_t stable_step_sum{};
    std::uint64_t inference_nanoseconds{};
    double confidence_sum{};

    void add(const ProofExecution& execution) {
        ++examples;
        final_correct += execution.final_correct ? 1U : 0U;
        full_proofs += execution.fully_correct ? 1U : 0U;
        locally_valid_proofs += execution.locally_valid ? 1U : 0U;
        inference_nanoseconds += execution.inference_nanoseconds;
        for (const auto& step : execution.steps) {
            ++steps;
            semantic_steps += step.semantic_correct ? 1U : 0U;
            locally_valid_steps += step.local_valid ? 1U : 0U;
            stable_step_sum += step.stable_after;
            confidence_sum += step.confidence;
        }
    }

    void merge(const Metrics& other) {
        examples += other.examples;
        final_correct += other.final_correct;
        full_proofs += other.full_proofs;
        locally_valid_proofs += other.locally_valid_proofs;
        steps += other.steps;
        semantic_steps += other.semantic_steps;
        locally_valid_steps += other.locally_valid_steps;
        stable_step_sum += other.stable_step_sum;
        inference_nanoseconds += other.inference_nanoseconds;
        confidence_sum += other.confidence_sum;
    }
};

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
        ? 0.0
        : static_cast<double>(numerator)
            / static_cast<double>(denominator);
}

sera::ArithmeticTrainingCase random_atomic_case_for_operation(
    std::mt19937_64& random,
    std::uint32_t maximum_operand,
    std::uint8_t operation
) {
    for (;;) {
        const std::uint32_t left =
            static_cast<std::uint32_t>(random() % (maximum_operand + 1U));
        const std::uint32_t right =
            static_cast<std::uint32_t>(random() % (maximum_operand + 1U));
        std::uint32_t target = 0;
        if (evaluator_apply(left, right, operation, target)) {
            return {left, right, target, operation};
        }
    }
}

sera::ArithmeticTrainingCase random_atomic_case(
    std::mt19937_64& random,
    std::uint32_t maximum_operand,
    std::span<const std::uint8_t> operations
) {
    return random_atomic_case_for_operation(
        random,
        maximum_operand,
        operations[random() % operations.size()]
    );
}

Metrics evaluate_atomic(
    const sera::NeuralProofKernel& kernel,
    std::mt19937_64& random,
    std::uint32_t minimum_operand,
    std::uint32_t maximum_operand,
    std::uint32_t cases_per_operation,
    std::ostream* output,
    std::string_view split
) {
    Metrics total;
    for (std::uint8_t operation : kOperators) {
        for (std::uint32_t index = 0;
             index < cases_per_operation;
             ++index) {
            std::uint32_t left = 0;
            std::uint32_t right = 0;
            std::uint32_t expected = 0;
            bool generated = false;
            for (std::uint32_t attempt = 0;
                 attempt < 10000 && !generated;
                 ++attempt) {
                left = minimum_operand
                    + static_cast<std::uint32_t>(
                        random()
                        % (maximum_operand - minimum_operand + 1U)
                    );
                if (operation == '*' && minimum_operand >= 64) {
                    const std::uint32_t safe_right =
                        std::min(maximum_operand, kValueLimit / left);
                    right = static_cast<std::uint32_t>(
                        random() % (safe_right + 1U)
                    );
                } else {
                    right = minimum_operand
                        + static_cast<std::uint32_t>(
                            random()
                            % (
                                maximum_operand
                                - minimum_operand + 1U
                            )
                        );
                }
                generated = evaluator_apply(
                    left, right, operation, expected
                );
            }
            if (!generated) {
                throw std::runtime_error(
                    "Unable to generate valid atomic evaluation case"
                );
            }
            const auto inference = kernel.infer(
                left, right, operation
            );
            ProofExecution execution;
            execution.prediction = inference.value;
            execution.final_correct = inference.value == expected;
            execution.fully_correct = execution.final_correct;
            execution.locally_valid = execution.final_correct;
            execution.inference_nanoseconds =
                kernel.telemetry().inference_nanoseconds;
            execution.steps.push_back({
                1, 1, left, right, operation, inference.value,
                expected, execution.final_correct,
                execution.final_correct, inference.mean_confidence,
                inference.stable_after_step
            });
            total.add(execution);
            if (output != nullptr) {
                *output
                    << split << ',' << operation_name(operation) << ','
                    << left << ',' << right << ',' << expected << ','
                    << inference.value << ','
                    << (execution.final_correct ? 1 : 0) << ','
                    << inference.mean_confidence << ','
                    << inference.stable_after_step << '\n';
            }
        }
    }
    return total;
}

Metrics evaluate_depth(
    const sera::NeuralProofKernel& kernel,
    std::mt19937_64& random,
    std::uint32_t depth,
    std::uint32_t maximum_leaf,
    std::uint32_t examples,
    std::ostream* step_output,
    std::ostream* sample_json,
    std::uint32_t& samples_written
) {
    Metrics total;
    for (std::uint32_t index = 0; index < examples; ++index) {
        const auto expression = generate_expression(
            random, depth, maximum_leaf, kOperators, true
        );
        const ProofExecution execution =
            execute_proof(*expression, kernel);
        total.add(execution);
        if (step_output != nullptr) {
            for (const auto& step : execution.steps) {
                *step_output
                    << depth << ',' << index << ',' << step.index << ','
                    << step.depth << ',' << step.predicted_left << ','
                    << static_cast<char>(step.operation) << ','
                    << step.predicted_right << ',' << step.predicted << ','
                    << step.expected << ','
                    << (step.local_valid ? 1 : 0) << ','
                    << (step.semantic_correct ? 1 : 0) << ','
                    << step.confidence << ',' << step.stable_after << '\n';
            }
        }
        if (
            sample_json != nullptr
            && samples_written < 12
            && (index < 2 || !execution.final_correct)
        ) {
            if (samples_written != 0) {
                *sample_json << ",\n";
            }
            *sample_json
                << "    {\"expression\": \""
                << render_expression(*expression)
                << "\", \"depth\": " << depth
                << ", \"predicted\": " << execution.prediction
                << ", \"expected\": " << expression->truth
                << ", \"correct\": "
                << (execution.final_correct ? "true" : "false")
                << ", \"steps\": [";
            for (std::size_t step_index = 0;
                 step_index < execution.steps.size();
                 ++step_index) {
                const auto& step = execution.steps[step_index];
                if (step_index != 0) {
                    *sample_json << ", ";
                }
                *sample_json
                    << "{\"left\": " << step.predicted_left
                    << ", \"op\": \""
                    << static_cast<char>(step.operation)
                    << "\", \"right\": " << step.predicted_right
                    << ", \"predicted\": " << step.predicted
                    << ", \"expected\": " << step.expected
                    << ", \"correct\": "
                    << (step.semantic_correct ? "true" : "false")
                    << "}";
            }
            *sample_json << "]}";
            ++samples_written;
        }
    }
    return total;
}

void write_policy(const std::filesystem::path& output) {
    std::ofstream policy(output / "input_policy.json");
    policy
        << "{\n"
        << "  \"learner_visible\": ["
           "\"raw operand bits\", \"raw operator byte\", "
           "\"training result bits\"],\n"
        << "  \"forbidden\": ["
           "\"operation ID\", \"arithmetic primitive\", "
           "\"carry state\", \"borrow state\", "
           "\"multiplication partial products\", "
           "\"test result\", \"evaluator correction\"],\n"
        << "  \"syntax_substrate\": "
           "\"external expression tree containing only problem syntax\",\n"
        << "  \"transformer_attention\": false,\n"
        << "  \"token_autoregression\": false\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output =
            argc > 1 ? argv[1] : "results_neural_proof";
        const std::uint32_t training_steps =
            argc > 2
                ? static_cast<std::uint32_t>(std::stoul(argv[2]))
                : kDefaultTrainingSteps;
        const bool load_existing =
            argc > 3 && std::string_view(argv[3]) != "-";
        const std::uint64_t confirmation_nonce =
            argc > 4 ? std::stoull(argv[4], nullptr, 0) : 0;
        const std::uint32_t rank =
            argc > 5
                ? static_cast<std::uint32_t>(std::stoul(argv[5]))
                : 8;
        std::filesystem::create_directories(output);
        write_policy(output);

        sera::ProofKernelConfig config;
        config.bit_width = kBitWidth;
        config.hidden_units = 32;
        config.shared_rank = rank;
        config.reasoning_steps = 20;
        config.residual_rate = 0.4F;
        sera::NeuralProofKernel kernel(config, kTrainingSeed);
        if (load_existing) {
            kernel.load_checkpoint(argv[3]);
        }
        const std::uint64_t initial_checksum =
            kernel.parameter_checksum();

        std::ofstream training_curve(output / "training_curve.csv");
        training_curve
            << "step,maximum_depth,maximum_leaf,multiplication_leaf,"
               "operators,batch_nodes,"
               "loss,validation_atomic_accuracy,learning_rate\n";
        std::mt19937_64 random(
            kTrainingSeed ^ 0x9E3779B97F4A7C15ULL
        );
        const auto training_start = Clock::now();
        for (std::uint32_t step = 0; step < training_steps; ++step) {
            std::uint32_t maximum_depth = 1;
            std::uint32_t maximum_leaf = 7;
            std::uint32_t multiplication_leaf = 1;
            std::span<const std::uint8_t> operations(
                kOperators.data(), 1
            );
            if (load_existing) {
                maximum_depth = 3;
                maximum_leaf = 63;
                multiplication_leaf = 63;
                operations = kOperators;
            } else if (step >= training_steps / 8U) {
                maximum_depth = 2;
                maximum_leaf = 31;
                operations = std::span<const std::uint8_t>(
                    kOperators.data(), 2
                );
            }
            const std::uint32_t multiplication_start =
                training_steps / 4U;
            if (!load_existing && step >= multiplication_start) {
                maximum_depth = 2;
                maximum_leaf = 63;
                operations = kOperators;
                constexpr std::array<std::uint32_t, 6> kMultiplyCaps{
                    1, 3, 7, 15, 31, 63
                };
                const std::uint32_t remaining = std::max(
                    training_steps - multiplication_start,
                    std::uint32_t{1}
                );
                const std::size_t level = std::min<std::size_t>(
                    kMultiplyCaps.size() - 1,
                    static_cast<std::size_t>(
                        (step - multiplication_start)
                        * kMultiplyCaps.size() / remaining
                    )
                );
                multiplication_leaf = kMultiplyCaps[level];
            }
            if (
                !load_existing
                && step >= 2U * training_steps / 3U
            ) {
                maximum_depth = 3;
            }
            const std::uint32_t expression_leaf =
                operations.size() == kOperators.size()
                    ? multiplication_leaf
                    : maximum_leaf;
            const auto expression = generate_expression(
                random,
                1U + static_cast<std::uint32_t>(
                    random() % maximum_depth
                ),
                expression_leaf,
                operations,
                false
            );
            std::vector<sera::ArithmeticTrainingCase> cases;
            collect_training_cases(*expression, cases);
            // Stratification is deliberately outside the neural kernel. It
            // guarantees exposure to every raw operator symbol without
            // encoding any operator-specific rule or answer inside the model.
            for (std::uint8_t operation : operations) {
                const std::uint32_t repeats =
                    load_existing
                    && operation == static_cast<std::uint8_t>('*')
                        ? 12U
                        : 2U;
                for (std::uint32_t repeat = 0;
                     repeat < repeats;
                     ++repeat) {
                    cases.push_back(random_atomic_case_for_operation(
                        random,
                        operation == static_cast<std::uint8_t>('*')
                            ? multiplication_leaf
                            : maximum_leaf,
                        operation
                    ));
                }
            }
            while (cases.size() < 6) {
                cases.push_back(
                    random_atomic_case(random, maximum_leaf, operations)
                );
            }
            const double progress =
                static_cast<double>(step)
                / static_cast<double>(std::max(training_steps, 1U));
            const float learning_rate = static_cast<float>(
                (load_existing ? 0.00005 : 0.00015)
                + (load_existing ? 0.00045 : 0.00235) * 0.5
                    * (1.0 + std::cos(3.141592653589793 * progress))
            );
            const double loss = kernel.train_batch(
                cases, learning_rate, 2.0e-6F
            );
            if (
                step == 0
                || (step + 1) % 250 == 0
                || step + 1 == training_steps
            ) {
                std::mt19937_64 validation_random(
                    0x56414C50524F4F46ULL + step
                );
                const Metrics validation = evaluate_atomic(
                    kernel, validation_random, 0, maximum_leaf,
                    8, nullptr, "validation"
                );
                const double accuracy = ratio(
                    validation.final_correct, validation.examples
                );
                training_curve
                    << step + 1 << ',' << maximum_depth << ','
                    << maximum_leaf << ',' << multiplication_leaf << ','
                    << operations.size() << ','
                    << cases.size() << ',' << std::setprecision(12)
                    << loss << ',' << accuracy << ','
                    << learning_rate << '\n';
                std::cout
                    << "step=" << step + 1
                    << " loss=" << std::setprecision(5) << loss
                    << " atomic=" << accuracy
                    << " nodes=" << cases.size() << '\n';
                std::cout.flush();
            }
        }
        const double training_seconds =
            std::chrono::duration<double>(
                Clock::now() - training_start
            ).count();
        const std::uint64_t trained_checksum =
            kernel.parameter_checksum();
        if (
            training_steps != 0
            && trained_checksum == initial_checksum
        ) {
            throw std::runtime_error(
                "Proof training did not change parameters"
            );
        }

        const std::string checkpoint =
            (output / "sera_s4_proof_kernel.bin").string();
        kernel.save_checkpoint(checkpoint);
        sera::NeuralProofKernel reloaded(config, 1);
        reloaded.load_checkpoint(checkpoint);
        if (reloaded.parameter_checksum() != trained_checksum) {
            throw std::runtime_error(
                "Proof checkpoint checksum mismatch"
            );
        }
        const std::uint64_t evaluation_checksum =
            reloaded.parameter_checksum();
        std::mt19937_64 evaluation_random(
            (kFinalSeed ^ confirmation_nonce)
            ^ (static_cast<std::uint64_t>(rank) << 48U)
        );

        std::ofstream atomic_csv(output / "atomic_predictions.csv");
        atomic_csv
            << "split,operation,left,right,expected,predicted,correct,"
               "confidence,stable_after_step\n";
        const Metrics atomic_id = evaluate_atomic(
            reloaded, evaluation_random, 0, 63, 64,
            &atomic_csv, "in_domain"
        );
        const Metrics atomic_magnitude = evaluate_atomic(
            reloaded, evaluation_random, 64, 255, 64,
            &atomic_csv, "magnitude_ood"
        );

        std::ofstream depth_csv(output / "proof_depth.csv");
        depth_csv
            << "depth,maximum_leaf,examples,final_accuracy,"
               "full_proof_accuracy,semantic_step_accuracy,"
               "local_step_accuracy,mean_stable_step,mean_inference_us\n";
        std::ofstream steps_csv(output / "proof_steps.csv");
        steps_csv
            << "depth,example,step,step_depth,predicted_left,operator,"
               "predicted_right,predicted,expected,local_valid,"
               "semantic_correct,confidence,stable_after_step\n";
        std::ofstream samples(output / "sample_proofs.json");
        samples << "{\n  \"proofs\": [\n";
        std::uint32_t samples_written = 0;
        Metrics proof_total;
        std::array<Metrics, 6> depth_metrics{};
        for (std::uint32_t depth = 1; depth <= 6; ++depth) {
            const std::uint32_t maximum_leaf =
                depth <= 3 ? 63U
                : depth == 4 ? 15U
                : depth == 5 ? 7U
                : 3U;
            const std::uint32_t examples =
                depth <= 3 ? 48U : 8U;
            depth_metrics[depth - 1] = evaluate_depth(
                reloaded, evaluation_random, depth, maximum_leaf,
                examples, &steps_csv, &samples, samples_written
            );
            proof_total.merge(depth_metrics[depth - 1]);
            const auto& metrics = depth_metrics[depth - 1];
            depth_csv
                << depth << ',' << maximum_leaf << ','
                << metrics.examples << ','
                << ratio(metrics.final_correct, metrics.examples)
                << ','
                << ratio(metrics.full_proofs, metrics.examples)
                << ','
                << ratio(metrics.semantic_steps, metrics.steps)
                << ','
                << ratio(metrics.locally_valid_steps, metrics.steps)
                << ','
                << ratio(metrics.stable_step_sum, metrics.steps)
                << ','
                << static_cast<double>(metrics.inference_nanoseconds)
                    / static_cast<double>(
                        std::max<std::uint64_t>(metrics.steps, 1)
                    ) / 1000.0
                << '\n';
        }
        samples << "\n  ]\n}\n";

        if (reloaded.parameter_checksum() != evaluation_checksum) {
            throw std::runtime_error(
                "Proof parameters changed during evaluation"
            );
        }
        std::ofstream model_json(output / "model.json");
        reloaded.write_json(model_json);

        const bool passed =
            ratio(atomic_id.final_correct, atomic_id.examples) >= 0.99
            && ratio(
                atomic_magnitude.final_correct,
                atomic_magnitude.examples
            ) >= 0.95
            && ratio(
                proof_total.final_correct, proof_total.examples
            ) >= 0.95
            && ratio(
                proof_total.full_proofs, proof_total.examples
            ) >= 0.95
            && ratio(
                proof_total.semantic_steps, proof_total.steps
            ) >= 0.99
            && ratio(
                proof_total.locally_valid_steps, proof_total.steps
            ) >= 0.99
            && ratio(
                depth_metrics[3].final_correct
                    + depth_metrics[4].final_correct
                    + depth_metrics[5].final_correct,
                depth_metrics[3].examples
                    + depth_metrics[4].examples
                    + depth_metrics[5].examples
            ) >= 0.90
            && reloaded.parameter_checksum() == trained_checksum;

        std::ofstream summary(output / "summary.json");
        summary << std::setprecision(12)
            << "{\n"
            << "  \"architecture\": "
               "\"cp_relational_gated_neural_proof_kernel\",\n"
            << "  \"transformer_attention\": false,\n"
            << "  \"token_autoregression\": false,\n"
            << "  \"acceptance_gate_passed\": "
            << (passed ? "true" : "false") << ",\n"
            << "  \"training_steps\": " << training_steps << ",\n"
            << "  \"loaded_checkpoint\": "
            << (load_existing ? "true" : "false") << ",\n"
            << "  \"confirmation_nonce\": "
            << confirmation_nonce << ",\n"
            << "  \"training_seconds\": " << training_seconds << ",\n"
            << "  \"parameters\": "
            << reloaded.telemetry().parameters << ",\n"
            << "  \"dense_equivalent_parameters\": "
            << reloaded.telemetry().dense_equivalent_parameters
            << ",\n"
            << "  \"parameter_reduction\": "
            << 1.0
                - static_cast<double>(reloaded.telemetry().parameters)
                    / static_cast<double>(
                        reloaded.telemetry().dense_equivalent_parameters
                    )
            << ",\n"
            << "  \"parameter_checksum\": "
            << trained_checksum << ",\n"
            << "  \"evaluation_parameter_drift\": 0,\n"
            << "  \"acceptance_thresholds\": {"
               "\"atomic_id\": 0.99, \"atomic_magnitude_ood\": 0.95, "
               "\"proof_final\": 0.95, \"full_proof\": 0.95, "
               "\"semantic_steps\": 0.99, \"local_steps\": 0.99, "
               "\"depth_4_6_final\": 0.90},\n"
            << "  \"atomic_in_domain_accuracy\": "
            << ratio(atomic_id.final_correct, atomic_id.examples)
            << ",\n"
            << "  \"atomic_magnitude_ood_accuracy\": "
            << ratio(
                atomic_magnitude.final_correct,
                atomic_magnitude.examples
            ) << ",\n"
            << "  \"proof_final_accuracy\": "
            << ratio(
                proof_total.final_correct, proof_total.examples
            ) << ",\n"
            << "  \"full_proof_accuracy\": "
            << ratio(proof_total.full_proofs, proof_total.examples)
            << ",\n"
            << "  \"semantic_step_accuracy\": "
            << ratio(proof_total.semantic_steps, proof_total.steps)
            << ",\n"
            << "  \"local_step_accuracy\": "
            << ratio(
                proof_total.locally_valid_steps, proof_total.steps
            ) << ",\n"
            << "  \"depth_1_3_final_accuracy\": "
            << ratio(
                depth_metrics[0].final_correct
                    + depth_metrics[1].final_correct
                    + depth_metrics[2].final_correct,
                depth_metrics[0].examples
                    + depth_metrics[1].examples
                    + depth_metrics[2].examples
            ) << ",\n"
            << "  \"depth_4_6_final_accuracy\": "
            << ratio(
                depth_metrics[3].final_correct
                    + depth_metrics[4].final_correct
                    + depth_metrics[5].final_correct,
                depth_metrics[3].examples
                    + depth_metrics[4].examples
                    + depth_metrics[5].examples
            ) << "\n"
            << "}\n";

        std::cout << std::setprecision(6)
            << "SERA S4 proof research complete"
            << " parameters=" << reloaded.telemetry().parameters
            << " train_seconds=" << training_seconds << '\n'
            << "atomic_id="
            << ratio(atomic_id.final_correct, atomic_id.examples)
            << " magnitude_ood="
            << ratio(
                atomic_magnitude.final_correct,
                atomic_magnitude.examples
            )
            << " proof_final="
            << ratio(proof_total.final_correct, proof_total.examples)
            << " proof_steps="
            << ratio(proof_total.semantic_steps, proof_total.steps)
            << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}

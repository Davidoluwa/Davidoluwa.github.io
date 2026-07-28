// SERA S5 step-by-step proof evaluation.
//
// The kernel answers one node. Reasoning is what happens when the same frozen
// parameter set is applied at every internal node of an expression tree, in
// post-order, consuming its own predicted child values. Nothing is regenerated
// as text and no search is performed: the proof is the evaluation order.
//
// Two accuracies are reported per step and they are not the same thing:
//   * semantic  -- the step's value equals the true value of that subtree;
//   * local     -- the step is a consistent transformation of the (possibly
//                  wrong) child values it was actually given.
// A model can be locally flawless and semantically wrong, so reporting only
// one of them would overstate the result.
//
// Usage: sera_s5_proof <out_dir> <channels> <seed> <steps>

#include "sera/interaction_workspace_kernel.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

std::uint32_t kWidth = 12;
std::uint32_t kLimit = (1U << 12) - 1U;
constexpr std::uint8_t kOperators[3] = {'+', '-', '*'};
constexpr std::uint32_t kMaxDepth = 6;

// External label evaluator. It defines truth for training targets and for
// scoring. It is never consulted by the kernel.
bool evaluate(
    std::uint8_t operation,
    std::uint32_t left,
    std::uint32_t right,
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
    if (wide > kLimit) {
        return false;
    }
    result = static_cast<std::uint32_t>(wide);
    return true;
}

struct Expression {
    bool leaf{true};
    std::uint32_t value{};
    std::uint8_t operation{};
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    std::uint32_t truth{};
};

std::unique_ptr<Expression> generate(
    std::mt19937_64& random,
    std::uint32_t depth,
    std::uint32_t maximum_leaf
) {
    if (depth == 0) {
        auto node = std::make_unique<Expression>();
        std::uniform_int_distribution<std::uint32_t> pick(0, maximum_leaf);
        node->value = pick(random);
        node->truth = node->value;
        return node;
    }
    // Rejection sampling until the subtree is admissible under the same
    // validity convention used everywhere else.
    for (int attempt = 0; attempt < 64; ++attempt) {
        auto node = std::make_unique<Expression>();
        node->leaf = false;
        std::uniform_int_distribution<std::size_t> pick_op(0, 2);
        node->operation = kOperators[pick_op(random)];
        node->left = generate(random, depth - 1, maximum_leaf);
        node->right = generate(random, depth - 1, maximum_leaf);
        std::uint32_t truth = 0;
        if (evaluate(node->operation, node->left->truth, node->right->truth, truth)) {
            node->truth = truth;
            return node;
        }
    }
    auto node = std::make_unique<Expression>();
    std::uniform_int_distribution<std::uint32_t> pick(0, maximum_leaf);
    node->value = pick(random);
    node->truth = node->value;
    return node;
}

std::string render(const Expression& expression) {
    if (expression.leaf) {
        return std::to_string(expression.value);
    }
    return "(" + render(*expression.left) + " "
        + static_cast<char>(expression.operation) + " "
        + render(*expression.right) + ")";
}

struct ProofStep {
    std::uint32_t index{};
    std::uint32_t left{};
    std::uint32_t right{};
    std::uint8_t operation{};
    std::uint32_t predicted{};
    std::uint32_t expected{};
    bool semantic_correct{};
    bool local_valid{};
    float confidence{};
};

struct ProofExecution {
    std::vector<ProofStep> steps;
    std::uint32_t prediction{};
    bool final_correct{};
    bool full_proof_correct{};
};

std::uint32_t execute(
    const Expression& expression,
    const sera::InteractionWorkspaceKernel& kernel,
    ProofExecution& execution
) {
    if (expression.leaf) {
        return expression.value;
    }
    // Post-order: children are resolved first, and the parent consumes the
    // model's own predictions, never the true child values.
    const std::uint32_t left = execute(*expression.left, kernel, execution);
    const std::uint32_t right = execute(*expression.right, kernel, execution);
    const auto inference = kernel.infer(left, right, expression.operation);

    std::uint32_t local_truth = 0;
    const bool local_defined =
        evaluate(expression.operation, left, right, local_truth);

    ProofStep step;
    step.index = static_cast<std::uint32_t>(execution.steps.size());
    step.left = left;
    step.right = right;
    step.operation = expression.operation;
    step.predicted = inference.value;
    step.expected = expression.truth;
    step.semantic_correct = inference.value == expression.truth;
    step.local_valid = local_defined && inference.value == local_truth;
    step.confidence = inference.mean_confidence;
    execution.steps.push_back(step);
    return inference.value;
}

ProofExecution run_proof(
    const Expression& expression,
    const sera::InteractionWorkspaceKernel& kernel
) {
    ProofExecution execution;
    execution.prediction = execute(expression, kernel, execution);
    execution.final_correct = execution.prediction == expression.truth;
    execution.full_proof_correct = execution.final_correct;
    for (const auto& step : execution.steps) {
        if (!step.semantic_correct) {
            execution.full_proof_correct = false;
        }
    }
    return execution;
}

struct Metrics {
    std::size_t trees{};
    std::size_t final_correct{};
    std::size_t full_correct{};
    std::size_t steps{};
    std::size_t semantic_steps{};
    std::size_t local_steps{};

    void add(const ProofExecution& execution) {
        trees += 1;
        final_correct += execution.final_correct ? 1 : 0;
        full_correct += execution.full_proof_correct ? 1 : 0;
        for (const auto& step : execution.steps) {
            steps += 1;
            semantic_steps += step.semantic_correct ? 1 : 0;
            local_steps += step.local_valid ? 1 : 0;
        }
    }

    [[nodiscard]] double ratio(std::size_t part, std::size_t whole) const {
        return whole == 0 ? 0.0
            : static_cast<double>(part) / static_cast<double>(whole);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <out_dir> <channels> <seed> <steps> [bit_width]\n";
        return 2;
    }
    try {
        const std::filesystem::path output = argv[1];
        const auto channels = static_cast<std::uint32_t>(std::stoul(argv[2]));
        const auto seed = static_cast<std::uint64_t>(std::stoull(argv[3]));
        const auto steps = static_cast<std::uint32_t>(std::stoul(argv[4]));
        if (argc >= 6) {
            kWidth = static_cast<std::uint32_t>(std::stoul(argv[5]));
            kLimit = (1U << kWidth) - 1U;
        }
        std::filesystem::create_directories(output);

        sera::WorkspaceKernelConfig config;
        config.bit_width = kWidth;
        config.channels = channels;
        config.shared_rank = 5;
        config.reasoning_steps = 2U * kWidth;
        sera::InteractionWorkspaceKernel kernel(config, seed);

        // Training pool: every admissible atomic case. Internal proof nodes
        // are drawn from the same distribution by construction, so the kernel
        // sees node-shaped work without ever seeing a tree.
        std::vector<sera::ArithmeticTrainingCase> pools[3];
        for (std::size_t index = 0; index < 3; ++index) {
            for (std::uint32_t left = 0; left <= kLimit; ++left) {
                for (std::uint32_t right = 0; right <= kLimit; ++right) {
                    std::uint32_t target = 0;
                    if (evaluate(kOperators[index], left, right, target)) {
                        pools[index].push_back(
                            {left, right, target, kOperators[index]}
                        );
                    }
                }
            }
        }

        std::mt19937_64 random(seed ^ 0x5EED5EEDULL);
        std::vector<sera::ArithmeticTrainingCase> batch;
        std::ofstream curve(output / "training_curve.csv");
        curve << "step,loss\n";
        for (std::uint32_t step = 0; step < steps; ++step) {
            batch.clear();
            for (std::size_t item = 0; item < 24; ++item) {
                auto& pool = pools[item % 3];
                std::uniform_int_distribution<std::size_t> pick(
                    0, pool.size() - 1
                );
                batch.push_back(pool[pick(random)]);
            }
            const double progress =
                static_cast<double>(step) / static_cast<double>(steps);
            const float rate = static_cast<float>(
                6.0e-3 * (0.5 * (1.0 + std::cos(3.14159265358979 * progress)))
                    + 2.0e-4
            );
            const double loss = kernel.train_batch(batch, rate, 0.0F);
            if (step % 50 == 0 || step + 1 == steps) {
                curve << step << "," << loss << "\n";
            }
        }

        kernel.save_checkpoint((output / "sera_s5_kernel.bin").string());
        std::ofstream model(output / "model.json");
        kernel.write_json(model);

        // Parameters are frozen from here on; the drift check proves it.
        const auto checksum_before = kernel.parameter_checksum();

        std::ofstream depth_csv(output / "proof_depth.csv");
        depth_csv << "depth,trees,final_accuracy,full_proof_accuracy,"
                     "semantic_step_accuracy,local_step_accuracy\n";
        std::ofstream transcript(output / "proof_transcripts.txt");
        std::ofstream summary(output / "summary.json");
        summary.precision(12);

        Metrics overall;
        std::vector<Metrics> by_depth(kMaxDepth + 1);
        std::mt19937_64 eval_random(0xE7A11ULL ^ seed);

        for (std::uint32_t depth = 1; depth <= kMaxDepth; ++depth) {
            Metrics metrics;
            for (std::uint32_t index = 0; index < 96; ++index) {
                const auto expression = generate(eval_random, depth, kWidth == 6 ? 7U : 15U);
                const auto execution = run_proof(*expression, kernel);
                metrics.add(execution);
                overall.add(execution);

                // Human-readable derivation for the first few trees.
                if (index < 3) {
                    transcript << "depth " << depth << "  "
                               << render(*expression) << "  = "
                               << expression->truth << "\n";
                    for (const auto& step : execution.steps) {
                        transcript
                            << "  step " << step.index + 1 << ": "
                            << step.left << ' '
                            << static_cast<char>(step.operation) << ' '
                            << step.right << " = " << step.predicted
                            << (step.semantic_correct
                                    ? "   [correct]"
                                    : "   [semantically wrong, expected "
                                        + std::to_string(step.expected) + "]")
                            << (step.local_valid ? "" : " [locally invalid]")
                            << "\n";
                    }
                    transcript << "  answer " << execution.prediction
                               << (execution.final_correct ? "  OK" : "  WRONG")
                               << "\n\n";
                }
            }
            by_depth[depth] = metrics;
            depth_csv << depth << ',' << metrics.trees << ','
                      << metrics.ratio(metrics.final_correct, metrics.trees)
                      << ','
                      << metrics.ratio(metrics.full_correct, metrics.trees)
                      << ','
                      << metrics.ratio(metrics.semantic_steps, metrics.steps)
                      << ','
                      << metrics.ratio(metrics.local_steps, metrics.steps)
                      << '\n';
            std::cout << "depth " << depth
                      << " final " << metrics.ratio(metrics.final_correct, metrics.trees)
                      << " semantic " << metrics.ratio(metrics.semantic_steps, metrics.steps)
                      << " local " << metrics.ratio(metrics.local_steps, metrics.steps)
                      << std::endl;
        }

        const auto checksum_after = kernel.parameter_checksum();

        summary << "{\n";
        summary << "  \"architecture\": "
                << "\"shift_equivariant_interaction_workspace\",\n";
        summary << "  \"bit_width\": " << kWidth << ",\n";
        summary << "  \"channels\": " << channels << ",\n";
        summary << "  \"seed\": " << seed << ",\n";
        summary << "  \"training_steps\": " << steps << ",\n";
        summary << "  \"parameters\": " << kernel.telemetry().parameters
                << ",\n";
        summary << "  \"cp_equivalent_relation_parameters\": "
                << kernel.telemetry().cp_equivalent_parameters << ",\n";
        summary << "  \"evaluation_parameter_drift\": "
                << (checksum_before == checksum_after ? 0 : 1) << ",\n";
        summary << "  \"proof_final_accuracy\": "
                << overall.ratio(overall.final_correct, overall.trees) << ",\n";
        summary << "  \"full_proof_accuracy\": "
                << overall.ratio(overall.full_correct, overall.trees) << ",\n";
        summary << "  \"semantic_step_accuracy\": "
                << overall.ratio(overall.semantic_steps, overall.steps) << ",\n";
        summary << "  \"local_step_accuracy\": "
                << overall.ratio(overall.local_steps, overall.steps) << ",\n";
        summary << "  \"depth_4_6_final_accuracy\": ";
        {
            std::size_t part = 0;
            std::size_t whole = 0;
            for (std::uint32_t depth = 4; depth <= kMaxDepth; ++depth) {
                part += by_depth[depth].final_correct;
                whole += by_depth[depth].trees;
            }
            summary << (whole == 0 ? 0.0
                        : static_cast<double>(part) / static_cast<double>(whole))
                    << "\n";
        }
        summary << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "s5 proof run failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

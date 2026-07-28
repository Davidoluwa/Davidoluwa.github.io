// SERA S5 controlled comparison.
//
// Tests the rank-obstruction prediction directly. The S4 kernel reaches its
// operands only through a rank-R CP contraction, which is a rank-R
// factorization of a tensor T[p][i][j]. Multiplication needs
// T[p][i][j] = 1[i + j == p], whose CP rank is 2W - 1. The prediction is that
// S4 multiplication accuracy has a knee at shared_rank = 2W - 1 and cannot be
// solved below it, while S5 -- which materializes the exact bilinear term on a
// 2-D lattice and routes it with a tied, translation-equivariant stencil --
// solves it with parameters that do not grow with W at all.
//
// Usage:
//   sera_s5_comparison <out_dir> <arch: s4|s5> <capacity> <seed> <steps>
// where <capacity> is shared_rank for s4 and channels for s5.
//
// Evaluation is exhaustive: every operand pair admissible under the shared
// validity convention (result must fit in W bits; subtraction must not go
// negative) is tested. No sampling, no held-out estimate.

#include "sera/interaction_workspace_kernel.hpp"
#include "sera/neural_proof_kernel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 6;
constexpr std::uint32_t kLimit = (1U << kWidth) - 1U;
constexpr std::uint8_t kOperators[3] = {'+', '-', '*'};

// The single source of arithmetic truth. It labels data and scores
// predictions; it is never visible to either model.
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

// Every admissible case for one operator, in a fixed order.
std::vector<sera::ArithmeticTrainingCase> exhaustive_cases(
    std::uint8_t operation
) {
    std::vector<sera::ArithmeticTrainingCase> cases;
    for (std::uint32_t left = 0; left <= kLimit; ++left) {
        for (std::uint32_t right = 0; right <= kLimit; ++right) {
            std::uint32_t target = 0;
            if (evaluate(operation, left, right, target)) {
                cases.push_back({left, right, target, operation});
            }
        }
    }
    return cases;
}

// Operator-stratified sampling: each batch draws equally from the three
// operator pools, so multiplication is never starved by the fact that fewer
// of its operand pairs are admissible.
class BatchSampler {
public:
    explicit BatchSampler(std::uint64_t seed) : engine_(seed) {
        for (std::size_t index = 0; index < 3; ++index) {
            pools_[index] = exhaustive_cases(kOperators[index]);
        }
    }

    void fill(std::vector<sera::ArithmeticTrainingCase>& batch, std::size_t size) {
        batch.clear();
        for (std::size_t item = 0; item < size; ++item) {
            const std::size_t pool = item % 3;
            std::uniform_int_distribution<std::size_t> pick(
                0, pools_[pool].size() - 1
            );
            batch.push_back(pools_[pool][pick(engine_)]);
        }
    }

private:
    std::mt19937_64 engine_;
    std::vector<sera::ArithmeticTrainingCase> pools_[3];
};

struct OperatorScore {
    std::size_t total{};
    std::size_t correct{};
    // Cases where both operands are at least 2. Products by 0 or 1 are
    // solvable by copying an operand and would otherwise inflate the score.
    std::size_t nontrivial_total{};
    std::size_t nontrivial_correct{};

    [[nodiscard]] double nontrivial_accuracy() const {
        return nontrivial_total == 0 ? 0.0
            : static_cast<double>(nontrivial_correct)
                / static_cast<double>(nontrivial_total);
    }

    [[nodiscard]] double accuracy() const {
        return total == 0 ? 0.0 : static_cast<double>(correct)
            / static_cast<double>(total);
    }
};

template <typename Kernel>
std::vector<OperatorScore> evaluate_exhaustively(const Kernel& kernel) {
    std::vector<OperatorScore> scores(3);
    for (std::size_t index = 0; index < 3; ++index) {
        for (const auto& example : exhaustive_cases(kOperators[index])) {
            const auto inference = kernel.infer(
                example.left, example.right, example.raw_operator
            );
            const bool nontrivial = example.left >= 2 && example.right >= 2;
            scores[index].total += 1;
            scores[index].nontrivial_total += nontrivial ? 1 : 0;
            if (inference.value == example.target) {
                scores[index].correct += 1;
                scores[index].nontrivial_correct += nontrivial ? 1 : 0;
            }
        }
    }
    return scores;
}

template <typename Kernel>
void run(
    Kernel& kernel,
    const std::string& label,
    std::uint64_t parameters,
    const std::string& out_dir,
    std::uint64_t seed,
    std::uint32_t steps
) {
    BatchSampler sampler(seed ^ 0xA5A5A5A5ULL);
    std::vector<sera::ArithmeticTrainingCase> batch;
    std::ofstream curve(out_dir + "/training_curve.csv");
    curve << "step,loss\n";

    for (std::uint32_t step = 0; step < steps; ++step) {
        sampler.fill(batch, 24);
        // Cosine decay from 6e-3; no schedule tuning per architecture.
        const double progress =
            static_cast<double>(step) / static_cast<double>(steps);
        const float rate = static_cast<float>(
            6.0e-3 * (0.5 * (1.0 + std::cos(3.14159265358979 * progress)))
                + 2.0e-4
        );
        const double loss = kernel.train_batch(batch, rate, 0.0F);
        if (step % 50 == 0 || step + 1 == steps) {
            curve << step << "," << loss << "\n";
            if (step % 500 == 0) {
                std::cout << label << " step " << step << " loss " << loss
                          << std::endl;
            }
        }
    }

    const auto checksum_before = kernel.parameter_checksum();
    const auto scores = evaluate_exhaustively(kernel);
    const auto checksum_after = kernel.parameter_checksum();

    std::size_t total = 0;
    std::size_t correct = 0;
    for (const auto& score : scores) {
        total += score.total;
        correct += score.correct;
    }

    std::ofstream summary(out_dir + "/summary.json");
    summary.precision(12);
    summary << "{\n";
    summary << "  \"label\": \"" << label << "\",\n";
    summary << "  \"bit_width\": " << kWidth << ",\n";
    summary << "  \"seed\": " << seed << ",\n";
    summary << "  \"training_steps\": " << steps << ",\n";
    summary << "  \"parameters\": " << parameters << ",\n";
    summary << "  \"evaluation_parameter_drift\": "
            << (checksum_before == checksum_after ? 0 : 1) << ",\n";
    summary << "  \"exhaustive\": true,\n";
    summary << "  \"addition_accuracy\": " << scores[0].accuracy() << ",\n";
    summary << "  \"subtraction_accuracy\": " << scores[1].accuracy() << ",\n";
    summary << "  \"multiplication_accuracy\": " << scores[2].accuracy()
            << ",\n";
    summary << "  \"multiplication_nontrivial_accuracy\": "
            << scores[2].nontrivial_accuracy() << ",\n";
    summary << "  \"addition_nontrivial_accuracy\": "
            << scores[0].nontrivial_accuracy() << ",\n";
    summary << "  \"subtraction_nontrivial_accuracy\": "
            << scores[1].nontrivial_accuracy() << ",\n";
    summary << "  \"multiplication_cases\": " << scores[2].total << ",\n";
    summary << "  \"multiplication_nontrivial_cases\": "
            << scores[2].nontrivial_total << ",\n";
    summary << "  \"overall_accuracy\": "
            << static_cast<double>(correct) / static_cast<double>(total)
            << "\n";
    summary << "}\n";

    std::cout << label << " | params " << parameters
              << " | add " << scores[0].accuracy()
              << " | sub " << scores[1].accuracy()
              << " | mul " << scores[2].accuracy()
              << " | mul(nontrivial) " << scores[2].nontrivial_accuracy()
              << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: " << argv[0]
                  << " <out_dir> <s4|s5> <capacity> <seed> <steps>\n";
        return 2;
    }
    try {
        const std::string out_dir = argv[1];
        const std::string arch = argv[2];
        const auto capacity =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const auto seed = static_cast<std::uint64_t>(std::stoull(argv[4]));
        const auto steps = static_cast<std::uint32_t>(std::stoul(argv[5]));
        std::filesystem::create_directories(out_dir);

        if (arch == "s4") {
            sera::ProofKernelConfig config;
            config.bit_width = kWidth;
            config.hidden_units = 32;
            config.shared_rank = capacity;
            config.reasoning_steps = 20;
            config.residual_rate = 0.4F;
            sera::NeuralProofKernel kernel(config, seed);
            const auto label =
                "s4_rank" + std::to_string(capacity) + "_seed"
                + std::to_string(seed);
            run(
                kernel, label, kernel.telemetry().parameters, out_dir, seed,
                steps
            );
        } else if (arch == "s5") {
            sera::WorkspaceKernelConfig config;
            config.bit_width = kWidth;
            config.channels = capacity;
            config.shared_rank = 5;
            config.reasoning_steps = 14;
            sera::InteractionWorkspaceKernel kernel(config, seed);
            const auto label =
                "s5_channels" + std::to_string(capacity) + "_seed"
                + std::to_string(seed);
            run(
                kernel, label, kernel.telemetry().parameters, out_dir, seed,
                steps
            );
        } else {
            std::cerr << "unknown architecture: " << arch << "\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "comparison failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

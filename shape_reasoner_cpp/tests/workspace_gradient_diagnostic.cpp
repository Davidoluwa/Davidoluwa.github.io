// Diagnostic: per-tensor analytic-vs-numeric gradient comparison, swept over
// finite-difference step sizes. Used to separate float32 differencing noise
// from a genuine backward-pass defect.
#include "sera/interaction_workspace_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    sera::WorkspaceKernelConfig config;
    config.bit_width = 4;
    config.channels = 5;
    config.shared_rank = 3;
    config.reasoning_steps = 3;

    const std::vector<sera::ArithmeticTrainingCase> batch{
        {3, 2, 5, '+'}, {7, 5, 2, '-'}, {3, 3, 9, '*'}
    };

    const std::size_t channels = config.channels;
    const std::size_t rank = config.shared_rank;
    const std::size_t features = 15;
    const std::size_t taps = 10;
    const std::vector<std::pair<std::string, std::size_t>> blocks{
        {"input", channels * features},
        {"input_bias", channels},
        {"projection", rank * channels},
        {"candidate", taps * channels * rank},
        {"candidate_bias", channels},
        {"update", taps * channels * rank},
        {"update_bias", channels},
        {"output", channels},
        {"output_bias", 1}
    };

    for (double epsilon : {1.0e-3, 3.0e-3, 1.0e-2}) {
        sera::InteractionWorkspaceKernel kernel(config, 0x9E3779B97F4A7C15ULL);
        const auto analytic = kernel.flat_gradient(batch);
        auto values = kernel.flat_parameters();
        std::printf("\n--- epsilon = %g ---\n", epsilon);

        std::size_t cursor = 0;
        for (const auto& block : blocks) {
            double worst = 0.0;
            const std::size_t stride = std::max<std::size_t>(block.second / 8, 1);
            for (std::size_t k = 0; k < block.second; k += stride) {
                const std::size_t index = cursor + k;
                const double original = values[index];
                values[index] = original + epsilon;
                kernel.set_flat_parameters(values);
                const double up = kernel.batch_loss(batch);
                values[index] = original - epsilon;
                kernel.set_flat_parameters(values);
                const double down = kernel.batch_loss(batch);
                values[index] = original;
                kernel.set_flat_parameters(values);

                const double numeric = (up - down) / (2.0 * epsilon);
                const double scale = std::max(
                    {1.0e-3, std::abs(numeric), std::abs(analytic[index])}
                );
                worst = std::max(worst, std::abs(numeric - analytic[index]) / scale);
            }
            std::printf("%-16s worst rel err %.4e\n", block.first.c_str(), worst);
            cursor += block.second;
        }
    }
    return 0;
}

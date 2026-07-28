#include "sera/interaction_workspace_kernel.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

sera::WorkspaceKernelConfig small_config() {
    sera::WorkspaceKernelConfig config;
    config.bit_width = 4;
    config.channels = 5;
    config.shared_rank = 3;
    config.reasoning_steps = 3;
    return config;
}

const std::vector<sera::ArithmeticTrainingCase>& sample_batch() {
    static const std::vector<sera::ArithmeticTrainingCase> batch{
        {3, 2, 5, '+'},
        {7, 5, 2, '-'},
        {3, 3, 9, '*'}
    };
    return batch;
}

// Central finite differences against the analytic gradient. This is the only
// guard that the hand-written backward pass through the gated stencil,
// the shared projection, and the anchor tap is actually correct.
void check_gradient() {
    sera::InteractionWorkspaceKernel kernel(
        small_config(), 0x9E3779B97F4A7C15ULL
    );
    const auto analytic = kernel.flat_gradient(sample_batch());
    auto values = kernel.flat_parameters();
    require(
        analytic.size() == values.size(),
        "Gradient and parameter vectors disagree in length"
    );

    std::mt19937_64 engine(12345);
    std::uniform_int_distribution<std::size_t> pick(0, values.size() - 1);
    // Parameters are stored in float32, so a small step is dominated by
    // round-off cancellation rather than truncation error. 1e-2 is the regime
    // where the central difference is actually resolvable; see
    // tests/workspace_gradient_diagnostic.cpp for the epsilon sweep.
    const double epsilon = 1.0e-2;
    double worst = 0.0;

    for (int trial = 0; trial < 60; ++trial) {
        const std::size_t index = pick(engine);
        const double original = values[index];

        values[index] = original + epsilon;
        kernel.set_flat_parameters(values);
        const double up = kernel.batch_loss(sample_batch());

        values[index] = original - epsilon;
        kernel.set_flat_parameters(values);
        const double down = kernel.batch_loss(sample_batch());

        values[index] = original;
        kernel.set_flat_parameters(values);

        const double numeric = (up - down) / (2.0 * epsilon);
        const double scale =
            std::max({1.0e-3, std::abs(numeric), std::abs(analytic[index])});
        const double relative = std::abs(numeric - analytic[index]) / scale;
        worst = std::max(worst, relative);
    }

    std::cout << "worst relative gradient error: " << worst << "\n";
    require(
        worst < 5.0e-3,
        "Analytic gradient disagrees with finite differences"
    );
}

void check_determinism_and_learning() {
    sera::InteractionWorkspaceKernel kernel(
        small_config(), 0x5151515151515151ULL
    );
    const auto first = kernel.infer(3, 2, '+');
    const auto second = kernel.infer(3, 2, '+');
    require(
        first.value == second.value && first.logits == second.logits,
        "Repeated inference was not deterministic"
    );
    require(
        std::isfinite(first.mean_confidence),
        "Inference confidence was non-finite"
    );
    require(
        first.stable_after_step <= small_config().reasoning_steps,
        "Stability step exceeded the reasoning horizon"
    );

    const double before = kernel.batch_loss(sample_batch());
    for (int step = 0; step < 200; ++step) {
        kernel.train_batch(sample_batch(), 0.02F);
    }
    const double after = kernel.batch_loss(sample_batch());
    std::cout << "loss " << before << " -> " << after << "\n";
    require(after < before, "Training did not reduce the batch loss");
}

void check_bit_width_guard() {
    sera::InteractionWorkspaceKernel kernel(small_config());
    const std::vector<sera::ArithmeticTrainingCase> overflow{{99, 1, 100, '+'}};
    require_throws(
        [&] { kernel.train_batch(overflow, 0.01F); },
        "Out-of-range training case was not rejected"
    );
    sera::WorkspaceKernelConfig bad = small_config();
    bad.channels = 0;
    require_throws(
        [&] { sera::InteractionWorkspaceKernel broken(bad); },
        "Invalid configuration was not rejected"
    );
}

void check_checkpoint_round_trip() {
    const auto path =
        std::filesystem::temp_directory_path() / "sera_s5_round_trip.bin";
    sera::InteractionWorkspaceKernel kernel(
        small_config(), 0x1234ABCD5678EF90ULL
    );
    for (int step = 0; step < 25; ++step) {
        kernel.train_batch(sample_batch(), 0.02F);
    }
    const auto expected = kernel.parameter_checksum();
    const auto reference = kernel.infer(3, 3, '*');
    kernel.save_checkpoint(path.string());

    sera::InteractionWorkspaceKernel restored(
        small_config(), 0xFFFFFFFFFFFFFFFFULL
    );
    restored.load_checkpoint(path.string());
    require(
        restored.parameter_checksum() == expected,
        "Checkpoint round trip was not bit exact"
    );
    const auto replayed = restored.infer(3, 3, '*');
    require(
        replayed.value == reference.value
            && replayed.logits == reference.logits,
        "Restored kernel did not reproduce inference"
    );

    sera::WorkspaceKernelConfig other = small_config();
    other.channels += 1;
    sera::InteractionWorkspaceKernel mismatched(other);
    require_throws(
        [&] { mismatched.load_checkpoint(path.string()); },
        "Shape-mismatched checkpoint was accepted"
    );
    std::filesystem::remove(path);
}

void check_inference_is_pure() {
    sera::InteractionWorkspaceKernel kernel(small_config(), 0xABCDEF0123456789ULL);
    const auto before = kernel.parameter_checksum();
    for (std::uint32_t left = 0; left < 16; ++left) {
        for (std::uint32_t right = 0; right < 16; ++right) {
            (void)kernel.infer(left, right, '*');
        }
    }
    require(
        kernel.parameter_checksum() == before,
        "Evaluation mutated model parameters"
    );
}

// The architectural claim: parameter count must not grow with bit width.
void check_width_independent_parameters() {
    sera::WorkspaceKernelConfig narrow = small_config();
    sera::WorkspaceKernelConfig wide = small_config();
    wide.bit_width = 12;
    sera::InteractionWorkspaceKernel a(narrow);
    sera::InteractionWorkspaceKernel b(wide);
    require(
        a.telemetry().parameters == b.telemetry().parameters,
        "Parameter count changed with bit width"
    );
}

}  // namespace

int main() {
    try {
        check_gradient();
        check_determinism_and_learning();
        check_bit_width_guard();
        check_checkpoint_round_trip();
        check_inference_is_pure();
        check_width_independent_parameters();
    } catch (const std::exception& error) {
        std::cerr << "workspace kernel test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "workspace kernel test passed\n";
    return 0;
}

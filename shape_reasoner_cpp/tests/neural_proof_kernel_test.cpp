#include "sera/neural_proof_kernel.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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

}  // namespace

int main() {
    try {
        sera::ProofKernelConfig config;
        config.bit_width = 6;
        config.hidden_units = 8;
        config.shared_rank = 3;
        config.reasoning_steps = 4;
        config.residual_rate = 0.4F;

        sera::NeuralProofKernel kernel(config, 0x5445535453454544ULL);
        const auto before = kernel.parameter_checksum();
        const auto deterministic_a = kernel.infer(5, 3, '+');
        const auto deterministic_b = kernel.infer(5, 3, '+');
        require(
            deterministic_a.value == deterministic_b.value
                && deterministic_a.logits == deterministic_b.logits,
            "Repeated inference was not deterministic"
        );
        require(
            std::isfinite(deterministic_a.mean_confidence),
            "Inference confidence was non-finite"
        );

        const std::vector<sera::ArithmeticTrainingCase> batch{
            {1, 1, 2, '+'},
            {3, 1, 2, '-'},
            {2, 3, 6, '*'}
        };
        const double loss = kernel.train_batch(batch, 1.0e-3F);
        require(std::isfinite(loss), "Training loss was non-finite");
        require(
            kernel.parameter_checksum() != before,
            "Training did not update parameters"
        );
        require(
            kernel.telemetry().optimizer_steps == 1
                && kernel.telemetry().training_cases == batch.size(),
            "Training telemetry was not updated"
        );

        require_throws(
            [&] { static_cast<void>(kernel.infer(64, 0, '+')); },
            "Out-of-range inference was accepted"
        );
        const std::vector<sera::ArithmeticTrainingCase> invalid{
            {64, 0, 0, '+'}
        };
        require_throws(
            [&] { static_cast<void>(
                kernel.train_batch(invalid, 1.0e-3F)
            ); },
            "Out-of-range training case was accepted"
        );

        const auto checkpoint = std::filesystem::temp_directory_path()
            / "sera_neural_proof_kernel_test.bin";
        const auto truncated = std::filesystem::temp_directory_path()
            / "sera_neural_proof_kernel_truncated.bin";
        kernel.save_checkpoint(checkpoint.string());
        sera::NeuralProofKernel restored(config, 7);
        restored.load_checkpoint(checkpoint.string());
        require(
            restored.parameter_checksum() == kernel.parameter_checksum(),
            "Checkpoint checksum did not round-trip"
        );
        require(
            restored.telemetry().training_cases
                == kernel.telemetry().training_cases,
            "Checkpoint lost training-case telemetry"
        );
        require(
            restored.infer(5, 3, '+').logits
                == kernel.infer(5, 3, '+').logits,
            "Checkpoint changed inference"
        );

        {
            std::ofstream out(truncated, std::ios::binary);
            const std::uint32_t junk = 0x12345678U;
            out.write(
                reinterpret_cast<const char*>(&junk),
                sizeof(junk)
            );
        }
        require_throws(
            [&] { restored.load_checkpoint(truncated.string()); },
            "Truncated checkpoint was accepted"
        );
        std::filesystem::remove(checkpoint);
        std::filesystem::remove(truncated);

        std::cout << "neural_proof_kernel_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neural_proof_kernel_test: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}

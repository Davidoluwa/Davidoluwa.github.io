#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <vector>

namespace sera {

struct BooleanExample {
    std::vector<std::uint8_t> input;
    std::vector<std::uint8_t> output;
};

struct BooleanClause {
    std::uint64_t bits{};
    std::uint64_t wildcard_mask{};
};

enum class BooleanProgramKind : std::uint8_t {
    clause_cover,
    algebraic_normal_form
};

struct BooleanOutputProgram {
    BooleanProgramKind kind{BooleanProgramKind::clause_cover};
    bool complement{};
    std::vector<BooleanClause> clauses;
    std::uint64_t anf_coefficients{};
};

struct BooleanLearningTelemetry {
    std::uint32_t input_bits{};
    std::uint32_t output_bits{};
    std::uint32_t examples{};
    std::uint64_t candidate_implicants{};
    std::uint64_t combination_attempts{};
    std::uint64_t anf_programs_evaluated{};
    std::uint32_t selected_clauses{};
    std::uint32_t selected_anf_terms{};
    std::uint32_t selected_literals{};
    std::uint32_t training_errors{};
    std::uint64_t wall_nanoseconds{};
};

class BooleanProgramLearner {
public:
    void train(std::span<const BooleanExample> examples);

    [[nodiscard]] std::vector<std::uint8_t> predict(
        std::span<const std::uint8_t> input
    ) const;

    [[nodiscard]] const std::vector<BooleanOutputProgram>& programs()
        const noexcept;

    [[nodiscard]] const BooleanLearningTelemetry& telemetry() const noexcept;

    void write_json(std::ostream& out) const;

private:
    std::uint32_t input_bits_{};
    std::uint32_t output_bits_{};
    std::vector<BooleanOutputProgram> programs_;
    BooleanLearningTelemetry telemetry_;
};

}  // namespace sera

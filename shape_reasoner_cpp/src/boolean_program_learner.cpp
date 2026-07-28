#include "sera/boolean_program_learner.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace sera {
namespace {

using Clock = std::chrono::steady_clock;

struct Implicant {
    std::uint64_t bits{};
    std::uint64_t wildcard_mask{};

    auto operator<=>(const Implicant&) const = default;
};

struct CoverSolution {
    bool valid{};
    std::uint32_t literals{};
    std::uint32_t clauses{};
    std::vector<std::size_t> selected;
};

struct AlgebraicSolution {
    std::uint64_t coefficients{};
    std::uint32_t cost{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t terms{std::numeric_limits<std::uint32_t>::max()};
};

std::uint64_t encode_input(std::span<const std::uint8_t> input) {
    std::uint64_t encoded = 0;
    for (std::size_t bit = 0; bit < input.size(); ++bit) {
        if (input[bit] > 1) {
            throw std::invalid_argument("Boolean input values must be 0 or 1");
        }
        encoded |= static_cast<std::uint64_t>(input[bit]) << bit;
    }
    return encoded;
}

bool matches(const Implicant& implicant, std::uint64_t value) {
    return ((value ^ implicant.bits) & ~implicant.wildcard_mask) == 0;
}

std::uint32_t literal_count(
    const Implicant& implicant,
    std::uint32_t input_bits
) {
    return input_bits
        - static_cast<std::uint32_t>(
            std::popcount(implicant.wildcard_mask)
        );
}

std::vector<Implicant> prime_implicants(
    std::uint32_t input_bits,
    std::uint64_t target_examples,
    std::uint64_t opposite_examples,
    std::uint64_t& combination_attempts
) {
    const std::uint64_t states = 1ULL << input_bits;
    std::vector<Implicant> current;
    current.reserve(states);
    for (std::uint64_t value = 0; value < states; ++value) {
        if ((opposite_examples & (1ULL << value)) == 0) {
            current.push_back({value, 0});
        }
    }
    std::vector<Implicant> primes;
    while (!current.empty()) {
        std::vector<unsigned char> combined(current.size(), 0);
        std::vector<Implicant> next;
        for (std::size_t i = 0; i < current.size(); ++i) {
            for (std::size_t j = i + 1; j < current.size(); ++j) {
                ++combination_attempts;
                if (
                    current[i].wildcard_mask
                    != current[j].wildcard_mask
                ) {
                    continue;
                }
                const std::uint64_t difference =
                    (current[i].bits ^ current[j].bits)
                    & ~current[i].wildcard_mask;
                if (std::popcount(difference) != 1) {
                    continue;
                }
                combined[i] = 1;
                combined[j] = 1;
                const Implicant merged{
                    current[i].bits & ~difference,
                    current[i].wildcard_mask | difference
                };
                next.push_back(merged);
            }
        }
        for (std::size_t i = 0; i < current.size(); ++i) {
            if (combined[i] != 0) {
                continue;
            }
            bool covers_target = false;
            for (std::uint64_t value = 0; value < states; ++value) {
                if (
                    (target_examples & (1ULL << value)) != 0
                    && matches(current[i], value)
                ) {
                    covers_target = true;
                    break;
                }
            }
            if (covers_target) {
                primes.push_back(current[i]);
            }
        }
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        current = std::move(next);
    }
    std::sort(primes.begin(), primes.end());
    primes.erase(std::unique(primes.begin(), primes.end()), primes.end());
    return primes;
}

std::uint64_t covered_targets(
    const Implicant& implicant,
    std::uint64_t target_examples,
    std::uint32_t input_bits
) {
    std::uint64_t covered = 0;
    const std::uint64_t states = 1ULL << input_bits;
    for (std::uint64_t value = 0; value < states; ++value) {
        if (
            (target_examples & (1ULL << value)) != 0
            && matches(implicant, value)
        ) {
            covered |= 1ULL << value;
        }
    }
    return covered;
}

bool better(const CoverSolution& lhs, const CoverSolution& rhs) {
    if (!lhs.valid) {
        return false;
    }
    if (!rhs.valid) {
        return true;
    }
    if (lhs.literals != rhs.literals) {
        return lhs.literals < rhs.literals;
    }
    return lhs.clauses < rhs.clauses;
}

CoverSolution solve_cover(
    std::uint64_t remaining,
    std::span<const Implicant> primes,
    std::span<const std::uint64_t> covers,
    std::uint32_t input_bits,
    std::unordered_map<std::uint64_t, CoverSolution>& memo
) {
    if (remaining == 0) {
        return {true, 0, 0, {}};
    }
    const auto cached = memo.find(remaining);
    if (cached != memo.end()) {
        return cached->second;
    }
    std::uint64_t chosen_bit = 0;
    std::size_t fewest_choices = std::numeric_limits<std::size_t>::max();
    for (std::uint32_t bit = 0; bit < 64; ++bit) {
        const std::uint64_t marker = 1ULL << bit;
        if ((remaining & marker) == 0) {
            continue;
        }
        std::size_t choices = 0;
        for (std::uint64_t cover : covers) {
            choices += (cover & marker) != 0 ? 1U : 0U;
        }
        if (choices < fewest_choices) {
            fewest_choices = choices;
            chosen_bit = marker;
        }
    }
    CoverSolution best;
    for (std::size_t index = 0; index < primes.size(); ++index) {
        if ((covers[index] & chosen_bit) == 0) {
            continue;
        }
        CoverSolution tail = solve_cover(
            remaining & ~covers[index],
            primes,
            covers,
            input_bits,
            memo
        );
        if (!tail.valid) {
            continue;
        }
        tail.literals += literal_count(primes[index], input_bits);
        ++tail.clauses;
        tail.selected.push_back(index);
        if (better(tail, best)) {
            best = std::move(tail);
        }
    }
    memo.emplace(remaining, best);
    return best;
}

[[maybe_unused]] BooleanOutputProgram learn_target_cover(
    std::uint32_t input_bits,
    std::uint64_t target_examples,
    std::uint64_t opposite_examples,
    bool complement,
    BooleanLearningTelemetry& telemetry
) {
    BooleanOutputProgram program;
    program.complement = complement;
    if (target_examples == 0) {
        return program;
    }
    const auto primes = prime_implicants(
        input_bits,
        target_examples,
        opposite_examples,
        telemetry.combination_attempts
    );
    telemetry.candidate_implicants += primes.size();
    std::vector<std::uint64_t> covers;
    covers.reserve(primes.size());
    for (const auto& prime : primes) {
        covers.push_back(
            covered_targets(prime, target_examples, input_bits)
        );
    }
    std::unordered_map<std::uint64_t, CoverSolution> memo;
    const CoverSolution solution = solve_cover(
        target_examples,
        primes,
        covers,
        input_bits,
        memo
    );
    if (!solution.valid) {
        throw std::runtime_error("No consistent Boolean cover exists");
    }
    std::vector<std::size_t> selected = solution.selected;
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    for (std::size_t index : selected) {
        program.clauses.push_back({
            primes[index].bits,
            primes[index].wildcard_mask
        });
    }
    return program;
}

std::uint32_t program_literals(
    const BooleanOutputProgram& program,
    std::uint32_t input_bits
) {
    if (program.kind == BooleanProgramKind::algebraic_normal_form) {
        std::uint32_t literals = 0;
        std::uint64_t remaining = program.anf_coefficients;
        while (remaining != 0) {
            const std::uint32_t monomial =
                static_cast<std::uint32_t>(std::countr_zero(remaining));
            literals += static_cast<std::uint32_t>(
                std::popcount(monomial)
            );
            remaining &= remaining - 1;
        }
        return literals;
    }
    std::uint32_t literals = 0;
    for (const auto& clause : program.clauses) {
        literals += input_bits
            - static_cast<std::uint32_t>(
                std::popcount(clause.wildcard_mask)
            );
    }
    return literals;
}

std::uint32_t algebraic_cost(std::uint64_t coefficients) {
    std::uint32_t cost = 0;
    std::uint64_t remaining = coefficients;
    while (remaining != 0) {
        const std::uint32_t monomial =
            static_cast<std::uint32_t>(std::countr_zero(remaining));
        const std::uint32_t degree =
            static_cast<std::uint32_t>(std::popcount(monomial));
        cost += 2 + degree + (degree > 1 ? degree - 1 : 0);
        remaining &= remaining - 1;
    }
    return cost;
}

BooleanOutputProgram learn_algebraic_normal_form(
    std::uint32_t input_bits,
    const std::map<std::uint64_t, std::vector<std::uint8_t>>& examples,
    std::uint32_t output_bit,
    BooleanLearningTelemetry& telemetry,
    std::vector<AlgebraicSolution>* version_space = nullptr
) {
    const std::uint32_t monomials = 1U << input_bits;
    struct Row {
        std::uint64_t coefficients{};
        bool result{};
    };
    std::vector<Row> rows;
    rows.reserve(examples.size());
    for (const auto& [input, output] : examples) {
        Row row;
        for (std::uint32_t monomial = 0;
             monomial < monomials;
             ++monomial) {
            if (
                (input & monomial)
                == static_cast<std::uint64_t>(monomial)
            ) {
                row.coefficients |= 1ULL << monomial;
            }
        }
        row.result = output[output_bit] != 0;
        rows.push_back(row);
    }

    std::vector<std::uint32_t> pivots;
    std::size_t pivot_row = 0;
    for (std::uint32_t column = 0;
         column < monomials && pivot_row < rows.size();
         ++column) {
        std::size_t selected = pivot_row;
        while (
            selected < rows.size()
            && (rows[selected].coefficients & (1ULL << column)) == 0
        ) {
            ++selected;
        }
        if (selected == rows.size()) {
            continue;
        }
        std::swap(rows[pivot_row], rows[selected]);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (
                row != pivot_row
                && (rows[row].coefficients & (1ULL << column)) != 0
            ) {
                rows[row].coefficients ^=
                    rows[pivot_row].coefficients;
                rows[row].result =
                    rows[row].result != rows[pivot_row].result;
            }
        }
        pivots.push_back(column);
        ++pivot_row;
    }
    for (std::size_t row = pivot_row; row < rows.size(); ++row) {
        if (rows[row].coefficients == 0 && rows[row].result) {
            throw std::runtime_error(
                "Boolean examples are inconsistent over GF(2)"
            );
        }
    }
    std::uint64_t particular = 0;
    for (std::size_t row = 0; row < pivots.size(); ++row) {
        if (rows[row].result) {
            particular |= 1ULL << pivots[row];
        }
    }
    std::vector<unsigned char> is_pivot(monomials, 0);
    for (std::uint32_t pivot : pivots) {
        is_pivot[pivot] = 1;
    }
    std::vector<std::uint64_t> nullspace;
    for (std::uint32_t free = 0; free < monomials; ++free) {
        if (is_pivot[free] != 0) {
            continue;
        }
        std::uint64_t vector = 1ULL << free;
        for (std::size_t row = 0; row < pivots.size(); ++row) {
            if ((rows[row].coefficients & (1ULL << free)) != 0) {
                vector |= 1ULL << pivots[row];
            }
        }
        nullspace.push_back(vector);
    }
    if (nullspace.size() > 22) {
        throw std::runtime_error(
            "ANF version space exceeds bounded exhaustive search"
        );
    }
    const std::uint64_t candidates = 1ULL << nullspace.size();
    telemetry.anf_programs_evaluated += candidates;
    AlgebraicSolution best;
    std::uint64_t coefficients = particular;
    std::uint64_t previous_gray = 0;
    for (std::uint64_t index = 0; index < candidates; ++index) {
        const std::uint64_t gray = index ^ (index >> 1U);
        if (index != 0) {
            const std::uint64_t changed = gray ^ previous_gray;
            coefficients ^= nullspace[std::countr_zero(changed)];
        }
        previous_gray = gray;
        const std::uint32_t cost = algebraic_cost(coefficients);
        const std::uint32_t terms =
            static_cast<std::uint32_t>(std::popcount(coefficients));
        if (
            cost < best.cost
            || (cost == best.cost && terms < best.terms)
        ) {
            best = {coefficients, cost, terms};
        }
        if (version_space != nullptr) {
            version_space->push_back({coefficients, cost, terms});
        }
    }
    if (version_space != nullptr) {
        std::sort(
            version_space->begin(), version_space->end(),
            [](const AlgebraicSolution& lhs, const AlgebraicSolution& rhs) {
                if (lhs.cost != rhs.cost) {
                    return lhs.cost < rhs.cost;
                }
                if (lhs.terms != rhs.terms) {
                    return lhs.terms < rhs.terms;
                }
                return lhs.coefficients < rhs.coefficients;
            }
        );
    }
    BooleanOutputProgram program;
    program.kind = BooleanProgramKind::algebraic_normal_form;
    program.anf_coefficients = best.coefficients;
    return program;
}

bool evaluate_program(
    const BooleanOutputProgram& program,
    std::uint64_t encoded
) {
    if (program.kind == BooleanProgramKind::algebraic_normal_form) {
        bool result = false;
        std::uint64_t remaining = program.anf_coefficients;
        while (remaining != 0) {
            const std::uint32_t monomial =
                static_cast<std::uint32_t>(std::countr_zero(remaining));
            result = result != (
                (encoded & monomial)
                == static_cast<std::uint64_t>(monomial)
            );
            remaining &= remaining - 1;
        }
        return result;
    }
    bool matched = false;
    for (const auto& clause : program.clauses) {
        if (
            ((encoded ^ clause.bits) & ~clause.wildcard_mask) == 0
        ) {
            matched = true;
            break;
        }
    }
    return program.complement ? !matched : matched;
}

[[maybe_unused]] std::uint32_t description_cost(
    const BooleanOutputProgram& program,
    std::uint32_t input_bits
) {
    if (program.kind == BooleanProgramKind::algebraic_normal_form) {
        return algebraic_cost(program.anf_coefficients);
    }
    return program_literals(program, input_bits)
        + static_cast<std::uint32_t>(program.clauses.size() * 2)
        + (program.complement ? 1U : 0U);
}

[[maybe_unused]] std::uint32_t prediction_errors(
    const BooleanOutputProgram& program,
    const std::map<std::uint64_t, std::vector<std::uint8_t>>& examples,
    std::uint32_t output_bit
) {
    std::uint32_t errors = 0;
    for (const auto& [input, output] : examples) {
        errors +=
            evaluate_program(program, input)
                != (output[output_bit] != 0)
            ? 1U : 0U;
    }
    return errors;
}

[[maybe_unused]] std::pair<std::uint64_t, std::uint64_t> class_masks(
    const std::map<std::uint64_t, std::vector<std::uint8_t>>& examples,
    std::uint32_t output_bit
) {
    std::uint64_t positive = 0;
    std::uint64_t negative = 0;
    for (const auto& [input, output] : examples) {
        if (output[output_bit] != 0) {
            positive |= 1ULL << input;
        } else {
            negative |= 1ULL << input;
        }
    }
    return {positive, negative};
}

[[maybe_unused]] std::uint32_t shared_monomial_cost(
    std::uint64_t monomials
) {
    std::uint32_t cost = 0;
    while (monomials != 0) {
        const std::uint32_t monomial =
            static_cast<std::uint32_t>(std::countr_zero(monomials));
        const std::uint32_t degree =
            static_cast<std::uint32_t>(std::popcount(monomial));
        cost += degree <= 1 ? 1U : degree;
        monomials &= monomials - 1;
    }
    return cost;
}

}  // namespace

void BooleanProgramLearner::train(
    std::span<const BooleanExample> examples
) {
    if (examples.empty()) {
        throw std::invalid_argument("Boolean training set is empty");
    }
    input_bits_ = static_cast<std::uint32_t>(
        examples.front().input.size()
    );
    output_bits_ = static_cast<std::uint32_t>(
        examples.front().output.size()
    );
    if (
        input_bits_ == 0 || input_bits_ > 6 || output_bits_ == 0
    ) {
        throw std::invalid_argument(
            "Boolean learner supports 1..6 input bits and nonempty outputs"
        );
    }
    std::map<std::uint64_t, std::vector<std::uint8_t>> unique;
    for (const auto& example : examples) {
        if (
            example.input.size() != input_bits_
            || example.output.size() != output_bits_
        ) {
            throw std::invalid_argument(
                "Boolean example dimensions are inconsistent"
            );
        }
        for (std::uint8_t value : example.output) {
            if (value > 1) {
                throw std::invalid_argument(
                    "Boolean output values must be 0 or 1"
                );
            }
        }
        const std::uint64_t encoded = encode_input(example.input);
        const auto [found, inserted] = unique.emplace(
            encoded, example.output
        );
        if (!inserted && found->second != example.output) {
            throw std::invalid_argument(
                "Conflicting outputs for one Boolean input"
            );
        }
    }
    const auto wall_start = Clock::now();
    telemetry_ = {};
    telemetry_.input_bits = input_bits_;
    telemetry_.output_bits = output_bits_;
    telemetry_.examples = static_cast<std::uint32_t>(unique.size());
    programs_.clear();
    programs_.reserve(output_bits_);
    for (std::uint32_t output_bit = 0;
         output_bit < output_bits_;
         ++output_bit) {
        // ANF is the canonical complete Boolean representation: every truth
        // table has one unique full-data expansion. Under partial data, choose
        // the minimum-description member of its affine version space.
        programs_.push_back(learn_algebraic_normal_form(
            input_bits_, unique, output_bit, telemetry_
        ));
    }
    for (const auto& program : programs_) {
        telemetry_.selected_clauses += static_cast<std::uint32_t>(
            program.clauses.size()
        );
        if (program.kind == BooleanProgramKind::algebraic_normal_form) {
            telemetry_.selected_anf_terms += static_cast<std::uint32_t>(
                std::popcount(program.anf_coefficients)
            );
        }
        telemetry_.selected_literals += program_literals(
            program, input_bits_
        );
    }
    for (const auto& [input, expected] : unique) {
        for (std::uint32_t output_bit = 0;
             output_bit < output_bits_;
             ++output_bit) {
            const bool predicted = evaluate_program(
                programs_[output_bit], input
            );
            telemetry_.training_errors +=
                predicted != (expected[output_bit] != 0) ? 1U : 0U;
        }
    }
    telemetry_.wall_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - wall_start
        ).count()
    );
}

std::vector<std::uint8_t> BooleanProgramLearner::predict(
    std::span<const std::uint8_t> input
) const {
    if (input.size() != input_bits_ || programs_.empty()) {
        throw std::invalid_argument(
            "Boolean prediction input dimension mismatch or untrained model"
        );
    }
    const std::uint64_t encoded = encode_input(input);
    std::vector<std::uint8_t> output(output_bits_, 0);
    for (std::size_t bit = 0; bit < programs_.size(); ++bit) {
        output[bit] = evaluate_program(programs_[bit], encoded) ? 1U : 0U;
    }
    return output;
}

const std::vector<BooleanOutputProgram>&
BooleanProgramLearner::programs() const noexcept {
    return programs_;
}

const BooleanLearningTelemetry&
BooleanProgramLearner::telemetry() const noexcept {
    return telemetry_;
}

void BooleanProgramLearner::write_json(std::ostream& out) const {
    out << "{\n"
        << "  \"model_family\": \"minimum_description_boolean_program\",\n"
        << "  \"input_bits\": " << input_bits_ << ",\n"
        << "  \"output_bits\": " << output_bits_ << ",\n"
        << "  \"programs\": [\n";
    for (std::size_t output = 0; output < programs_.size(); ++output) {
        const auto& program = programs_[output];
        out << "    {\"output_wire\": " << output
            << ", \"kind\": \""
            << (
                program.kind == BooleanProgramKind::algebraic_normal_form
                    ? "algebraic_normal_form"
                    : "clause_cover"
            )
            << "\""
            << ", \"complement\": "
            << (program.complement ? "true" : "false")
            << ", \"anf_coefficients\": "
            << program.anf_coefficients
            << ", \"clauses\": [";
        for (std::size_t index = 0;
             index < program.clauses.size();
             ++index) {
            if (index != 0) {
                out << ", ";
            }
            out << "{\"bits\": " << program.clauses[index].bits
                << ", \"wildcard_mask\": "
                << program.clauses[index].wildcard_mask << "}";
        }
        out << "]}";
        if (output + 1 != programs_.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ],\n"
        << "  \"telemetry\": {\n"
        << "    \"candidate_implicants\": "
        << telemetry_.candidate_implicants << ",\n"
        << "    \"combination_attempts\": "
        << telemetry_.combination_attempts << ",\n"
        << "    \"anf_programs_evaluated\": "
        << telemetry_.anf_programs_evaluated << ",\n"
        << "    \"selected_clauses\": "
        << telemetry_.selected_clauses << ",\n"
        << "    \"selected_anf_terms\": "
        << telemetry_.selected_anf_terms << ",\n"
        << "    \"selected_literals\": "
        << telemetry_.selected_literals << ",\n"
        << "    \"training_errors\": "
        << telemetry_.training_errors << "\n"
        << "  }\n}\n";
}

}  // namespace sera

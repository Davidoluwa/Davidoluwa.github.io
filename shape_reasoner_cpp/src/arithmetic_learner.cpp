#include "sera/arithmetic_learner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <stdexcept>

namespace sera {
namespace {

struct BasisTerm {
    std::uint32_t left_power{};
    std::uint32_t right_power{};
};

double integer_power(double value, std::uint32_t power) {
    double result = 1.0;
    for (std::uint32_t i = 0; i < power; ++i) {
        result *= value;
    }
    return result;
}

double evaluate_basis(
    const ArithmeticExample& example,
    const BasisTerm& term,
    double input_scale
) {
    const double left = static_cast<double>(example.left) / input_scale;
    const double right = static_cast<double>(example.right) / input_scale;
    return integer_power(left, term.left_power)
        * integer_power(right, term.right_power);
}

double evaluate_basis(
    double left,
    double right,
    const BasisTerm& term,
    double input_scale
) {
    return integer_power(left / input_scale, term.left_power)
        * integer_power(right / input_scale, term.right_power);
}

std::vector<double> solve_linear_system(
    std::vector<double> matrix,
    std::vector<double> rhs,
    std::size_t dimensions
) {
    for (std::size_t column = 0; column < dimensions; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < dimensions; ++row) {
            if (
                std::abs(matrix[row * dimensions + column])
                > std::abs(matrix[pivot * dimensions + column])
            ) {
                pivot = row;
            }
        }
        const double pivot_value =
            matrix[pivot * dimensions + column];
        if (
            !std::isfinite(pivot_value)
            || std::abs(pivot_value) < 1.0e-15
        ) {
            throw std::runtime_error("Polynomial normal equations are singular");
        }
        if (pivot != column) {
            for (std::size_t j = column; j < dimensions; ++j) {
                std::swap(
                    matrix[column * dimensions + j],
                    matrix[pivot * dimensions + j]
                );
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        for (std::size_t row = column + 1; row < dimensions; ++row) {
            const double multiplier =
                matrix[row * dimensions + column]
                / matrix[column * dimensions + column];
            matrix[row * dimensions + column] = 0.0;
            for (std::size_t j = column + 1; j < dimensions; ++j) {
                matrix[row * dimensions + j] -=
                    multiplier * matrix[column * dimensions + j];
            }
            rhs[row] -= multiplier * rhs[column];
        }
    }
    std::vector<double> solution(dimensions, 0.0);
    for (std::size_t reverse = 0; reverse < dimensions; ++reverse) {
        const std::size_t row = dimensions - reverse - 1;
        double value = rhs[row];
        for (std::size_t column = row + 1; column < dimensions; ++column) {
            value -= matrix[row * dimensions + column] * solution[column];
        }
        solution[row] = value / matrix[row * dimensions + row];
    }
    return solution;
}

std::vector<double> fit_coefficients(
    std::span<const ArithmeticExample> examples,
    std::span<const BasisTerm> basis,
    std::span<const std::size_t> selected,
    double input_scale,
    double target_scale,
    double ridge
) {
    const std::size_t count = selected.size();
    std::vector<double> normal(count * count, 0.0);
    std::vector<double> rhs(count, 0.0);
    std::vector<double> row(count, 0.0);
    for (const auto& example : examples) {
        for (std::size_t i = 0; i < count; ++i) {
            row[i] = evaluate_basis(
                example, basis[selected[i]], input_scale
            );
        }
        const double target = example.result / target_scale;
        for (std::size_t i = 0; i < count; ++i) {
            rhs[i] += row[i] * target;
            for (std::size_t j = 0; j < count; ++j) {
                normal[i * count + j] += row[i] * row[j];
            }
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        normal[i * count + i] += ridge;
    }
    return solve_linear_system(
        std::move(normal), std::move(rhs), count
    );
}

double mean_squared_error(
    std::span<const ArithmeticExample> examples,
    std::span<const BasisTerm> basis,
    std::span<const std::size_t> selected,
    std::span<const double> coefficients,
    double input_scale,
    double target_scale
) {
    if (examples.empty()) {
        return 0.0;
    }
    double squared = 0.0;
    for (const auto& example : examples) {
        double normalized = 0.0;
        for (std::size_t i = 0; i < selected.size(); ++i) {
            normalized += coefficients[i] * evaluate_basis(
                example, basis[selected[i]], input_scale
            );
        }
        const double error =
            normalized * target_scale - example.result;
        squared += error * error;
    }
    return squared / static_cast<double>(examples.size());
}

std::uint64_t split_hash(const ArithmeticExample& example) {
    std::uint64_t value =
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(example.left) + 0x10000
        );
    value = value * 0x9E3779B185EBCA87ULL
        + static_cast<std::uint64_t>(
            static_cast<std::int64_t>(example.right) + 0x10000
        );
    value ^= static_cast<std::uint64_t>(example.operator_id)
        * 0xC2B2AE3D27D4EB4FULL;
    value ^= value >> 29U;
    value *= 0x165667B19E3779F9ULL;
    return value ^ (value >> 32U);
}

}  // namespace

ArithmeticRelationLibrary::ArithmeticRelationLibrary(
    ArithmeticTrainingOptions options
) : options_(options) {
    if (
        options_.maximum_total_degree == 0
        || options_.maximum_terms == 0
    ) {
        throw std::invalid_argument(
            "Polynomial degree and term budget must be positive"
        );
    }
    if (
        !(options_.ridge > 0.0) || !std::isfinite(options_.ridge)
        || !(options_.minimum_relative_improvement >= 0.0)
        || !std::isfinite(options_.minimum_relative_improvement)
    ) {
        throw std::invalid_argument("Invalid arithmetic training options");
    }
}

void ArithmeticRelationLibrary::train(
    std::span<const ArithmeticExample> examples
) {
    if (examples.empty()) {
        throw std::invalid_argument("Arithmetic training set is empty");
    }
    std::map<std::uint32_t, std::vector<ArithmeticExample>> grouped;
    for (const auto& example : examples) {
        if (!std::isfinite(example.result)) {
            throw std::invalid_argument(
                "Arithmetic training result must be finite"
            );
        }
        grouped[example.operator_id].push_back(example);
    }
    std::vector<BasisTerm> basis;
    for (std::uint32_t degree = 0;
         degree <= options_.maximum_total_degree;
         ++degree) {
        for (std::uint32_t left_power = 0;
             left_power <= degree;
             ++left_power) {
            basis.push_back({
                left_power,
                degree - left_power
            });
        }
    }

    models_.clear();
    telemetry_.clear();
    for (const auto& [operator_id, operator_examples] : grouped) {
        if (operator_examples.size() < basis.size() + 4) {
            throw std::invalid_argument(
                "Insufficient examples for polynomial discovery"
            );
        }
        double input_scale = 1.0;
        double target_scale = 1.0;
        for (const auto& example : operator_examples) {
            input_scale = std::max(
                input_scale,
                static_cast<double>(
                    std::max(std::abs(example.left), std::abs(example.right))
                )
            );
            target_scale = std::max(
                target_scale, std::abs(example.result)
            );
        }
        std::vector<ArithmeticExample> fit;
        std::vector<ArithmeticExample> validation;
        fit.reserve(operator_examples.size());
        validation.reserve(operator_examples.size() / 5 + 1);
        for (const auto& example : operator_examples) {
            if (split_hash(example) % 5U == 0U) {
                validation.push_back(example);
            } else {
                fit.push_back(example);
            }
        }
        if (validation.empty() || fit.size() < basis.size()) {
            throw std::runtime_error(
                "Deterministic fit/validation split is degenerate"
            );
        }

        std::vector<std::size_t> selected{0};
        std::vector<double> coefficients = fit_coefficients(
            fit, basis, selected, input_scale, target_scale, options_.ridge
        );
        double current_validation = mean_squared_error(
            validation, basis, selected, coefficients,
            input_scale, target_scale
        );
        while (
            selected.size() < options_.maximum_terms
            && selected.size() < basis.size()
            && current_validation > 1.0e-20
        ) {
            std::size_t best_candidate =
                std::numeric_limits<std::size_t>::max();
            double best_validation =
                std::numeric_limits<double>::infinity();
            std::vector<double> best_coefficients;
            for (std::size_t candidate = 1;
                 candidate < basis.size();
                 ++candidate) {
                if (
                    std::find(
                        selected.begin(), selected.end(), candidate
                    ) != selected.end()
                ) {
                    continue;
                }
                std::vector<std::size_t> proposed = selected;
                proposed.push_back(candidate);
                const auto proposed_coefficients = fit_coefficients(
                    fit, basis, proposed, input_scale, target_scale,
                    options_.ridge
                );
                const double proposed_validation = mean_squared_error(
                    validation, basis, proposed, proposed_coefficients,
                    input_scale, target_scale
                );
                if (proposed_validation < best_validation) {
                    best_validation = proposed_validation;
                    best_candidate = candidate;
                    best_coefficients = proposed_coefficients;
                }
            }
            const double required =
                current_validation
                * (1.0 - options_.minimum_relative_improvement);
            if (
                best_candidate == std::numeric_limits<std::size_t>::max()
                || !(best_validation < required)
            ) {
                break;
            }
            selected.push_back(best_candidate);
            coefficients = std::move(best_coefficients);
            current_validation = best_validation;
        }

        const auto selection_coefficients = coefficients;
        const double fit_mse = mean_squared_error(
            fit, basis, selected, selection_coefficients,
            input_scale, target_scale
        );
        const double validation_mse = mean_squared_error(
            validation, basis, selected, selection_coefficients,
            input_scale, target_scale
        );
        coefficients = fit_coefficients(
            operator_examples, basis, selected, input_scale, target_scale,
            options_.ridge
        );
        const double all_mse = mean_squared_error(
            operator_examples, basis, selected, coefficients,
            input_scale, target_scale
        );

        Model model;
        model.operator_id = operator_id;
        model.input_scale = input_scale;
        model.target_scale = target_scale;
        RelationTrainingTelemetry item;
        item.operator_id = operator_id;
        item.examples = static_cast<std::uint32_t>(
            operator_examples.size()
        );
        item.fit_examples = static_cast<std::uint32_t>(fit.size());
        item.validation_examples = static_cast<std::uint32_t>(
            validation.size()
        );
        item.input_scale = input_scale;
        item.target_scale = target_scale;
        item.fit_mse = fit_mse;
        item.validation_mse = validation_mse;
        item.all_training_mse = all_mse;
        for (std::size_t i = 0; i < selected.size(); ++i) {
            const PolynomialTerm term{
                basis[selected[i]].left_power,
                basis[selected[i]].right_power,
                coefficients[i]
            };
            model.terms.push_back(term);
            item.terms.push_back(term);
        }
        models_.push_back(std::move(model));
        telemetry_.push_back(std::move(item));
    }
}

double ArithmeticRelationLibrary::predict(
    std::uint32_t operator_id,
    double left,
    double right
) const {
    const auto found = std::find_if(
        models_.begin(), models_.end(),
        [&](const Model& model) {
            return model.operator_id == operator_id;
        }
    );
    if (found == models_.end()) {
        throw std::out_of_range("Unknown learned arithmetic operator");
    }
    double normalized = 0.0;
    for (const auto& term : found->terms) {
        normalized += term.coefficient * evaluate_basis(
            left,
            right,
            {term.left_power, term.right_power},
            found->input_scale
        );
    }
    return normalized * found->target_scale;
}

bool ArithmeticRelationLibrary::has_operator(
    std::uint32_t operator_id
) const noexcept {
    return std::any_of(
        models_.begin(), models_.end(),
        [&](const Model& model) {
            return model.operator_id == operator_id;
        }
    );
}

const std::vector<RelationTrainingTelemetry>&
ArithmeticRelationLibrary::telemetry() const noexcept {
    return telemetry_;
}

void ArithmeticRelationLibrary::write_json(std::ostream& out) const {
    out << std::setprecision(17);
    out << "{\n  \"model_family\": \"sparse_polynomial_relation\",\n"
        << "  \"maximum_total_degree\": "
        << options_.maximum_total_degree << ",\n"
        << "  \"operators\": [\n";
    for (std::size_t index = 0; index < models_.size(); ++index) {
        const auto& model = models_[index];
        const auto& training = telemetry_[index];
        out << "    {\n"
            << "      \"operator_id\": " << model.operator_id << ",\n"
            << "      \"training_examples\": "
            << training.examples << ",\n"
            << "      \"validation_mse\": "
            << training.validation_mse << ",\n"
            << "      \"input_scale\": "
            << model.input_scale << ",\n"
            << "      \"target_scale\": "
            << model.target_scale << ",\n"
            << "      \"terms\": [\n";
        for (std::size_t term_index = 0;
             term_index < model.terms.size();
             ++term_index) {
            const auto& term = model.terms[term_index];
            const double raw_coefficient =
                term.coefficient * model.target_scale
                / integer_power(
                    model.input_scale,
                    term.left_power + term.right_power
                );
            out << "        {\"name\": \""
                << polynomial_term_name(
                    term.left_power, term.right_power
                )
                << "\", \"left_power\": " << term.left_power
                << ", \"right_power\": " << term.right_power
                << ", \"normalized_coefficient\": "
                << term.coefficient
                << ", \"raw_coefficient\": "
                << raw_coefficient << "}";
            if (term_index + 1 != model.terms.size()) {
                out << ',';
            }
            out << '\n';
        }
        out << "      ]\n    }";
        if (index + 1 != models_.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n}\n";
}

std::string polynomial_term_name(
    std::uint32_t left_power,
    std::uint32_t right_power
) {
    if (left_power == 0 && right_power == 0) {
        return "1";
    }
    std::string name;
    if (left_power != 0) {
        name += "left";
        if (left_power != 1) {
            name += "^" + std::to_string(left_power);
        }
    }
    if (right_power != 0) {
        if (!name.empty()) {
            name += "*";
        }
        name += "right";
        if (right_power != 1) {
            name += "^" + std::to_string(right_power);
        }
    }
    return name;
}

}  // namespace sera

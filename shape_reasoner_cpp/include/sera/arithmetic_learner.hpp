#pragma once

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sera {

struct ArithmeticExample {
    std::uint32_t operator_id{};
    int left{};
    int right{};
    double result{};
};

struct PolynomialTerm {
    std::uint32_t left_power{};
    std::uint32_t right_power{};
    double coefficient{};
};

struct RelationTrainingTelemetry {
    std::uint32_t operator_id{};
    std::uint32_t examples{};
    std::uint32_t fit_examples{};
    std::uint32_t validation_examples{};
    double input_scale{};
    double target_scale{};
    double fit_mse{};
    double validation_mse{};
    double all_training_mse{};
    std::vector<PolynomialTerm> terms;
};

struct ArithmeticTrainingOptions {
    std::uint32_t maximum_total_degree{3};
    std::uint32_t maximum_terms{6};
    double ridge{1.0e-12};
    double minimum_relative_improvement{1.0e-8};
};

class ArithmeticRelationLibrary {
public:
    explicit ArithmeticRelationLibrary(
        ArithmeticTrainingOptions options = {}
    );

    void train(std::span<const ArithmeticExample> examples);

    [[nodiscard]] double predict(
        std::uint32_t operator_id,
        double left,
        double right
    ) const;

    [[nodiscard]] bool has_operator(std::uint32_t operator_id) const noexcept;

    [[nodiscard]] const std::vector<RelationTrainingTelemetry>& telemetry()
        const noexcept;

    void write_json(std::ostream& out) const;

private:
    struct Model {
        std::uint32_t operator_id{};
        double input_scale{1.0};
        double target_scale{1.0};
        std::vector<PolynomialTerm> terms;
    };

    ArithmeticTrainingOptions options_;
    std::vector<Model> models_;
    std::vector<RelationTrainingTelemetry> telemetry_;
};

[[nodiscard]] std::string polynomial_term_name(
    std::uint32_t left_power,
    std::uint32_t right_power
);

}  // namespace sera

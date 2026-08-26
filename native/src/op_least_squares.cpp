// The `least_squares` operation: wire decoding/encoding and fault mapping
// only. The mathematics lives in scl/least_squares.hpp + least_squares.cpp.
//
// PARAMETERS VERSUS INPUTS -- got right at first write, because it bit
// once already on the state-estimation observations.
//   PARAMETERS (configuration bytes -> parameters_identity):
//     the rank tolerance, and the DECISION to weight. Both are solver
//     configuration: they say how to fit, not what was measured.
//   INPUTS (input payload -> input_identity):
//     X, y, and the WEIGHTS. A weight vector is per-observation data
//     about the observations -- it has one entry per row and changes when
//     the data changes -- so it is input, even though the choice to use
//     weighting at all is a parameter. Putting weights in the
//     configuration would mint a new parameters_identity per dataset and
//     make the model useless for comparing fits across data.
//
// configuration: exactly 24 bytes, little-endian
//     offset  0  int32    weighted          0 unweighted, 1 weights supplied
//     offset  4  int32    reserved          must be 0
//     offset  8  int32    reserved2         must be 0
//     offset 12  int32    reserved3         must be 0
//     offset 16  float64  rank_tolerance    relative singular-value cutoff, > 0
//
// input:  int32 n_rows, int32 n_cols, then
//         n_rows * n_cols float64  design matrix X, ROW-MAJOR
//         n_rows          float64  response y
//         n_rows          float64  weights   -- present only when weighted == 1
//   The shape travels with the DATA rather than in the configuration,
//   because a shape is a property of the observations, not a fitting
//   choice: two datasets of different size are different inputs, not
//   different parameters.
//
// output: n_cols float64 -- the coefficients, in column order

#include "scl/bytes.hpp"
#include "scl/least_squares.hpp"
#include "scl/operation.hpp"
#include "scl/protocol.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

namespace scl {
namespace {

struct Configuration {
    LeastSquaresParameters parameters;
};

Configuration decode_configuration(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 24) {
        std::ostringstream os;
        os << "configuration must be exactly 24 bytes (int32 weighted, 3 x int32 reserved, "
           << "float64 rank_tolerance), got " << bytes.size();
        throw OperationValidationError(os.str());
    }
    Configuration configuration;
    const int32_t weighted = read_int32_le(bytes, 0);
    if (weighted != 0 && weighted != 1) {
        std::ostringstream os;
        os << "weighted must be 0 or 1, got " << weighted;
        throw OperationValidationError(os.str());
    }
    configuration.parameters.weighted = weighted == 1;

    for (std::size_t offset : {std::size_t{4}, std::size_t{8}, std::size_t{12}}) {
        const int32_t reserved = read_int32_le(bytes, offset);
        if (reserved != 0) {
            std::ostringstream os;
            os << "reserved field at offset " << offset << " must be 0, got " << reserved;
            throw OperationValidationError(os.str());
        }
    }

    const double tolerance = read_double_le(bytes, 16);
    if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) {
        std::ostringstream os;
        os << "rank_tolerance must be finite and in (0, 1), got " << tolerance;
        throw OperationValidationError(os.str());
    }
    configuration.parameters.rank_tolerance = tolerance;
    return configuration;
}

LeastSquaresProblem decode_input(const std::vector<uint8_t>& bytes, bool weighted) {
    if (bytes.size() < 8) {
        throw OperationValidationError(
            "input must begin with int32 n_rows and int32 n_cols (8 bytes)");
    }
    const int32_t rows = read_int32_le(bytes, 0);
    const int32_t cols = read_int32_le(bytes, 4);
    if (rows <= 0 || cols <= 0) {
        std::ostringstream os;
        os << "n_rows and n_cols must both be positive, got n_rows=" << rows
           << " n_cols=" << cols;
        throw OperationValidationError(os.str());
    }

    LeastSquaresProblem problem;
    problem.n_rows = static_cast<std::size_t>(rows);
    problem.n_cols = static_cast<std::size_t>(cols);

    const std::size_t values = problem.n_rows * problem.n_cols + problem.n_rows +
                                (weighted ? problem.n_rows : 0);
    const std::size_t expected = 8 + values * 8;
    if (bytes.size() != expected) {
        std::ostringstream os;
        os << "input must be exactly " << expected << " bytes for n_rows=" << rows
           << " n_cols=" << cols << (weighted ? " with weights" : " without weights")
           << ", got " << bytes.size();
        throw OperationValidationError(os.str());
    }

    std::size_t offset = 8;
    auto take = [&bytes, &offset](std::size_t count) {
        std::vector<double> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(read_double_le(bytes, offset));
            offset += 8;
        }
        return out;
    };
    problem.design = take(problem.n_rows * problem.n_cols);
    problem.response = take(problem.n_rows);
    if (weighted) {
        problem.weights = take(problem.n_rows);
    }

    for (double value : problem.design) {
        if (!std::isfinite(value)) {
            throw OperationValidationError("design matrix contains a non-finite value");
        }
    }
    for (double value : problem.response) {
        if (!std::isfinite(value)) {
            throw OperationValidationError("response vector contains a non-finite value");
        }
    }
    return problem;
}

std::vector<uint8_t> encode_output(const std::vector<double>& coefficients) {
    std::vector<uint8_t> out;
    out.reserve(coefficients.size() * 8);
    for (double value : coefficients) {
        write_double_le(out, value);
    }
    return out;
}

}  // namespace

OperationOutcome run_least_squares(const OperationRequest& request) {
    try {
        std::string unavailable = backend_unavailable_reason(request.backend);
        if (!unavailable.empty()) {
            return OperationOutcome::halted(kFaultBackendUnavailable, unavailable);
        }

        const Configuration configuration = decode_configuration(request.configuration);
        const LeastSquaresProblem problem =
            decode_input(request.input, configuration.parameters.weighted);

        const auto compute_start = std::chrono::steady_clock::now();
        const LeastSquaresResult result =
            solve_least_squares(problem, configuration.parameters);
        const auto compute_end = std::chrono::steady_clock::now();
        const double compute_seconds =
            std::chrono::duration<double>(compute_end - compute_start).count();

        std::vector<Metric> metrics = {
            {"native_compute_seconds", compute_seconds},
            {"n_cols", static_cast<double>(problem.n_cols)},
            {"n_rows", static_cast<double>(problem.n_rows)},
        };
        if (!result.ok) {
            return OperationOutcome::halted(kFaultValidation, result.error, metrics);
        }

        // Conditioning is REPORTED, not left for the caller to infer. This
        // is why the solver is an SVD: an ill-conditioned fit returns a
        // plausible answer, and the only thing that makes it visible is
        // publishing the number alongside it.
        metrics.push_back({"condition_number", result.condition_number});
        metrics.push_back({"effective_rank", static_cast<double>(result.effective_rank)});
        metrics.push_back({"jacobi_sweeps", static_cast<double>(result.sweeps)});
        metrics.push_back({"smallest_singular_value", result.singular_values.back()});
        metrics.push_back({"largest_singular_value", result.singular_values.front()});

        return OperationOutcome::completed(encode_output(result.coefficients), metrics);
    } catch (const OperationValidationError& error) {
        return OperationOutcome::halted(kFaultValidation, error.what());
    } catch (const std::exception& error) {
        return OperationOutcome::halted(kFaultComputation, error.what());
    }
}

}  // namespace scl

// The `kalman_filter_linear` operation: wire decoding/encoding and fault
// mapping only. The mathematics lives in scl/kalman.hpp + kalman.cpp.
//
// PARAMETERS VERSUS INPUTS, and the third axis on top of both.
//
//   PARAMETERS (configuration bytes -> parameters_identity):
//     the two tolerances, the two dimensions, and the two PROVENANCE
//     discriminants. Model matrices F, H, Q, R are also parameters --
//     they say how to filter, not what was measured.
//   INPUTS (input payload -> input_identity):
//     the measurement stream, x0 and P0. The stream is the data; the
//     initial condition travels with it because it is a statement about
//     the run rather than about the model.
//
// THE THIRD AXIS (docs/SCL_CONTRACT.md 6.3): Q and R both participate and
// are indistinguishable on the annotating/participating axis, but R may be
// measurement-derived and Q never can be. The discriminant is in the
// CONFIGURATION so it reaches parameters_identity, and the source identity
// it guards is in the INPUT, because an identity of a supplied artifact is
// a reference to data.
//
// configuration: 40 bytes + F + H + Q + R, little-endian
//     offset  0  int32    state_dimension        n > 0
//     offset  4  int32    measurement_dimension  m > 0
//     offset  8  int32    process_provenance     MUST be 0 (asserted)
//     offset 12  int32    measurement_provenance 0 asserted, 1 supplied
//     offset 16  int32    reserved               must be 0
//     offset 20  int32    reserved2              must be 0
//     offset 24  float64  symmetry_tolerance     >= 0
//     offset 32  float64  psd_tolerance          >= 0
//     offset 40  float64  F[n*n], H[m*n], Q[n*n], R[m*m]  row-major, in order
//
// input:  int32 steps, int32 source_identity_length, then
//         n            float64  initial state x0
//         n*n          float64  initial covariance P0, row-major
//         steps*m      float64  measurements, row-major, ORDER SIGNIFICANT
//         source_identity_length bytes  UTF-8 -- GUARDED by the R
//                                       discriminant; must be 0 when R is
//                                       `asserted` (clause 2)
//
// output: steps * n float64 -- the posterior state trajectory
//
// ORDERING IS REQUIRED AND SIGNIFICANT for this modality, unlike
// least_squares where it is explicitly not. Two identical measurement
// multisets in different orders are DIFFERENT inputs here, and the
// row-major stream encoding is what makes that true of the bytes.

#include "scl/bytes.hpp"
#include "scl/kalman.hpp"
#include "scl/operation.hpp"
#include "scl/protocol.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

namespace scl {
namespace {

struct Configuration {
    KalmanParameters parameters;
    std::size_t n = 0;
    std::size_t m = 0;
    NoiseProvenance process_provenance = NoiseProvenance::kAsserted;
    NoiseProvenance measurement_provenance = NoiseProvenance::kAsserted;
    std::vector<double> F, H, Q, R;
};

NoiseProvenance decode_provenance(int32_t raw, const char* which) {
    if (raw == 0) return NoiseProvenance::kAsserted;
    if (raw == 1) return NoiseProvenance::kSupplied;
    std::ostringstream os;
    os << which << " provenance must be 0 (asserted) or 1 (supplied), got " << raw;
    throw OperationValidationError(os.str());
}

Configuration decode_configuration(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 40) {
        std::ostringstream os;
        os << "configuration must be at least 40 bytes (2 x int32 dimensions, "
              "2 x int32 provenance, 2 x int32 reserved, float64 symmetry_tolerance, "
              "float64 psd_tolerance) followed by F, H, Q and R, got " << bytes.size();
        throw OperationValidationError(os.str());
    }
    Configuration c;
    const int32_t n = read_int32_le(bytes, 0);
    const int32_t m = read_int32_le(bytes, 4);
    if (n <= 0) throw OperationValidationError(
        "state_dimension must be positive, got " + std::to_string(n));
    if (m <= 0) throw OperationValidationError(
        "measurement_dimension must be positive, got " + std::to_string(m));

    const int32_t process_raw = read_int32_le(bytes, 8);
    c.process_provenance = decode_provenance(process_raw, "process noise Q");
    c.measurement_provenance = decode_provenance(read_int32_le(bytes, 12), "measurement noise R");

    for (std::size_t offset : {std::size_t(16), std::size_t(20)}) {
        const int32_t reserved = read_int32_le(bytes, offset);
        if (reserved != 0) {
            std::ostringstream os;
            os << "reserved configuration word at offset " << offset << " must be 0, got "
               << reserved << " -- a reserved field that is read and ignored would let two "
                  "identical requests differ in identity";
            throw OperationValidationError(os.str());
        }
    }

    c.parameters.symmetry_tolerance = read_double_le(bytes, 24);
    c.parameters.psd_tolerance = read_double_le(bytes, 32);
    if (!(c.parameters.symmetry_tolerance >= 0.0) || !(c.parameters.psd_tolerance >= 0.0)) {
        throw OperationValidationError(
            "symmetry_tolerance and psd_tolerance must be finite and non-negative");
    }

    c.n = static_cast<std::size_t>(n);
    c.m = static_cast<std::size_t>(m);
    const std::size_t expected =
        40 + 8 * (c.n * c.n + c.m * c.n + c.n * c.n + c.m * c.m);
    if (bytes.size() != expected) {
        std::ostringstream os;
        os << "configuration must be exactly " << expected << " bytes for n=" << c.n
           << ", m=" << c.m << " (40 header + F " << (c.n * c.n) << " + H " << (c.m * c.n)
           << " + Q " << (c.n * c.n) << " + R " << (c.m * c.m) << " float64), got "
           << bytes.size();
        throw OperationValidationError(os.str());
    }

    std::size_t at = 40;
    auto take = [&](std::size_t count) {
        std::vector<double> out(count);
        for (std::size_t i = 0; i < count; ++i, at += 8) out[i] = read_double_le(bytes, at);
        return out;
    };
    c.F = take(c.n * c.n);
    c.H = take(c.m * c.n);
    c.Q = take(c.n * c.n);
    c.R = take(c.m * c.m);
    return c;
}

KalmanProblem decode_input(const std::vector<uint8_t>& bytes, const Configuration& c) {
    if (bytes.size() < 8) {
        throw OperationValidationError(
            "input must begin with int32 steps and int32 source_identity_length, got " +
            std::to_string(bytes.size()) + " bytes");
    }
    const int32_t steps = read_int32_le(bytes, 0);
    const int32_t id_len = read_int32_le(bytes, 4);
    if (steps <= 0) {
        throw OperationValidationError(
            "steps must be positive, got " + std::to_string(steps) +
            " -- an empty measurement stream is a validation fault, not an empty success");
    }
    if (id_len < 0) {
        throw OperationValidationError(
            "source_identity_length must be non-negative, got " + std::to_string(id_len));
    }

    KalmanProblem p;
    p.state_dimension = c.n;
    p.measurement_dimension = c.m;
    p.steps = static_cast<std::size_t>(steps);
    p.transition = c.F;
    p.observation = c.H;
    p.process_noise.dimension = c.n;
    p.process_noise.matrix = c.Q;
    p.process_noise.provenance = c.process_provenance;
    p.measurement_noise.dimension = c.m;
    p.measurement_noise.matrix = c.R;
    p.measurement_noise.provenance = c.measurement_provenance;

    const std::size_t doubles = c.n + c.n * c.n + p.steps * c.m;
    const std::size_t expected = 8 + 8 * doubles + static_cast<std::size_t>(id_len);
    if (bytes.size() != expected) {
        std::ostringstream os;
        os << "input must be exactly " << expected << " bytes (8 header + x0 " << c.n
           << " + P0 " << (c.n * c.n) << " + measurements " << (p.steps * c.m)
           << " float64 + " << id_len << " identity bytes), got " << bytes.size();
        throw OperationValidationError(os.str());
    }

    std::size_t at = 8;
    auto take = [&](std::size_t count) {
        std::vector<double> out(count);
        for (std::size_t i = 0; i < count; ++i, at += 8) out[i] = read_double_le(bytes, at);
        return out;
    };
    p.initial_state = take(c.n);
    p.initial_covariance = take(c.n * c.n);
    p.measurements = take(p.steps * c.m);
    p.measurement_noise.source_identity.assign(bytes.begin() + at, bytes.end());
    return p;
}

std::vector<uint8_t> encode_output(const std::vector<double>& states) {
    std::vector<uint8_t> out;
    out.reserve(states.size() * 8);
    for (double v : states) write_double_le(out, v);
    return out;
}

}  // namespace

OperationOutcome run_kalman_filter_linear(const OperationRequest& request) {
    try {
        std::string unavailable = backend_unavailable_reason(request.backend);
        if (!unavailable.empty()) {
            return OperationOutcome::halted(kFaultBackendUnavailable, unavailable);
        }

        const Configuration configuration = decode_configuration(request.configuration);
        const KalmanProblem problem = decode_input(request.input, configuration);

        const auto start = std::chrono::steady_clock::now();
        const KalmanResult result = run_kalman_filter(problem, configuration.parameters);
        const auto end = std::chrono::steady_clock::now();

        std::vector<Metric> metrics = {
            {"native_compute_seconds", std::chrono::duration<double>(end - start).count()},
            {"measurement_dimension", static_cast<double>(problem.measurement_dimension)},
            {"state_dimension", static_cast<double>(problem.state_dimension)},
            {"steps", static_cast<double>(result.steps)},
            // REPORTED, like the condition number for least_squares. A
            // posterior approaching singularity is a fact about the model
            // and the arithmetic, and the only thing that makes it visible
            // is publishing it beside the answer.
            {"smallest_posterior_eigenvalue", result.smallest_posterior_eigenvalue},
            // The provenance answer an auditor asks for, as a metric rather
            // than something to be reconstructed from the request bytes.
            {"measurement_noise_is_supplied",
             configuration.measurement_provenance == NoiseProvenance::kSupplied ? 1.0 : 0.0},
        };

        return OperationOutcome::completed(encode_output(result.states), metrics);
    } catch (const OperationValidationError& error) {
        return OperationOutcome::halted(kFaultValidation, error.what());
    } catch (const KalmanValidationError& error) {
        return OperationOutcome::halted(kFaultValidation, error.what());
    } catch (const std::exception& error) {
        return OperationOutcome::halted(kFaultComputation, error.what());
    }
}

}  // namespace scl

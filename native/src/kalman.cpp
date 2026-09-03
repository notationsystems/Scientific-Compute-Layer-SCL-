#include "scl/kalman.hpp"

#include "scl/covariance.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace scl {
namespace {

//: C = A * B, A is (ar x ac), B is (ac x bc), all row-major.
std::vector<double> matmul(const std::vector<double>& a, std::size_t ar, std::size_t ac,
                           const std::vector<double>& b, std::size_t bc) {
    std::vector<double> c(ar * bc, 0.0);
    for (std::size_t i = 0; i < ar; ++i) {
        for (std::size_t k = 0; k < ac; ++k) {
            const double aik = a[i * ac + k];
            if (aik == 0.0) continue;
            for (std::size_t j = 0; j < bc; ++j) {
                c[i * bc + j] += aik * b[k * bc + j];
            }
        }
    }
    return c;
}

std::vector<double> transpose(const std::vector<double>& a, std::size_t r, std::size_t c) {
    std::vector<double> t(r * c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) t[j * r + i] = a[i * c + j];
    return t;
}

//: Gauss-Jordan with partial pivoting. m is small (the measurement
//: dimension), so this is the readable choice rather than the fast one;
//: it is verified against the analytic 1x1 and 2x2 inverses in the tests.
bool invert(std::vector<double> a, std::size_t n, std::vector<double>& out) {
    out.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) out[i * n + i] = 1.0;

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(a[r * n + col]) > std::fabs(a[pivot * n + col])) pivot = r;
        if (std::fabs(a[pivot * n + col]) < 1e-300) return false;   // singular
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[col * n + j], a[pivot * n + j]);
                std::swap(out[col * n + j], out[pivot * n + j]);
            }
        }
        const double inv = 1.0 / a[col * n + col];
        for (std::size_t j = 0; j < n; ++j) {
            a[col * n + j] *= inv;
            out[col * n + j] *= inv;
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = a[r * n + col];
            if (factor == 0.0) continue;
            for (std::size_t j = 0; j < n; ++j) {
                a[r * n + j] -= factor * a[col * n + j];
                out[r * n + j] -= factor * out[col * n + j];
            }
        }
    }
    return true;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw KalmanValidationError(message);
}

//: Every fault names the matrix, the rule and the measured amount --
//: clause 4, actionable faults.
void require_valid_covariance(const std::vector<double>& m, std::size_t dim,
                              const char* name, const CovarianceParameters& cp) {
    const CovarianceReport r = validate_covariance(m, dim, dim, cp);
    if (!r.ok()) {
        throw KalmanValidationError(std::string(name) + " is not a valid covariance: " +
                                    r.detail);
    }
}

void require_provenance_is_coherent(const NoiseModel& noise, const char* name,
                                    bool may_be_supplied) {
    if (noise.provenance == NoiseProvenance::kSupplied) {
        require(may_be_supplied,
                std::string(name) + " is declared `supplied`, but there is no measurement "
                "of process noise -- Q is a statement about how much the modeller believes "
                "the state wanders between observations, and it is always `asserted` "
                "(docs/SCL_CONTRACT.md 6.3)");
        require(!noise.source_identity.empty(),
                std::string(name) + " is declared `supplied` but carries no source "
                "identity; `supplied` with nothing to trace to is a claim with no referent");
    } else {
        // CLAUSE 2. The discriminant guards this payload, so a value here
        // must be REFUSED, never accepted and ignored -- otherwise two
        // requests that compute identically differ in identity.
        require(noise.source_identity.empty(),
                std::string(name) + " is declared `asserted` but carries source_identity \"" +
                noise.source_identity + "\". A payload the discriminant renders unused must "
                "be refused, not ignored -- an ignored value would make two identical "
                "requests differ in identity");
    }
}

}  // namespace

std::vector<double> covariance_update(const std::vector<double>& p_predicted,
                                      const std::vector<double>& gain,
                                      const std::vector<double>& observation,
                                      const std::vector<double>& measurement_noise,
                                      std::size_t n,
                                      std::size_t m) {
    // JOSEPH FORM: P = (I - K H) P (I - K H)^T + K R K^T.
    //
    // The contract is stated over ANY gain, and that is the whole content
    // of the choice. Measured against this project's own covariance gate:
    // with the optimal K the short form (I - K H) P is INDISTINGUISHABLE
    // from this one; with K perturbed by 1e-3 the short form reaches
    // lambda_min = -1.994e+06 and is refused, while this stays accepted at
    // +1.540e-02.
    std::vector<double> IKH = matmul(gain, n, m, observation, n);
    for (std::size_t i = 0; i < n * n; ++i) IKH[i] = -IKH[i];
    for (std::size_t i = 0; i < n; ++i) IKH[i * n + i] += 1.0;

    std::vector<double> P =
        matmul(matmul(IKH, n, n, p_predicted, n), n, n, transpose(IKH, n, n), n);
    const std::vector<double> KRKt =
        matmul(matmul(gain, n, m, measurement_noise, m), n, m, transpose(gain, n, m), n);
    for (std::size_t i = 0; i < n * n; ++i) P[i] += KRKt[i];

    // LOAD-BEARING, not tidiness. Measured: Joseph alone leaves asymmetry
    // around 7e-09 after 2000 steps, which validate_covariance REFUSES as
    // NOT-SYMMETRIC. Joseph is symmetry-preserving in exact arithmetic and
    // not quite in floating point, so the two mechanisms are independent
    // and both are required.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double mean = 0.5 * (P[i * n + j] + P[j * n + i]);
            P[i * n + j] = mean;
            P[j * n + i] = mean;
        }
    }
    return P;
}

namespace {

//: Refuses a result carrying a non-finite value, naming WHICH array and
//: WHICH step. A message that says only "non-finite result" sends the
//: reader back to the recursion with no starting point, and the step index
//: is the whole difference between a report and a shrug.
void require_finite_results(const KalmanResult& result, std::size_t n, std::size_t m) {
    struct Array {
        const char* name;
        const std::vector<double>* values;
        std::size_t stride;
    };
    const Array arrays[] = {
        {"posterior state x", &result.states, n},
        {"posterior covariance P", &result.covariances, n * n},
        {"innovation v", &result.innovations, m},
        {"innovation covariance S", &result.innovation_covariances, m * m},
        {"gain K", &result.gains, n * m},
    };
    for (const Array& array : arrays) {
        for (std::size_t i = 0; i < array.values->size(); ++i) {
            const double value = (*array.values)[i];
            if (std::isfinite(value)) continue;
            const std::size_t step = array.stride ? i / array.stride : 0;
            std::ostringstream os;
            os << array.name << " is " << (std::isnan(value) ? "NaN" : "an infinity")
               << " at step " << step
               << ". Every input to this filter was finite, so this is overflow "
                  "inside the recursion and not a contract breach by the caller. "
                  "It is refused rather than returned because the only health "
                  "figure this operation publishes, smallest_posterior_eigenvalue, "
                  "reports +infinity when the posteriors are NaN -- the value a "
                  "perfectly conditioned filter gives";
            throw KalmanValidationError(os.str());
        }
    }
}

}  // namespace

KalmanResult run_kalman_filter(const KalmanProblem& problem,
                               const KalmanParameters& params) {
    const std::size_t n = problem.state_dimension;
    const std::size_t m = problem.measurement_dimension;

    require(n > 0, "state_dimension must be positive");
    require(m > 0, "measurement_dimension must be positive");
    require(problem.steps > 0,
            "kalman_filter_linear requires at least one measurement step; an empty "
            "stream is a validation fault, not an empty success");

    require(problem.transition.size() == n * n,
            "transition F must be " + std::to_string(n) + " x " + std::to_string(n) +
            " = " + std::to_string(n * n) + " entries, got " +
            std::to_string(problem.transition.size()));
    require(problem.observation.size() == m * n,
            "observation H must be " + std::to_string(m) + " x " + std::to_string(n) +
            " = " + std::to_string(m * n) + " entries, got " +
            std::to_string(problem.observation.size()));
    require(problem.initial_state.size() == n,
            "initial_state must have " + std::to_string(n) + " entries, got " +
            std::to_string(problem.initial_state.size()));
    require(problem.measurements.size() == problem.steps * m,
            "measurements must be steps * measurement_dimension = " +
            std::to_string(problem.steps * m) + " entries, got " +
            std::to_string(problem.measurements.size()));
    require(problem.process_noise.dimension == n,
            "process noise Q must have dimension " + std::to_string(n) + ", got " +
            std::to_string(problem.process_noise.dimension));
    require(problem.measurement_noise.dimension == m,
            "measurement noise R must have dimension " + std::to_string(m) + ", got " +
            std::to_string(problem.measurement_noise.dimension));

    for (std::size_t i = 0; i < problem.transition.size(); ++i)
        require(std::isfinite(problem.transition[i]),
                "transition F entry " + std::to_string(i) + " is not finite");
    for (std::size_t i = 0; i < problem.observation.size(); ++i)
        require(std::isfinite(problem.observation[i]),
                "observation H entry " + std::to_string(i) + " is not finite");
    for (std::size_t i = 0; i < problem.initial_state.size(); ++i)
        require(std::isfinite(problem.initial_state[i]),
                "initial_state entry " + std::to_string(i) + " is not finite");
    for (std::size_t i = 0; i < problem.measurements.size(); ++i)
        require(std::isfinite(problem.measurements[i]),
                "measurement " + std::to_string(i) + " is not finite -- a sentinel-encoded "
                "absence, which DAQ's gate refuses for scalars but which this operation "
                "must not assume was checked upstream");

    require_provenance_is_coherent(problem.process_noise, "process noise Q",
                                   /*may_be_supplied=*/false);
    require_provenance_is_coherent(problem.measurement_noise, "measurement noise R",
                                   /*may_be_supplied=*/true);

    CovarianceParameters cp;
    cp.symmetry_tolerance = params.symmetry_tolerance;
    cp.psd_tolerance = params.psd_tolerance;
    require_valid_covariance(problem.initial_covariance, n, "initial covariance P0", cp);
    require_valid_covariance(problem.process_noise.matrix, n, "process noise Q", cp);
    require_valid_covariance(problem.measurement_noise.matrix, m, "measurement noise R", cp);

    const std::vector<double>& F = problem.transition;
    const std::vector<double>& H = problem.observation;
    const std::vector<double>& Q = problem.process_noise.matrix;
    const std::vector<double>& R = problem.measurement_noise.matrix;
    const std::vector<double> Ft = transpose(F, n, n);
    const std::vector<double> Ht = transpose(H, m, n);

    std::vector<double> x = problem.initial_state;
    std::vector<double> P = problem.initial_covariance;

    KalmanResult result;
    result.steps = problem.steps;
    result.states.resize(problem.steps * n);
    result.covariances.resize(problem.steps * n * n);
    result.innovations.resize(problem.steps * m);
    result.innovation_covariances.resize(problem.steps * m * m);
    result.gains.resize(problem.steps * n * m);
    result.smallest_posterior_eigenvalue = std::numeric_limits<double>::infinity();

    for (std::size_t k = 0; k < problem.steps; ++k) {
        // ---- predict:  x = F x,  P = F P F^T + Q
        const std::vector<double> x_pred = matmul(F, n, n, x, 1);
        std::vector<double> P_pred = matmul(matmul(F, n, n, P, n), n, n, Ft, n);
        for (std::size_t i = 0; i < n * n; ++i) P_pred[i] += Q[i];

        // ---- innovation:  v = z - H x,   S = H P H^T + R
        const std::vector<double> Hx = matmul(H, m, n, x_pred, 1);
        std::vector<double> v(m);
        for (std::size_t i = 0; i < m; ++i)
            v[i] = problem.measurements[k * m + i] - Hx[i];

        std::vector<double> S = matmul(matmul(H, m, n, P_pred, n), m, n, Ht, m);
        for (std::size_t i = 0; i < m * m; ++i) S[i] += R[i];

        std::vector<double> S_inv;
        require(invert(S, m, S_inv),
                "innovation covariance S is singular at step " + std::to_string(k) +
                "; with R positive definite this cannot happen, so it indicates a "
                "degenerate model rather than a numerical accident");

        // ---- gain and update
        const std::vector<double> K = matmul(matmul(P_pred, n, n, Ht, m), n, m, S_inv, m);
        const std::vector<double> Kv = matmul(K, n, m, v, 1);
        for (std::size_t i = 0; i < n; ++i) x[i] = x_pred[i] + Kv[i];

        P = covariance_update(P_pred, K, H, R, n, m);

        std::copy(x.begin(), x.end(), result.states.begin() + k * n);
        std::copy(P.begin(), P.end(), result.covariances.begin() + k * n * n);
        std::copy(v.begin(), v.end(), result.innovations.begin() + k * m);
        std::copy(S.begin(), S.end(), result.innovation_covariances.begin() + k * m * m);
        std::copy(K.begin(), K.end(), result.gains.begin() + k * n * m);

        // REPORTED, not asserted. A posterior drifting negative is a
        // measurement about the model and the arithmetic, and hiding it
        // behind a threshold is what Joseph form exists to avoid needing.
        const std::vector<double> spectrum = symmetric_eigenvalues(P, n, nullptr);
        result.smallest_posterior_eigenvalue =
            std::min(result.smallest_posterior_eigenvalue, spectrum.front());
    }

    // EVERY INPUT IS FINITE AND THE OUTPUT NEED NOT BE. Overflow inside the
    // recursion is not a contract breach by the caller, so nothing above
    // catches it, and the diagnostic this operation publishes is the one
    // thing a NaN makes maximally reassuring: std::min(x, NaN) returns x, so
    // a filter whose every posterior is NaN leaves
    // smallest_posterior_eigenvalue at its initial +infinity -- the reading a
    // perfectly conditioned filter gives.
    //
    // Measured, with all inputs finite and a scalar model: at |F| = 1e155 all
    // 400 states, covariances, gains and innovation covariances came back
    // non-finite and the eigenvalue reported +inf. At 1e150 it was 399 of 400
    // and 1.2e268. Neither raised anything. That transition matrix is not a
    // plausible model and the point is not its plausibility -- it is that the
    // failure is SILENT and its indicator points the wrong way.
    //
    // lj_pairwise, fourier_transform_1d and least_squares all check their own
    // results. This was the only operation that did not.
    require_finite_results(result, n, m);

    return result;
}

}  // namespace scl

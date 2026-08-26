// THE ANALYTIC ANCHORS from
// architecture/kalman_validation_preregistration.yaml.
//
// The three innovation statistics check CONSISTENCY -- that the filter
// agrees with its own predictions. They are necessary and not sufficient:
// a filter can be self-consistent and still wrong. These check
// CORRECTNESS against closed form, with no simulation involved, so they
// separate an implementation error from a modelling one before any
// statistic is run.

#include "scl/kalman.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0, g_checks = 0;
void check(bool c, const char* e, const char* f, int l) {
    ++g_checks;
    if (!c) { ++g_failures; std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", e, f, l); }
}
#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)
bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

scl::KalmanProblem scalar_problem(double q, double r, double p0, std::size_t steps) {
    scl::KalmanProblem p;
    p.state_dimension = 1;
    p.measurement_dimension = 1;
    p.transition = {1.0};
    p.observation = {1.0};
    p.process_noise.dimension = 1;
    p.process_noise.matrix = {q};
    p.measurement_noise.dimension = 1;
    p.measurement_noise.matrix = {r};
    p.initial_state = {0.0};
    p.initial_covariance = {p0};
    p.steps = steps;
    p.measurements.assign(steps, 0.0);
    return p;
}

// ANCHOR 1. Steady-state gain, tolerance 1.0e-9.
//
// The recursion P- = P+ + Q, K = P-/(P-+R), P+ = (1-K)P- has a fixed
// point. Substituting P+ = P- - Q into P+ = P- R/(P- + R) gives
//
//     P-^2 - Q P- - Q R = 0   =>   P- = (Q + sqrt(Q^2 + 4 Q R)) / 2
//
// P HERE IS THE PRIOR, and that is worth stating because the posterior is
// the natural misreading: the posterior fixed point solves a DIFFERENT
// quadratic, P+^2 + Q P+ - Q R = 0. Verified by independent iteration
// before this test was written -- for Q = 0.01, R = 0.25 the recursion
// converges to prior 0.055249378105604 and posterior 0.045249378105604,
// and the closed form above returns the prior.
void test_steady_state_gain_matches_closed_form() {
    const double q = 0.01, r = 0.25, tol = 1.0e-9;
    const double prior_ss = (q + std::sqrt(q * q + 4.0 * q * r)) / 2.0;
    const double gain_ss = prior_ss / (prior_ss + r);

    const scl::KalmanResult res =
        scl::run_kalman_filter(scalar_problem(q, r, 1.0, 4000), scl::KalmanParameters{});

    CHECK(close(res.gains.back(), gain_ss, tol));
    // the posterior converges to the OTHER root, which is what makes the
    // prior/posterior distinction a real check rather than a comment
    const double posterior_ss = (-q + std::sqrt(q * q + 4.0 * q * r)) / 2.0;
    CHECK(close(res.covariances.back(), posterior_ss, tol));
    CHECK(!close(res.covariances.back(), prior_ss, 1e-3));

    // and S = H P- H^T + R = P- + R at steady state
    CHECK(close(res.innovation_covariances.back(), prior_ss + r, tol));
}

// ANCHOR 2. Zero process noise, tolerance 1.0e-9.
//
// With Q = 0 the information adds: 1/P_k = 1/P_0 + k/R, so
// P_k = P_0 R / (R + k P_0) exactly, and the gain falls like 1/k.
void test_zero_process_noise_converges_as_one_over_k() {
    const double r = 0.25, p0 = 1.0, tol = 1.0e-9;
    const std::size_t steps = 500;
    const scl::KalmanResult res =
        scl::run_kalman_filter(scalar_problem(0.0, r, p0, steps), scl::KalmanParameters{});

    for (std::size_t k : {std::size_t(1), std::size_t(10), std::size_t(100), steps}) {
        const double expected = p0 * r / (r + static_cast<double>(k) * p0);
        CHECK(close(res.covariances[k - 1], expected, tol));
    }
    // the estimate is the sample mean of the measurements: with all
    // measurements zero it stays exactly zero, and the gain decays
    CHECK(close(res.states.back(), 0.0, tol));
    CHECK(res.gains.back() < res.gains.front());
    CHECK(close(res.gains.back(), p0 / (r + static_cast<double>(steps) * p0), 1e-6));
}

// ANCHOR 2b. The same case driven by a NON-zero constant measurement, so
// "converges to the sample mean" is actually exercised rather than
// satisfied by everything being zero.
void test_zero_process_noise_estimate_is_the_sample_mean() {
    const double r = 0.25, p0 = 1e12, tol = 1e-6;   // diffuse prior
    const std::size_t steps = 200;
    scl::KalmanProblem p = scalar_problem(0.0, r, p0, steps);
    double running = 0.0;
    for (std::size_t k = 0; k < steps; ++k) {
        p.measurements[k] = 1.0 + 0.5 * std::sin(static_cast<double>(k));
        running += p.measurements[k];
    }
    const scl::KalmanResult res = scl::run_kalman_filter(p, scl::KalmanParameters{});
    CHECK(close(res.states.back(), running / static_cast<double>(steps), tol));
}

// ANCHOR 3. Worthless measurements, tolerance 1.0e-6.
//
// As R grows the gain goes to zero and the filter follows its prediction
// -- it ignores a measurement it has been told carries no information.
void test_a_worthless_measurement_is_ignored() {
    const double tol = 1.0e-6;
    scl::KalmanProblem p = scalar_problem(1e-12, 1e12, 1.0, 50);
    for (std::size_t k = 0; k < p.steps; ++k) p.measurements[k] = 1000.0;
    const scl::KalmanResult res = scl::run_kalman_filter(p, scl::KalmanParameters{});
    CHECK(res.gains.back() < tol);
    CHECK(close(res.states.back(), 0.0, 1e-3));   // never moved toward 1000
}

// --- THE CONTRACT FAULTS, each refused by name ---------------------------
void test_the_contract_faults() {
    scl::KalmanParameters params;

    auto refused = [&](scl::KalmanProblem p, const char* fragment) {
        try {
            scl::run_kalman_filter(p, params);
            return false;
        } catch (const scl::KalmanValidationError& e) {
            return std::string(e.what()).find(fragment) != std::string::npos;
        }
    };

    // empty stream is a fault, not an empty success (clause 3)
    scl::KalmanProblem empty = scalar_problem(0.01, 0.25, 1.0, 0);
    CHECK(refused(empty, "at least one measurement"));

    // an indefinite R
    scl::KalmanProblem bad_r = scalar_problem(0.01, -1.0, 1.0, 5);
    CHECK(refused(bad_r, "not positive semidefinite"));

    // a non-finite measurement -- DAQ's gate refuses this for scalars, and
    // this operation does not assume that was checked upstream
    scl::KalmanProblem nan_z = scalar_problem(0.01, 0.25, 1.0, 5);
    nan_z.measurements[2] = std::nan("");
    CHECK(refused(nan_z, "not finite"));

    // 6.3: Q may never be `supplied`
    scl::KalmanProblem supplied_q = scalar_problem(0.01, 0.25, 1.0, 5);
    supplied_q.process_noise.provenance = scl::NoiseProvenance::kSupplied;
    supplied_q.process_noise.source_identity = "sha256:deadbeef";
    CHECK(refused(supplied_q, "no measurement of process noise"));

    // CLAUSE 2: a payload the discriminant renders unused is REFUSED
    scl::KalmanProblem ignored_payload = scalar_problem(0.01, 0.25, 1.0, 5);
    ignored_payload.measurement_noise.provenance = scl::NoiseProvenance::kAsserted;
    ignored_payload.measurement_noise.source_identity = "sha256:deadbeef";
    CHECK(refused(ignored_payload, "must be refused, not ignored"));

    // and `supplied` with nothing to trace to
    scl::KalmanProblem empty_source = scalar_problem(0.01, 0.25, 1.0, 5);
    empty_source.measurement_noise.provenance = scl::NoiseProvenance::kSupplied;
    CHECK(refused(empty_source, "no source identity"));

    // R legitimately supplied WITH a source is accepted
    scl::KalmanProblem good = scalar_problem(0.01, 0.25, 1.0, 5);
    good.measurement_noise.provenance = scl::NoiseProvenance::kSupplied;
    good.measurement_noise.source_identity = "sha256:abc123";
    bool accepted = true;
    try { scl::run_kalman_filter(good, params); } catch (...) { accepted = false; }
    CHECK(accepted);
}

// --- THE JOSEPH FORM IS ENFORCED, NOT MERELY CHOSEN ----------------------
//
// Replacing Joseph with the short form P = (I - K H) P survived every
// other test here, because none of them ran a regime where the two
// differ. Measured: in benign conditions they are bit-identical; with a
// diffuse prior and a near-exact measurement the short form drives
// lambda_min to EXACTLY ZERO -- the covariance goes singular -- while
// Joseph retains a small positive value.
//
// A singular P means the filter believes it knows one direction of the
// state perfectly; the gain in that direction is then identically zero and
// no later measurement can correct it. Silent and permanent.
void test_the_joseph_form_keeps_the_covariance_nonsingular() {
    const std::size_t n = 2, steps = 5000;
    scl::KalmanProblem p;
    p.state_dimension = n;
    p.measurement_dimension = 1;
    p.transition = {1.0, 1.0,
                    0.0, 1.0};
    p.observation = {1.0, 0.0};
    p.process_noise.dimension = n;
    p.process_noise.matrix = {0.0, 0.0, 0.0, 0.0};      // q = 0
    p.measurement_noise.dimension = 1;
    p.measurement_noise.matrix = {1e-16};               // near-exact measurement
    p.initial_state = {0.0, 0.0};
    p.initial_covariance = {1e10, 0.0, 0.0, 1e10};      // diffuse prior
    p.steps = steps;
    p.measurements.assign(steps, 0.0);

    const scl::KalmanResult r = scl::run_kalman_filter(p, scl::KalmanParameters{});

    // The discriminating assertion: STRICTLY positive. The short form
    // reaches exactly 0.0 here; Joseph holds ~2.4e-27.
    CHECK(r.smallest_posterior_eigenvalue > 0.0);
    CHECK(r.smallest_posterior_eigenvalue < 1e-20);   // and it IS this regime
}

}  // namespace

int main() {
    test_steady_state_gain_matches_closed_form();
    test_zero_process_noise_converges_as_one_over_k();
    test_zero_process_noise_estimate_is_the_sample_mean();
    test_a_worthless_measurement_is_ignored();
    test_the_joseph_form_keeps_the_covariance_nonsingular();
    test_the_contract_faults();
    std::printf("kalman analytic: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

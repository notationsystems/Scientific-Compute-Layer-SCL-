// THE ANALYTIC ANCHORS from
// architecture/kalman_validation_preregistration.yaml.
//
// The three innovation statistics check CONSISTENCY -- that the filter
// agrees with its own predictions. They are necessary and not sufficient:
// a filter can be self-consistent and still wrong. These check
// CORRECTNESS against closed form, with no simulation involved, so they
// separate an implementation error from a modelling one before any
// statistic is run.

#include "scl/covariance.hpp"
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

// --- THE COVARIANCE UPDATE'S CONTRACT, STATED OVER ANY GAIN --------------
//
// THE FIRST VERSION OF THIS TEST WAS WEAK AND IS REPLACED. It asserted
// that lambda_min stays strictly positive in one extreme regime -- which
// worked, but only because the short form underflowed to exactly 0.0
// there. That is a discriminator that happens to fire, not the property.
// Re-measured on a 3-state model at optimal K, the two forms are
// IDENTICAL to every printed digit, so the old test was pinning an
// accident of underflow.
//
// The real guarantee is:
//
//     FOR ANY K, the result is a valid covariance whenever P_pred is.
//
// Joseph holds it; (I - K H) P holds it only when K is exactly optimal.
// Stated over the property rather than over the implementation: a later
// equivalent form is free to replace Joseph, and this test should still
// pass. It does NOT assert "Joseph is used".
//
// Judged by validate_covariance -- this project's own gate -- rather than
// by a threshold invented here.
void test_the_covariance_update_holds_for_any_gain() {
    const std::size_t n = 3, m = 1;
    const std::vector<double> H = {1.0, 0.0, 0.0};
    const std::vector<double> R = {0.25};
    const std::vector<double> P_pred = {4.0, 1.0, 0.5,
                                        1.0, 3.0, 0.25,
                                        0.5, 0.25, 2.0};
    const scl::CovarianceParameters cp;
    CHECK(scl::validate_covariance(P_pred, n, n, cp).ok());   // the premise

    // the optimal gain for this P_pred, computed here rather than taken
    // from the filter, so the test does not inherit what it is checking
    const double S = P_pred[0] + R[0];
    const std::vector<double> K_opt = {P_pred[0] / S, P_pred[3] / S, P_pred[6] / S};

    // THE OPTIMAL GAIN IS THE EASY CASE, and every update form passes it.
    CHECK(scl::validate_covariance(
              scl::covariance_update(P_pred, K_opt, H, R, n, m), n, n, cp).ok());

    // ANY OTHER GAIN is the contract. Scaled, zero, doubled, negative --
    // a suboptimal gain is a worse ESTIMATE, never an invalid covariance.
    for (double factor : {0.0, 0.5, 0.999, 1.001, 1.5, 2.0, -1.0, 10.0}) {
        std::vector<double> K = K_opt;
        for (double& k : K) k *= factor;
        const std::vector<double> P = scl::covariance_update(P_pred, K, H, R, n, m);
        const scl::CovarianceReport rep = scl::validate_covariance(P, n, n, cp);
        CHECK(rep.ok());
        if (!rep.ok()) std::fprintf(stderr, "  gain factor %g: %s\n", factor,
                                    rep.detail.c_str());
    }
}

// The same property under ITERATION, which is where a defect accumulates
// rather than appearing at once. A gain perturbed by 1e-3 drives the short
// form to lambda_min = -1.994e+06 over 2000 steps; the contract above must
// keep the covariance inside the gate for the whole run.
void test_the_contract_survives_two_thousand_perturbed_steps() {
    const std::size_t n = 3, m = 1;
    const std::vector<double> H = {1.0, 0.0, 0.0};
    const std::vector<double> R = {0.25};
    const std::vector<double> F = {1.0, 1.0, 0.5,
                                   0.0, 1.0, 1.0,
                                   0.0, 0.0, 1.0};
    std::vector<double> P(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) P[i * n + i] = 1e8;
    const scl::CovarianceParameters cp;

    double worst = 1e300;
    for (int step = 0; step < 2000; ++step) {
        // predict, with a small Q so the recursion does not degenerate
        std::vector<double> Pp(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = 0; b < n; ++b)
                        Pp[i * n + j] += F[i * n + a] * P[a * n + b] * F[j * n + b];
        for (std::size_t i = 0; i < n; ++i) Pp[i * n + i] += 0.01;

        const double S = Pp[0] + R[0];
        // DELIBERATELY SUBOPTIMAL by 1e-3 -- the regime the operation
        // itself cannot reach, and exactly why this is tested here on the
        // function rather than through the operation.
        std::vector<double> K = {Pp[0] / S * 1.001, Pp[3] / S * 1.001, Pp[6] / S * 1.001};
        P = scl::covariance_update(Pp, K, H, R, n, m);

        const scl::CovarianceReport rep = scl::validate_covariance(P, n, n, cp);
        CHECK(rep.ok());
        if (!rep.ok()) {
            std::fprintf(stderr, "  step %d: %s\n", step, rep.detail.c_str());
            break;
        }
        worst = std::min(worst, rep.smallest_eigenvalue);
    }
    CHECK(worst > 0.0);
}

// --- AND THE SAME CONTRACT AT THE OPTIMAL GAIN, WHICH IS A DIFFERENT
// --- REGIME AND CATCHES A DIFFERENT MECHANISM.
//
// TWO MECHANISMS, TWO REGIMES, and this was found by mutation rather than
// by design. Removing the explicit symmetrisation survived the perturbed
// -gain test above: with a suboptimal gain P stays large, so the RELATIVE
// symmetry budget (1e-10 x scale) is loose and roundoff asymmetry fits
// inside it.
//
// At the OPTIMAL gain the covariance converges to small values, the budget
// tightens with it, and asymmetry of 1.41e-08 is refused. Measured both
// ways through the real function:
//
//     with symmetrisation      max_asym 0.000e+00   accepted
//     without                  max_asym 1.411e-08   NOT-SYMMETRIC
//
// So Joseph and the symmetrisation are independent: Joseph holds PSD under
// any gain, symmetrisation holds symmetry under the optimal one, and
// neither substitutes for the other. Stated as the same property -- the
// result is a valid covariance -- in the regime that can see it.
void test_the_contract_also_holds_at_the_optimal_gain_over_many_steps() {
    const std::size_t n = 3, m = 1;
    const std::vector<double> H = {1.0, 0.0, 0.0};
    const std::vector<double> R = {0.25};
    const std::vector<double> F = {1.0, 1.0, 0.5,
                                   0.0, 1.0, 1.0,
                                   0.0, 0.0, 1.0};
    std::vector<double> P(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) P[i * n + i] = 1e8;
    const scl::CovarianceParameters cp;

    double worst_asymmetry = 0.0;
    for (int step = 0; step < 2000; ++step) {
        std::vector<double> Pp(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = 0; b < n; ++b)
                        Pp[i * n + j] += F[i * n + a] * P[a * n + b] * F[j * n + b];
        for (std::size_t i = 0; i < n; ++i) Pp[i * n + i] += 0.01;

        const double S = Pp[0] + R[0];
        const std::vector<double> K = {Pp[0] / S, Pp[3] / S, Pp[6] / S};   // OPTIMAL
        P = scl::covariance_update(Pp, K, H, R, n, m);

        const scl::CovarianceReport rep = scl::validate_covariance(P, n, n, cp);
        CHECK(rep.ok());
        if (!rep.ok()) {
            std::fprintf(stderr, "  optimal-gain step %d: %s\n", step, rep.detail.c_str());
            break;
        }
        worst_asymmetry = std::max(worst_asymmetry, rep.max_asymmetry);
    }
    // exactly symmetric, not merely within budget -- the mechanism is a
    // forced average, so anything else means it did not run
    CHECK(worst_asymmetry == 0.0);
}

}  // namespace

int main() {
    test_steady_state_gain_matches_closed_form();
    test_zero_process_noise_converges_as_one_over_k();
    test_zero_process_noise_estimate_is_the_sample_mean();
    test_a_worthless_measurement_is_ignored();
    test_the_covariance_update_holds_for_any_gain();
    test_the_contract_survives_two_thousand_perturbed_steps();
    test_the_contract_also_holds_at_the_optimal_gain_over_many_steps();
    test_the_contract_faults();
    std::printf("kalman analytic: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

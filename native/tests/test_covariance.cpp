// The covariance contract, validated against INDEPENDENT mathematics.
//
// Never against this implementation's own output. Every expected value
// below comes from an analytic eigenvalue formula or from an invariant of
// the spectrum (trace, determinant) that holds regardless of how the
// eigenvalues were computed.

#include "scl/covariance.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", expr, file, line);
    }
}
#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// --- ORACLE 1: analytic --------------------------------------------------
// [[a, b], [b, a]] has eigenvalues a+b and a-b, exactly. Derived by hand,
// not by running the solver.
void test_analytic_two_by_two() {
    const double a = 2.0, b = 3.0;
    const std::vector<double> m = {a, b, b, a};
    int sweeps = 0;
    const std::vector<double> e = scl::symmetric_eigenvalues(m, 2, &sweeps);
    CHECK(e.size() == 2);
    CHECK(close(e[0], a - b, 1e-12));   // -1
    CHECK(close(e[1], a + b, 1e-12));   //  5
    CHECK(sweeps > 0);
}

// --- ORACLE 2: a diagonal matrix IS its own spectrum ----------------------
void test_diagonal_is_its_own_spectrum() {
    const std::vector<double> m = {4.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0, 0.0, 9.0};
    const std::vector<double> e = scl::symmetric_eigenvalues(m, 3, nullptr);
    CHECK(close(e[0], 1.0, 1e-12));
    CHECK(close(e[1], 4.0, 1e-12));
    CHECK(close(e[2], 9.0, 1e-12));
}

// --- ORACLE 3: spectral invariants ---------------------------------------
// sum(lambda) == trace and prod(lambda) == det, for ANY correct solver.
void test_trace_and_determinant_invariants() {
    const std::vector<double> m = {6.0, 2.0, 1.0,
                                   2.0, 5.0, 3.0,
                                   1.0, 3.0, 7.0};
    const std::vector<double> e = scl::symmetric_eigenvalues(m, 3, nullptr);

    const double trace = m[0] + m[4] + m[8];
    const double sum = std::accumulate(e.begin(), e.end(), 0.0);
    CHECK(close(sum, trace, 1e-10));

    // determinant by cofactor expansion, computed independently
    const double det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
    const double product = e[0] * e[1] * e[2];
    CHECK(close(product, det, 1e-9));
}

// --- THE MEASUREMENT THAT CHOSE THE SOLVER -------------------------------
// A and B have IDENTICAL singular spectra and opposite PSD verdicts. This
// is the case an SVD-based check cannot answer, and the reason this file
// uses a symmetric eigensolver. If it ever passes for both, the solver has
// silently become sign-blind.
void test_the_case_an_svd_cannot_distinguish() {
    scl::CovarianceParameters p;
    const std::vector<double> A = {2.0, 3.0, 3.0, 2.0};   // eigenvalues -1, 5
    const std::vector<double> B = {3.0, 2.0, 2.0, 3.0};   // eigenvalues  1, 5

    const scl::CovarianceReport ra = scl::validate_covariance(A, 2, 2, p);
    const scl::CovarianceReport rb = scl::validate_covariance(B, 2, 2, p);

    CHECK(!ra.ok());
    CHECK(ra.fault == scl::CovarianceFault::kNotPositiveSemidefinite);
    CHECK(rb.ok());

    // identical singular magnitudes, which is what an SVD would have seen
    CHECK(close(std::fabs(ra.smallest_eigenvalue), std::fabs(rb.smallest_eigenvalue), 1e-12));
    CHECK(close(std::fabs(ra.largest_eigenvalue), std::fabs(rb.largest_eigenvalue), 1e-12));
    // and opposite signs, which is what the eigensolver sees and it does not
    CHECK(ra.smallest_eigenvalue < 0.0);
    CHECK(rb.smallest_eigenvalue > 0.0);
}

// --- THE FIVE RULES, each refused by its own fault ------------------------
void test_each_rule_has_its_own_fault_code() {
    scl::CovarianceParameters p;

    // 2. ragged: declared shape does not match the element count
    CHECK(scl::validate_covariance({1.0, 0.0, 0.0}, 2, 2, p).fault ==
          scl::CovarianceFault::kNotRectangular);

    // 3. rectangular but not square
    CHECK(scl::validate_covariance({1.0, 0.0, 0.0, 1.0, 0.0, 0.0}, 2, 3, p).fault ==
          scl::CovarianceFault::kNotSquare);

    // empty
    CHECK(scl::validate_covariance({}, 0, 0, p).fault == scl::CovarianceFault::kEmpty);

    // 1. non-finite entry
    const double nan = std::nan("");
    CHECK(scl::validate_covariance({1.0, 0.0, 0.0, nan}, 2, 2, p).fault ==
          scl::CovarianceFault::kNonFiniteEntry);
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(scl::validate_covariance({1.0, 0.0, 0.0, inf}, 2, 2, p).fault ==
          scl::CovarianceFault::kNonFiniteEntry);

    // 4. asymmetric
    CHECK(scl::validate_covariance({1.0, 0.5, 0.4, 1.0}, 2, 2, p).fault ==
          scl::CovarianceFault::kNotSymmetric);

    // 5. not PSD
    CHECK(scl::validate_covariance({2.0, 3.0, 3.0, 2.0}, 2, 2, p).fault ==
          scl::CovarianceFault::kNotPositiveSemidefinite);

    // and a legitimate covariance passes all five
    CHECK(scl::validate_covariance({4.0, 1.0, 1.0, 4.0}, 2, 2, p).ok());
}

// --- THE ORDER OF THE RULES IS PART OF THE CONTRACT ----------------------
// A matrix that breaks several rules must report the FIRST one, because a
// spectrum computed over a ragged or asymmetric matrix is a number that
// means nothing.
void test_shape_is_reported_before_entries_before_symmetry_before_psd() {
    scl::CovarianceParameters p;
    const double nan = std::nan("");

    // ragged AND non-finite -> ragged
    CHECK(scl::validate_covariance({1.0, nan, 0.0}, 2, 2, p).fault ==
          scl::CovarianceFault::kNotRectangular);
    // non-finite AND asymmetric -> non-finite
    CHECK(scl::validate_covariance({1.0, 0.5, 0.4, nan}, 2, 2, p).fault ==
          scl::CovarianceFault::kNonFiniteEntry);
    // asymmetric AND (would be) non-PSD -> asymmetric
    const scl::CovarianceReport r =
        scl::validate_covariance({2.0, 3.0, 2.5, 2.0}, 2, 2, p);
    CHECK(r.fault == scl::CovarianceFault::kNotSymmetric);
    CHECK(r.eigenvalues.empty());   // never computed, so never reported
}

// --- TOLERANCES ARE PARAMETERS, AND THEY CHANGE THE ANSWER ---------------
void test_both_tolerances_change_which_inputs_are_accepted() {
    // symmetric to 1e-9, which a strict check rejects and a looser one admits
    const std::vector<double> nearly = {4.0, 1.0, 1.0 + 1e-9, 4.0};
    scl::CovarianceParameters strict;
    strict.symmetry_tolerance = 1e-12;
    scl::CovarianceParameters loose;
    loose.symmetry_tolerance = 1e-6;
    CHECK(scl::validate_covariance(nearly, 2, 2, strict).fault ==
          scl::CovarianceFault::kNotSymmetric);
    CHECK(scl::validate_covariance(nearly, 2, 2, loose).ok());

    // PSD by -1e-9, the roundoff case a real measured covariance produces
    const std::vector<double> grazing = {1.0 - 1e-9, 1.0, 1.0, 1.0};
    scl::CovarianceParameters psd_strict;
    psd_strict.psd_tolerance = 0.0;
    scl::CovarianceParameters psd_loose;
    psd_loose.psd_tolerance = 1e-6;
    CHECK(scl::validate_covariance(grazing, 2, 2, psd_strict).fault ==
          scl::CovarianceFault::kNotPositiveSemidefinite);
    CHECK(scl::validate_covariance(grazing, 2, 2, psd_loose).ok());
}

// --- SCALE INVARIANCE ----------------------------------------------------
// The same covariance in metres and in millimetres must get the same
// verdict. An ABSOLUTE tolerance silently tightens as units shrink.
void test_the_verdict_does_not_depend_on_the_unit() {
    // THIS TEST WAS REWRITTEN AFTER A MUTATION SURVIVED IT. The first
    // version scaled an exactly-symmetric matrix and an obviously-broken
    // one, and asserted the verdicts matched. They did -- under BOTH a
    // relative and an absolute tolerance, because neither example sits
    // anywhere near the line. Replacing the relative budget with an
    // absolute one changed nothing it could see, so it tested the claim
    // without discriminating on it: architecture/proof_integrity.yaml, in
    // this file's own tests.
    //
    // The discriminating case is ROUNDOFF-LEVEL asymmetry at a LARGE
    // scale, which is what a real covariance in small units looks like.
    scl::CovarianceParameters p;   // symmetry_tolerance 1e-10, relative

    // relative asymmetry 1e-11 -- inside tolerance at any scale
    const std::vector<double> metres = {4.0, 1.0, 1.0 + 4e-11, 4.0};
    std::vector<double> millimetres = {4.0e6, 1.0e6, 1.0e6 + 4e-5, 4.0e6};

    CHECK(scl::validate_covariance(metres, 2, 2, p).ok());
    // absolute asymmetry here is 4e-5, four hundred thousand times the
    // 1e-10 tolerance. An ABSOLUTE rule refuses this legitimate covariance
    // purely for being expressed in millimetres; the relative rule admits
    // it, which is the whole reason the rule is relative.
    const scl::CovarianceReport mm = scl::validate_covariance(millimetres, 2, 2, p);
    CHECK(mm.ok());
    CHECK(mm.max_asymmetry > 1e-6);   // genuinely large in absolute terms

    // and a REAL asymmetry -- 1 part in 4 -- is refused at both scales
    const std::vector<double> bad_m = {4.0, 1.0, 2.0, 4.0};
    std::vector<double> bad_mm(bad_m.size());
    for (std::size_t i = 0; i < bad_m.size(); ++i) bad_mm[i] = bad_m[i] * 1e6;
    CHECK(scl::validate_covariance(bad_m, 2, 2, p).fault ==
          scl::CovarianceFault::kNotSymmetric);
    CHECK(scl::validate_covariance(bad_mm, 2, 2, p).fault ==
          scl::CovarianceFault::kNotSymmetric);
}

// --- THE SPECTRUM IS REPORTED, NOT JUST TESTED ---------------------------
void test_a_refusal_still_carries_the_measurement() {
    scl::CovarianceParameters p;
    const scl::CovarianceReport r =
        scl::validate_covariance({2.0, 3.0, 3.0, 2.0}, 2, 2, p);
    CHECK(!r.ok());
    CHECK(r.eigenvalues.size() == 2);
    CHECK(close(r.smallest_eigenvalue, -1.0, 1e-12));
    CHECK(close(r.largest_eigenvalue, 5.0, 1e-12));
    CHECK(close(r.condition_number, 5.0, 1e-9));
    CHECK(!r.detail.empty());
    // "how negative" is a different fact from "was negative"
    const scl::CovarianceReport worse =
        scl::validate_covariance({0.0, 3.0, 3.0, 0.0}, 2, 2, p);
    CHECK(worse.smallest_eigenvalue < r.smallest_eigenvalue);
}

// --- A SINGULAR BUT LEGITIMATE COVARIANCE --------------------------------
// Rank-deficient is NOT the same as invalid: a perfectly correlated pair
// has a zero eigenvalue and is a real covariance.
void test_a_rank_deficient_covariance_is_admitted() {
    scl::CovarianceParameters p;
    const scl::CovarianceReport r =
        scl::validate_covariance({1.0, 1.0, 1.0, 1.0}, 2, 2, p);
    CHECK(r.ok());
    CHECK(close(r.smallest_eigenvalue, 0.0, 1e-12));
    CHECK(close(r.largest_eigenvalue, 2.0, 1e-12));
    CHECK(std::isinf(r.condition_number));   // reported, not hidden
}

}  // namespace

int main() {
    test_analytic_two_by_two();
    test_diagonal_is_its_own_spectrum();
    test_trace_and_determinant_invariants();
    test_the_case_an_svd_cannot_distinguish();
    test_each_rule_has_its_own_fault_code();
    test_shape_is_reported_before_entries_before_symmetry_before_psd();
    test_both_tolerances_change_which_inputs_are_accepted();
    test_the_verdict_does_not_depend_on_the_unit();
    test_a_refusal_still_carries_the_measurement();
    test_a_rank_deficient_covariance_is_admitted();

    std::printf("covariance: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

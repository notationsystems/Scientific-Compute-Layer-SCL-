#pragma once
// THE COVARIANCE CONTRACT. What SCL requires a covariance matrix to BE.
//
// WHY THIS FILE EXISTS AT ALL. The acquisition layer's aligned-observation
// gate was asked whether its element-type refusal reaches matrix cells. It
// does, for the rules that are decided -- and it formally DECLINED to
// supply two that a covariance needs, by implementing each and reverting:
//
//   * a matrix entry is not required to be numeric there. A categorical
//     string is admitted as a leaf exactly as it is as a scalar cell,
//     because that gate answers ALIGNABILITY and not fittability.
//   * shape is not decided there at all. Raggedness, dimensionality,
//     symmetry and positive-semidefiniteness were named as this contract's
//     to define, and both admissions are pinned as tests on that side, so
//     closing either later fails loudly rather than being discovered.
//
// So a covariance-bearing R with categorical-string entries, or with an
// empty row, passes DAQ's gate today. NONE of the five rules below may be
// assumed to have been checked upstream. They are checked here.
//
// ------------------------------------------------------------------
// THE FIVE RULES
// ------------------------------------------------------------------
//
//   1. NUMERIC ENTRY.   Every entry is a finite double. Not NaN, not an
//                       infinity. A non-finite entry is a sentinel-encoded
//                       absence wearing a number's type.
//   2. RECTANGULAR.     Every row has the same length. A ragged matrix has
//                       no well-defined transpose, so rules 4 and 5 cannot
//                       even be stated over one.
//   3. SQUARE.          n rows, n columns. A covariance relates a set of
//                       components to itself.
//   4. SYMMETRIC        within a stated tolerance. cov(i,j) and cov(j,i)
//                       are the same quantity; a measured one differs only
//                       by roundoff.
//   5. POSITIVE         within a stated tolerance. Every variance of every
//      SEMIDEFINITE     linear combination is non-negative -- which is what
//                       a covariance MEANS, not an extra condition on one.
//
// Rules 4 and 5 carry TOLERANCES, and the tolerances are PARAMETERS rather
// than constants. This is not a convenience. A measured covariance is
// symmetric-up-to-roundoff and PSD-up-to-roundoff; a strict check rejects
// legitimate input and a loose one admits a broken model, and where that
// line sits is a modelling assertion the caller makes. It changes which
// inputs are accepted, so it changes the answer, so it PARTICIPATES in
// parameters_identity (docs/SCL_CONTRACT.md 6.1).
//
// ------------------------------------------------------------------
// WHY A SYMMETRIC EIGENSOLVER AND NOT THE SVD ALREADY IN THE SUBSTRATE
// ------------------------------------------------------------------
//
// least_squares elected SVD because the singular spectrum IS the
// conditioning measurement -- the operation REPORTS conditioning instead
// of leaving a caller to infer it. The same argument is what this check
// needs: make the failure visible rather than return a verdict.
//
// But the singular spectrum cannot see PSD, and this was measured before
// the choice rather than assumed:
//
//     A = [[2, 3],      eigenvalues  -1, 5     NOT positive semidefinite
//          [3, 2]]      singular      5, 1
//
//     B = [[3, 2],      eigenvalues   1, 5     IS positive semidefinite
//          [2, 3]]      singular      5, 1
//
// Identical singular spectra, opposite verdicts. For a symmetric matrix
// sigma_i = |lambda_i|, so the SVD discards exactly the sign that PSD is a
// question about -- and sigma >= 0 holds for EVERY real matrix, so a
// "check" against it is vacuous in the strict sense this project already
// has a name for.
//
// The sign is recoverable as sign(v_i^T A v_i) per singular vector, but a
// symmetric eigensolver returns it directly, and it is the SAME algorithm
// family already read and verified here: two-sided Jacobi rotations, where
// least_squares uses one-sided. So the substrate argument holds in full --
// small code, verifiable by reading, spectrum reported as metrics -- with
// the operator corrected to one that can answer the question asked.
//
// EIGENVALUES ARE REPORTED, NOT JUST TESTED. lambda_min, lambda_max and
// the condition number come out as metrics, so a caller sees HOW negative
// or HOW ill-conditioned, not merely that a threshold was crossed. A
// covariance that fails by 1e-18 and one that fails by -3.0 are different
// facts about a model and must not arrive as the same boolean.

#include <cstddef>
#include <string>
#include <vector>

namespace scl {

//: Which of the five rules a matrix broke. A caller acts differently on a
//: ragged input (a shape bug) than on a slightly-negative eigenvalue (a
//: model or tolerance question), so they are never merged into one code.
enum class CovarianceFault {
    kNone = 0,
    kNotRectangular,    //: rows of differing length
    kNotSquare,         //: rectangular but n_rows != n_cols
    kEmpty,             //: zero-dimensional; a covariance over nothing
    kNonFiniteEntry,    //: NaN or an infinity
    kNotSymmetric,      //: asymmetry beyond symmetry_tolerance
    kNotPositiveSemidefinite,  //: lambda_min below -psd_tolerance * scale
};

struct CovarianceParameters {
    //: Largest tolerated |A(i,j) - A(j,i)|, RELATIVE to the largest entry
    //: magnitude. Relative rather than absolute so the rule is
    //: scale-invariant: a covariance in metres and the same covariance in
    //: millimetres must receive the same verdict, and an absolute
    //: tolerance silently tightens as units shrink.
    double symmetry_tolerance = 1e-10;

    //: How negative the smallest eigenvalue may be before the matrix is
    //: refused, RELATIVE to the largest eigenvalue magnitude. Same
    //: scale-invariance argument. Zero means exact PSD, which real
    //: measured covariances essentially never satisfy.
    double psd_tolerance = 1e-10;
};

//: The spectrum, always returned -- on success and on failure alike. A
//: refusal that does not say how far out of tolerance the input was is a
//: verdict, not a measurement.
struct CovarianceReport {
    CovarianceFault fault = CovarianceFault::kNone;
    std::string detail;                 //: actionable, names the offending index
    std::size_t dimension = 0;
    std::vector<double> eigenvalues;    //: ascending; empty if shape failed first
    double smallest_eigenvalue = 0.0;
    double largest_eigenvalue = 0.0;
    double max_asymmetry = 0.0;         //: max |A(i,j) - A(j,i)|, absolute
    double condition_number = 0.0;      //: |lambda_max| / |lambda_min|, inf if singular
    int jacobi_sweeps = 0;

    bool ok() const { return fault == CovarianceFault::kNone; }
};

//: Signed eigenvalues of a symmetric matrix, ascending, by two-sided
//: cyclic Jacobi. `matrix` is row-major n*n and is NOT checked for
//: symmetry here -- validate_covariance does that first, deliberately, so
//: this stays one job.
std::vector<double> symmetric_eigenvalues(const std::vector<double>& matrix,
                                          std::size_t n,
                                          int* sweeps_out = nullptr);

//: Apply all five rules, in the order stated: shape before entries before
//: symmetry before PSD. The order matters and is part of the contract --
//: eigenvalues of a ragged or asymmetric matrix are not defined, so
//: reporting a PSD failure over one would be reporting a number that
//: means nothing.
CovarianceReport validate_covariance(const std::vector<double>& matrix,
                                     std::size_t n_rows,
                                     std::size_t n_cols,
                                     const CovarianceParameters& params);

}  // namespace scl

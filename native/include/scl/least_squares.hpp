#pragma once
// Linear least squares: minimize ||W^(1/2) (X b - y)||_2.
//
// WHY SVD, AND NOT THE OTHER THREE. This workload was elected precisely
// because its failure mode is SILENT: an ill-conditioned fit returns a
// plausible wrong answer, where the transform's failures are loud. So the
// solver is chosen on what it makes VISIBLE, not on speed.
//
//   normal equations (X^T X b = X^T y, via Cholesky)
//       REJECTED. kappa(X^T X) = kappa(X)^2 -- it squares the condition
//       number, so it degrades fastest on exactly the inputs that are
//       already hardest, and it degrades quietly. Choosing it here would
//       build the silent failure into the substrate.
//
//   Householder QR
//       Backward stable and does NOT square kappa. A good solver. But it
//       needs column pivoting to handle rank deficiency, and it yields no
//       condition number -- so a caller cannot tell a healthy fit from a
//       barely-determined one without doing more work elsewhere.
//
//   SVD
//       ADOPTED. It handles rank deficiency natively (the minimum-norm
//       solution falls out), and the singular spectrum IS the conditioning
//       measurement: sigma_max / sigma_min comes out as a by-product, so
//       the operation can REPORT conditioning as a metric instead of
//       leaving the caller to infer it. That is the whole point -- it
//       converts the silent failure into a visible one.
//
// ONE-SIDED JACOBI rather than Golub-Reinsch: a quarter of the code, so
// it can be verified by reading, and it has high RELATIVE accuracy on
// small singular values. That accuracy is not incidental here -- small
// singular values are exactly what rank determination and the Hilbert
// case turn on.
//
// The rank tolerance is a PARAMETER, never a hidden default: it changes
// the answer for a rank-deficient system, so it participates in
// parameters_identity (see docs/SCL_CONTRACT.md 6.1).
//
// ---------------------------------------------------------------------
// WHAT THE CONDITIONING METRICS DO NOT SEE. Recorded 2026-08-26, measured
// rather than reasoned, because the paragraph above promises a caller
// that this operation "converts the silent failure into a visible one"
// and there is one failure of exactly that shape it cannot see.
//
// RANK AND CONDITIONING ARE PROPERTIES OF THE DESIGN MATRIX'S COLUMNS.
// A dependence among the DATA ROWS -- one observation that is an exact
// function of two others -- leaves the design matrix full rank and well
// conditioned, so every diagnostic this operation reports comes back
// healthy. It is not a defect: the solver is looking at the columns, and
// the dependence is not in the columns.
//
// MEASURED, on the acquisition layer's polymer row (Mn, Mw and a
// polydispersity index PDI = Mw/Mn arriving together from one
// instrument), fitted in logs so the third row is exactly the second
// minus the first:
//
//     design           [[1, 0], [0, 1], [-1, 1]]     3 rows, 2 columns
//     effective_rank                        2        FULL RANK
//     condition_number                   1.4378      excellent
//     smallest / largest singular value  36.21 / 52.07
//     residuals                          ~1e-15
//     weighted chi^2                      2.1e-26
//
// The conditioning metric does not merely fail to warn -- at 1.44 it
// ACTIVELY REASSURES. And because the third row is reproduced exactly by
// any fit of the first two, chi^2 is identically zero FOR ALL DATA, not
// only for this row: a caller computing goodness of fit reads perfect
// agreement between three measurements, and a reduced chi^2 divides by a
// degrees-of-freedom count that is nominally 1 and truly 0.
//
// THE CALLER'S OBLIGATION, since this operation cannot discharge it:
// establish that the rows are independent measurements before fitting.
// SCL is handed (X, y, weights) and has no access to what produced them;
// nothing in the wire protocol can carry "row 3 is a function of rows 1
// and 2", and inventing a field for it here would be SCL asserting a
// property of acquisition it cannot verify.
//
// WHY IT IS RECORDED RATHER THAN FIXED. A rank check on the augmented
// matrix [X | y] would detect this row -- and would also fire on any
// exactly-fitting dataset, including a legitimate one, so it converts a
// caller obligation into a false refusal. The boundary is stated instead,
// and tests/test_least_squares_row_dependence.py pins it by showing the
// diagnostics silent rather than asserting they would be.
// ---------------------------------------------------------------------
//
// CORRELATED ERRORS: THE CALLER WHITENS, AND THAT IS EXACT. Recorded
// 2026-08-26 beside the boundary above, because they are the same family
// -- what the caller must establish before calling -- and because the
// acquisition layer can now MEASURE a correlation (its
// science/replicate_pairing.py recovers it from paired replicate runs),
// so the question of what to do with one stopped being hypothetical.
//
// The wire carries WEIGHTS, one per row: a diagonal. No off-diagonal
// term is expressible, so a covariance cannot be handed to this
// operation. It does not need to be. For a known Sigma = L L^T, the
// caller solves L X~ = X and L y~ = y and fits the whitened problem
// ordinarily; the result IS the generalized least-squares estimate, not
// an approximation to it.
//
// MEASURED, through the CLI, on a 3x2 problem with corr = 0.8 between
// two of the errors:
//
//     pre-whitened through this operation   2.119565   3.069565
//     analytic GLS, computed independently  2.119565   3.069565
//         agreement                         8.9e-16
//     the same fit with weights only,
//     the correlation ignored               2.116667   3.066667
//         error                             2.9e-03
//
// Note where that error lands: in the COEFFICIENTS, not merely in their
// uncertainties. Ignoring a correlation moves the answer.
//
// So no extension is proposed. What was missing is that nothing said
// this, and a caller holding a covariance had no way to learn from the
// operation that whitening is the route. tests/test_least_squares_
// correlated_errors.py measures the agreement rather than asserting it.
// ---------------------------------------------------------------------

#include <cstddef>
#include <string>
#include <vector>

namespace scl {

struct LeastSquaresProblem {
    std::size_t n_rows = 0;
    std::size_t n_cols = 0;
    std::vector<double> design;    //: row-major, n_rows * n_cols
    std::vector<double> response;  //: n_rows
    std::vector<double> weights;   //: empty when unweighted, else n_rows
};

struct LeastSquaresParameters {
    //: Relative cutoff on the singular values: sigma_j is treated as zero
    //: when sigma_j <= rank_tolerance * sigma_max. Participating.
    double rank_tolerance = 1e-12;
    bool weighted = false;
};

struct LeastSquaresResult {
    bool ok = false;
    std::string error;
    std::vector<double> coefficients;     //: n_cols
    std::vector<double> singular_values;  //: n_cols, descending
    double condition_number = 0.0;        //: sigma_max / sigma_min_kept
    std::size_t effective_rank = 0;
    std::size_t sweeps = 0;
};

//: Singular values of a row-major m x n matrix (m >= n), descending.
//: Exposed for testing the decomposition independently of the solve.
std::vector<double> singular_values_of(const std::vector<double>& row_major,
                                        std::size_t n_rows, std::size_t n_cols);

LeastSquaresResult solve_least_squares(const LeastSquaresProblem& problem,
                                        const LeastSquaresParameters& parameters);

}  // namespace scl

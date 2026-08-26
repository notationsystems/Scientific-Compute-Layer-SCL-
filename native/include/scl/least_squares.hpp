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

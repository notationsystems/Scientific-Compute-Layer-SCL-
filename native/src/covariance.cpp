#include "scl/covariance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace scl {
namespace {

constexpr int kMaxSweeps = 60;

std::string index_pair(std::size_t i, std::size_t j) {
    std::ostringstream os;
    os << "(" << i << ", " << j << ")";
    return os.str();
}

}  // namespace

std::vector<double> symmetric_eigenvalues(const std::vector<double>& matrix,
                                          std::size_t n,
                                          int* sweeps_out) {
    // TWO-SIDED CYCLIC JACOBI. least_squares uses the one-sided form on a
    // rectangular design matrix; this is the same rotation applied from
    // both sides of a symmetric one, which is what preserves the SIGN of
    // each eigenvalue. That sign is the entire reason this exists rather
    // than a call into the existing SVD -- see the header.
    std::vector<double> a = matrix;
    if (n == 0) {
        if (sweeps_out) *sweeps_out = 0;
        return {};
    }

    int sweep = 0;
    for (; sweep < kMaxSweeps; ++sweep) {
        // Off-diagonal Frobenius norm: the quantity Jacobi drives to zero.
        double off = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if (off <= std::numeric_limits<double>::epsilon() *
                       std::numeric_limits<double>::epsilon()) {
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (apq == 0.0) continue;

                // Same rotation as the one-sided solver, written the same
                // way for the same reason: computing t from zeta rather
                // than from a difference of squares avoids cancellation
                // when the two diagonal entries are close.
                const double zeta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double sign = zeta >= 0.0 ? 1.0 : -1.0;
                const double t = sign / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;

                for (std::size_t k = 0; k < n; ++k) {   // rotate rows p, q
                    const double akp = a[k * n + p];
                    const double akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {   // and columns p, q
                    const double apk = a[p * n + k];
                    const double aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    if (sweeps_out) *sweeps_out = sweep;

    std::vector<double> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = a[i * n + i];
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

CovarianceReport validate_covariance(const std::vector<double>& matrix,
                                     std::size_t n_rows,
                                     std::size_t n_cols,
                                     const CovarianceParameters& params) {
    CovarianceReport report;

    // RULE 2 and 3, before anything else. A ragged or non-square matrix has
    // no transpose and no spectrum, so symmetry and PSD cannot be stated
    // over it -- reporting either would be reporting a number that means
    // nothing. The caller supplies the shape it believes it has; a
    // mismatch against the actual element count is the ragged case, since
    // a flattened ragged matrix is exactly a wrong element count.
    if (n_rows == 0 || n_cols == 0) {
        report.fault = CovarianceFault::kEmpty;
        report.detail = "covariance has a zero dimension (" +
                        std::to_string(n_rows) + " x " + std::to_string(n_cols) +
                        "); a covariance over no components is not a covariance";
        return report;
    }
    if (matrix.size() != n_rows * n_cols) {
        report.fault = CovarianceFault::kNotRectangular;
        std::ostringstream os;
        os << "covariance declares " << n_rows << " x " << n_cols << " = "
           << (n_rows * n_cols) << " entries but carries " << matrix.size()
           << " -- a ragged matrix has no transpose, so symmetry and "
              "positive-semidefiniteness cannot be stated over it";
        report.detail = os.str();
        return report;
    }
    if (n_rows != n_cols) {
        report.fault = CovarianceFault::kNotSquare;
        report.detail = "covariance must be square, got " + std::to_string(n_rows) +
                        " x " + std::to_string(n_cols) +
                        "; a covariance relates a set of components to itself";
        return report;
    }

    const std::size_t n = n_rows;
    report.dimension = n;

    // RULE 1. Every entry finite, checked before the spectrum: a single
    // NaN propagates through every rotation and would surface as an
    // uninterpretable eigenvalue rather than as the entry that caused it.
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double v = matrix[i * n + j];
            if (!std::isfinite(v)) {
                report.fault = CovarianceFault::kNonFiniteEntry;
                std::ostringstream os;
                os << "covariance entry " << index_pair(i, j) << " is "
                   << (std::isnan(v) ? "NaN" : "an infinity")
                   << " -- a sentinel-encoded absence wearing a number's type. "
                      "DAQ's table gate refuses this, but its numeric-entry rule "
                      "does not reach matrix leaves by contract, so it is checked here";
                report.detail = os.str();
                return report;
            }
            scale = std::max(scale, std::fabs(v));
        }
    }

    // RULE 4. Symmetry, relative to the largest entry magnitude so the
    // verdict does not change with the unit the covariance is expressed in.
    std::size_t worst_i = 0, worst_j = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double gap = std::fabs(matrix[i * n + j] - matrix[j * n + i]);
            if (gap > report.max_asymmetry) {
                report.max_asymmetry = gap;
                worst_i = i;
                worst_j = j;
            }
        }
    }
    const double symmetry_budget = params.symmetry_tolerance * std::max(scale, 1.0);
    if (report.max_asymmetry > symmetry_budget) {
        report.fault = CovarianceFault::kNotSymmetric;
        std::ostringstream os;
        os << "covariance is not symmetric: entries " << index_pair(worst_i, worst_j)
           << " and " << index_pair(worst_j, worst_i) << " differ by "
           << report.max_asymmetry << ", above the tolerated "
           << symmetry_budget << " (symmetry_tolerance " << params.symmetry_tolerance
           << " relative to largest magnitude " << scale << ")";
        report.detail = os.str();
        return report;
    }

    // RULE 5. Only now is the spectrum meaningful.
    report.eigenvalues = symmetric_eigenvalues(matrix, n, &report.jacobi_sweeps);
    report.smallest_eigenvalue = report.eigenvalues.front();
    report.largest_eigenvalue = report.eigenvalues.back();

    double largest_magnitude = 0.0;
    double smallest_magnitude = std::numeric_limits<double>::infinity();
    for (double lambda : report.eigenvalues) {
        largest_magnitude = std::max(largest_magnitude, std::fabs(lambda));
        smallest_magnitude = std::min(smallest_magnitude, std::fabs(lambda));
    }
    report.condition_number = smallest_magnitude > 0.0
                                  ? largest_magnitude / smallest_magnitude
                                  : std::numeric_limits<double>::infinity();

    const double psd_budget = params.psd_tolerance * std::max(largest_magnitude, 1.0);
    if (report.smallest_eigenvalue < -psd_budget) {
        report.fault = CovarianceFault::kNotPositiveSemidefinite;
        std::ostringstream os;
        os << "covariance is not positive semidefinite: smallest eigenvalue "
           << report.smallest_eigenvalue << " is below the tolerated "
           << (-psd_budget) << " (psd_tolerance " << params.psd_tolerance
           << " relative to largest magnitude " << largest_magnitude
           << "). A negative eigenvalue means some linear combination of the "
              "components has negative variance, which no measurement can have";
        report.detail = os.str();
        // the spectrum is RETAINED on failure: how negative is a different
        // fact about a model than merely that it was negative.
    }
    return report;
}

}  // namespace scl

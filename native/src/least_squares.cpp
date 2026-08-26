#include "scl/least_squares.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace scl {
namespace {

//: One-sided Jacobi sweeps until every column pair is orthogonal to
//: working precision. `columns` is n_rows x n_cols in COLUMN-major order
//: (each column contiguous), which is what the rotation touches; `v`
//: accumulates the right singular vectors.
std::size_t jacobi_orthogonalize(std::vector<double>& columns, std::vector<double>& v,
                                  std::size_t n_rows, std::size_t n_cols) {
    const std::size_t kMaxSweeps = 60;
    // Rotate while any pair is measurably non-orthogonal. The threshold is
    // relative to the column norms, so it scales with the data rather than
    // assuming anything about magnitude.
    const double kOrthogonal = 1e-15;

    std::size_t sweeps = 0;
    for (; sweeps < kMaxSweeps; ++sweeps) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n_cols; ++p) {
            for (std::size_t q = p + 1; q < n_cols; ++q) {
                double* ap = &columns[p * n_rows];
                double* aq = &columns[q * n_rows];

                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (std::size_t i = 0; i < n_rows; ++i) {
                    alpha += ap[i] * ap[i];
                    beta += aq[i] * aq[i];
                    gamma += ap[i] * aq[i];
                }
                if (gamma == 0.0 || alpha == 0.0 || beta == 0.0) {
                    continue;
                }
                if (std::fabs(gamma) <= kOrthogonal * std::sqrt(alpha * beta)) {
                    continue;
                }

                // The rotation that annihilates gamma. Written with the
                // stable branch for t (adding the larger root's reciprocal)
                // so cancellation cannot amplify a small gamma.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double sign = zeta >= 0.0 ? 1.0 : -1.0;
                const double t = sign / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;

                for (std::size_t i = 0; i < n_rows; ++i) {
                    const double tp = ap[i], tq = aq[i];
                    ap[i] = c * tp - s * tq;
                    aq[i] = s * tp + c * tq;
                }
                double* vp = &v[p * n_cols];
                double* vq = &v[q * n_cols];
                for (std::size_t i = 0; i < n_cols; ++i) {
                    const double tp = vp[i], tq = vq[i];
                    vp[i] = c * tp - s * tq;
                    vq[i] = s * tp + c * tq;
                }
                rotated = true;
            }
        }
        if (!rotated) {
            ++sweeps;
            break;
        }
    }
    return sweeps;
}

struct Decomposition {
    std::vector<double> columns;  //: column-major, columns[j] = sigma_j * u_j
    std::vector<double> v;        //: column-major n_cols x n_cols
    std::vector<double> sigma;
    std::vector<std::size_t> order;
    std::size_t sweeps = 0;
};

Decomposition decompose(const std::vector<double>& row_major, std::size_t n_rows,
                         std::size_t n_cols) {
    Decomposition d;
    d.columns.assign(n_rows * n_cols, 0.0);
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            d.columns[j * n_rows + i] = row_major[i * n_cols + j];
        }
    }
    d.v.assign(n_cols * n_cols, 0.0);
    for (std::size_t j = 0; j < n_cols; ++j) {
        d.v[j * n_cols + j] = 1.0;
    }

    d.sweeps = jacobi_orthogonalize(d.columns, d.v, n_rows, n_cols);

    d.sigma.assign(n_cols, 0.0);
    for (std::size_t j = 0; j < n_cols; ++j) {
        double norm = 0.0;
        for (std::size_t i = 0; i < n_rows; ++i) {
            const double value = d.columns[j * n_rows + i];
            norm += value * value;
        }
        d.sigma[j] = std::sqrt(norm);
    }

    d.order.resize(n_cols);
    std::iota(d.order.begin(), d.order.end(), std::size_t{0});
    std::sort(d.order.begin(), d.order.end(),
              [&d](std::size_t a, std::size_t b) { return d.sigma[a] > d.sigma[b]; });
    return d;
}

}  // namespace

std::vector<double> singular_values_of(const std::vector<double>& row_major,
                                        std::size_t n_rows, std::size_t n_cols) {
    if (n_rows == 0 || n_cols == 0 || row_major.size() != n_rows * n_cols) {
        return {};
    }
    Decomposition d = decompose(row_major, n_rows, n_cols);
    std::vector<double> sorted;
    sorted.reserve(n_cols);
    for (std::size_t index : d.order) {
        sorted.push_back(d.sigma[index]);
    }
    return sorted;
}

LeastSquaresResult solve_least_squares(const LeastSquaresProblem& problem,
                                        const LeastSquaresParameters& parameters) {
    LeastSquaresResult result;
    const std::size_t m = problem.n_rows, n = problem.n_cols;

    if (m == 0 || n == 0) {
        result.error = "least squares needs at least one row and one column";
        return result;
    }
    if (problem.design.size() != m * n) {
        result.error = "design matrix size does not match n_rows * n_cols";
        return result;
    }
    if (problem.response.size() != m) {
        result.error = "response vector length does not match n_rows";
        return result;
    }
    if (m < n) {
        result.error = "underdetermined system: n_rows must be >= n_cols";
        return result;
    }
    if (parameters.weighted && problem.weights.size() != m) {
        result.error = "weighted fit needs exactly n_rows weights";
        return result;
    }

    // Weighting is applied by scaling each row by sqrt(w_i), which is what
    // makes the ordinary solve minimize the WEIGHTED residual. A negative
    // weight has no square root and no meaning as a weight.
    std::vector<double> design = problem.design;
    std::vector<double> response = problem.response;
    if (parameters.weighted) {
        for (std::size_t i = 0; i < m; ++i) {
            const double w = problem.weights[i];
            if (!(w >= 0.0) || !std::isfinite(w)) {
                result.error = "weights must be finite and non-negative";
                return result;
            }
            const double scale = std::sqrt(w);
            for (std::size_t j = 0; j < n; ++j) {
                design[i * n + j] *= scale;
            }
            response[i] *= scale;
        }
    }

    Decomposition d = decompose(design, m, n);
    result.sweeps = d.sweeps;

    result.singular_values.reserve(n);
    for (std::size_t index : d.order) {
        result.singular_values.push_back(d.sigma[index]);
    }
    const double sigma_max = result.singular_values.front();
    if (!(sigma_max > 0.0)) {
        result.error = "design matrix is entirely zero; no direction is determined";
        return result;
    }

    const double cutoff = parameters.rank_tolerance * sigma_max;
    double smallest_kept = sigma_max;
    result.coefficients.assign(n, 0.0);

    // b = V * Sigma^+ * U^T * y, accumulated one retained direction at a
    // time. A direction below the cutoff contributes NOTHING rather than
    // contributing an enormous amount -- that is what makes this the
    // minimum-norm solution and what keeps a rank-deficient system from
    // returning a huge plausible answer.
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t index = d.order[k];
        const double sigma = d.sigma[index];
        if (sigma <= cutoff) {
            continue;
        }
        double projection = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            projection += d.columns[index * m + i] * response[i];
        }
        // columns[index] is sigma * u, so dividing twice by sigma gives
        // (u^T y) / sigma without ever forming u separately.
        const double coefficient = projection / (sigma * sigma);
        for (std::size_t j = 0; j < n; ++j) {
            result.coefficients[j] += coefficient * d.v[index * n + j];
        }
        ++result.effective_rank;
        smallest_kept = sigma;
    }

    if (result.effective_rank == 0) {
        result.error = "every singular value fell below the rank tolerance";
        return result;
    }
    for (double value : result.coefficients) {
        if (!std::isfinite(value)) {
            result.error = "solve produced a non-finite coefficient";
            return result;
        }
    }

    result.condition_number = sigma_max / smallest_kept;
    result.ok = true;
    return result;
}

}  // namespace scl

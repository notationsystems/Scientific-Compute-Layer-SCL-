"""`least_squares`, validated against INDEPENDENT MATHEMATICS.

This workload was elected because its failure mode is SILENT: an
ill-conditioned fit returns a plausible wrong answer, where the
transform's failures are loud. So the validation is built around exposing
that, not around agreeing with a second implementation:

  * known beta on a constructed X -- the answer is known before the solve
  * residual orthogonality, X^T r ~ 0 -- the DEFINING property of a
    least-squares solution, and the strongest check available that needs
    no second implementation
  * rank-deficient systems -- where a naive solver returns something huge
    and plausible
  * a Hilbert matrix -- notoriously ill-conditioned, to check that the
    operation REPORTS the conditioning rather than hiding it

No second least-squares implementation is used as an oracle anywhere.
"""

from __future__ import annotations

import math

import pytest

from scl.client import SCLRequest, run_scl_request
from scl.least_squares import (
    DEFAULT_RANK_TOLERANCE,
    decode_least_squares_output,
    encode_least_squares_configuration,
    encode_least_squares_input,
    normal_equation_residual,
    residuals,
)

OPERATION = "least_squares"


def fit(design, response, cli_path, weights=None, rank_tolerance=DEFAULT_RANK_TOLERANCE):
    request = SCLRequest(
        OPERATION, "cpu",
        encode_least_squares_configuration(rank_tolerance, weighted=weights is not None),
        encode_least_squares_input(design, response, weights))
    return run_scl_request(request, cli_path=cli_path)


def coefficients(result):
    assert result.status == "completed", result.detail
    return decode_least_squares_output(result.output)


# ============================================ known beta on constructed X

@pytest.mark.parametrize("beta", [[2.0, -3.0], [0.0, 1.0], [1e3, 1e-3], [-7.25, 0.5]])
def test_exact_fit_recovers_the_beta_it_was_built_from(beta, cli_path):
    """y = X*beta EXACTLY, so the residual is zero and the answer is known
    before the solve. Nothing here is compared to another solver."""
    design = [[1.0, float(i)] for i in range(1, 9)]
    response = [beta[0] + beta[1] * row[1] for row in design]
    got = coefficients(fit(design, response, cli_path))
    for expected, actual in zip(beta, got):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_an_overdetermined_noisy_fit_satisfies_orthogonality(cli_path):
    """THE defining property: the residual is orthogonal to every column
    of X. True of the least-squares solution and of nothing else, so it
    validates the answer without knowing the answer."""
    design = [[1.0, float(i), float(i * i)] for i in range(1, 12)]
    response = [3.0 - 0.5 * row[1] + 0.25 * row[2] + (0.1 if i % 3 else -0.2)
                for i, row in enumerate(design)]
    got = coefficients(fit(design, response, cli_path))

    scale = max(abs(v) for row in design for v in row) * max(abs(v) for v in response)
    for component in normal_equation_residual(design, response, got):
        assert abs(component) < 1e-9 * scale, "X^T r is not zero: this is not the LS solution"


def test_the_fit_is_optimal_against_deliberate_perturbations(cli_path):
    """Optimality checked directly: perturbing any coefficient in either
    direction must not reduce the sum of squares. A wrong-but-plausible
    answer fails this even when it looks reasonable."""
    design = [[1.0, float(i)] for i in range(1, 10)]
    response = [2.0 + 0.7 * row[1] + (0.3 if i % 2 else -0.3) for i, row in enumerate(design)]
    got = coefficients(fit(design, response, cli_path))

    def sum_of_squares(b):
        return sum(r * r for r in residuals(design, response, b))

    best = sum_of_squares(got)
    for index in range(len(got)):
        for delta in (1e-6, -1e-6, 1e-3, -1e-3):
            perturbed = list(got)
            perturbed[index] += delta
            assert sum_of_squares(perturbed) >= best * (1 - 1e-12)


# ==================================================== rank deficiency

def test_a_duplicated_column_is_reported_as_rank_deficient(cli_path):
    """An exactly collinear column means the system does not determine a
    unique beta. The operation must SAY so through effective_rank rather
    than return a large arbitrary answer."""
    design = [[1.0, float(i), float(i)] for i in range(1, 8)]   # cols 1 and 2 identical
    response = [1.0 + 2.0 * float(i) for i in range(1, 8)]
    result = fit(design, response, cli_path)
    got = coefficients(result)

    assert result.metrics["effective_rank"] == 2, "three columns, only two independent"
    assert result.metrics["n_cols"] == 3
    # the minimum-norm solution still FITS -- it is a valid least-squares
    # answer, just not a unique one
    for component in normal_equation_residual(design, response, got):
        assert abs(component) < 1e-8


def test_the_rank_deficient_answer_is_minimum_norm_not_arbitrary(cli_path):
    """With EXACT duplication this is weaker than it looks -- the null
    direction's projection is itself ~0, so a naive solve is
    self-limiting and stays bounded too. Found by mutation: dropping the
    cutoff did not fail this test. Kept as a floor, with the real check
    below."""
    design = [[1.0, float(i), float(i)] for i in range(1, 8)]
    response = [1.0 + 2.0 * float(i) for i in range(1, 8)]
    got = coefficients(fit(design, response, cli_path))
    assert all(abs(value) < 10.0 for value in got), f"coefficients blew up: {got}"


def test_the_cutoff_decides_the_rank_and_the_reported_conditioning(cli_path):
    """The real demonstration, on NEAR-collinearity rather than exact, and
    on what actually moves.

    Column 3 is column 2 plus a genuinely independent direction scaled by
    1e-8 -- near-null, but not a scalar multiple of anything. Running the
    same data at two tolerances shows the cutoff doing the work.

    WHAT I EXPECTED AND DID NOT GET: the coefficients exploding. They do
    not, because this `y` lies in the span of the first two columns, so
    the near-null direction's projection is itself ~0. Measured, not
    assumed -- the first version of this test asserted a blow-up that
    never happens.

    WHAT ACTUALLY MOVES is the REPORTED CONDITION NUMBER, by seven orders
    of magnitude. That is precisely the argument for choosing an SVD: the
    fit looks fine either way, and the only thing that distinguishes a
    barely-determined system from a healthy one is the number the
    operation publishes alongside it."""
    design = [[1.0, float(i), float(i) + 1e-8 * float(i * i)] for i in range(1, 9)]
    response = [1.0 + 2.0 * float(i) for i in range(1, 9)]

    kept = fit(design, response, cli_path, rank_tolerance=1e-14)
    dropped = fit(design, response, cli_path, rank_tolerance=1e-6)

    assert kept.metrics["effective_rank"] == 3
    assert dropped.metrics["effective_rank"] == 2, "the near-null direction must be cut"

    assert kept.metrics["condition_number"] > 1e7
    assert dropped.metrics["condition_number"] < 1e2
    assert kept.metrics["condition_number"] > 1e5 * dropped.metrics["condition_number"]

    # both fit the data; the difference is what you can TELL about them
    for result in (kept, dropped):
        for component in normal_equation_residual(design, response, coefficients(result)):
            assert abs(component) < 1e-6


def test_rank_tolerance_changes_the_answer_and_therefore_participates(cli_path):
    """A parameter that cannot change the result does not need to be in
    parameters_identity. This one can, so it does."""
    design = [[1.0, float(i), float(i) + 1e-8 * float(i * i)] for i in range(1, 9)]
    response = [1.0 + 2.0 * float(i) for i in range(1, 9)]
    tight = fit(design, response, cli_path, rank_tolerance=1e-14)
    loose = fit(design, response, cli_path, rank_tolerance=1e-6)
    assert tight.output != loose.output, (
        "the tolerance must be able to change the computed output, or it is annotating")
    assert tight.computation_identity != loose.computation_identity


def test_a_zero_design_matrix_is_refused_rather_than_fitted(cli_path):
    design = [[0.0, 0.0] for _ in range(4)]
    result = fit(design, [1.0, 2.0, 3.0, 4.0], cli_path)
    assert result.status == "halted"
    assert "no direction is determined" in result.detail


# ============================== conditioning is REPORTED, not hidden

def _hilbert(n):
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


@pytest.mark.parametrize("n", [4, 6, 8])
def test_the_hilbert_condition_number_is_reported_and_grows(n, cli_path):
    """A Hilbert matrix is the standard ill-conditioning probe. The
    operation must publish a condition number that GROWS with n -- that
    number is what turns a silent degradation into a visible one."""
    design = _hilbert(n)
    response = [1.0] * n
    result = fit(design, response, cli_path)
    assert result.status == "completed", result.detail
    assert result.metrics["condition_number"] > 10 ** (1.5 * n - 3), (
        f"H_{n} should be severely ill-conditioned, got "
        f"{result.metrics['condition_number']:.3e}")


def test_condition_number_is_monotone_in_hilbert_size(cli_path):
    seen = [fit(_hilbert(n), [1.0] * n, cli_path).metrics["condition_number"]
            for n in (3, 5, 7)]
    assert seen == sorted(seen), f"conditioning must worsen with size, got {seen}"


def test_a_well_conditioned_fit_reports_a_small_condition_number(cli_path):
    """The control: without it, a large number everywhere would look like
    a working detector."""
    design = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    result = fit(design, [1.0, 2.0, 3.0], cli_path)
    assert result.metrics["condition_number"] < 10.0


def test_normal_equations_would_have_squared_this(cli_path):
    """WHY THE SOLVER CHOICE MATTERS, measured rather than asserted.

    Normal equations solve X^T X b = X^T y, and kappa(X^T X) = kappa(X)^2.
    Here the reported kappa(X) is already large; squaring it would exceed
    float64's ~1e16 resolution entirely, so a normal-equations solve would
    be operating on a matrix numerically indistinguishable from singular
    -- and would return an answer anyway."""
    result = fit(_hilbert(8), [1.0] * 8, cli_path)
    kappa = result.metrics["condition_number"]
    assert kappa > 1e9
    assert kappa ** 2 > 1e16, (
        "the squared condition number exceeds double precision: this is the "
        "input on which normal equations fail silently and SVD does not")


# ============================================================ weighting

def test_weighting_changes_the_fit_in_the_direction_of_the_weights(cli_path):
    """A heavily weighted point must pull the fit toward itself. Checked
    by direction rather than by a second implementation's number."""
    design = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    response = [0.0, 1.0, 10.0]
    plain = coefficients(fit(design, response, cli_path))
    pulled = coefficients(fit(design, response, cli_path, weights=[1.0, 1.0, 1000.0]))

    def predict(b, x):
        return b[0] + b[1] * x

    assert abs(predict(pulled, 2.0) - 10.0) < abs(predict(plain, 2.0) - 10.0)


def test_uniform_weights_reproduce_the_unweighted_fit(cli_path):
    """w = 1 everywhere is the unweighted problem, so it must give the
    same answer -- a property of the formulation, not a coincidence."""
    design = [[1.0, float(i)] for i in range(1, 7)]
    response = [1.0 + 2.0 * row[1] + (0.1 if i % 2 else -0.1) for i, row in enumerate(design)]
    plain = coefficients(fit(design, response, cli_path))
    weighted = coefficients(fit(design, response, cli_path, weights=[1.0] * 6))
    for a, b in zip(plain, weighted):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def test_a_zero_weight_excludes_a_row_entirely(cli_path):
    design = [[1.0, float(i)] for i in range(1, 6)]
    response = [1.0 + 2.0 * row[1] for row in design]
    response[-1] = 1e6                       # a wild outlier...
    got = coefficients(fit(design, response, cli_path,
                           weights=[1.0, 1.0, 1.0, 1.0, 0.0]))   # ...weighted out
    assert math.isclose(got[0], 1.0, abs_tol=1e-8)
    assert math.isclose(got[1], 2.0, abs_tol=1e-8)


@pytest.mark.parametrize("bad", [[-1.0, 1.0, 1.0], [float("nan"), 1.0, 1.0]])
def test_invalid_weights_are_refused(bad, cli_path):
    design = [[1.0, float(i)] for i in range(1, 4)]
    result = fit(design, [1.0, 2.0, 3.0], cli_path, weights=bad)
    assert result.status == "halted" and result.exit_code == 11
    assert "weight" in result.detail.lower()


# =================================================== validation faults

def test_an_underdetermined_system_is_refused(cli_path):
    result = fit([[1.0, 2.0, 3.0]], [1.0], cli_path)
    assert result.status == "halted" and result.exit_code == 11
    assert "n_rows must be >= n_cols" in result.detail


@pytest.mark.parametrize("tolerance", [0.0, -1e-9, 1.0, 2.0, float("inf")])
def test_rank_tolerance_outside_the_open_unit_interval_is_refused(tolerance, cli_path):
    design = [[1.0, float(i)] for i in range(1, 5)]
    result = fit(design, [1.0, 2.0, 3.0, 4.0], cli_path, rank_tolerance=tolerance)
    assert result.status == "halted" and result.exit_code == 11
    assert "rank_tolerance" in result.detail


def test_a_non_finite_design_value_is_refused(cli_path):
    design = [[1.0, float("inf")], [1.0, 2.0], [1.0, 3.0]]
    result = fit(design, [1.0, 2.0, 3.0], cli_path)
    assert result.status == "halted" and result.exit_code == 11
    assert "design matrix" in result.detail


# ======================================================== identity

def test_rank_tolerance_is_a_parameter_and_the_data_is_input():
    """The split, asserted rather than assumed: tolerance and the decision
    to weight live in the configuration; X, y and the weights live in the
    input."""
    design = [[1.0, float(i)] for i in range(1, 5)]
    response = [1.0, 2.0, 3.0, 4.0]

    base = SCLRequest(OPERATION, "cpu", encode_least_squares_configuration(1e-12),
                      encode_least_squares_input(design, response))
    looser = SCLRequest(OPERATION, "cpu", encode_least_squares_configuration(1e-6),
                        encode_least_squares_input(design, response))
    other_data = SCLRequest(OPERATION, "cpu", encode_least_squares_configuration(1e-12),
                            encode_least_squares_input(design, [1.0, 2.0, 3.0, 5.0]))

    assert base.parameters_identity() != looser.parameters_identity()
    assert base.input_identity() == looser.input_identity()
    assert base.input_identity() != other_data.input_identity()
    assert base.parameters_identity() == other_data.parameters_identity()


def test_weights_belong_to_the_input_not_the_configuration():
    """The distinction that bit once on the state-estimation observations:
    the CHOICE to weight is a parameter, the weights themselves are data."""
    design = [[1.0, float(i)] for i in range(1, 4)]
    response = [1.0, 2.0, 3.0]
    first = SCLRequest(OPERATION, "cpu", encode_least_squares_configuration(weighted=True),
                       encode_least_squares_input(design, response, [1.0, 1.0, 1.0]))
    second = SCLRequest(OPERATION, "cpu", encode_least_squares_configuration(weighted=True),
                        encode_least_squares_input(design, response, [1.0, 1.0, 5.0]))
    assert first.parameters_identity() == second.parameters_identity(), (
        "changing the weight VALUES must not change the parameter identity")
    assert first.input_identity() != second.input_identity(), (
        "changing the weight values must change the INPUT identity")

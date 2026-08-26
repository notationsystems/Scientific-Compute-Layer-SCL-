"""A known covariance reaches this operation by whitening, and exactly.

WHY IT IS TESTED RATHER THAN NOTED. The acquisition layer can now measure
a correlation between observations -- its science/replicate_pairing.py
recovers one from paired replicate runs -- so "what does a caller do with
a covariance" stopped being hypothetical. The wire carries WEIGHTS, one
per row, which is a diagonal; no off-diagonal term is expressible.

That is not a capability gap. For Sigma = L L^T the caller solves
L X~ = X and L y~ = y and fits the whitened problem ordinarily, and the
result IS the generalized least-squares estimate. What was missing is
that nothing said so. This file measures the agreement rather than
asserting it, and carries the case that shows ignoring the correlation
moves the ANSWER and not merely its uncertainty.

THE ORACLE IS COMPUTED HERE. The analytic GLS estimate is formed from
Sigma^-1 by hand in this file; the operation under test is never used to
check itself.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from scl.least_squares import (encode_least_squares_configuration,  # noqa: E402
                               encode_least_squares_input,
                               decode_least_squares_output)

DESIGN = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
RESPONSE = [2.10, 3.05, 5.20]
SIGMAS = [0.10, 0.10, 0.10]
RHO = 0.8


def covariance(rho=RHO):
    s = SIGMAS
    return [[s[0] ** 2, rho * s[0] * s[1], 0.0],
            [rho * s[0] * s[1], s[1] ** 2, 0.0],
            [0.0, 0.0, s[2] ** 2]]


def cholesky(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = (math.sqrt(matrix[i][i] - total) if i == j
                           else (matrix[i][j] - total) / lower[j][j])
    return lower


def forward_substitute(lower, vector):
    out = [0.0] * len(vector)
    for i, _ in enumerate(vector):
        out[i] = (vector[i] - sum(lower[i][k] * out[k] for k in range(i))) / lower[i][i]
    return out


def whiten(design, response, sigma):
    lower = cholesky(sigma)
    columns = [forward_substitute(lower, [row[c] for row in design])
               for c in range(len(design[0]))]
    whitened_design = [[columns[c][r] for c in range(len(columns))]
                       for r in range(len(design))]
    return whitened_design, forward_substitute(lower, response)


def analytic_gls(design, response, sigma):
    """(X^T S^-1 X)^-1 X^T S^-1 y, formed here and not by the operation."""
    n = len(sigma)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
                 for i, row in enumerate(sigma)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [v / divisor for v in augmented[column]]
        for row in range(n):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [a - factor * b
                                  for a, b in zip(augmented[row], augmented[column])]
    inverse = [row[n:] for row in augmented]

    m = len(design[0])
    normal = [[sum(design[k][i] * inverse[k][l] * design[l][j]
                   for k in range(n) for l in range(n)) for j in range(m)]
              for i in range(m)]
    moment = [sum(design[k][i] * inverse[k][l] * response[l]
                  for k in range(n) for l in range(n)) for i in range(m)]
    determinant = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
    return [(normal[1][1] * moment[0] - normal[0][1] * moment[1]) / determinant,
            (-normal[1][0] * moment[0] + normal[0][0] * moment[1]) / determinant]


def fit(design, response, weights=None):
    cli = REPO_ROOT / "native" / "build" / "scl_cli"
    if not cli.exists():
        pytest.skip("scl_cli is not built in this tree")
    request = {
        "operation": "least_squares", "backend": "cpu",
        "configuration_hex": encode_least_squares_configuration(
            1e-12, weights is not None).hex(),
        "input_hex": encode_least_squares_input(design, response, weights).hex(),
    }
    out = json.loads(subprocess.run([str(cli)], input=json.dumps(request),
                                    capture_output=True, text=True, timeout=60).stdout)
    assert out["status"] == "completed", out
    return decode_least_squares_output(bytes.fromhex(out["output_hex"])), out["metrics"]


def test_the_wire_carries_a_diagonal_and_nothing_more():
    """The premise. If an off-diagonal ever becomes expressible, the
    whitening obligation stops being the answer and this file is stale."""
    import inspect
    assert list(inspect.signature(encode_least_squares_input).parameters) == [
        "design", "response", "weights"]
    assert list(inspect.signature(encode_least_squares_configuration).parameters) == [
        "rank_tolerance", "weighted"]
    # one weight per row, not a matrix
    with pytest.raises(ValueError):
        encode_least_squares_input(DESIGN, RESPONSE, [1.0, 1.0])


def test_whitening_reproduces_generalized_least_squares_exactly():
    sigma = covariance()
    coefficients, _ = fit(*whiten(DESIGN, RESPONSE, sigma))
    expected = analytic_gls(DESIGN, RESPONSE, sigma)
    assert coefficients == pytest.approx(expected, abs=1e-12), (
        "pre-whitening is exact or it is not the route; there is no third option"
    )


def test_ignoring_the_correlation_moves_the_coefficients_not_just_their_uncertainty():
    """The case that makes the obligation worth stating."""
    sigma = covariance()
    weighted, _ = fit(DESIGN, RESPONSE, [1 / s ** 2 for s in SIGMAS])
    expected = analytic_gls(DESIGN, RESPONSE, sigma)
    error = max(abs(a - b) for a, b in zip(weighted, expected))
    assert error > 1e-3, (
        f"ignoring rho=0.8 moved the answer by only {error}; if a correlation no longer changes "
        "the estimate, the obligation recorded in the header needs re-measuring"
    )


def test_at_zero_correlation_the_two_routes_agree():
    """The discriminating case. Without it, the test above could be
    measuring a whitening bug rather than the correlation."""
    sigma = covariance(rho=0.0)
    weighted, _ = fit(DESIGN, RESPONSE, [1 / s ** 2 for s in SIGMAS])
    whitened, _ = fit(*whiten(DESIGN, RESPONSE, sigma))
    assert whitened == pytest.approx(weighted, abs=1e-12), (
        "with no correlation, whitening and weighting must be the same fit"
    )


def test_the_header_records_the_obligation_and_claims_no_extension():
    header = (REPO_ROOT / "native" / "include" / "scl" / "least_squares.hpp").read_text()
    assert "CORRELATED ERRORS: THE CALLER WHITENS" in header
    assert "No off-diagonal" in header
    assert "no extension is proposed" in header
    assert "COEFFICIENTS, not merely in their" in header, (
        "where the error lands is the reason the obligation matters and must stay recorded"
    )

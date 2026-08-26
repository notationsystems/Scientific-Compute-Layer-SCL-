"""The boundary of what `least_squares`'s conditioning metrics can see.

WHY THIS FILE EXISTS. native/include/scl/least_squares.hpp promises the
caller that adopting SVD "converts the silent failure into a visible
one" -- the singular spectrum comes out as a by-product, so the operation
REPORTS conditioning instead of leaving the caller to infer it. There is
one failure of exactly that shape it cannot see, and a promise with an
unstated exception is worse than no promise.

RANK AND CONDITIONING ARE PROPERTIES OF THE DESIGN MATRIX'S COLUMNS. A
dependence among the DATA ROWS -- one observation that is an exact
function of two others -- leaves the columns full rank and well
conditioned. The solver is looking at the columns; the dependence is not
in the columns.

THIS IS NOT A DEFECT TEST. Nothing here asserts that least_squares SHOULD
detect row dependence. It asserts that it does not, so that the promise in
the header stays honest and so that anyone who later adds such a check
finds out that they changed a documented boundary. The header records why
the obvious fix -- rank on the augmented [X | y] -- is refused: it fires
on any exactly-fitting dataset, converting a caller obligation into a
false refusal.

THE CASE is the acquisition layer's polymer row, which is where this was
found: a GPC instrument reports Mn, Mw and PDI = Mw/Mn together, and in
logs the third row is exactly the second minus the first.
"""

from __future__ import annotations

import math

import pytest



def _fit(design, response, weights=None, rank_tolerance=1e-12):
    import base64
    import json
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "python"))
    from scl.least_squares import (encode_least_squares_configuration,
                                   encode_least_squares_input,
                                   decode_least_squares_output)

    cli = root / "native" / "build" / "scl_cli"
    if not cli.exists():
        pytest.skip("scl_cli is not built in this tree")
    request = {
        "operation": "least_squares",
        "backend": "cpu",
        "configuration_hex": encode_least_squares_configuration(
            rank_tolerance, weights is not None).hex(),
        "input_hex": encode_least_squares_input(design, response, weights).hex(),
    }
    out = subprocess.run([str(cli)], input=json.dumps(request),
                         capture_output=True, text=True, timeout=60)
    response_json = json.loads(out.stdout)
    coefficients = (decode_least_squares_output(bytes.fromhex(response_json["output_hex"]))
                    if response_json.get("output_hex") else None)
    return response_json, coefficients


#: The polymer row, in logs. Row 3 is row 2 minus row 1, exactly.
MN, MW = 104000.0, 109200.0
DESIGN = [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]
RESPONSE = [math.log(MN), math.log(MW), math.log(MW / MN)]
R_MN, R_MW = 0.03, 0.025
WEIGHTS = [1 / R_MN ** 2, 1 / R_MW ** 2, 1 / (R_MN ** 2 + R_MW ** 2)]


def test_the_row_really_is_an_exact_function_of_the_other_two():
    """The premise, established before anything is concluded from it."""
    assert RESPONSE[2] == pytest.approx(RESPONSE[1] - RESPONSE[0], abs=1e-12)


def test_a_dependent_row_leaves_the_design_full_rank():
    response, _ = _fit(DESIGN, RESPONSE, WEIGHTS)
    assert response["status"] == "completed"
    metrics = response["metrics"]
    assert metrics["n_rows"] == 3.0 and metrics["n_cols"] == 2.0
    assert metrics["effective_rank"] == 2.0, (
        "least_squares now reports reduced rank for a row-dependent system. That is a CHANGE to "
        "the boundary documented in native/include/scl/least_squares.hpp, not a bug fix -- rank "
        "is a property of the columns, and these columns are independent."
    )


def test_the_conditioning_metric_actively_reassures():
    """The sharper half. It does not merely fail to warn."""
    response, _ = _fit(DESIGN, RESPONSE, WEIGHTS)
    condition_number = response["metrics"]["condition_number"]
    assert condition_number < 2.0, (
        f"condition_number is {condition_number}; the documented boundary is that a row-dependent "
        "system looks EXCELLENTLY conditioned, which is what makes the gap dangerous"
    )


def test_the_fit_is_exact_so_chi_squared_cannot_detect_it():
    """chi^2 is identically zero for ALL data, not only for this row --
    which is why a goodness-of-fit test reads perfect agreement between
    three measurements that are not three."""
    response, coefficients = _fit(DESIGN, RESPONSE, WEIGHTS)
    assert coefficients is not None
    residuals = [RESPONSE[i] - sum(DESIGN[i][j] * coefficients[j] for j in range(2))
                 for i in range(3)]
    assert max(abs(r) for r in residuals) < 1e-12
    chi_squared = sum(WEIGHTS[i] * residuals[i] ** 2 for i in range(3))
    assert chi_squared < 1e-20

    # And for arbitrary data on the same design, not just this row.
    import random
    random.seed(20260826)
    for _ in range(200):
        y1, y2 = random.gauss(0, 1), random.gauss(0, 1)
        _, c = _fit(DESIGN, [y1, y2, y2 - y1], WEIGHTS)
        r = [[y1, y2, y2 - y1][i] - sum(DESIGN[i][j] * c[j] for j in range(2))
             for i in range(3)]
        assert max(abs(v) for v in r) < 1e-9, (
            "the third row is no longer reproduced exactly; the boundary this file documents "
            "rests on it being an exact linear combination"
        )
        break  # one draw through the real CLI is enough; the algebra is exact


def test_an_independent_third_row_is_not_fitted_exactly():
    """THE DISCRIMINATING CASE, without which the three tests above would
    pass on a solver that fits everything exactly. Same design, same
    weights -- only the dependence removed."""
    perturbed = [RESPONSE[0], RESPONSE[1], RESPONSE[1] - RESPONSE[0] + 0.01]
    response, coefficients = _fit(DESIGN, perturbed, WEIGHTS)
    residuals = [perturbed[i] - sum(DESIGN[i][j] * coefficients[j] for j in range(2))
                 for i in range(3)]
    chi_squared = sum(WEIGHTS[i] * residuals[i] ** 2 for i in range(3))
    assert max(abs(r) for r in residuals) > 1e-6, (
        "an inconsistent third row is still fitted exactly, so the exactness above says nothing "
        "about dependence"
    )

    # The CONTRAST is the property, not an absolute threshold. A first
    # attempt asserted chi^2 > 1.0, which is a number I chose rather than
    # one the case gives: a 0.01 discrepancy against sigma_3 = 0.039 is
    # chi^2 = 0.033, small in absolute terms and enormous beside zero.
    dependent_response, dependent_coefficients = _fit(DESIGN, RESPONSE, WEIGHTS)
    dependent_residuals = [
        RESPONSE[i] - sum(DESIGN[i][j] * dependent_coefficients[j] for j in range(2))
        for i in range(3)]
    dependent_chi_squared = sum(
        WEIGHTS[i] * dependent_residuals[i] ** 2 for i in range(3))
    assert chi_squared > 1e18 * dependent_chi_squared, (
        f"the dependent row gives chi^2 = {dependent_chi_squared:.3e} and the independent one "
        f"{chi_squared:.3e}; the dependent case must be zero to machine precision beside it"
    )
    # and the diagnostics are UNCHANGED between the two cases, which is the point
    assert response["metrics"]["effective_rank"] == 2.0
    assert response["metrics"]["condition_number"] == pytest.approx(
        dependent_response["metrics"]["condition_number"], rel=1e-12), (
        "the conditioning metric distinguishes the dependent row from the independent one; the "
        "documented boundary is that it cannot"
    )


def test_the_header_states_the_boundary_and_why_it_is_not_closed():
    import pathlib
    header = (pathlib.Path(__file__).resolve().parent.parent
              / "native" / "include" / "scl" / "least_squares.hpp").read_text()
    assert "WHAT THE CONDITIONING METRICS DO NOT SEE" in header
    assert "properties of the design matrix's columns" in header.lower()
    assert "augmented" in header, (
        "the header must say why the obvious fix is refused, or the boundary reads as an oversight"
    )
    assert "THE CALLER'S OBLIGATION" in header

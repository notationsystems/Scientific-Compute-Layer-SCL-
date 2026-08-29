"""SCL-4: the row-dependence boundary, against a real vendor report's rows.

The boundary recorded in native/include/scl/least_squares.hpp was measured
on a constructed polymer row. Anchor 2 supplies a real one: Mn, Mw and
Mw/Mn in one table, per injection, with the ratio exactly determined by
the other two.

THE NUMBERS ARE UNVERIFIED AND THE CONCLUSION DOES NOT REST ON THEM. They
are quoted from the analysis document, which the brief itself records as
unverified against the rendered PDF. The row-dependence property is
STRUCTURAL -- it holds for any (Mn, Mw, Mw/Mn) triple -- so what is
asserted here would hold whatever the true values are. Anything that DID
depend on the values would be untrustworthy until the PDF is retrieved,
and nothing here does.
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

#: Anchor 2, Impact Analytical R190048 -- AS QUOTED, NOT AS VERIFIED.
INJECTIONS = (("injection_1", 23479.0, 30985.0), ("injection_2", 26459.0, 33070.0))
R_MN, R_MW = 0.03, 0.025
DESIGN = [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]
WEIGHTS = [1 / R_MN ** 2, 1 / R_MW ** 2, 1 / (R_MN ** 2 + R_MW ** 2)]


def fit(response):
    cli = REPO_ROOT / "native" / "build" / "scl_cli"
    if not cli.exists():
        pytest.skip("scl_cli is not built in this tree")
    request = {"operation": "least_squares", "backend": "cpu",
               "configuration_hex": encode_least_squares_configuration(1e-12, True).hex(),
               "input_hex": encode_least_squares_input(DESIGN, response, WEIGHTS).hex()}
    out = json.loads(subprocess.run([str(cli)], input=json.dumps(request),
                                    capture_output=True, text=True, timeout=60).stdout)
    coefficients = decode_least_squares_output(bytes.fromhex(out["output_hex"]))
    residuals = [response[i] - sum(DESIGN[i][j] * coefficients[j] for j in range(2))
                 for i in range(3)]
    chi_squared = sum(WEIGHTS[i] * residuals[i] ** 2 for i in range(3))
    return out["metrics"], chi_squared


def rows_for(mn, mw):
    return [math.log(mn), math.log(mw), math.log(mw / mn)]


@pytest.mark.parametrize("name,mn,mw", INJECTIONS)
def test_the_diagnostics_stay_silent_on_a_real_dependent_row(name, mn, mw):
    metrics, chi_squared = fit(rows_for(mn, mw))
    assert metrics["effective_rank"] == 2.0, name
    assert metrics["condition_number"] < 2.0, (
        f"{name}: condition_number {metrics['condition_number']} -- the boundary says a real "
        "dependent row looks excellently conditioned"
    )
    assert chi_squared < 1e-20, name


def test_the_discriminating_case_still_separates_on_the_real_rows():
    """Same design, same weights, dependence removed. Without this the
    three assertions above would pass on a solver that fits everything."""
    _, mn, mw = INJECTIONS[0]
    dependent_metrics, dependent_chi = fit(rows_for(mn, mw))
    perturbed = rows_for(mn, mw)
    perturbed[2] += 0.01
    metrics, chi_squared = fit(perturbed)

    assert chi_squared > 1e18 * dependent_chi, (
        f"dependent chi2 {dependent_chi:.3e} vs independent {chi_squared:.3e}"
    )
    # and the diagnostics do NOT move, which is the boundary itself
    assert metrics["effective_rank"] == dependent_metrics["effective_rank"]
    assert metrics["condition_number"] == pytest.approx(
        dependent_metrics["condition_number"], rel=1e-12), (
        "the conditioning metric now distinguishes a dependent row from an independent one; the "
        "documented boundary has moved and least_squares.hpp must be re-measured"
    )


def test_the_aggregate_dispersity_is_not_the_ratio_of_the_means():
    """The finding the brief says must survive ingest, recomputed here.

    NOTE A DISCREPANCY, recorded rather than smoothed: the brief states
    the mean of the ratios as 1.2850, and computing it from the brief's
    OWN quoted Mn and Mw gives 1.2848. The two cannot both come from these
    figures, so at least one is rounded or transcribed from elsewhere --
    which is exactly why the anchor needs verifying against the rendered
    PDF before any of it is treated as data.
    """
    mean_of_ratios = sum(mw / mn for _, mn, mw in INJECTIONS) / len(INJECTIONS)
    ratio_of_means = (sum(mw for _, _, mw in INJECTIONS) / len(INJECTIONS)) / (
        sum(mn for _, mn, _ in INJECTIONS) / len(INJECTIONS))

    assert mean_of_ratios != pytest.approx(ratio_of_means, abs=1e-6), (
        "the two aggregates coincide for these values, so this anchor cannot demonstrate the "
        "derived-as-measured failure and a different one is needed"
    )
    assert mean_of_ratios == pytest.approx(1.2848, abs=5e-4)
    assert ratio_of_means == pytest.approx(1.2827, abs=5e-4)
    assert abs(mean_of_ratios - ratio_of_means) == pytest.approx(0.0021, abs=5e-4)

"""Kalman validation against PRE-REGISTERED criteria.

This file contains no thresholds. Every number it compares against is read
from architecture/kalman_validation_preregistration.yaml, whose git history
shows it predates native/src/kalman.cpp. A test carrying its own thresholds
can have them adjusted in the commit that makes it pass, and the diff looks
like tuning; a test that reads them cannot.

The native binary computes the statistics and reaches no verdict. The
verdict is here, against numbers it did not produce.

WHAT IS NOT TESTED, deliberately: agreement between the filtered estimate
and the simulated truth. That conflates the filter, the model and the
simulator, and passes when two errors cancel.
"""

from __future__ import annotations

import math
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "architecture" / "kalman_validation_preregistration.yaml"
BINARY = REPO_ROOT / "native" / "build" / "tests" / "kalman_validate"

yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def prereg():
    return yaml.safe_load(PREREG_PATH.read_text())


@pytest.fixture(scope="module")
def runs(prereg):
    """One run per pre-registered seed. ALL of them must pass; the
    pre-registration says so, and the reason it says so is that one seed
    cannot distinguish a correct filter from a lucky one."""
    if not BINARY.exists():
        pytest.skip(f"{BINARY} not built")
    n = prereg["machine_readable"]["sample_count_N"]
    out = {}
    for seed in prereg["determinism"]["seeds"]:
        proc = subprocess.run([str(BINARY), str(seed), str(n)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"seed {seed}: {proc.stderr}"
        parsed = {"mean": {}, "rho": {}}
        for line in proc.stdout.split("\n"):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "mean":
                parsed["mean"][int(parts[1])] = float(parts[2])
            elif parts[0] == "rho":
                parsed["rho"][int(parts[1])] = float(parts[2])
            elif parts[0] in ("nis", "min_posterior_eigenvalue", "steps"):
                parsed[parts[0]] = float(parts[1])
        out[seed] = parsed
    return out


# --------------------------------------------------- the mechanism itself --

def test_the_preregistration_predates_the_implementation(prereg):
    """The claim that makes every threshold below meaningful, checked
    against git rather than taken on the file's word."""
    rev = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H",
         "--", "architecture/kalman_validation_preregistration.yaml"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    kalman = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", "native/src/kalman.cpp"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    if not rev.stdout.strip() or not kalman.stdout.strip():
        pytest.skip("one of the two files is not yet committed")
    prereg_commit = rev.stdout.split()[-1]
    kalman_commit = kalman.stdout.split()[-1]
    if prereg_commit == kalman_commit:
        pytest.fail("the pre-registration and the implementation were added in the "
                    "SAME commit, so the thresholds do not demonstrably predate the run")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prereg_commit, kalman_commit],
        cwd=str(REPO_ROOT), capture_output=True)
    assert ancestor.returncode == 0, (
        "the implementation does not descend from the commit that added the "
        "pre-registration -- the thresholds cannot be shown to predate the run")


def test_the_derived_tolerances_match_the_prose_that_states_them(prereg):
    """The machine-readable block is a re-encoding, not a second source.
    If the two disagree, the harness is applying a threshold nobody wrote
    down in words -- which is how a pre-registration quietly stops binding."""
    mr = prereg["machine_readable"]
    n, m, k = mr["sample_count_N"], mr["measurement_dimension_m"], mr["sigma_multiplier"]

    assert mr["derived_mean_and_whiteness_tolerance"] == pytest.approx(
        k / math.sqrt(n), abs=1e-15)
    assert mr["derived_nis_lower"] == pytest.approx(m - k * math.sqrt(2 * m / n), abs=1e-15)
    assert mr["derived_nis_upper"] == pytest.approx(m + k * math.sqrt(2 * m / n), abs=1e-15)

    # and the prose still says the same rounded numbers
    prose = prereg["statistics"]["innovation_covariance"]["tolerance"]
    assert f"{mr['derived_nis_lower']:.4f}" in prose
    assert f"{mr['derived_nis_upper']:.4f}" in prose
    assert f"{mr['derived_mean_and_whiteness_tolerance']:.4f}" in \
        prereg["statistics"]["innovation_whiteness"]["tolerance"]


def test_the_nis_reissue_retains_the_prior_value(prereg):
    """This file's own rule: a changed number is a reissue with the prior
    retained. The NIS band was corrected before any run and must show it."""
    entry = prereg["statistics"]["innovation_covariance"]
    assert "reissued" in entry, "the NIS band was corrected; the record must say so"
    assert "PRIOR VALUE RETAINED" in entry["reissued"]
    assert "1.9051" in entry["reissued"], "the superseded band must remain readable"


# ------------------------------------------------------- the three statistics --

def test_innovation_mean_is_zero(prereg, runs):
    tol = prereg["machine_readable"]["derived_mean_and_whiteness_tolerance"]
    for seed, run in runs.items():
        for component, mu in run["mean"].items():
            assert abs(mu) <= tol, (
                f"seed {seed}, component {component}: normalised innovation mean {mu} "
                f"exceeds the pre-registered {tol}. A biased filter -- typically a wrong "
                f"H, a wrong transition, or an unmodelled input.")


def test_innovation_covariance_matches_what_the_filter_predicts(prereg, runs):
    mr = prereg["machine_readable"]
    lo, hi = mr["derived_nis_lower"], mr["derived_nis_upper"]
    for seed, run in runs.items():
        nis = run["nis"]
        assert lo <= nis <= hi, (
            f"seed {seed}: NIS {nis} outside the pre-registered [{lo}, {hi}]. Above the "
            f"band the filter is OVERCONFIDENT -- its P or R too small; below, overly "
            f"conservative. Both are model errors an estimate-vs-truth test would hide.")


def test_innovations_are_white(prereg, runs):
    tol = prereg["machine_readable"]["derived_mean_and_whiteness_tolerance"]
    lags = prereg["machine_readable"]["lags_L"]
    for seed, run in runs.items():
        assert set(run["rho"]) == set(range(1, lags + 1))
        for lag, rho in run["rho"].items():
            assert abs(rho) <= tol, (
                f"seed {seed}, lag {lag}: autocorrelation {rho} exceeds the "
                f"pre-registered {tol}. Structure the filter did not remove -- the "
                f"strongest single indicator that the state model is wrong.")


def test_all_three_seeds_were_run_because_the_preregistration_requires_it(prereg, runs):
    assert prereg["determinism"]["all_three_must_pass"] is True
    assert set(runs) == set(prereg["determinism"]["seeds"]), (
        "the pre-registration requires every listed seed to pass; running a subset "
        "is how a seed gets selected after the fact")


def test_the_posterior_covariance_never_loses_positive_semidefiniteness(runs):
    """Not a pre-registered statistic -- a structural property of the
    Joseph form, checked because the short update form violates it
    silently after a few hundred steps."""
    for seed, run in runs.items():
        assert run["min_posterior_eigenvalue"] > 0.0, (
            f"seed {seed}: the posterior covariance reached a non-positive eigenvalue "
            f"({run['min_posterior_eigenvalue']}), which the Joseph form exists to prevent")

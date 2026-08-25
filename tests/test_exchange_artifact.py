"""The Phase 2 exchange artifact and its pinned canonicalization.

The joint prompt makes the exchange artifacts content-addressed and binds
the joint decision record to their hashes. That guarantee is only as good
as the serializer's determinism, so these tests lock it:

  * regeneration is byte-identical (a hash that moves is a hash that
    proves nothing);
  * the shared fixture matches its recorded digest, which is how the DAQ
    repository verifies it agrees with this encoding before Phase 2 is
    considered complete;
  * canonical output round-trips through an INDEPENDENT parser (PyYAML)
    back to the exact input object -- this is what caught the real bug
    that `1e+16` and `1e-5` resolve as STRINGS under YAML 1.1's float
    grammar, silently turning numbers into text in a content-addressed
    artifact;
  * the artifact contains no workload selection, which §2 forbids.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCHANGE = REPO_ROOT / "architecture" / "exchange"
sys.path.insert(0, str(EXCHANGE))

import canonical_yaml as cy  # noqa: E402
from build_scl_requirements import DOCUMENT, SUBSTRATE, WORKLOADS  # noqa: E402

yaml = pytest.importorskip("yaml")


# --- canonicalization ----------------------------------------------------

def test_serialization_is_deterministic():
    first = cy.canonical_bytes(DOCUMENT)
    second = cy.canonical_bytes(DOCUMENT)
    assert first == second
    assert cy.canonical_sha256(DOCUMENT).startswith("sha256:")


def test_shared_fixture_matches_its_recorded_digest():
    """How the two repositories confirm they agree on the encoding."""
    fixture = EXCHANGE / "canonicalization_fixture.yaml"
    recorded = (EXCHANGE / "canonicalization_fixture.sha256").read_text().strip()
    assert cy.canonical_sha256(cy.FIXTURE) == recorded
    assert cy.file_sha256(fixture) == recorded


def test_canonical_output_round_trips_through_an_independent_parser():
    """Independent implementation, not our own reader -- floats must come
    back as floats, not strings (the YAML 1.1 exponent-grammar trap)."""
    assert yaml.safe_load(cy.canonical_dump(cy.FIXTURE)) == cy.FIXTURE
    assert yaml.safe_load(cy.canonical_dump(DOCUMENT)) == DOCUMENT


@pytest.mark.parametrize("value", [1e16, 1e-5, 1e-4, 1.0, 0.0, -7.25, 0.3333333333333333, 1.5e20])
def test_every_float_round_trips_as_a_float(value):
    parsed = yaml.safe_load(cy.canonical_dump({"v": value}))["v"]
    assert isinstance(parsed, float), f"{value!r} serialized to a non-float"
    assert parsed == value


@pytest.mark.parametrize("text", ["true", "false", "null", "yes", "no", "on", "off",
                                   "1.5", "42", "", "needs: quoting", " leading",
                                   "-dash", "#hash", "~"])
def test_ambiguous_strings_survive_as_strings(text):
    parsed = yaml.safe_load(cy.canonical_dump({"v": text}))["v"]
    assert parsed == text, f"{text!r} did not survive as a string (got {parsed!r})"


def test_keys_are_sorted_at_every_level():
    document = {"b": {"z": 1, "a": 2}, "a": [{"beta": 1, "alpha": 2}]}
    lines = cy.canonical_dump(document).splitlines()
    assert lines[0].startswith("a:")
    assert "  - alpha: 2" in lines          # sorted inside a sequence entry
    nested = [ln.strip() for ln in lines]
    assert nested.index("a: 2") < nested.index("z: 1")
    assert yaml.safe_load(cy.canonical_dump(document)) == document


def test_single_letter_y_and_n_keys_are_quoted_conservatively():
    """Deliberate over-quoting. YAML 1.1 resolves bare `y`/`n` as booleans
    and implementations disagree on whether to honour that (PyYAML does
    not; other parsers do). Over-quoting always round-trips; under-quoting
    silently turns a key into `True`/`False` in a content-addressed
    artifact, so the ambiguous case is quoted rather than gambled on."""
    document = {"n": 1, "y": 2}
    text = cy.canonical_dump(document)
    assert '"n": 1' in text and '"y": 2' in text
    assert yaml.safe_load(text) == document


def test_non_finite_floats_are_refused():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            cy.canonical_dump({"v": bad})


# --- the artifact itself -------------------------------------------------

def test_committed_artifact_matches_regeneration():
    """The committed bytes are exactly what the generator produces, so the
    recorded hash refers to the file actually in the repository."""
    committed = (EXCHANGE / "scl_requirements.yaml").read_bytes()
    assert committed == cy.canonical_bytes(DOCUMENT)
    recorded = (EXCHANGE / "scl_requirements.sha256").read_text().strip()
    assert recorded == cy.canonical_sha256(DOCUMENT)
    assert recorded == cy.file_sha256(EXCHANGE / "scl_requirements.yaml")


def test_generator_is_reproducible_from_a_clean_run():
    before = (EXCHANGE / "scl_requirements.yaml").read_bytes()
    subprocess.run([sys.executable, "build_scl_requirements.py"], cwd=str(EXCHANGE),
                   check=True, capture_output=True)
    assert (EXCHANGE / "scl_requirements.yaml").read_bytes() == before


def test_artifact_contains_no_workload_selection():
    """§2: measured facts and requirements only. A requirements artifact
    that ranked or recommended would be making the joint decision on the
    decision record's behalf."""
    assert DOCUMENT["contains_workload_selection"] is False
    text = (EXCHANGE / "scl_requirements.yaml").read_text().lower()
    for forbidden in ("recommend", "we should", "selected_workload", "highest_leverage", "ranking"):
        assert forbidden not in text, f"artifact appears to make a selection: {forbidden!r}"


def test_every_workload_separates_the_three_parameter_provenances():
    for name, spec in WORKLOADS.items():
        for field in ("observation_requirements", "model_parameters", "computational_parameters"):
            assert spec[field], f"{name} is missing {field}"


def test_kalman_records_that_q_is_asserted_never_acquired():
    """§8: Q is the parameter that silently sets the estimate's
    confidence. It must be recorded as asserted, never DAQ-derived."""
    model = " ".join(WORKLOADS["kalman_filter_linear"]["model_parameters"]).lower()
    assert "q" in model and "never supplied by daq" in model
    observation = " ".join(WORKLOADS["kalman_filter_linear"]["observation_requirements"]).lower()
    assert "q" not in observation.split()
    assert WORKLOADS["kalman_filter_linear"]["uncertainty_requirements"].startswith("structured")


def test_substrate_classifications_use_the_fixed_vocabulary():
    allowed = {"EXISTING", "REUSABLE", "SMALL EXTENSION", "MISSING", "OUT OF SCOPE"}
    for name, entry in SUBSTRATE.items():
        assert entry["classification"] in allowed, f"{name}: {entry['classification']}"
        assert entry["evidence"], f"{name} has no traced evidence"


def test_implemented_workloads_report_no_missing_primitives():
    for name, spec in WORKLOADS.items():
        if spec["status_in_scl"] == "IMPLEMENTED":
            assert spec["primitives_missing"] == [], f"{name} claims IMPLEMENTED but lists gaps"


def test_linear_algebra_family_is_recorded_as_missing():
    """The measured fact that drives the whole exchange: no matrix,
    solve, or decomposition capability exists."""
    for primitive in ("matrices", "matrix_multiplication", "transpose",
                       "linear_solves", "decompositions"):
        assert SUBSTRATE[primitive]["classification"] == "MISSING"
    for workload in ("least_squares", "pca", "kalman_filter_linear"):
        assert WORKLOADS[workload]["primitives_missing"], f"{workload} should report gaps"


def test_cuda_state_does_not_claim_gpu_execution():
    assert DOCUMENT["cuda_state"]["gpu_executed"] is False
    assert DOCUMENT["cuda_state"]["compiled"] is True

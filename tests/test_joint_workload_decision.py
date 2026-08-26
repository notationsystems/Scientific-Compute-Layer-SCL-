"""The joint decision record, held to the two things it exists to provide:
a binding to artifacts neither side can retroactively alter, and an honest
account of who authored it.

The second matters as much as the first here. An earlier version of this
decision was correctly demoted to a DAQ proposal, on the grounds that a
session with write access to one repository and read-only access to the
other was one party writing both sides. This record was authored with BOTH
credentials, which is weaker than that failure and is still not
independence -- so it says so, and this file refuses to let it stop saying
so. Provenance recorded honestly beats provenance implied.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCHANGE = REPO_ROOT / "architecture" / "exchange"
DECISIONS = REPO_ROOT / "architecture" / "decisions"
RECORD = DECISIONS / "2026-08-26-joint-workload-decision.yaml"
sys.path.insert(0, str(EXCHANGE))

import canonical_yaml as cy  # noqa: E402

yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def record():
    return yaml.safe_load(RECORD.read_text())


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_record_is_a_fixed_point_of_the_shared_serializer(record):
    """If it is not, its own digest depends on who re-emitted it."""
    assert cy.canonical_bytes(record) == RECORD.read_bytes()


def test_the_recorded_digest_matches_the_bytes(record):
    recorded = (DECISIONS / "2026-08-26-joint-workload-decision.sha256").read_text().strip()
    assert recorded == _digest(RECORD)


def test_both_input_hashes_bind_to_real_artifacts(record):
    """The one guarantee the record exists to provide. Checked against
    whichever artifacts this repository actually holds -- each repository
    owns one and mirrors the other."""
    for key, filename in (("capabilities_artifact_hash", "daq_capabilities.yaml"),
                          ("requirements_artifact_hash", "scl_requirements.yaml")):
        path = EXCHANGE / filename
        if path.exists():
            assert record[key] == _digest(path), f"{filename} does not match the bound hash"


def test_the_agreement_fixture_hash_is_bound_too(record):
    """Without it, the two artifacts could be encoded differently and both
    still match their own sidecars."""
    assert record["canonicalization_fixture_hash"] == _digest(
        EXCHANGE / "canonicalization_fixture.yaml")


# ---------------------------------------------------- authorship honesty

def test_the_record_states_plainly_that_one_agent_held_both_pens(record):
    provenance = record["authorship_provenance"]
    statement = provenance["statement"].lower()
    assert "one session" in statement and "one agent" in statement
    assert "both repositories" in statement
    assert provenance["why_it_is_stated"]


def test_the_record_does_not_claim_independence(record):
    provenance = record["authorship_provenance"]
    not_claimed = " ".join(provenance["what_is_NOT_claimed"]).lower()
    assert "two independent parties" in not_claimed
    convergence = provenance["prior_convergence_noted_and_discounted"].lower()
    assert "not independent confirmation" in convergence, (
        "a concurrent agreement from the same inputs under the same rule is not evidence")


def test_the_mitigation_is_the_hashes_and_not_the_authorship(record):
    """The honest mitigation: a reader who distrusts the authorship can
    re-derive both inputs and check the digests."""
    mitigation = record["authorship_provenance"]["what_actually_mitigates_it"].lower()
    assert "not the authorship" in mitigation
    assert "re-derive" in mitigation or "regenerate" in mitigation
    assert record["verification_instructions"]


# ------------------------------------------------------- the decision

def test_the_decision_is_stated_with_its_rule(record):
    assert record["decision"]["workload"] == "fourier_transform_1d"
    assert record["decision"]["daq_extension"] == "none"
    assert record["selection_rule"]
    assert len(record["rationale"]) >= 4


def test_the_deferral_records_that_it_is_not_a_prohibition(record):
    """Kalman is deferred on sequencing and a second blocking requirement,
    not forbidden -- and the coupling argument is carried, because that is
    the part invisible to a reader in three months."""
    deferred = record["what_this_decision_defers_and_why"]
    assert deferred["this_record_does_not_foreclose_it"] is True
    coupling = deferred["the_non_scalar_extension"].lower()
    assert "one extension, not two" in coupling
    assert "silent" in coupling and "loud" in coupling, "the coupling argument must be carried"
    assert "must lead" in coupling
    assert len(deferred["what_would_have_to_be_true_first"]) >= 3


def test_the_record_binds_reissue_not_edit(record):
    assert "REISSUED rather than edited" in record["binding_rule"]

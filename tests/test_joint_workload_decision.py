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
    assert "declined to make it" in not_claimed, (
        "DAQ framed the choice; framing is not assent to the election made from it")


def test_the_mitigation_is_the_hashes_and_not_the_authorship(record):
    """The honest mitigation: a reader who distrusts the authorship can
    re-derive both inputs and check the digests."""
    mitigation = record["authorship_provenance"]["what_actually_mitigates_it"].lower()
    assert "not the authorship" in mitigation
    assert "re-derive" in mitigation or "regenerate" in mitigation
    assert record["verification_instructions"]


# ------------------------------------------------------- the decision

def test_the_decision_elects_one_of_the_two_framed_options(record):
    """DAQ framed the choice and declined to make it. A record that does
    not elect is another proposal."""
    choice = record["the_actual_choice"]
    assert choice["elected"] in ("option_a", "option_b")
    assert choice["option_a"] and choice["option_b"]
    assert record["decision"]["workload"] == "least_squares"


def test_the_rule_defect_is_recorded_as_a_defect(record):
    """Not as a preference, and not as a tradeoff. The rule cannot reach
    the highest-leverage candidate by construction, which is a property of
    the rule rather than a judgement about candidates."""
    defect = record["the_rule_defect"]
    assert defect["status"] == "RECORDED_AS_A_DEFECT"
    assert "by construction" in defect["statement"]
    assert "adjacency" in defect["why_it_is_a_defect_and_not_a_preference"]
    assert defect["the_correction"], "a defect recorded without its repair is a complaint"


def test_fourier_is_recorded_as_withdrawn_on_completion(record):
    """The trigger for the whole re-measure: recommending something
    already built is not a decision."""
    assert "ON COMPLETION" in record["the_rule_defect"]["how_it_surfaced"]
    assert "not on merit" in record["the_rule_defect"]["how_it_surfaced"]


def test_the_unelected_option_records_what_it_would_have_cost(record):
    assert record["what_convolution_would_have_cost"]
    assert "exactly where they are" in record["what_convolution_would_have_cost"]


def test_the_coupling_argument_is_carried_not_just_the_selection(record):
    """The part that would be invisible to whoever picks up Kalman."""
    carried = record["carried_forward_not_resolved"]
    coupling = carried["the_non_scalar_coupling"].lower()
    assert "one extension" in coupling
    assert "silent" in coupling and "loud" in coupling
    assert "must lead" in coupling
    assert carried["the_pass_through"] and carried["recursive_depth"]


def test_the_preconditions_record_that_the_reissue_came_first(record):
    """Ordering was load-bearing: a decision written before the
    canonicalization fix would bind digests about to change."""
    joined = " ".join(record["preconditions_met"]).lower()
    assert "resolved first" in joined
    assert "about to change" in joined


def test_the_record_binds_reissue_not_edit(record):
    assert "REISSUED rather than edited" in record["binding_rule"]


# --------------------------------------------------------------------------
# THE REISSUE CHAIN. Added after the second reissue shipped a link that
# pointed AT ITSELF: `previous_canonicalization_fixture_hash` named the new
# fixture rather than the one it replaced. Both repositories' suites were
# fully green, because nothing anywhere asserted on the chain -- the record
# CLAIMED to name its predecessor and no test ever read the claim.
#
# Same shape as the two defects clause 10 was written for, and it is worth
# naming as a third: a property stated in an artifact is not a property
# enforced by one. The record binds the artifacts it names; until now
# nothing bound the record to the record it replaced.
# --------------------------------------------------------------------------


def _fixture_hash_history():
    """Distinct values of the record's fixture hash, newest committed first.

    Returns None when history is unavailable (a shallow clone), so the
    weaker half of the check still runs rather than the whole thing being
    silently skipped."""
    import subprocess

    relative = RECORD.relative_to(REPO_ROOT).as_posix()
    log = subprocess.run(
        ["git", "log", "--format=%H", "--", relative],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if log.returncode != 0 or not log.stdout.strip():
        return None

    seen = []
    for commit in log.stdout.split():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if blob.returncode != 0:
            continue
        value = yaml.safe_load(blob.stdout).get("canonicalization_fixture_hash")
        if value and (not seen or seen[-1] != value):
            seen.append(value)
    return seen or None


def test_the_reissue_chain_does_not_point_at_itself(record):
    """The defect that motivated this block, stated as its own case.

    A record whose predecessor is itself has no predecessor: the chain
    terminates at the link that was supposed to extend it."""
    assert (record["reissue"]["previous_canonicalization_fixture_hash"]
            != record["canonicalization_fixture_hash"]), (
        "previous_canonicalization_fixture_hash names the CURRENT fixture -- "
        "the chain link points at itself and records no predecessor"
    )


def test_the_named_predecessor_is_the_one_git_actually_replaced(record):
    """The strong form: not merely that the link differs from the current
    hash, but that it is the value this reissue REPLACED.

    Checked against committed history rather than against the record's own
    say-so -- a record is not an oracle for its own provenance."""
    history = _fixture_hash_history()
    if history is None:
        pytest.skip("no git history for the record (shallow clone)")

    current = record["canonicalization_fixture_hash"]
    # Working-tree value first; git dedups to the same sequence once landed.
    chain = history if history[0] == current else [current] + history
    if len(chain) < 2:
        pytest.skip("the record has only ever had one fixture hash")

    assert record["reissue"]["previous_canonicalization_fixture_hash"] == chain[1], (
        f"the record names {record['reissue']['previous_canonicalization_fixture_hash']} "
        f"as its predecessor, but the value it replaced was {chain[1]}"
    )

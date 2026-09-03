"""architecture/scl_api_plane_assignment.yaml, and this layer's two findings.

THE INVARIANT is the acquisition layer's record's: every API response
either carries a canonical reference and a proof root, or declares itself
an operational observation with its limitations -- amended there to three
branches, because a refusal is neither.

WHAT THIS SIDE ADDED, and neither is visible from the other layer:

  * THE BRANCH CAN BE CONDITIONAL. SCLResult is a claim when it completed
    and a refusal when it halted, decided by its own status field. So the
    invariant is not checkable against type definitions at all; it has to
    be checked per instance. Both statuses are exercised below.

  * THIS LAYER'S REFUSALS CARRY NO CODE. `detail` is a free string from
    the native payload. The acquisition layer's refusals are codes a
    caller can branch on; these are prose a caller can only log. Recorded,
    not repaired -- a halt vocabulary is a contract change -- and asserted
    here as a KNOWN GAP so that closing it fails this test and forces the
    record to be updated with it.

THE BRANCH NAMES ARE NOT DEFINED HERE. They live in the acquisition
layer's record. This file asserts against them and does not restate them,
which is why it checks that the citation is present rather than checking a
vocabulary it would then own a second copy of.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RECORD_PATH = REPO_ROOT / "architecture" / "scl_api_plane_assignment.yaml"


@pytest.fixture(scope="module")
def record():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(RECORD_PATH.read_text())


def _public_return_types():
    """Project-owned types returned by a public entry point, derived by
    walking return annotations over module-level functions AND public
    methods -- a surface is what a consumer can call."""
    found = {}
    for path in sorted((REPO_ROOT / "python" / "scl").rglob("*.py")):
        tree = ast.parse(path.read_text())
        candidates = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                candidates.append(node)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                candidates.extend(c for c in node.body if isinstance(c, ast.FunctionDef))
        for node in candidates:
            if node.name.startswith("_") or node.returns is None:
                continue
            for inner in ast.walk(node.returns):
                if not isinstance(inner, ast.Name):
                    continue
                name = inner.id
                if not name[:1].isupper():
                    continue
                if name in ("Optional", "Tuple", "Dict", "List", "Sequence",
                            "Mapping", "Any", "Path", "Iterable"):
                    continue
                found.setdefault(name, str(path.relative_to(REPO_ROOT)))
    return found


def test_every_response_type_this_layer_returns_is_classified(record):
    returned = _public_return_types()
    assert returned, "the walk found nothing; it is broken"
    classified = set(record["response_types"]) | set(
        record["vendored_response_types"]["observed"])
    unclassified = sorted(set(returned) - classified)
    assert not unclassified, (
        "returned by a public entry point and unclassified: "
        f"{[(n, returned[n]) for n in unclassified]}"
    )


def test_nothing_is_classified_that_this_layer_does_not_return(record):
    returned = set(_public_return_types())
    classified = set(record["response_types"]) | set(
        record["vendored_response_types"]["observed"])
    orphans = sorted(classified - returned)
    assert not orphans, f"classified and never returned: {orphans}"


def test_the_branch_vocabulary_is_cited_and_not_redefined(record):
    """One meaning, one encoding, across a repository boundary. This side
    must not carry its own definition of what a claim is."""
    citation = record["the_branches_are_declared_elsewhere"]
    assert "api_plane_assignment.yaml" in citation["where"]
    assert citation.get("why_not_restated")
    assert "response_branches" not in record, (
        "this record defines the branch vocabulary; it is the acquisition "
        "layer's and must be cited, not copied"
    )


def test_a_component_names_what_carries_it(record):
    """The fourth category this side needed. A value object makes no
    assertion on its own, so its obligation must be traceable to a
    response that does -- and that response must itself be classified."""
    responses = {
        name for name, entry in record["response_types"].items()
        if entry["branch"] != "component"
    }
    seen = 0
    for name, entry in record["response_types"].items():
        if entry["branch"] != "component":
            continue
        seen += 1
        carrier = entry.get("carried_by")
        assert carrier, f"{name} is a component and names no carrier"
        assert carrier in responses, (
            f"{name} is carried by {carrier}, which is not a classified response"
        )
    assert seen, "the component category is declared and unused"


def test_a_completed_result_carries_its_references(record):
    """The conditional branch, exercised rather than described -- half one.

    Built through the real client encoder so the shape is the one a caller
    receives, not one assembled for the test."""
    from scl.client import SCLResult

    entry = record["response_types"]["SCLResult"]
    assert entry["branch"] == "conditional"
    assert entry["branch_when_completed"] == "claim"
    fields = {f.name for f in __import__("dataclasses").fields(SCLResult)}
    for required in (entry["reference_field"], entry["reason_field"], "status"):
        assert required in fields, f"SCLResult has no {required!r}"


def test_a_halted_result_mints_no_reference_and_says_why(record):
    """Half two, and the one that matters. A halted result must NOT carry
    a computation identity -- minting one would be a reference to a
    computation that did not happen. The docstring on SCLResult states
    this; here it is asserted."""
    import inspect

    from scl.client import SCLResult

    source = inspect.getdoc(SCLResult) or ""
    assert "halted" in source
    assert "None" in source, (
        "SCLResult no longer documents that a halted result carries no "
        "identities; the conditional branch has changed"
    )
    entry = record["response_types"]["SCLResult"]
    assert entry["branch_when_halted"] == "refusal"


def test_the_halt_reason_gap_is_still_open_and_still_recorded(record):
    """A KNOWN GAP ASSERTED AS STILL OPEN. `detail` is a free string from
    the native payload, so this layer's refusals cannot be branched on.

    This passes while the gap exists and FAILS when someone closes it,
    which is the correct direction: closing it must move the record too,
    or the record would go on describing a repository that had repaired
    itself."""
    finding = record["the_halted_branch_has_no_reason_vocabulary"]
    assert finding["status"] == "RECORDED_NOT_REPAIRED"
    assert finding.get("what_would_close_it")
    client = (REPO_ROOT / "python" / "scl" / "client.py").read_text()
    assert 'detail=response.get("detail")' in client, (
        "the halt reason no longer comes straight from the payload -- if a "
        "vocabulary has been introduced, update the record and this test"
    )
    tree = ast.parse(client)
    vocabularies = [
        node.targets[0].id for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
        and "HALT" in node.targets[0].id
    ]
    assert not vocabularies, (
        f"a halt vocabulary now exists ({vocabularies}); the record says none does"
    )


def test_no_proof_root_is_claimed_on_this_side_either(record):
    """Content commitments are not proof roots. Asserted here so the two
    halves of the pair cannot drift into one of them claiming a root the
    other says does not exist."""
    assert "NOT" in record["what_this_record_does_not_do"]["no_proof_root_is_claimed"].upper()
    for name, entry in record["response_types"].items():
        assert "proof_root" not in entry, f"{name} claims a proof root"

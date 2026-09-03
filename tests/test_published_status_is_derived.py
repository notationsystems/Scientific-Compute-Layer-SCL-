"""No claim this repository PUBLISHES about its own state may be typed.

WHAT WENT WRONG, measured 2026-09-03. architecture/exchange/scl_requirements.yaml
is the artifact the acquisition layer reads to learn what SCL can do and
what is owed. Its `least_squares` entry said:

    status_in_scl: NOT_IMPLEMENTED
    blocking_requirements[2].statement:
        "Matrices, matrix multiplication, transpose, decompositions and
         linear solves must exist in SCL. None of them do."
    blocking_requirements[2].measured_basis:
        "substrate_inventory classifies all five as MISSING"

All five exist. The operation is in the registry, has a kernel, a Python
binding and five test files. And one field away in the SAME entry,
`primitives_missing` read `[]` -- because that field had already been made
derived, with a comment saying why:

    "A list that must agree with another list is a list that will stop
     agreeing, so it is computed from the substrate instead."

The repair was applied to the list and not to its neighbours. A STATUS
that must agree with a registry is the same object as a list that must
agree with a list, and it stopped agreeing in the same way -- silently,
in the artifact a counterparty reads, while every digest check passed.

WHAT IS ASSERTED. Not that the current values are right, which would be
this file restating them. That each published self-claim is a FUNCTION of
something that cannot be wrong about it: the status of a workload comes
from the operation registry, and the status of a requirement naming
primitives comes from the substrate inventory. Both are recomputed here
independently of the generator, so a generator that starts hand-writing
either one fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCHANGE = REPO_ROOT / "architecture" / "exchange"
BUILDER = EXCHANGE / "build_scl_requirements.py"


def _requirements():
    pyyaml = pytest.importorskip("yaml")
    return pyyaml.safe_load((EXCHANGE / "scl_requirements.yaml").read_text())


def _registered_operations():
    """Parsed here rather than imported from the generator, so this is a
    second reading of the registry and not a restatement of the first."""
    source = (REPO_ROOT / "native" / "src" / "operation_registry.cpp").read_text()
    table = source.split("kOperations[] = {", 1)[1].split("};", 1)[0]
    return frozenset(re.findall(r'\{"([a-z0-9_]+)"', table))


def test_the_registry_is_readable_and_not_empty():
    """The premise. If the parse returned nothing, every status below
    would read NOT_IMPLEMENTED and the test would pass over a broken
    derivation -- the vacuous shape this pair keeps finding."""
    operations = _registered_operations()
    assert len(operations) >= 4, f"parsed {sorted(operations)} from the registry"


def test_every_published_workload_status_matches_the_registry():
    for name, workload in _requirements()["workloads"].items():
        expected = "IMPLEMENTED" if name in _registered_operations() else "NOT_IMPLEMENTED"
        assert workload["status_in_scl"] == expected, (
            f"{name} is published as {workload['status_in_scl']} and the "
            f"operation registry says {expected}"
        )


def test_least_squares_in_particular_is_published_as_built():
    """Named, because it is the row that was wrong and because a general
    check passing does not prove the specific case that motivated it was
    the one it fixed -- the re-run-the-probe rule from
    architecture/proof_integrity.yaml, applied to the artifact."""
    workloads = _requirements()["workloads"]
    assert "least_squares" in _registered_operations()
    assert workloads["least_squares"]["status_in_scl"] == "IMPLEMENTED"
    assert (REPO_ROOT / "native" / "src" / "least_squares.cpp").exists()
    assert (REPO_ROOT / "python" / "scl" / "least_squares.py").exists()


def test_having_every_primitive_does_not_publish_a_workload_as_built():
    """The discriminating case, and the reason the derivation reads the
    REGISTRY rather than `primitives_missing == []`. pca needs no missing
    primitive and is not implemented. A derivation from primitives would
    publish it as built, which is the same class of false claim in the
    other direction."""
    workloads = _requirements()["workloads"]
    pca = workloads["pca"]
    assert pca["primitives_missing"] == [], (
        "pca now has a missing primitive, so it no longer separates "
        "`has the parts` from `is built`; another workload must take its place here"
    )
    assert pca["status_in_scl"] == "NOT_IMPLEMENTED"
    assert "pca" not in _registered_operations()


def test_every_requirement_that_names_primitives_has_a_derived_status():
    """The neighbour that was left hand-written. A requirement listing the
    primitives that satisfy it must have its status computed from the
    substrate, and the outstanding set published beside it so the claim
    can be checked rather than believed."""
    document = _requirements()
    substrate = document["substrate_inventory"]
    seen = 0
    for name, workload in document["workloads"].items():
        for entry in workload["blocking_requirements"]:
            if "primitives_still_missing" not in entry:
                continue
            seen += 1
            outstanding = sorted(
                p for p in entry["primitives_still_missing"]
                if substrate[p]["classification"] == "MISSING"
            )
            assert entry["primitives_still_missing"] == outstanding, (
                f"{name}/{entry['requirement']} publishes a missing set the "
                "substrate does not support"
            )
            expected = "UNSATISFIED" if outstanding else "SATISFIED"
            assert entry["status"] == expected, (
                f"{name}/{entry['requirement']} is published {entry['status']} "
                f"with {outstanding or 'no'} primitives outstanding"
            )
    assert seen, "no requirement names the primitives that would satisfy it"


def test_the_generator_does_not_accept_a_hand_written_workload_status():
    """The property over the SOURCE, because a value can be derived today
    and hand-written tomorrow by anyone adding a workload. `workload()`
    must not take a status at all -- there is no keyword to pass one
    through, so the next author cannot supply one without deleting this
    test's subject."""
    source = BUILDER.read_text()
    signature = source.split("def workload(", 1)[1].split("):", 1)[0]
    assert "status" not in signature, (
        "workload() accepts a status again; it is derived from the registry "
        "and a parameter is a way to disagree with it"
    )
    assert '"status_in_scl": "IMPLEMENTED" if name in REGISTERED' in source

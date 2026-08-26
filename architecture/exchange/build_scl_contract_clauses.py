#!/usr/bin/env python3
"""Emit SCL's half of the three-party invariant register.

WHY THIS EXISTS. The register is derived in the acquisition repository,
because that is the only party that holds the vendored core. It can read
its own invariants.yaml and it can read the core's tree. It CANNOT read
this repository, so SCL's invariant source -- the OPERATION CONTRACT --
was not reachable from the register at all. A three-party register with
one party's source unreachable is a two-party register with a footnote.

WHAT SCL'S INVARIANT SOURCE ACTUALLY IS, stated plainly because it is not
the same shape as the other two parties':

    DAQ  architecture/invariants.yaml     ids + rules + STATUSES, machine
                                          readable, one entry per rule
    SCL  native/include/scl/operation.hpp a numbered comment block, and a
                                          suite that enumerates the
                                          registry FROM THE BINARY
    STE  nowhere                          referenced by number, defined in
                                          a brief the tree does not hold

So SCL has no status vocabulary. A clause is not `enforced` or
`vacuously_enforced`; it either has a dedicated test that has been shown
capable of failing, or it does not. This artifact therefore reports what
SCL can honestly say -- clause, title, the tests that claim it, and
whether a mutation has ever been shown to break it -- rather than
borrowing a vocabulary that means something else next door.

DERIVED, NOT RETYPED. The clause numbers and titles are parsed out of the
header itself, the test names come from the conformance suite by reading
its `def test_clauseN_...` definitions, and the mutation coverage comes
from the mutation checker's own table. Retyping any of the three would
make this artifact a second copy that can disagree with the first.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from canonical_yaml import canonical_bytes  # noqa: E402

CONTRACT_HEADER = REPO / "native" / "include" / "scl" / "operation.hpp"
CONFORMANCE_SUITE = REPO / "tests" / "test_operation_registry_contract.py"
MUTATION_CHECKER = REPO / "tests" / "mutation_check_operation_contract.py"


def clauses_from_the_header():
    """(number, title) for every clause, parsed from the contract itself.

    The header's form is `//  N. TITLE.` with continuation lines indented
    further, so a clause opener is a comment line whose first token is a
    number followed by a period.
    """
    found = {}
    for line in CONTRACT_HEADER.read_text().splitlines():
        match = re.match(r"^//\s{0,2}(\d+)\.\s+([A-Z][A-Z ,'-]*[A-Z])[.,]", line)
        if match:
            found[int(match.group(1))] = match.group(2).strip()
    return found


def tests_claiming_each_clause():
    """clause number -> the conformance tests that name it."""
    claimed = {}
    for line in CONFORMANCE_SUITE.read_text().splitlines():
        match = re.match(r"^def (test_clause(\d+)_[a-z0-9_]+)\(", line)
        if match:
            claimed.setdefault(int(match.group(2)), []).append(match.group(1))
    return {number: sorted(names) for number, names in claimed.items()}


def clauses_a_mutation_has_broken():
    """clause number -> the tests a planted mutation was REQUIRED to fail.

    Read from the mutation checker's table rather than from its output, so
    this artifact does not depend on a run having happened recently. What
    it records is which clauses have a mutation written for them at all --
    a weaker claim than 'the mutation was caught', and labelled as such.
    """
    source = MUTATION_CHECKER.read_text()
    table = source[source.index("MUTATIONS = ["):]
    covered = {}
    for entry in re.finditer(r"\(\s*(\d+),\s*\"([^\"]+)\",", table):
        covered.setdefault(int(entry.group(1)), []).append(entry.group(2))
    return {number: sorted(set(v)) for number, v in covered.items()}


TITLES = clauses_from_the_header()
TESTED = tests_claiming_each_clause()
MUTATED = clauses_a_mutation_has_broken()

CLAUSES = {}
for number in sorted(TITLES):
    tests = TESTED.get(number, [])
    mutations = MUTATED.get(number, [])
    CLAUSES[f"clause_{number:02d}"] = {
        "number": number,
        "title": TITLES[number],
        "dedicated_tests": tests,
        "mutations_written_against_it": mutations,
        "coverage": (
            "tested_and_mutation_checked" if tests and mutations
            else "tested_only" if tests
            else "no_dedicated_test"
        ),
    }

UNTESTED = sorted(k for k, v in CLAUSES.items() if v["coverage"] == "no_dedicated_test")

DOCUMENT = {
    "extends": "core@1.0.0",
    "artifact": "scl_contract_clauses",
    "owner": "scl",
    "purpose": (
        "SCL's invariant source, in the form the three-party register can join on. Emitted here "
        "because the register is derived in the acquisition repository, which cannot read this one."
    ),
    "source_of_truth": {
        "clauses": "native/include/scl/operation.hpp",
        "conformance": "tests/test_operation_registry_contract.py",
        "mutation_check": "tests/mutation_check_operation_contract.py",
        "derived_not_retyped": (
            "clause numbers and titles are parsed from the header, test names are read from the "
            "conformance suite's definitions, and mutation coverage is read from the checker's own "
            "table. Nothing here is typed twice."
        ),
    },
    "why_there_is_no_status_field": (
        "the acquisition repository's invariants carry a status -- enforced, vacuously_enforced, "
        "represented_unenforced, absent. SCL's clauses do not have one, and inventing a mapping "
        "would make the register join on a word that means something different on each side. What "
        "SCL can say instead is factual: which tests claim a clause, and whether a mutation has "
        "ever been written to break it."
    ),
    "what_the_conformance_suite_enumerates": (
        "the registry FROM THE BINARY rather than a hardcoded list, so an operation added tomorrow "
        "is held to every clause the moment it is registered. That is why SCL reports clause "
        "coverage rather than per-operation coverage: the per-operation axis is closed by "
        "construction."
    ),
    "clause_count": len(CLAUSES),
    "clauses": CLAUSES,
    "clauses_with_no_dedicated_test": UNTESTED,
    "clause_10_is_the_named_exception": (
        "clause 10 is PROBE THE RULE, NOT THE TESTS. It is a procedure the other clauses are "
        "checked by, not a property of a registry entry, so a dedicated test asserting it would be "
        "a test asserting that testing happened. It is reported here as having no dedicated test "
        "rather than being quietly excluded from the count, because a clause nobody checks and a "
        "clause that cannot be checked look identical in a list that hides one of them."
    ),
}

if __name__ == "__main__":
    payload = canonical_bytes(DOCUMENT)
    (HERE / "scl_contract_clauses.yaml").write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    (HERE / "scl_contract_clauses.sha256").write_text(digest + "\n")
    print("wrote scl_contract_clauses.yaml")
    print("clauses:", len(CLAUSES), "no dedicated test:", UNTESTED)
    print(digest)

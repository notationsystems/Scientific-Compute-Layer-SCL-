"""No test module may bind the same test name twice.

WHY THIS EXISTS. A merge landed three test functions that were already
defined later in the same file. Python keeps the LAST binding, so the
earlier three were dead code -- and pytest, which collects the module's
namespace rather than its source, reported the file green with 82 tests
while three of them did not exist. Nothing was red. Nothing could be: a
shadowed test is not a failing test, it is an absent one.

One of the three mattered. It called `repo_loads`, a name this repository
does not define, so it could never have run even unshadowed -- it was
pasted from the sibling repository where that name is a local import. A
two-parser agreement check on the fixture had been WRITTEN, reviewed, and
committed, and had never once executed.

This is the same shape as the reissue chain that pointed at itself, the
mutation harness that verified a diff was applied rather than that it
changed behaviour, and the guard that asserted absence-of-violation rather
than emptiness: a property that is stated somewhere is not thereby
enforced anywhere. The specific lesson here is narrower and worth keeping
separate -- a GREEN SUITE IS NOT EVIDENCE THAT THE TESTS IN IT RAN.

Deliberately source-level, via AST rather than via the imported module:
the imported module is exactly where the evidence has already been
destroyed, because by then the shadowed definition is gone.

Enforces the invariant `a_check_must_be_shown_capable_of_failing`
(architecture/invariants.yaml in the acquisition repository); the class
this instance belongs to is recorded in architecture/proof_integrity.yaml,
held byte-identical in both repositories.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest

TESTS = pathlib.Path(__file__).resolve().parent
MODULES = sorted(TESTS.glob("test_*.py"))


def _duplicate_bindings(path: pathlib.Path) -> dict:
    """Every module-level name bound more than once, with its line numbers.

    Functions and classes both, since a shadowed fixture or helper is the
    same defect one layer down."""
    tree = ast.parse(path.read_text())
    lines = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines[node.name].append(node.lineno)
    return {name: at for name, at in lines.items() if len(at) > 1}


def test_there_are_test_modules_to_check():
    """The guard's own domain is non-empty.

    Asserted rather than assumed: a parametrization over an empty glob
    passes vacuously, which is the failure this whole file is about."""
    assert MODULES, "no test modules found -- this guard would pass vacuously"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_level_name_is_bound_twice(path):
    duplicates = _duplicate_bindings(path)
    assert not duplicates, "\n".join(
        f"{path.name}: '{name}' is defined at lines {at} -- Python keeps only "
        f"line {at[-1]}, so the earlier {len(at) - 1} are dead code that pytest "
        f"will never run and never report"
        for name, at in sorted(duplicates.items())
    )

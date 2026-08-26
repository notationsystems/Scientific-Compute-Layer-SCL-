"""Hand-authored YAML, held to the one construct measured to diverge.

WHY THIS EXISTS, AND WHY IT IS THE WEAKER CHECK. The canonical emitter
refuses a form it cannot render identically for both readers -- but it
governs EMITTED artifacts, and the architecture YAML in this repository is
written by hand. Nothing the emitter does touches a file it did not write.

The acquisition repository runs the strong check: it holds a second,
dependency-free parser and asserts that both read every YAML file
identically. This repository has no second parser, so it cannot. It
asserts the SYNTACTIC property instead, and says so rather than implying
equivalent coverage.

WHAT DIVERGES, MEASURED across both readers rather than assumed:

    "k": {}          AGREE    both read an empty mapping
    "k": []          AGREE    both read an empty sequence
    "k": [1, 2]      DIVERGE  PyYAML [1, 2]   -- minimal reader "[1, 2]"
    "k": ["a"]       DIVERGE  PyYAML ['a']    -- minimal reader '["a"]'
    "k": {"a": 1}    DIVERGE  PyYAML {'a': 1} -- minimal reader '{"a": 1}'
    block sequence   AGREE
    "k": true        AGREE    both type it as a bool
    "k": 1.0         AGREE    both type it as a float

So it is exactly the NON-EMPTY flow collection, and it fails silently: a
list on one side, a string on the other, no error anywhere. The empty flow
forms are the emitter's documented exception and are safe.

This is not hypothetical. A flow sequence was hand-written into an
architecture artifact in this phase and read as one string by the minimal
reader; the sibling repository's two-parser check caught it. This test is
what catches it on THIS side, before it reaches a digest.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "architecture" / "exchange"))

yaml = pytest.importorskip("yaml")


def _hand_authored_yaml():
    """DERIVED, not listed. An enumeration of directories is the failure
    architecture/proof_integrity.yaml names -- and it cost exactly this
    check its coverage once already, in the sibling repository, where the
    enumerated set missed the hash-bearing joint decision record."""
    return tuple(sorted(
        path for path in REPO_ROOT.rglob("*.yaml")
        if not any(part in {".git", "build", "build_cuda", "CMakeFiles"}
                   for part in path.parts)
    ))


def _non_empty_flow_collections(text: str):
    """Flow collections with at least one entry, found through PyYAML's
    own event stream.

    The event stream rather than a regex, deliberately: this file's own
    prose contains `[1, 2]` and `{"a": 1}` inside quoted strings, and a
    regex over source text would flag the documentation of the defect as
    an instance of it."""
    events = list(yaml.parse(text))
    out = []
    for index, event in enumerate(events):
        starts_flow = (
            isinstance(event, (yaml.SequenceStartEvent, yaml.MappingStartEvent))
            and event.flow_style
        )
        if not starts_flow:
            continue
        closer = (yaml.SequenceEndEvent if isinstance(event, yaml.SequenceStartEvent)
                  else yaml.MappingEndEvent)
        if not isinstance(events[index + 1], closer):     # empty ones agree
            out.append(event.start_mark.line + 1)
    return out


def test_there_is_hand_authored_yaml_to_check():
    """The domain, asserted non-empty before anything is asserted about
    it. A sweep over no files reports no violations."""
    assert _hand_authored_yaml(), "no YAML found -- the check below is vacuous"


@pytest.mark.parametrize("path", _hand_authored_yaml(),
                         ids=lambda p: p.name)
def test_no_hand_authored_yaml_uses_a_non_empty_flow_collection(path):
    lines = _non_empty_flow_collections(path.read_text())
    assert not lines, (
        f"{path.relative_to(REPO_ROOT)} uses a non-empty flow collection at "
        f"line(s) {lines}. PyYAML reads a collection there; the sibling "
        f"repository's dependency-free reader reads ONE STRING, with no error "
        f"from either. Write it as a block sequence or mapping."
    )


def test_every_yaml_we_hold_is_on_the_surface_the_sibling_verifies():
    """The cross-repository dependency, made explicit and checked.

    Every YAML file here is byte-identical in the acquisition repository,
    which runs the real two-parser agreement check over all of them. That
    is what makes the weaker syntactic check above sufficient -- and it is
    only true while it is true. A file added here and not shared would be
    covered by nothing stronger than the flow-collection scan, and this
    test is where that decision has to be made deliberately.

    REWRITTEN 2026-08-26, when verify_pair_landed's SHARED tuple became a
    DERIVED intersection of the two trees. This test used to read that
    tuple, which made it depend on an enumeration to prove an absence --
    and the enumeration was missing three files at the moment it was
    replaced, one of them the pair checker itself.

    It cannot be replaced by reading the intersection, because that needs
    both repositories and only one is present here. So it asserts the
    property one repository CAN establish alone: every YAML this side
    holds lives on the shared surface. Byte-identity across that surface
    is then established at landing time, by the check that has both trees.
    """
    import verify_pair_landed as vpl

    surface = (REPO_ROOT / vpl.SHARED_SURFACE).resolve()
    off_surface = sorted(
        path.relative_to(REPO_ROOT) for path in _hand_authored_yaml()
        if surface not in path.resolve().parents
    )
    assert not off_surface, (
        f"these YAML files are held here but are not on the shared surface "
        f"({vpl.SHARED_SURFACE}/), so the sibling never holds them and its two-parser check "
        f"never sees them: {off_surface}. Either move them onto the surface or state here why "
        "the syntactic check alone is enough for them."
    )


def test_the_shared_surface_is_not_empty_here():
    """The domain again: a surface this repository holds nothing on would
    make the check above vacuously true."""
    import verify_pair_landed as vpl

    surface = (REPO_ROOT / vpl.SHARED_SURFACE).resolve()
    assert surface.is_dir(), f"{vpl.SHARED_SURFACE}/ does not exist here"
    held = _hand_authored_yaml()
    on_surface = [p for p in held if surface in p.resolve().parents]
    assert on_surface and len(on_surface) == len(held), (
        "the shared surface holds no YAML, so the check above asserts nothing"
    )


# ---------------------------------------------------------------------------
# THE SECOND DIVERGENCE CLASS, added after the flow-collection check PASSED a
# file the sibling's two-parser check then refused.
#
# That is the syntactic check's documented weakness arriving in practice, on
# the first artifact written after it existed: a Kalman pre-registration whose
# TOLERANCES read as strings under one parser and floats under the other.
#
#     k: 1e-9        PyYAML '1e-9'  (a STRING)   minimal reader 1e-09  DIVERGE
#     k: 1.0e-9      PyYAML 1e-09                minimal reader 1e-09  AGREE
#     k: "1e-9"      PyYAML '1e-9'               minimal reader '1e-9' AGREE
#
# YAML 1.1's float grammar -- which PyYAML's safe_load implements -- requires
# BOTH a decimal point in the mantissa and an explicit exponent sign. Without
# them the scalar is not a float, so it stays a string. The canonical emitter
# has known this for phases and formats around it; it governs what it EMITS,
# and nothing it knows reaches a file written by hand.
#
# Detectable here WITHOUT a second parser, which is why it belongs in this
# file: if PyYAML returns a str for an unquoted scalar that looks numeric,
# the two YAML versions disagree about it by construction.
# ---------------------------------------------------------------------------

_NUMERIC_LOOKING = __import__("re").compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
)


def _implicitly_typed_traps(text: str):
    """Unquoted scalars that LOOK numeric but that PyYAML types as str.

    Uses the event stream so the style of each scalar is known: a quoted
    scalar is unambiguous and is not a trap, and this file's own prose
    contains `1e-9` inside quoted strings."""
    traps = []
    for event in yaml.parse(text):
        if not isinstance(event, yaml.ScalarEvent):
            continue
        if event.style is not None:        # quoted or block: unambiguous
            continue
        if _NUMERIC_LOOKING.match(event.value) and not _is_yaml_float(event.value):
            traps.append((event.start_mark.line + 1, event.value))
    return traps


def _is_yaml_float(value: str) -> bool:
    """Whether PyYAML's YAML 1.1 resolver types this scalar as a number."""
    return isinstance(yaml.safe_load(f"v: {value}\n")["v"], (int, float))


@pytest.mark.parametrize("path", _hand_authored_yaml(), ids=lambda p: p.name)
def test_no_hand_authored_yaml_uses_a_version_dependent_number(path):
    traps = _implicitly_typed_traps(path.read_text())
    assert not traps, (
        f"{path.relative_to(REPO_ROOT)} contains scalars that look numeric but "
        f"that YAML 1.1 and YAML 1.2 type differently: {traps}. Write the "
        f"mantissa with a decimal point and the exponent with a sign "
        f"(1.0e-9, not 1e-9), or quote it if it is genuinely a string."
    )

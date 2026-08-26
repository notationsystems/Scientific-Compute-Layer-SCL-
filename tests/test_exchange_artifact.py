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
    assert lines[0].startswith('"a":')
    assert '  - "alpha": 2' in lines        # sorted inside a sequence entry
    nested = [ln.strip() for ln in lines]
    assert nested.index('"a": 2') < nested.index('"z": 1')
    assert yaml.safe_load(cy.canonical_dump(document)) == document


def test_every_string_is_quoted_values_and_keys_alike():
    """The corrected rule. "Quote only where required" was the DEFECT:
    YAML implicit typing lets two conformant parsers agree on the bytes and
    disagree on a scalar's TYPE, so a byte-identical artifact could
    hash-bind a different typed structure on each side. Unconditional
    quoting closes the class; the previous rule closed six known cases and
    left the rest passing incidentally."""
    document = {"n": 1, "y": 2, "plain": "unremarkable"}
    text = cy.canonical_dump(document)
    assert '"n": 1' in text and '"y": 2' in text
    assert '"plain": "unremarkable"' in text, "even an unambiguous string is quoted"
    assert yaml.safe_load(text) == document


TRAP_SCALARS = [
    "2026-08-25", "2026-08-25T12:00:00Z", "1:30:00", "0x1F", ".inf", ".nan",
    "yes", "no", "on", "off", "null", "~", "007", "0o777", "+5", "1_000", "True",
]


@pytest.mark.parametrize("scalar", TRAP_SCALARS)
def test_implicit_typing_traps_survive_as_strings(scalar):
    """The six measured divergences AND the ones that used to pass
    incidentally. The incidental passes are pinned deliberately: they are
    the ones that regress silently on a parser upgrade, because nothing
    was holding them."""
    for document in ({"v": scalar}, {scalar: "v"}):
        parsed = yaml.safe_load(cy.canonical_dump(document))
        assert isinstance(next(iter(parsed)), str), f"{scalar!r} as a KEY changed type"
        assert parsed == document, f"{scalar!r} did not survive as a string"


def test_the_fixture_pins_the_trap_class_itself():
    """Pinned in the SHARED fixture, so the agreement check between the two
    repositories exercises the class rather than either side's tests."""
    traps = cy.FIXTURE["implicit_typing_traps"]
    assert set(traps.values()) >= set(TRAP_SCALARS)
    reparsed = yaml.safe_load(cy.canonical_dump(cy.FIXTURE))["implicit_typing_traps"]
    assert all(isinstance(v, str) for v in reparsed.values())
    assert reparsed == traps


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


def test_every_required_primitive_is_classified_somewhere():
    """THE PROPERTY, not the inventory. Two lists -- what a workload
    REQUIRES and what the substrate CLASSIFIES -- and until this test
    nothing checked they agree.

    Found by closing the Kalman row: covariance_propagation and
    psd_handling were named in kalman_filter_linear's primitives_required
    and had NO substrate entry at all. The artifact required something it
    never inventoried, and every suite was green. Adding the two entries
    fixes that instance; asserting the correspondence fixes the class."""
    unclassified = {
        (workload, primitive)
        for workload, body in WORKLOADS.items()
        for primitive in body.get("primitives_required", ())
        if primitive not in SUBSTRATE
    }
    assert not unclassified, (
        f"required but never classified: {sorted(unclassified)}")


def test_a_missing_primitive_and_a_reported_gap_agree():
    """The other direction of the same correspondence: a workload reports
    a primitive as missing only if the substrate says it is missing, and
    reports every one that is.

    Stated as an equality rather than as a list of which primitives are
    MISSING today, because that list moved the moment Kalman landed --
    matrices, matrix_multiplication, transpose and linear_solves are all
    EXISTS now, and a test naming them would have had to be edited in the
    commit that made them true."""
    for workload, body in WORKLOADS.items():
        required = set(body.get("primitives_required", ()))
        reported = set(body.get("primitives_missing", ()))
        actually_missing = {p for p in required
                            if SUBSTRATE[p]["classification"] == "MISSING"}
        assert reported == actually_missing, (
            f"{workload}: reports {sorted(reported)} missing, substrate says "
            f"{sorted(actually_missing)}")


def test_cuda_state_does_not_claim_gpu_execution():
    assert DOCUMENT["cuda_state"]["gpu_executed"] is False
    assert DOCUMENT["cuda_state"]["compiled"] is True


# --- blocking requirements ------------------------------------------------

def _blocking_rows():
    for name, spec in WORKLOADS.items():
        for row in spec["blocking_requirements"]:
            yield name, row


def test_blocking_requirement_rows_use_a_fixed_shape_and_vocabulary():
    """A requirement arrives in the other repository as something to ANSWER
    only if it is structured. Prose in `notes` arrives as something to
    read."""
    required = {"requirement", "owner", "statement", "measured_basis",
                "consequence_if_unmet", "status"}
    for name, row in _blocking_rows():
        assert required <= set(row), f"{name}/{row.get('requirement')} is missing {required - set(row)}"
        assert row["owner"] in ("daq", "scl"), f"{name}: unknown owner {row['owner']!r}"
        assert row["status"] in ("SATISFIED", "UNSATISFIED"), f"{name}: {row['status']}"
        assert row["measured_basis"], f"{name}/{row['requirement']} states no measured basis"


def test_kalman_carries_both_blocking_dependencies_explicitly():
    """The two are independent -- structured R is a DATA-SHAPE gap and
    recursive depth is an INVARIANT gap -- so satisfying either leaves the
    other untouched. Both must appear as their own rows, not merged into
    one 'Kalman is hard' note."""
    rows = {row["requirement"]: row for row in WORKLOADS["kalman_filter_linear"]["blocking_requirements"]}
    assert "structured_measurement_uncertainty" in rows
    assert "recursive_generation_depth" in rows
    for row in rows.values():
        assert row["owner"] == "daq"
        # STATUS IS NOT PINNED TO A VALUE -- it moved from UNSATISFIED to
        # SATISFIED when DAQ closed both rows, and a test asserting the old
        # value would have had to be edited in the commit that made it
        # false. What is pinned is the OBLIGATION that comes with the
        # value: a row may only claim SATISFIED if it says who satisfied it
        # and how this side verified it, so the status cannot be flipped
        # without evidence travelling with it.
        assert row["status"] in ("UNSATISFIED", "SATISFIED")
        if row["status"] == "SATISFIED":
            assert row.get("satisfied_by"), \
                f"{row['requirement']} claims SATISFIED without naming what satisfied it"
            assert row.get("verified_how"), \
                f"{row['requirement']} claims SATISFIED without saying how this side checked it"
            assert "measured" in row["verified_how"].lower() or \
                   "re-ran" in row["verified_how"].lower(), \
                f"{row['requirement']}'s verification must be a measurement, not a reading"
    depth = rows["recursive_generation_depth"]
    assert "vacuously_enforced" in depth["statement"] + depth["measured_basis"]
    assert "proposed_rule" in depth, "the depth correction must travel as a stated rule, not a hint"
    assert "composition" in depth["proposed_rule"], "the guard clause closes composition, not just recursion"


def test_the_three_unimplemented_linear_algebra_workloads_all_carry_rows():
    for workload in ("least_squares", "pca", "kalman_filter_linear"):
        assert WORKLOADS[workload]["blocking_requirements"], f"{workload} states no blocking requirement"


def test_fourier_is_recorded_as_satisfied_and_built():
    """The worked example of what a SATISFIED row looks like, so the shape
    is legible from a case that actually closed."""
    spec = WORKLOADS["fourier_transform_1d"]
    assert spec["status_in_scl"] == "IMPLEMENTED"
    rows = spec["blocking_requirements"]
    assert rows, "the one built workload should still show its requirements"
    assert all(row["status"] == "SATISFIED" for row in rows)


def test_the_annotating_parameter_rule_travels_with_units_named():
    """The generalization, not just the dt instance -- units are the next
    case and the artifact must say so where PCA's unit requirement can
    point back at it."""
    rule = DOCUMENT["identity_model"]["annotating_vs_participating_parameters"]
    assert "PARTICIPATING" in rule and "ANNOTATING" in rule
    assert "units are the next instance" in rule.lower()
    assert "must not" in rule.lower()
    pca_units = [row for row in WORKLOADS["pca"]["blocking_requirements"]
                 if row["requirement"] == "commensurable_units_or_explicit_scaling"]
    assert len(pca_units) == 1
    assert "relates_to" in pca_units[0], "the units instance should point back at the general rule"


# --- raised findings ------------------------------------------------------

def test_execution_record_divergence_is_raised_without_being_resolved():
    """SCL raises it and does not resolve it: proposing a second
    ExecutionRecord would be the parallel-architecture failure the design
    forbids."""
    finding = DOCUMENT["daq_execution_record_finding"]
    assert finding["status"] == "RAISED_UNRESOLVED"
    assert "daf_acquisition_only" in finding["finding"]
    assert finding["measured_basis"], "a finding with no measured basis is an opinion"
    assert "absorb this one, not sit beside it" in finding["daq_s_own_stated_position"]
    assert "not proposing one" in finding["scl_position"]


def test_absent_is_not_zero_is_a_candidate_not_a_proposal():
    """Three independent arrivals is mild evidence, and the artifact must
    say mild. It also has to carry the counter-consideration, or it is an
    argument wearing an observation's clothes."""
    entry = DOCUMENT["core_vocabulary_candidates"]["absent_is_not_zero"]
    assert entry["status"] == "CANDIDATE_ONLY_NOT_PROPOSED"
    assert entry["counter_consideration"]
    assert "uncertainty_kind=absent" in entry["observation"]
    assert "n_particles" in entry["observation"]


def test_unresolved_edges_are_recorded_with_why_they_are_not_solved_here():
    edges = DOCUMENT["unresolved_edges"]
    assert "comparability_is_weaker_than_identity" in edges
    for name, edge in edges.items():
        assert edge["status"] == "RECORDED_UNRESOLVED", name
        assert edge["why_it_is_not_solved_here"], name
    spectral = edges["comparability_is_weaker_than_identity"]
    assert "not comparable as spectra" in spectral["edge"].lower()
    assert "annotating" in spectral["rule_needed_when_a_comparison_layer_exists"].lower()


# ================================================ the COLLECTION class (§2.1)
#
# The always-quote rule closed the ambiguity class one level down, among
# scalars. This is the same defect one level up, measured after that fix
# landed and repaired the same way: emitter-side, by refusing the form.


def test_a_sequence_inside_a_sequence_is_refused_at_the_writer():
    """The measured divergence, closed at the writer rather than by
    teaching one reader to cope. The divergence was between PyYAML and the
    ACQUISITION repository's dependency-free reader, which is not present
    here -- so the refusal is what this suite can check, and it checks it
    because the emitter is shared and a change to it reopens the class on
    both sides."""
    with pytest.raises(TypeError, match="sequence directly inside a sequence"):
        cy.canonical_dump({"k": [[1, 2], [3]]})

    # the documented alternative round-trips
    wrapped = {"k": [{"row": [1, 2]}]}
    assert yaml.safe_load(cy.canonical_dump(wrapped)) == wrapped


# The three collection tests that stood here were DEAD CODE: byte-identical
# names redefined below, and Python keeps only the last binding. The suite
# reported 82 tests collected and green while three of them did not exist.
#
# One of the three was not merely redundant. Its body called `repo_loads`,
# a name this repository does not define -- it was pasted from the
# acquisition repository, where it is a local import of the dependency-free
# reader. Live, it would have raised NameError on every run. Shadowed, it
# was a two-parser agreement check that had never executed and could not.
# The check itself is right and it lives where both parsers do, in the
# acquisition repository's copy of this file.
#
# tests/test_no_test_name_is_shadowed.py now makes this class impossible to
# reintroduce silently in either repository.



# ---------------- the escape class, and the two shapes only this half pinned
#
# The collection tests above came from the other half of this reissue, which
# reached the same refusal independently and swept it across both parsers more
# thoroughly. What follows is what only this half measured: a THIRD class,
# beside the collection one and wider, plus the narrowness check and the
# non-finite guard.


def test_the_refusal_also_covers_the_shapes_that_used_to_CRASH_the_emitter():
    """The refusal above is the divergence case. These are the cases that
    were not a divergence at all -- the emitter could not represent them.
    An empty collection nested in a sequence fell through to the scalar
    formatter and raised `unsupported scalar type for canonical YAML:
    list`, so a legal document shape had no canonical form. They now hit
    the same named refusal, which is the difference between "we decided
    not to encode this" and "we did not notice"."""
    for shape in ([[], [1]], [(1,)], [1, ["x"]], [{"a": 1}, ["b"]]):
        with pytest.raises(TypeError, match="sequence directly inside a sequence"):
            cy.canonical_dump({"k": shape})


def test_the_refusal_is_narrow_and_the_legal_shapes_still_emit():
    """A mapping inside a sequence, and a sequence inside a mapping, are
    both fine and unaffected. Only sequence-directly-inside-sequence is
    refused, so the refusal costs nothing any real artifact does."""
    for document in (
        {"k": ["a", "b"]},
        {"k": [{"a": 1}, {"b": 2}]},
        {"k": {"a": {"b": 1}}},
        {"k": [{"a": [1, 2]}]},
        {"k": [{}, {"a": 1}]},
        {"k": {"a": {}, "b": []}},
        {"k": [{"b": {"c": ["x"]}}]},
    ):
        assert yaml.safe_load(cy.canonical_dump(document)) == document


def test_the_fixture_pins_the_collection_class_not_only_scalars():
    """The scalar class is pinned by `implicit_typing_traps`. Until Phase
    37 nothing pinned the shapes where the block renderer's dash-collapse
    actually runs, which is where the divergence lived."""
    shapes = cy.FIXTURE["collection_shapes"]
    # The UNION of two independently-authored fixtures. Both halves of this
    # reissue closed the collection class concurrently and reached the same
    # refusal; their fixtures differed, and coverage is the entire purpose
    # of this entry, so neither replaced the other.
    assert set(shapes) >= {
        # depth -- the deepest legal interleave the dash-collapse runs on
        "empty_seq_under_a_key_in_a_sequence",
        "map_in_seq_in_map_in_seq",
        "seq_under_a_key_in_a_map_in_a_seq",
        # breadth -- empty collections in every position, and the
        # documented alternative to a bare nested sequence
        "empty_map_in_a_sequence",
        "empty_map_value",
        "empty_sequence_value",
        "nested_empty_map",
        "nested_empty_sequence",
        "sequence_of_maps",
        "sequence_of_scalars",
        "wrapped_inner_sequence",
    }
    for value in shapes.values():
        assert yaml.safe_load(cy.canonical_dump({"v": value}))["v"] == value


# ==================================================== the ESCAPE class (§2.1)
#
# A different defect from the two above, and the difference decides the
# repair. The bytes here have exactly ONE correct meaning under YAML 1.2
# and this repository's reader (PyYAML) already returns it. The
# divergence was on the DAQ side, whose dependency-free minimal reader
# left every escape as literal characters -- so it was repaired THERE,
# reader-side, and no artifact or digest moved. Pinned here anyway,
# because the emitter that produces these escapes is shared and a change
# to it would reopen the class on both sides.


@pytest.mark.parametrize("value", [
    'he said "x"', "a\\b", 'a\\"b', "a\nb", "a\tb", "a\rb",
    "C:\\Users\\x", 'he said "hi" # not a comment', 'he said "hi": and more',
    "\\", 'ends with "', "",
])
def test_every_escape_the_emitter_can_produce_round_trips_identically(value):
    """`_quote` escapes exactly five sequences -- backslash, double quote,
    newline, carriage return, tab -- and the always-quote rule sends EVERY
    string through it. So any value containing one of those arrives at the
    reader escaped, and until Phase 37 the DAQ repository's minimal reader
    returned it with the backslashes still in place."""
    text = cy.canonical_dump({"k": value})
    assert yaml.safe_load(text)["k"] == value


def test_escaped_keys_round_trip_too():
    document = {'a "b"': 1, "c\\d": 2}
    assert yaml.safe_load(cy.canonical_dump(document)) == document


def test_the_emitter_refuses_a_non_finite_float():
    """Checked here rather than assumed: the shared serializer is already
    clean on the axis that produced the NaN finding elsewhere in this
    phase. `_format_float` raises rather than emitting `.nan`/`.inf`,
    which is the same writer-side rule applied one layer down."""
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            cy.canonical_dump({"k": value})

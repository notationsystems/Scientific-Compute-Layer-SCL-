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


# --- the COLLECTION half of the same class ------------------------------

COLLECTION_SHAPES = [
    ({"k": []}, "empty sequence value"),
    ({"k": {}}, "empty map value"),
    ({"k": {"inner": []}}, "nested empty sequence"),
    ({"k": {"inner": {}}}, "nested empty map"),
    ({"k": [{}, {"a": 1}]}, "empty map INSIDE a sequence"),
    ({"k": [1, 2, 3]}, "sequence of scalars"),
    ({"k": [{"a": 1}, {"b": 2}]}, "sequence of maps"),
    ({"k": [{"row": [1, 2]}, {"row": [3]}]}, "wrapped inner sequence"),
]


@pytest.mark.parametrize("document,label", COLLECTION_SHAPES,
                         ids=[label for _, label in COLLECTION_SHAPES])
def test_every_collection_shape_round_trips(document, label):
    """The always-quote fix closed the SCALAR half of "two encodings, one
    meaning". Nothing held the collection half, and the fixture pinned no
    collection shapes at all until these were added.

    `empty map INSIDE a sequence` previously made the emitter RAISE
    "unsupported scalar type" -- it could not represent a legal document
    shape. Loud rather than silent, so never a wrong answer, but a hole."""
    assert yaml.safe_load(cy.canonical_dump(document)) == document


def test_a_sequence_inside_a_sequence_is_refused_at_the_writer():
    """MEASURED cross-parser divergence, closed the same way the scalar
    class was. The block form `- - 1` is legal YAML that PyYAML reads and
    the acquisition repository's dependency-free reader REFUSES. One side
    able to read an artifact the other cannot is the same failure as two
    sides typing a scalar differently.

    Refused at the WRITER rather than tolerated -- a reader that copes is
    not a fix, it relocates the ambiguity (section 6.2)."""
    with pytest.raises(TypeError, match="sequence directly inside a sequence"):
        cy.canonical_dump({"k": [[1, 2], [3]]})
    with pytest.raises(TypeError, match="sequence directly inside a sequence"):
        cy.canonical_dump({"k": [[], 1]})

    # the documented alternative works, and is better practice anyway:
    # a bare nested sequence gives its elements no name
    wrapped = {"k": [{"row": [1, 2]}, {"row": [3]}]}
    assert yaml.safe_load(cy.canonical_dump(wrapped)) == wrapped


def test_the_fixture_pins_collection_shapes_too():
    """The agreement fixture is what the two repositories compare. If it
    pins only scalars, only scalars are agreed."""
    shapes = cy.FIXTURE["collection_shapes"]
    assert {"empty_map_in_a_sequence", "empty_map_value", "empty_sequence_value",
            "nested_empty_map", "nested_empty_sequence", "sequence_of_maps",
            "sequence_of_scalars", "wrapped_inner_sequence"} <= set(shapes)
    assert yaml.safe_load(cy.canonical_dump(cy.FIXTURE))["collection_shapes"] == shapes


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
        assert row["status"] == "UNSATISFIED"
        assert row["owner"] == "daq"
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

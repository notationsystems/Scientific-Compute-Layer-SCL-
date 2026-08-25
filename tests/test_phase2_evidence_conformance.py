"""Phase 2: does SCL's Lennard-Jones computed result conform to STE's
REAL, existing evidence/identity/canonical-state machinery, all the way
through to derived state -- without SCL creating any parallel evidence,
identity, or canonical-state system of its own?

Every test here uses STE's actual production code (evidence.pool,
evidence.admission, execution.dispatcher, experiment.step,
materials.model_state, materials.decision, retrieval.engine) imported
from a real local checkout -- no mocks for the architectural boundary
under test, mirroring STE's own tests/test_execution_dispatcher.py, whose
`_setup`/`_dispatcher`/`_run_loop` pattern this file deliberately reuses
with SCL substituted for the Rust engine as the dispatch backend.

Skips cleanly (conftest.requires_ste) if no STE checkout is available."""

from __future__ import annotations

import functools

import pytest

from conftest import requires_ste

pytestmark = requires_ste


@pytest.fixture(autouse=True)
def _skip_if_native_missing(cli_path):
    if not cli_path.exists():
        pytest.skip("scl_cli not built")


def _import_ste():
    from evidence.admission import admit_document, admit_referent
    from evidence.pool import EvidencePool
    from evidence.types import make_document, make_referent, make_source
    from execution.dispatcher import SpecificationDispatcher
    from experiment.policy import ExperimentPolicy
    from experiment.session import make_experiment_session
    from experiment.step import run_experiment_step
    from materials.candidates import generate_candidates
    from materials.decision import evaluate_program, make_criterion
    from materials.iteration import reevaluate_program
    from materials.model_state import resolve_model_state_key
    from materials.optimization import OptimizationPolicy
    from materials.program import make_material_program_query
    from materials.selection import SelectionPolicy
    from materials.utility import ExperimentUtilityInput
    from retrieval.engine import DeterministicRetrievalEngine

    return dict(
        admit_document=admit_document, admit_referent=admit_referent, EvidencePool=EvidencePool,
        make_document=make_document, make_referent=make_referent, make_source=make_source,
        SpecificationDispatcher=SpecificationDispatcher, ExperimentPolicy=ExperimentPolicy,
        make_experiment_session=make_experiment_session, run_experiment_step=run_experiment_step,
        generate_candidates=generate_candidates, evaluate_program=evaluate_program,
        make_criterion=make_criterion, reevaluate_program=reevaluate_program,
        resolve_model_state_key=resolve_model_state_key, OptimizationPolicy=OptimizationPolicy,
        make_material_program_query=make_material_program_query, SelectionPolicy=SelectionPolicy,
        ExperimentUtilityInput=ExperimentUtilityInput, DeterministicRetrievalEngine=DeterministicRetrievalEngine,
    )


def _benefit(ste, estimate):
    if estimate.estimate is not None:
        return ste["ExperimentUtilityInput"](benefit=estimate.estimate, cost=1.0)
    return ste["ExperimentUtilityInput"](benefit=1.0, cost=1.0)


def _setup(ste):
    pool = ste["EvidencePool"]()
    source = ste["make_source"](kind="computational_campaign", name="SCL")
    pool.put_source(source)
    doc = ste["make_document"](
        source_id=source.id, raw_content="scl phase 2 lj session",
        retrieval_method="manual_entry", retrieved_at="2026-08-25T00:00:00Z",
    )
    ste["admit_document"](pool, doc)
    pool.put_document(doc)
    for key, kind in (("process-lj-cell", "process"), ("formulation-argon-pair", "formulation")):
        referent = ste["make_referent"](natural_key=key, kind=kind)
        ste["admit_referent"](pool, referent)
        pool.put_referent(referent)
    query = ste["make_material_program_query"](
        ["formulation-argon-pair"], "process-lj-cell", ("interaction_energy",)
    )
    criterion = ste["make_criterion"]("interaction_energy", "<=", 0)
    engine = ste["DeterministicRetrievalEngine"]()
    iteration = ste["reevaluate_program"](pool, engine, query, (criterion,))
    session = ste["make_experiment_session"](pool, engine, iteration, document_id=doc.id)
    return pool, session, criterion, engine


def _dispatcher(ste, cli_path):
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    def spec_for(_candidate):
        return build_lj_specification(
            epsilon=1.0, sigma=1.0, cutoff=5.0, positions=[(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)],
            backend="cpu", cli_path=cli_path,
        )

    return ste["SpecificationDispatcher"](
        spec_for=spec_for, interpret=interpret_lj_result, extracted_at="2026-08-25T00:00:00Z",
        runner=functools.partial(run_scl_specification, cli_path=cli_path),
    )


def _run_loop(ste, cli_path):
    pool, session, criterion, engine = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    policy = ste["ExperimentPolicy"](
        selection_policy=ste["SelectionPolicy"](
            allowed_action_classes=None, allow_already_represented_context=True,
            allow_redundant=True, allow_not_determinable_feasibility=True, max_selected=None,
        ),
        optimization_policy=ste["OptimizationPolicy"](
            max_candidates=1, allowed_action_classes=None, allow_indeterminate_utility=True
        ),
        utility_input_source=functools.partial(_benefit, ste),
    )
    step = ste["run_experiment_step"](
        session, candidates, _dispatcher(ste, cli_path), policy, confidence=1.0
    )
    return ste, pool, step, criterion


# --- 1/10. The decisive test: the full real loop, SCL to derived state ----
# Acquisition (fixture pool) -> Evidence (admit_record/admit_
# experimental_result) -> Observation -> [Trust: none exists yet, see
# docs/PHASE2_AUDIT.md] -> Derived State (ModelState.update), exercised
# end to end with SCL as the ONLY thing that changed versus STE's own
# tests/test_execution_dispatcher.py -- the dispatch backend.

def test_full_loop_admits_and_transitions_state(cli_path):
    ste = _import_ste()
    pool, session, criterion, engine = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    policy = ste["ExperimentPolicy"](
        selection_policy=ste["SelectionPolicy"](
            allowed_action_classes=None, allow_already_represented_context=True,
            allow_redundant=True, allow_not_determinable_feasibility=True, max_selected=None,
        ),
        optimization_policy=ste["OptimizationPolicy"](
            max_candidates=1, allowed_action_classes=None, allow_indeterminate_utility=True
        ),
        utility_input_source=functools.partial(_benefit, ste),
    )
    step = ste["run_experiment_step"](
        session, candidates, _dispatcher(ste, cli_path), policy, confidence=1.0
    )

    # Evidence: the Observation was really admitted into the pool.
    assert pool.has_observation(step.observation.id)
    assert step.observation.extraction_method == "simulation:deterministic_native_execution"
    assert step.observation.content["evidence_class"] == "computed"

    # Firewall: raw execution bookkeeping stays in the Record, not the Observation.
    assert "computation" in step.dispatched.record_raw_content
    assert "computation" not in step.observation.content
    assert "occurrence" not in step.observation.content

    # Derived state: the NEW session.state is a real ModelState with one
    # sample in exactly the cell resolve_model_state_key predicts.
    key = ste["resolve_model_state_key"](
        step.result.formulation.id, step.result.property, step.chosen_candidate_id and
        next(c for c in candidates.candidates if c.id == step.chosen_candidate_id).target_context,
    )
    assert key in step.session.state.samples
    samples = step.session.state.samples[key]
    assert len(samples) == 1
    assert samples[0].observation_id == step.observation.id
    assert samples[0].value == step.observation.content["value"]


def test_evidence_identity_is_not_contaminated_by_execution_history(cli_path):
    """Two complete loops -- two subprocess invocations, two admissions:
    the admitted Observation id is IDENTICAL both times. Mirrors STE's
    own test_evidence_identity_is_not_contaminated_by_execution_history
    for the Rust engine, now proven for SCL."""
    ste = _import_ste()
    _, _, first, _ = _run_loop(ste, cli_path)
    _, _, second, _ = _run_loop(ste, cli_path)
    assert first.observation.id == second.observation.id
    assert first.result.candidate_id == second.result.candidate_id


def test_a_halting_scl_execution_admits_nothing(cli_path):
    """Mirrors STE's own test_a_halting_execution_admits_nothing_and_is_
    recorded_failed: an SCL computation that halts (here: coincident
    particles) must leave the pool exactly as it was -- no fabricated
    measurement, no partial admission."""
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    ste = _import_ste()
    pool, session, criterion, engine = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    policy = ste["ExperimentPolicy"](
        selection_policy=ste["SelectionPolicy"](
            allowed_action_classes=None, allow_already_represented_context=True,
            allow_redundant=True, allow_not_determinable_feasibility=True, max_selected=None,
        ),
        optimization_policy=ste["OptimizationPolicy"](
            max_candidates=1, allowed_action_classes=None, allow_indeterminate_utility=True
        ),
        utility_input_source=functools.partial(_benefit, ste),
    )

    def spec_for(_candidate):
        # coincident particles -> COMPUTATION fault, no output
        return build_lj_specification(1.0, 1.0, 5.0, [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)], cli_path=cli_path)

    broken_dispatcher = ste["SpecificationDispatcher"](
        spec_for=spec_for, interpret=interpret_lj_result, extracted_at="2026-08-25T00:00:00Z",
        runner=functools.partial(run_scl_specification, cli_path=cli_path),
    )
    fingerprint_before = pool.fingerprint()
    with pytest.raises(RuntimeError, match="no output, no measurement"):
        ste["run_experiment_step"](session, candidates, broken_dispatcher, policy, confidence=1.0)
    assert pool.fingerprint() == fingerprint_before, "nothing entered the pool on a halted SCL run"


# --- 2/3. execution identity vs evidence identity are separate spaces -----

def test_execution_identity_and_evidence_identity_are_distinct_namespaces(cli_path):
    from scl.ste_adapter import build_lj_specification, run_scl_specification

    spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    ste = _import_ste()
    pool, session, criterion, engine = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    from scl.ste_adapter import interpret_lj_result
    candidate = candidates.candidates[0]
    content = interpret_lj_result(candidate, result)

    # execution.commitments identities (raw-bytes scheme) never equal an
    # evidence.identity content_hash (canonical-JSON scheme) of anything
    # meaningfully comparable -- different hash INPUT framing entirely.
    from evidence.identity import content_hash
    evidence_style_hash = content_hash(content)
    assert result.computation_identity != evidence_style_hash
    assert result.output_identity != evidence_style_hash
    assert result.specification_identity != evidence_style_hash


# --- 4. provenance completeness --------------------------------------------

def test_provenance_traces_observation_back_through_record_to_specification(cli_path):
    ste = _import_ste()
    _, pool, step, _ = _run_loop(ste, cli_path)
    assert len(step.observation.record_ids) == 1
    record_id = step.observation.record_ids[0]
    record = pool.get_record(record_id) if hasattr(pool, "get_record") else None
    # Whether or not a getter exists, the raw content itself must name
    # the specification/program/input/computation identities that
    # produced this observation -- that is what "provenance" concretely
    # means at this seam (execution.dispatcher's own raw_content format).
    assert "specification " in step.dispatched.record_raw_content
    assert "program " in step.dispatched.record_raw_content
    assert "computation " in step.dispatched.record_raw_content
    if record is not None:
        assert record.raw_content == step.dispatched.record_raw_content


# --- 5/6/7. method block + quantity + uncertainty semantics ---------------

def test_method_block_marks_inapplicable_fields_explicitly(cli_path):
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    ste = _import_ste()
    _, session, _, _ = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    content = interpret_lj_result(candidates.candidates[0], result)

    method_block = content["method_block"]
    for applicable_field in ("potential", "potential_version", "cutoff", "boundary_conditions",
                              "numerical_precision", "system_definition", "backend"):
        assert method_block[applicable_field]["applicable"] is True

    for not_applicable_field in ("integration_configuration", "initialization", "temperature",
                                  "timestep", "equilibration", "sampling_time", "thermostat",
                                  "barostat", "convergence_criteria"):
        entry = method_block[not_applicable_field]
        assert entry["applicable"] is False
        assert entry["reason"]  # a stated reason, never a bare False


def test_quantities_carry_unit_and_explicit_absent_uncertainty(cli_path):
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0), (0.7, 0.7, 0.7)], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    ste = _import_ste()
    _, session, _, _ = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    content = interpret_lj_result(candidates.candidates[0], result)

    energy_q = content["quantities"]["total_energy"]
    assert energy_q["unit"] == "epsilon"
    assert energy_q["uncertainty_kind"] == "absent"
    assert energy_q["uncertainty"] is None

    forces = content["quantities"]["forces"]
    assert len(forces) == 3
    for force in forces:
        for axis in ("fx", "fy", "fz"):
            assert force[axis]["uncertainty_kind"] == "absent"
            assert force[axis]["uncertainty"] is None
            assert force[axis]["unit"] == "epsilon_per_sigma"


def test_quantity_rejects_fabricated_and_contradictory_uncertainty():
    from scl.quantity import Quantity

    with pytest.raises(ValueError, match="uncertainty_kind"):
        Quantity(value=1.0, unit="epsilon", uncertainty=None, uncertainty_kind="not_a_real_kind")
    with pytest.raises(ValueError, match="contradiction"):
        Quantity(value=1.0, unit="epsilon", uncertainty=0.1, uncertainty_kind="absent")
    with pytest.raises(ValueError, match="requires a numeric uncertainty"):
        Quantity(value=1.0, unit="epsilon", uncertainty=None, uncertainty_kind="stated")
    with pytest.raises(ValueError, match="unit"):
        Quantity(value=1.0, unit="", uncertainty=None, uncertainty_kind="absent")


# --- 8. validation stays advisory, never gates a state transition ---------

def test_evaluate_program_is_advisory_and_never_blocks_the_write_path(cli_path):
    """materials.decision.evaluate_program is real, existing validation
    machinery. Its own module docstring states it is "deterministic,
    side-effect-free, read-only"; confirmed structurally here rather than
    by running the full analyze_program pipeline (see
    test_rich_content_breaks_materials_analysis_comparison_grouping
    below for why that pipeline can't currently consume SCL's richer
    content) -- evaluate_program's module imports neither
    `evidence.pool` nor `materials.model_state`, so there is no code path
    by which calling it could have gated, required, or undone the write
    (admission + ModelState.update) that test_full_loop_admits_and_
    transitions_state already proved completes independently of it."""
    import inspect

    import materials.decision as decision_module

    source = inspect.getsource(decision_module)
    import_lines = "\n".join(line for line in source.splitlines() if line.strip().startswith(("import ", "from ")))
    assert "evidence.pool" not in import_lines
    assert "evidence import pool" not in import_lines
    assert "materials.model_state" not in import_lines
    assert "materials import model_state" not in import_lines
    assert decision_module.evaluate_program.__doc__ is not None
    assert "read-only" in decision_module.evaluate_program.__doc__


def test_rich_content_breaks_materials_analysis_comparison_grouping(cli_path):
    """A genuine, demonstrated conformance gap (Task 2:
    CONFLICT_REQUIRES_INVESTIGATION), found by actually running the real
    pipeline, not merely inspecting it: materials.analysis._comparison_
    context treats every Observation.content key except "property"/
    "value" as grouping context, and materials.analysis._group_by_
    comparison_context requires that whole context to be hashable
    (`tuple(sorted(context.items()))` as a dict key). SCL's Phase 2
    content deliberately nests `quantities`/`method_block`/`parameters`
    as dicts (Tasks 5/6's typed-quantity and method-block requirements)
    -- and dicts are not hashable, so analyze_program/analyze fails with
    TypeError the moment SCL's content reaches that code path.

    This is NOT something SCL should silently work around by flattening
    its content (that would undo Tasks 5/6) or by reaching into STE's
    materials/analysis.py (out of SCL's remit -- see docs/PHASE2_AUDIT.md
    §2's SCL/STE boundary). It is recorded here, reproduced honestly, as
    an upstream dependency: STE's own comparison-context mechanism would
    need to either exclude non-hashable content keys from grouping or
    provide a declared "context" sub-key analogous to "property"/"value"
    before richly-typed computed content (from SCL or any other backend)
    can flow through materials.analysis unmodified."""
    ste = _import_ste()
    _, pool, step, criterion = _run_loop(ste, cli_path)
    from materials.program import analyze_program, make_material_program_query
    from retrieval.engine import DeterministicRetrievalEngine

    query = make_material_program_query(["formulation-argon-pair"], "process-lj-cell", ("interaction_energy",))
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        analyze_program(pool, DeterministicRetrievalEngine(), query)


# --- 9. canonical write protection -----------------------------------------

def test_raw_scl_result_cannot_be_admitted_directly(cli_path):
    from scl.ste_adapter import build_lj_specification, run_scl_specification

    spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    ste = _import_ste()
    pool, _, _, _ = _setup(ste)
    # EvidencePool.put_observation expects a real evidence.types.Observation
    # (an id + record_ids + ...), not STE's own ExecutionResult. There is
    # no code path by which SCL's raw result type satisfies that shape.
    with pytest.raises(AttributeError):
        pool.put_observation(result)


def test_model_state_update_rejects_a_mismatched_candidate_and_result(cli_path):
    from materials.model_state import EMPTY_MODEL_STATE, update

    ste = _import_ste()
    _, _pool, step, _criterion = _run_loop(ste, cli_path)
    # Deliberately a DIFFERENT candidate from the one step.result actually
    # fulfills (different property, different requirement_ids) -- not
    # relying on generate_candidates() happening to yield more than one
    # candidate for this session's single-formulation/single-property query.
    referent = ste["make_referent"](natural_key="argon-like-pair-cell", kind="formulation")
    from materials.candidates import make_action_candidate

    wrong_candidate = make_action_candidate(
        action_class="scl_lj_pairwise_energy_forces",
        requirement_ids=("some-other-requirement-entirely",),
        formulation=referent, property="some_other_property", role="target", target_context={},
    )
    assert wrong_candidate.id != step.chosen_candidate_id
    with pytest.raises(AssertionError, match="does not match result.candidate_id"):
        update(EMPTY_MODEL_STATE, wrong_candidate, step.result, step.observation)


def test_content_addressing_makes_the_pool_append_only_not_mutable(cli_path):
    """The concrete, already-enforced mechanism behind
    class_assigned_at_ingest-style immutability in THIS codebase: ids are
    content hashes, EvidencePool has no update/delete method, so
    "reclassifying" an Observation's evidence_class after the fact
    produces a DIFFERENT Observation with a DIFFERENT id -- the original
    is untouched, still present, unchanged."""
    from evidence.types import make_observation

    ste = _import_ste()
    _, pool, step, _criterion = _run_loop(ste, cli_path)
    original = step.observation
    reclassified_content = dict(original.content)
    reclassified_content["evidence_class"] = "derived"  # attempted "reclassification"
    reclassified = make_observation(
        record_ids=original.record_ids, extraction_method=original.extraction_method,
        content=reclassified_content, confidence=1.0, extracted_at="2026-08-25T00:00:00Z",
    )
    assert reclassified.id != original.id
    assert pool.has_observation(original.id)
    stored_original = pool._observations[original.id] if hasattr(pool, "_observations") else None
    if stored_original is not None:
        assert stored_original.content["evidence_class"] == "computed"  # untouched


# --- 11. malformed computational result ------------------------------------

def test_interpret_rejects_a_halted_result(cli_path):
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    spec = build_lj_specification(1.0, 1.0, 5.0, [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    assert result.status == "halted"
    ste = _import_ste()
    _, session, _, _ = _setup(ste)
    candidates = ste["generate_candidates"](session.iteration.specification)
    with pytest.raises(ValueError, match="completed result"):
        interpret_lj_result(candidates.candidates[0], result)

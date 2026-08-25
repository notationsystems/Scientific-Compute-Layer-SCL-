"""Real STE <-> SCL integration: imports STE's ACTUAL execution.*,
evidence.types, materials.candidates modules from a local checkout (see
conftest.py) and proves scl.ste_adapter.run_scl_specification is a
working, drop-in ExecutionSpecification->ExecutionResult backend,
substitutable for execution.gromacs.run_gromacs_specification at the real
execution.dispatcher.SpecificationDispatcher seam -- no mocks, no fakes,
the genuine STE types end to end.

Skips cleanly (conftest.requires_ste) if no STE checkout is available."""

from __future__ import annotations

import functools
import math

import pytest

from conftest import requires_ste

pytestmark = requires_ste


@pytest.fixture(autouse=True)
def _skip_if_native_missing(cli_path):
    if not cli_path.exists():
        pytest.skip("scl_cli not built")


def _import_ste():
    from execution.commitments import canonical_u32, commit_hex, COMPUTATION_TAG, OUTPUT_TAG
    from execution.dispatcher import SpecificationDispatcher
    from execution.specification import ExecutionSpecification
    from evidence.types import make_referent
    from materials.candidates import make_action_candidate

    return {
        "canonical_u32": canonical_u32,
        "commit_hex": commit_hex,
        "COMPUTATION_TAG": COMPUTATION_TAG,
        "OUTPUT_TAG": OUTPUT_TAG,
        "SpecificationDispatcher": SpecificationDispatcher,
        "ExecutionSpecification": ExecutionSpecification,
        "make_referent": make_referent,
        "make_action_candidate": make_action_candidate,
    }


def test_completed_run_produces_a_real_ste_execution_result(cli_path):
    from scl.ste_adapter import build_lj_specification, run_scl_specification

    spec = build_lj_specification(
        epsilon=1.0, sigma=1.0, cutoff=5.0, positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        backend="cpu", cli_path=cli_path,
    )
    ste = _import_ste()
    assert isinstance(spec, ste["ExecutionSpecification"])

    result = run_scl_specification(spec, cli_path=cli_path)
    assert result.specification is spec
    assert result.status == "completed"
    assert result.output is not None
    assert result.computation_identity is not None
    assert result.specification_identity == spec.identity()
    assert result.program_identity == spec.program_identity()
    assert result.input_identity == spec.input_identity()

    from scl.client import decode_lj_output

    total_energy, _ = decode_lj_output(result.output)
    sr6 = (1.0 / 1.5) ** 6
    expected = 4.0 * (sr6 * sr6 - sr6)
    assert math.isclose(total_energy, expected, rel_tol=1e-12)


def test_backend_choice_is_part_of_the_program_identity(cli_path):
    from scl.ste_adapter import build_lj_specification

    cpu_spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], backend="cpu", cli_path=cli_path)
    cuda_spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], backend="cuda", cli_path=cli_path)
    assert cpu_spec.program_identity() != cuda_spec.program_identity()
    assert cpu_spec.input_identity() == cuda_spec.input_identity()


def test_kernel_version_is_part_of_the_program_mirrors_gromacs_precedent():
    """Same discipline as STE's own
    test_execution_gromacs.test_engine_version_is_part_of_the_program:
    an engine/kernel version bump must move program_identity, or a
    behavior change could hide under an unchanged identity."""
    from scl.ste_adapter import scl_program_descriptor
    from execution.specification import ExecutionSpecification

    a = ExecutionSpecification(scl_program_descriptor("scl-cli/0.1.0", "cpu"), b"", b"")
    b = ExecutionSpecification(scl_program_descriptor("scl-cli/0.2.0", "cpu"), b"", b"")
    assert a.program_identity() != b.program_identity()


def test_invalid_parameters_halt_with_no_output_and_no_computation_identity(cli_path):
    """Mirrors test_execution_gromacs.test_broken_topology_halts_at_grompp_
    with_no_output: a halted run never fabricates output or identity."""
    from scl.ste_adapter import FAULT_VALIDATION, run_scl_specification, scl_program_descriptor, scl_version_line
    from scl.client import encode_lj_configuration, encode_lj_positions
    from execution.specification import ExecutionSpecification

    version_line = scl_version_line(cli_path)
    spec = ExecutionSpecification(
        program=scl_program_descriptor(version_line, "cpu"),
        configuration=encode_lj_configuration(1.0, -1.0, 5.0),  # sigma < 0
        input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
    )
    result = run_scl_specification(spec, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == FAULT_VALIDATION
    assert result.output is None
    assert result.output_identity is None
    assert result.computation_identity is None
    assert result.detail is not None


def test_repeat_determinism_same_binary_same_machine(cli_path):
    from scl.ste_adapter import build_lj_specification, run_scl_specification

    spec = build_lj_specification(0.9, 1.05, 6.0, [(0.1, 0.2, 0.3), (1.4, -0.5, 0.2)], cli_path=cli_path)
    first = run_scl_specification(spec, cli_path=cli_path)
    second = run_scl_specification(spec, cli_path=cli_path)
    assert first.output == second.output
    assert first.computation_identity == second.computation_identity


def test_geometry_change_changes_input_and_computation_identity(cli_path):
    from scl.ste_adapter import build_lj_specification, run_scl_specification

    near = run_scl_specification(
        build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.3, 0, 0)], cli_path=cli_path), cli_path=cli_path
    )
    far = run_scl_specification(
        build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (2.8, 0, 0)], cli_path=cli_path), cli_path=cli_path
    )
    assert near.input_identity != far.input_identity
    assert near.program_identity == far.program_identity
    assert near.output != far.output
    assert near.computation_identity != far.computation_identity


def test_interpret_result_is_the_evidence_firewall(cli_path):
    """Phase 112b's EXECUTION IDENTITY != EVIDENCE IDENTITY firewall
    (execution/dispatcher.py's module docstring): interpret_lj_result's
    semantic content must carry ONLY the computed value/meaning -- now
    richer (Phase 2: evidence class, typed quantities, method block) but
    still never specification/occurrence/computation-identity
    bookkeeping, which must ride only in record_raw_content instead."""
    ste = _import_ste()
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    referent = ste["make_referent"](natural_key="argon-like-pair-cell", kind="formulation")
    candidate = ste["make_action_candidate"](
        action_class="scl_lj_pairwise_energy_forces",
        requirement_ids=("scl-phase1-smoke-requirement",),
        formulation=referent, property="interaction_energy", role="target", target_context={},
    )

    spec = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    content = interpret_lj_result(candidate, result)
    assert set(content.keys()) == {
        "property", "value", "evidence_class", "quantities", "method_block", "parameters",
    }
    assert content["evidence_class"] == "computed"
    assert isinstance(content["value"], float)
    serialized_keys = str(content.keys())
    for forbidden in ("computation", "specification", "occurrence", "program_identity"):
        assert forbidden not in serialized_keys


def test_scl_is_substitutable_at_the_real_specification_dispatcher_seam(cli_path):
    """The decisive proof: execution.dispatcher.SpecificationDispatcher
    (the SAME seam execution.gromacs.run_gromacs_specification plugs
    into, per its own module docstring's `runner` field) accepts
    run_scl_specification as `runner` with zero changes to the
    dispatcher, and dispatch() returns a real DispatchedMeasurement."""
    ste = _import_ste()
    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    referent = ste["make_referent"](natural_key="argon-like-pair-cell", kind="formulation")
    candidate = ste["make_action_candidate"](
        action_class="scl_lj_pairwise_energy_forces",
        requirement_ids=("scl-phase1-smoke-requirement",),
        formulation=referent,
        property="interaction_energy",
        role="target",
        target_context={},
    )

    def spec_for(_candidate):
        return build_lj_specification(1.0, 1.0, 5.0, [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)], cli_path=cli_path)

    def interpret(dispatched_candidate, result):
        return interpret_lj_result(dispatched_candidate, result)

    dispatcher = ste["SpecificationDispatcher"](
        spec_for=spec_for,
        interpret=interpret,
        extracted_at="2026-08-25T00:00:00Z",
        runner=functools.partial(run_scl_specification, cli_path=cli_path),
    )

    measurement = dispatcher.dispatch(candidate)
    assert "value" in measurement.content
    assert measurement.content["evidence_class"] == "computed"
    assert measurement.record_locator.startswith("execution:")
    assert "computation " in measurement.record_raw_content
    assert measurement.extraction_method == "simulation:deterministic_native_execution"

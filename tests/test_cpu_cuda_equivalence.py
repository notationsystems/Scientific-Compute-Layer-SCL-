"""Phase 3: CPU <-> CUDA scientific equivalence for the LJ pairwise
workload.

Environment fact, verified directly and unchanged since Phase 1 (see
docs/PHASE1_AUDIT.md §1, re-confirmed here): this development environment
has NO GPU (`lspci` shows no VGA/3D device, no `/dev/nvidia*`). What
CHANGED this phase: `nvidia-cuda-toolkit` (nvcc 12.0) is now installed,
so the CUDA kernel (`native/backends/cuda/lj_pairwise_cuda.cu`, written
but never built in Phase 1) can be COMPILED and LINKED for the first
time. It still cannot be GPU-EXECUTED -- there is no device to execute it
on. Every test below keeps that distinction explicit, per this phase's
own instruction: "clearly distinguish: compiled / linked / unit-tested /
GPU-executed. Do not conflate them."

CPU reference contract (Task 4 -- grounded in the actual implementation,
native/src/lj_pairwise.cpp, not invented):
  - potential: truncated Lennard-Jones 12-6, V(r) = 4*eps*[(s/r)^12-(s/r)^6]
  - cutoff: plain truncation (V=F=0 beyond cutoff; not shifted to zero at
    the boundary)
  - units: "reduced" -- energy in units of the caller-supplied epsilon,
    force in units of epsilon/sigma; SCL performs no unit conversion
  - precision: IEEE-754 double (float64) throughout, -fno-fast-math, no
    SIMD-reordering flags (native/CMakeLists.txt)
  - boundary conditions: open (no periodic images)
  - input ordering: positions are an ordered sequence; forces[i]
    corresponds to positions[i] by index (no reordering/sorting anywhere
    in the pipeline)
  - tolerance: bitwise-identical for repeated CPU runs (same binary, same
    machine -- Phase 1); no cross-backend tolerance has been established
    because no cross-backend comparison has ever been run (this file's
    whole subject)
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess

import pytest

from conftest import requires_ste
from scl.client import (
    SCLRequest,
    decode_lj_output,
    encode_lj_configuration,
    encode_lj_positions,
    run_scl_request,
)

CUDA_TOOLCHAIN_PRESENT = shutil.which("nvcc") is not None
requires_cuda_toolchain = pytest.mark.skipif(
    not CUDA_TOOLCHAIN_PRESENT,
    reason="no nvcc on PATH; environment gap, not an architectural failure (see docs/PHASE3_AUDIT.md)",
)


@pytest.fixture(autouse=True)
def _skip_if_no_cuda_binary(cuda_cli_path):
    if cuda_cli_path is None:
        pytest.skip("scl_cli could not be built with -DSCL_WITH_CUDA=ON (no nvcc, or the build failed)")


# --- compiled / linked -------------------------------------------------

@requires_cuda_toolchain
def test_cuda_kernel_compiles_and_links(cuda_cli_path):
    """COMPILED + LINKED, proven directly: the binary exists and answers
    --version. This is the ceiling of what can be claimed without a GPU
    -- see test_cuda_backend_reports_unavailable_not_a_fabricated_result
    below for what happens when this binary is actually asked to run the
    cuda backend."""
    proc = subprocess.run([str(cuda_cli_path), "--version"], capture_output=True, timeout=10)
    assert proc.returncode == 0
    assert proc.stdout.decode().strip()  # a real version string was printed


# --- honest non-claim: no GPU-executed numerical equivalence -----------

def test_cuda_backend_reports_unavailable_not_a_fabricated_result(cuda_cli_path):
    """The decisive honesty check for this phase: a scl_cli binary that
    WAS built with CUDA support, asked to run on the cuda backend, in an
    environment with NO GPU, must report BACKEND_UNAVAILABLE (exit_code
    12) -- never silently fall back to CPU, never fabricate a "cuda"
    result. Task 10's requirement made concrete and GPU-executed... as
    far as "GPU-executed" can mean anything when there is no GPU: this
    IS the real runtime cudaGetDeviceCount() path, genuinely executed,
    genuinely returning zero devices."""
    request = SCLRequest(
        operation="lj_pairwise_energy_forces", backend="cuda",
        parameters=encode_lj_configuration(1.0, 1.0, 5.0),
        input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
    )
    result = run_scl_request(request, cli_path=cuda_cli_path)
    assert result.status == "halted"
    assert result.exit_code == 12  # BACKEND_UNAVAILABLE
    assert result.backend_used == "cuda"  # echoes what was asked for, never silently "cpu"
    assert result.output is None
    assert "no CUDA device is visible" in result.detail


def test_cpu_backend_is_unaffected_by_cuda_being_compiled_in(cuda_cli_path, cli_path):
    """A CUDA-enabled build's CPU path must produce IDENTICAL results to
    the CPU-only build's CPU path -- compiling in CUDA support must not
    change CPU numerics (no shared mutable state, no macro-conditional
    math in lj_pairwise.cpp, which has no #ifdef SCL_WITH_CUDA at all)."""
    request = SCLRequest(
        operation="lj_pairwise_energy_forces", backend="cpu",
        parameters=encode_lj_configuration(0.9, 1.05, 6.0),
        input_payload=encode_lj_positions([(0.1, 0.2, 0.3), (1.4, -0.5, 0.2), (-0.6, 0.9, -1.1)]),
    )
    from_cuda_build = run_scl_request(request, cli_path=cuda_cli_path)
    from_cpu_build = run_scl_request(request, cli_path=cli_path)
    assert from_cuda_build.status == from_cpu_build.status == "completed"
    assert from_cuda_build.output == from_cpu_build.output  # bit-identical


# --- numerical equivalence scaffold (Task 6) ----------------------------
# Structured to run a REAL CPU-vs-CUDA numerical comparison the moment a
# GPU becomes available -- see the `if backend_available` branch below,
# never reached in this environment but not dead code: the CUDA CLI
# itself reports availability, so this test is honest under
# cross-environment reuse (e.g. re-run in CI with a real GPU runner)
# without any edit.

def test_numerical_equivalence_or_honest_absence(cuda_cli_path):
    """Compare CPU and CUDA energy/forces for identical input. In THIS
    environment (no GPU) this reduces to: assert the honest
    BACKEND_UNAVAILABLE outcome and go no further -- Task 6 explicitly
    forbids claiming equivalence that was not measured. On a machine
    with a real device, the same test body performs the actual
    comparison (absolute error, relative error, max elementwise error,
    RMS error) against a stated tolerance."""
    epsilon, sigma, cutoff = 1.0, 1.0, 6.0
    positions = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.6, 1.1, -0.3), (-0.9, 0.4, 0.8)]
    parameters = encode_lj_configuration(epsilon, sigma, cutoff)
    input_payload = encode_lj_positions(positions)

    cpu_request = SCLRequest(
        operation="lj_pairwise_energy_forces", backend="cpu",
        parameters=parameters, input_payload=input_payload,
    )
    cuda_request = SCLRequest(
        operation="lj_pairwise_energy_forces", backend="cuda",
        parameters=parameters, input_payload=input_payload,
    )

    cpu_result = run_scl_request(cpu_request, cli_path=cuda_cli_path)
    cuda_result = run_scl_request(cuda_request, cli_path=cuda_cli_path)
    assert cpu_result.status == "completed"

    if cuda_result.status == "halted":
        assert cuda_result.exit_code == 12
        pytest.skip(
            "no CUDA device available in this environment -- numerical equivalence NOT measured "
            "(honest absence, not a false pass; see docs/PHASE3_AUDIT.md 'Numerical Equivalence')"
        )

    # Unreached in this environment; real comparison for a future GPU-equipped run.
    cpu_energy, cpu_forces = decode_lj_output(cpu_result.output)
    cuda_energy, cuda_forces = decode_lj_output(cuda_result.output)

    abs_error = abs(cuda_energy - cpu_energy)
    rel_error = abs_error / abs(cpu_energy) if cpu_energy != 0 else abs_error
    assert abs_error < 1e-9 or rel_error < 1e-9

    max_elementwise_error = 0.0
    squared_errors = []
    for (cfx, cfy, cfz), (gfx, gfy, gfz) in zip(cpu_forces, cuda_forces):
        for c, g in ((cfx, gfx), (cfy, gfy), (cfz, gfz)):
            diff = abs(c - g)
            max_elementwise_error = max(max_elementwise_error, diff)
            squared_errors.append(diff * diff)
    rms_error = math.sqrt(sum(squared_errors) / len(squared_errors))
    assert max_elementwise_error < 1e-9
    assert rms_error < 1e-9


# --- identity separation across backend selection (Task 8) -------------

def test_backend_identity_is_separate_from_scientific_input_identity(cuda_cli_path):
    """scientific input identity != backend identity != execution
    identity -- extends Phase 1/2's coverage
    (test_backend_choice_is_part_of_the_program_identity), now exercised
    against the genuinely CUDA-capable binary."""
    positions = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
    parameters = encode_lj_configuration(1.0, 1.0, 5.0)
    input_payload = encode_lj_positions(positions)

    cpu_request = SCLRequest(operation="lj_pairwise_energy_forces", backend="cpu",
                              parameters=parameters, input_payload=input_payload)
    cuda_request = SCLRequest(operation="lj_pairwise_energy_forces", backend="cuda",
                               parameters=parameters, input_payload=input_payload)

    # Same scientific input identity regardless of backend (input bytes unchanged).
    assert cpu_request.input_identity() == cuda_request.input_identity()
    # Different operation identity: backend is part of "what will run" (folded
    # in deliberately, docs/SCL_CONTRACT.md §5), same discipline GROMACS uses
    # for its own engine version.
    assert cpu_request.operation_identity() != cuda_request.operation_identity()
    assert cpu_request.identity() != cuda_request.identity()

    cpu_result = run_scl_request(cpu_request, cli_path=cuda_cli_path)
    cuda_result = run_scl_request(cuda_request, cli_path=cuda_cli_path)
    # Execution-level: the completed CPU result carries a real computation
    # identity; the halted CUDA result carries none (Phase 1 discipline:
    # a halted run never fabricates output or identity) -- itself proof
    # that "ran on cuda" and "ran on cpu" are never conflated into one
    # shared execution identity.
    assert cpu_result.computation_identity is not None
    assert cuda_result.computation_identity is None


# --- failure semantics specific to backend selection (Task 10) ---------

def test_backend_unavailability_is_checked_before_input_validation(cuda_cli_path):
    """Documents the actual, deliberate fault-priority order in
    native/src/main.cpp: backend availability is checked BEFORE
    parameter/input validation, so a request that is ALSO invalid
    (negative sigma) still reports BACKEND_UNAVAILABLE, not VALIDATION --
    there is no point validating input for a backend that cannot run it.
    This is a real, observed behavior, recorded here as a locked
    contract rather than left implicit."""
    request = SCLRequest(
        operation="lj_pairwise_energy_forces", backend="cuda",
        parameters=encode_lj_configuration(1.0, -1.0, 5.0),  # ALSO invalid (sigma < 0)
        input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
    )
    result = run_scl_request(request, cli_path=cuda_cli_path)
    assert result.exit_code == 12  # BACKEND_UNAVAILABLE, not 11 (VALIDATION)


def test_unknown_backend_string_is_a_protocol_fault_not_a_cuda_fault(cuda_cli_path):
    """A malformed backend request through the CUDA-capable binary still
    goes through the ordinary protocol-fault path (unchanged from
    Phase 1) -- CUDA support does not introduce a new, looser parsing
    branch."""
    envelope = json.dumps({
        "operation": "lj_pairwise_energy_forces",
        "backend": "not_a_real_backend",
        "configuration_hex": encode_lj_configuration(1.0, 1.0, 5.0).hex(),
        "input_hex": encode_lj_positions([(0, 0, 0), (1, 0, 0)]).hex(),
    }).encode()
    proc = subprocess.run([str(cuda_cli_path)], input=envelope, capture_output=True, timeout=10)
    response = json.loads(proc.stdout)
    assert response["exit_code"] == 10  # PROTOCOL
    assert "unknown backend" in response["detail"]


# --- Task 9: STE conformance for the CUDA-selected path -------------------

@requires_ste
def test_cuda_selected_result_conforms_to_the_same_ste_execution_result_type(cuda_cli_path):
    """A CUDA-requesting ExecutionSpecification, run through
    run_scl_specification against the CUDA-capable binary, still produces
    STE's real ExecutionResult type -- halted, exit_code=12, no output,
    no computation_identity -- exactly the same shape Phase 1/2 already
    proved for a halted CPU computation fault. The backend that halted
    changes nothing about the TYPE the STE seam receives."""
    from scl.ste_adapter import build_lj_specification, run_scl_specification

    spec = build_lj_specification(
        epsilon=1.0, sigma=1.0, cutoff=5.0, positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
        backend="cuda", cli_path=cuda_cli_path,
    )
    result = run_scl_specification(spec, cli_path=cuda_cli_path)
    assert result.status == "halted"
    assert result.exit_code == 12
    assert result.output is None
    assert result.output_identity is None
    assert result.computation_identity is None
    # program_identity/input_identity are still computed -- a halted run
    # still names the request it answers, per execution/gromacs.py's own
    # discipline (Phase 1).
    assert result.program_identity == spec.program_identity()
    assert result.input_identity == spec.input_identity()


@requires_ste
def test_cuda_selection_through_the_real_experiment_step_admits_nothing(cuda_cli_path):
    """The full real loop (Phase 2's _run_loop pattern), with the
    dispatcher's spec_for choosing backend="cuda": run_experiment_step
    must raise (never fabricate a measurement) and the pool's fingerprint
    must be byte-identical before and after -- the SAME invariant Phase 2
    proved for a computation fault (coincident particles), now proved for
    a backend-unavailable fault specifically."""
    import functools

    from scl.ste_adapter import build_lj_specification, interpret_lj_result, run_scl_specification

    from evidence.admission import admit_document, admit_referent
    from evidence.pool import EvidencePool
    from evidence.types import make_document, make_referent, make_source
    from execution.dispatcher import SpecificationDispatcher
    from experiment.policy import ExperimentPolicy
    from experiment.session import make_experiment_session
    from experiment.step import run_experiment_step
    from materials.candidates import generate_candidates
    from materials.iteration import reevaluate_program
    from materials.decision import make_criterion
    from materials.optimization import OptimizationPolicy
    from materials.program import make_material_program_query
    from materials.selection import SelectionPolicy
    from materials.utility import ExperimentUtilityInput
    from retrieval.engine import DeterministicRetrievalEngine

    pool = EvidencePool()
    source = make_source(kind="computational_campaign", name="SCL-phase3")
    pool.put_source(source)
    doc = make_document(
        source_id=source.id, raw_content="scl phase 3 cuda-unavailable session",
        retrieval_method="manual_entry", retrieved_at="2026-08-25T00:00:00Z",
    )
    admit_document(pool, doc)
    pool.put_document(doc)
    for key, kind in (("process-lj-cell", "process"), ("formulation-argon-pair", "formulation")):
        referent = make_referent(natural_key=key, kind=kind)
        admit_referent(pool, referent)
        pool.put_referent(referent)
    query = make_material_program_query(["formulation-argon-pair"], "process-lj-cell", ("interaction_energy",))
    criterion = make_criterion("interaction_energy", "<=", 0)
    engine = DeterministicRetrievalEngine()
    iteration = reevaluate_program(pool, engine, query, (criterion,))
    session = make_experiment_session(pool, engine, iteration, document_id=doc.id)
    candidates = generate_candidates(session.iteration.specification)

    def spec_for(_candidate):
        return build_lj_specification(1.0, 1.0, 5.0, [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)],
                                       backend="cuda", cli_path=cuda_cli_path)

    dispatcher = SpecificationDispatcher(
        spec_for=spec_for, interpret=interpret_lj_result, extracted_at="2026-08-25T00:00:00Z",
        runner=functools.partial(run_scl_specification, cli_path=cuda_cli_path),
    )
    policy = ExperimentPolicy(
        selection_policy=SelectionPolicy(
            allowed_action_classes=None, allow_already_represented_context=True,
            allow_redundant=True, allow_not_determinable_feasibility=True, max_selected=None,
        ),
        optimization_policy=OptimizationPolicy(
            max_candidates=1, allowed_action_classes=None, allow_indeterminate_utility=True
        ),
        utility_input_source=lambda estimate: ExperimentUtilityInput(
            benefit=estimate.estimate if estimate.estimate is not None else 1.0, cost=1.0
        ),
    )

    fingerprint_before = pool.fingerprint()
    with pytest.raises(RuntimeError, match="no output, no measurement"):
        run_experiment_step(session, candidates, dispatcher, policy, confidence=1.0)
    assert pool.fingerprint() == fingerprint_before, "a backend-unavailable SCL run must admit nothing"

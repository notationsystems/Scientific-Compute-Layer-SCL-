"""The STE-facing seam: translates STE's own `ExecutionSpecification` /
`ExecutionResult` (execution/specification.py, execution/engine.py in the
Scientific Transformer Engine repo) into scl.client calls and back.

This module is written to be a DROP-IN alternative to
`execution.gromacs.run_gromacs_specification` in
`execution.dispatcher.SpecificationDispatcher.runner` -- same dimension
mapping, same identity discipline, same staged FAULT_* exit codes, same
posture on the trust spectrum (an external process; identities recomputed
here, never trusted from the child process). Read execution/gromacs.py
alongside this file; the parallel is deliberate.

DIMENSION MAPPING (mirrors GROMACS's program/configuration/input split,
execution/gromacs.py lines 25-30):

    program        WHAT would be computed: the kernel descriptor + the
                   scl_cli build's version line + which BACKEND ran
                   (cpu/cuda is a different "engine", exactly as two
                   GROMACS versions are two programs -- Phase 126 sec 8)
    configuration  the parameters GOVERNING the run: epsilon, sigma,
                   cutoff, as three little-endian float64
    input          the system the run is OVER: N particle positions, as
                   N*3 little-endian float64

TRUST SPECTRUM -- same tier as GROMACS, not the Rust engine's tier: SCL
knows nothing of STE's identities, so program_identity/input_identity are
computed by STE from spec (as always), and output_identity/computation_
identity are computed HERE, by this module, from the bytes scl_cli
returned -- never trusted from the child process. Nothing here is
zk-provable; nothing here claims bit-identical results across machines
(this is floating point). See docs/SCL_CONTRACT.md.

This module imports ONLY `execution.specification` and `execution.engine`
(plain data/identity types) -- never `evidence.*`, `materials.*`, or
`core.canonical`. SCL does not know canonical state exists.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Optional

from execution.commitments import COMPUTATION_TAG, OUTPUT_TAG, canonical_u32, commit_hex
from execution.engine import EngineProtocolError, ExecutionResult
from execution.specification import ExecutionSpecification

from .client import (
    SCLRequest,
    decode_lj_output,
    default_cli_path,
    encode_lj_configuration,
    encode_lj_positions,
    run_scl_request,
)
from .errors import SCLProtocolError, SCLTimeoutError

SCL_DESCRIPTOR_HEADER = b"ste.scl.lj-pairwise-energy-forces.v1"
_BACKEND_MARKER = b"\n[backend]\n"

#: Re-exported for STE-side tests, mirroring execution/gromacs.py's
#: FAULT_GROMPP etc. -- the single source of truth for the numbers is
#: native/include/scl/protocol.hpp; these must stay in sync with it
#: (test_ste_integration.py pins the agreement).
FAULT_PROTOCOL = 10
FAULT_VALIDATION = 11
FAULT_BACKEND_UNAVAILABLE = 12
FAULT_COMPUTATION = 13
FAULT_INTERNAL = 14


def scl_version_line(cli_path: Optional[pathlib.Path] = None) -> str:
    """The one line `scl_cli --version` prints -- mirrors
    execution.gromacs.gmx_version_line."""
    path = cli_path if cli_path is not None else default_cli_path()
    if not path.exists():
        raise SCLProtocolError(f"scl_cli binary not found at {path}")
    proc = subprocess.run([str(path), "--version"], capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise EngineProtocolError(f"{path} --version exited {proc.returncode}")
    line = proc.stdout.decode(errors="replace").strip()
    if not line:
        raise EngineProtocolError(f"{path} --version produced no output")
    return line


def scl_program_descriptor(version_line: str, backend: str) -> bytes:
    """Build the `program` bytes for an SCL lj_pairwise_energy_forces
    workload. The backend is part of the PROGRAM, not incidental: cpu and
    cuda are two different engines, exactly as two GROMACS versions are
    two programs (execution/gromacs.py, `gromacs_program_descriptor`)."""
    return SCL_DESCRIPTOR_HEADER + b"\n" + version_line.encode("utf-8") + _BACKEND_MARKER + backend.encode(
        "utf-8"
    )


def _split_descriptor(program: bytes) -> str:
    head, marker, backend = program.partition(_BACKEND_MARKER)
    if not head.startswith(SCL_DESCRIPTOR_HEADER) or not marker:
        raise EngineProtocolError("not an SCL lj-pairwise-energy-forces program descriptor")
    return backend.decode("utf-8")


def build_lj_specification(
    epsilon: float,
    sigma: float,
    cutoff: float,
    positions,
    backend: str = "cpu",
    cli_path: Optional[pathlib.Path] = None,
) -> ExecutionSpecification:
    """Convenience constructor: build a real ExecutionSpecification for an
    SCL LJ pairwise workload, the same way a caller would build one for
    execution.gromacs (see gromacs_program_descriptor's callers)."""
    version_line = scl_version_line(cli_path)
    program = scl_program_descriptor(version_line, backend)
    configuration = encode_lj_configuration(epsilon, sigma, cutoff)
    input_payload = encode_lj_positions(positions)
    return ExecutionSpecification(program=program, configuration=configuration, input_payload=input_payload)


def run_scl_specification(
    spec: ExecutionSpecification,
    cli_path: Optional[pathlib.Path] = None,
    timeout: float = 60.0,
) -> ExecutionResult:
    """Run one SCL computation for `spec` and return it as STE's own
    ExecutionResult -- the SAME type execution.engine.run_specification and
    execution.gromacs.run_gromacs_specification return, so downstream code
    (SpecificationDispatcher and everything past it) is backend-agnostic.

    Raises EngineProtocolError for channel-level failures (unreadable
    program descriptor, missing binary, hung process, unparseable
    response) -- never for a normal computational halt, which comes back
    as `ExecutionResult(status="halted", ...)` with `output=None`,
    exactly like a halted GROMACS run."""
    backend = _split_descriptor(spec.program)
    request = SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend=backend,
        parameters=spec.configuration,
        input_payload=spec.input_payload,
    )

    try:
        result = run_scl_request(request, cli_path=cli_path, timeout=timeout)
    except SCLTimeoutError as exc:
        raise EngineProtocolError(str(exc)) from exc
    except SCLProtocolError as exc:
        raise EngineProtocolError(str(exc)) from exc

    program_identity = spec.program_identity()
    input_identity = spec.input_identity()

    if result.status == "halted":
        return ExecutionResult(
            specification=spec,
            specification_identity=spec.identity(),
            program_identity=program_identity,
            input_identity=input_identity,
            engine_occurrence=0,
            status="halted",
            exit_code=result.exit_code,
            output=None,
            output_identity=None,
            computation_identity=None,
            detail=result.detail,
        )

    output = result.output
    output_identity = commit_hex(OUTPUT_TAG, [output])
    computation_identity = commit_hex(
        COMPUTATION_TAG,
        [
            bytes.fromhex(program_identity),
            bytes.fromhex(input_identity),
            bytes.fromhex(output_identity),
            canonical_u32(result.exit_code),
        ],
    )
    return ExecutionResult(
        specification=spec,
        specification_identity=spec.identity(),
        program_identity=program_identity,
        input_identity=input_identity,
        engine_occurrence=0,
        status="completed",
        exit_code=result.exit_code,
        output=output,
        output_identity=output_identity,
        computation_identity=computation_identity,
        detail=None,
    )


def interpret_lj_result(result: ExecutionResult) -> dict:
    """Turn a completed SCL ExecutionResult's raw `output` bytes into
    semantic content -- the firewall STE's dispatcher seam requires
    (Phase 112b, execution/dispatcher.py's module docstring):
    EXECUTION IDENTITY != EVIDENCE IDENTITY. This function is the
    `interpret` callable a caller would pass to
    `SpecificationDispatcher(interpret=interpret_lj_result, ...)` --
    it must be called by the CALLER, never by this adapter, so that the
    specification/occurrence/computation-identity bookkeeping stays out
    of the semantic content and rides only in the dispatcher's
    `record_raw_content` instead, exactly as it does for the Rust engine
    and for GROMACS."""
    if result.status != "completed" or result.output is None:
        raise ValueError("interpret_lj_result requires a completed result with output")
    total_energy, forces = decode_lj_output(result.output)
    return {"total_energy_reduced_units": total_energy, "forces_reduced_units": forces}

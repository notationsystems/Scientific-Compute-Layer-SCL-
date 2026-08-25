"""SCLRequest / SCLResult and the subprocess client that runs one against
the native `scl_cli` binary. STE-agnostic by design -- see package
docstring. This is Task 2's "SCL Request -> SCL Backend -> Computational
Result" made concrete for the one operation this Phase 1 substrate
implements: `lj_pairwise_energy_forces`.
"""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .errors import (
    SCLBackendUnavailableError,
    SCLComputationError,
    SCLInternalError,
    SCLProtocolError,
    SCLTimeoutError,
    SCLValidationError,
)
from .identity import (
    COMPUTATION_TAG,
    INPUT_TAG,
    OPERATION_TAG,
    OUTPUT_TAG,
    PARAMETERS_TAG,
    REQUEST_TAG,
    commit_hex,
)

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: exit_code -> exception type, for raise_for_result(). Kept as one table
#: so the fault vocabulary (native/include/scl/protocol.hpp) has exactly
#: one place it is mirrored on the Python side.
_FAULT_EXCEPTIONS = {
    10: SCLProtocolError,
    11: SCLValidationError,
    12: SCLBackendUnavailableError,
    13: SCLComputationError,
    14: SCLInternalError,
}


def default_cli_path() -> pathlib.Path:
    """Where native/CMakeLists.txt places the built binary."""
    return _PACKAGE_ROOT / "native" / "build" / "scl_cli"


def encode_lj_configuration(epsilon: float, sigma: float, cutoff: float) -> bytes:
    """Canonical encoding for the `lj_pairwise_energy_forces` operation's
    parameters: three little-endian float64. Matches
    native/src/main.cpp::decode_configuration exactly."""
    return struct.pack("<ddd", epsilon, sigma, cutoff)


def encode_lj_positions(positions: Sequence[Tuple[float, float, float]]) -> bytes:
    """Canonical encoding for particle positions: N * 3 little-endian
    float64. Matches native/src/main.cpp::decode_positions exactly."""
    out = b""
    for x, y, z in positions:
        out += struct.pack("<ddd", x, y, z)
    return out


def decode_lj_output(output: bytes) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Inverse of native/src/main.cpp::encode_output: 8 bytes total energy
    followed by N * 24 bytes of (fx, fy, fz)."""
    if len(output) < 8 or (len(output) - 8) % 24 != 0:
        raise SCLProtocolError(f"malformed lj_pairwise_energy_forces output: {len(output)} bytes")
    (total_energy,) = struct.unpack_from("<d", output, 0)
    forces: List[Tuple[float, float, float]] = []
    offset = 8
    while offset < len(output):
        forces.append(struct.unpack_from("<ddd", output, offset))
        offset += 24
    return total_energy, forces


@dataclass(frozen=True)
class SCLRequest:
    """One computational request to the SCL substrate: "run this operation,
    on this backend, with these parameters, over this input." A request,
    not an event -- identical requests share an identity; running one
    twice is two occurrences, tracked by the caller (STE's operation
    ledger for STE-originated requests), never by this type. Mirrors the
    shape of STE's own ExecutionSpecification one layer up."""

    operation: str
    backend: str  # "cpu" | "cuda"
    parameters: bytes
    input_payload: bytes

    def operation_identity(self) -> str:
        return commit_hex(OPERATION_TAG, [self.operation.encode("utf-8"), self.backend.encode("utf-8")])

    def parameters_identity(self) -> str:
        return commit_hex(PARAMETERS_TAG, [self.parameters])

    def input_identity(self) -> str:
        return commit_hex(INPUT_TAG, [self.input_payload])

    def identity(self) -> str:
        return commit_hex(
            REQUEST_TAG,
            [self.operation.encode("utf-8"), self.backend.encode("utf-8"), self.parameters, self.input_payload],
        )


@dataclass(frozen=True)
class SCLResult:
    """One answered SCLRequest. `output` is None exactly when `status` is
    "halted" -- and then `output_identity`/`computation_identity` are None
    too, mirroring STE's own ExecutionResult discipline: an absent output
    is never represented as b"" or any placeholder."""

    request: SCLRequest
    request_identity: str
    status: str  # "completed" | "halted"
    exit_code: int
    backend_used: str
    backend_version: str
    output: Optional[bytes]
    output_identity: Optional[str]
    computation_identity: Optional[str]
    detail: Optional[str]
    wall_clock_seconds: float
    native_compute_seconds: Optional[float]
    n_particles: Optional[int]


def raise_for_result(result: SCLResult) -> None:
    """Convert a halted SCLResult into the matching typed exception (see
    errors.py). No-op if `result.status == "completed"`. Optional
    convenience for callers that prefer exceptions over branching on
    `.status`; scl.ste_adapter does NOT call this, because STE's own
    ExecutionResult convention (execution/gromacs.py) is to return a
    halted result, not raise -- exactly the convention this function
    exists to make available for callers that want the other style."""
    if result.status == "completed":
        return
    exc_type = _FAULT_EXCEPTIONS.get(result.exit_code, SCLProtocolError)
    raise exc_type(
        f"{result.detail} (operation={result.request.operation!r}, "
        f"backend={result.request.backend!r}, exit_code={result.exit_code})"
    )


def run_scl_request(
    request: SCLRequest,
    cli_path: Optional[pathlib.Path] = None,
    timeout: float = 60.0,
) -> SCLResult:
    """Run `request` in a fresh scl_cli process and return the checked
    result. Checked, not trusted, in execution/engine.py's sense: every
    identity in the returned SCLResult is computed HERE, from bytes this
    layer already holds or the CLI's raw output bytes -- the CLI is never
    asked to assert an identity we accept unchecked.

    Raises SCLProtocolError/SCLTimeoutError for channel-level failures
    (missing binary, hung process, unparseable response). A normal
    computational halt (bad parameters, unavailable backend, a faulted
    computation) is returned as an SCLResult with status="halted" --
    see raise_for_result() to convert that into an exception instead."""
    path = cli_path if cli_path is not None else default_cli_path()
    if not path.exists():
        raise SCLProtocolError(
            f"scl_cli binary not found at {path}; build it with: "
            f"cmake -S native -B native/build && cmake --build native/build"
        )

    envelope = json.dumps(
        {
            "operation": request.operation,
            "backend": request.backend,
            "configuration_hex": request.parameters.hex(),
            "input_hex": request.input_payload.hex(),
        }
    ).encode("utf-8")

    start = time.monotonic()
    try:
        proc = subprocess.run([str(path)], input=envelope, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SCLTimeoutError(
            f"scl_cli did not answer within {timeout}s "
            f"(operation={request.operation!r}, backend={request.backend!r})"
        ) from exc
    wall_clock_seconds = time.monotonic() - start

    if proc.returncode != 0:
        raise SCLProtocolError(
            f"scl_cli exited catastrophically with code {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')!r}"
        )

    try:
        response = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SCLProtocolError(f"scl_cli produced non-JSON stdout: {proc.stdout[:400]!r}") from exc

    if not isinstance(response, dict):
        raise SCLProtocolError(f"scl_cli response is not a JSON object: {response!r}")
    for required in ("status", "exit_code", "backend_used", "backend_version"):
        if required not in response:
            raise SCLProtocolError(f"scl_cli response missing field {required!r}: {response!r}")

    status = response["status"]
    if status not in ("completed", "halted"):
        raise SCLProtocolError(f"scl_cli reported unrecognised status {status!r}: {response!r}")

    metrics = response.get("metrics") or {}
    exit_code = int(response["exit_code"])
    request_identity = request.identity()

    if status == "halted":
        return SCLResult(
            request=request,
            request_identity=request_identity,
            status="halted",
            exit_code=exit_code,
            backend_used=response["backend_used"],
            backend_version=response["backend_version"],
            output=None,
            output_identity=None,
            computation_identity=None,
            detail=response.get("detail"),
            wall_clock_seconds=wall_clock_seconds,
            native_compute_seconds=metrics.get("native_compute_seconds"),
            n_particles=metrics.get("n_particles"),
        )

    output_hex = response.get("output_hex")
    if not isinstance(output_hex, str):
        raise SCLProtocolError("scl_cli reported status=completed but output_hex is missing/null")
    output = bytes.fromhex(output_hex)
    output_identity = commit_hex(OUTPUT_TAG, [output])
    computation_identity = commit_hex(
        COMPUTATION_TAG,
        [
            bytes.fromhex(request.operation_identity()),
            bytes.fromhex(request.input_identity()),
            bytes.fromhex(output_identity),
            struct.pack("<I", exit_code),
        ],
    )
    return SCLResult(
        request=request,
        request_identity=request_identity,
        status="completed",
        exit_code=exit_code,
        backend_used=response["backend_used"],
        backend_version=response["backend_version"],
        output=output,
        output_identity=output_identity,
        computation_identity=computation_identity,
        detail=None,
        wall_clock_seconds=wall_clock_seconds,
        native_compute_seconds=metrics.get("native_compute_seconds"),
        n_particles=metrics.get("n_particles"),
    )

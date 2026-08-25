# The SCL Contract — Phase 1

Two layers, deliberately kept separate (see `python/scl/__init__.py`):

1. **`scl_cli` wire protocol** — the native process boundary. STE-agnostic.
2. **`ExecutionSpecification` ↔ `ExecutionResult` translation** — the
   STE-facing seam (`python/scl/ste_adapter.py`).

## 1. Conceptual flow (as specified in the Phase 1 brief)

```
STE Operation (ExecutionSpecification: program, configuration, input_payload)
    |
    v  scl.ste_adapter.run_scl_specification
SCL Request (SCLRequest: operation, backend, parameters, input_payload)
    |
    v  scl.client.run_scl_request  (subprocess, JSON on stdin/stdout)
SCL Backend (scl_cli process: cpu today; cuda compiled-in-but-untested — see PHASE1_AUDIT.md)
    |
    v
Computational Result (SCLResult: status, exit_code, output bytes, identities, metrics)
    |
    v  scl.ste_adapter.run_scl_specification (translates back)
STE ExecutionResult (specification_identity, program_identity, input_identity,
                      status, exit_code, output, output_identity,
                      computation_identity, detail)
    |
    v  caller-supplied interpret() function (e.g. scl.ste_adapter.interpret_lj_result)
Semantic content -> DispatchedMeasurement -> STE's OWN admission path
                     (SCL never calls this; the caller does)
```

## 2. The one operation this Phase 1 substrate implements

`lj_pairwise_energy_forces` — truncated Lennard-Jones pairwise potential
energy and per-particle forces for an N-particle system (see
`SCL_ARCHITECTURE.md` §5 and `PHASE1_AUDIT.md` §3 for why this operation
was chosen).

```
V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6],  r <= cutoff
V(r) = 0,                                        r >  cutoff   (plain truncation, not shifted)
```

## 3. `scl_cli` wire protocol

One JSON object on stdin, one JSON object on stdout, one process per
computation. The **process exit code is always 0** whenever the CLI
produced a well-formed JSON answer, *including a halted computation* —
mirroring STE's own Rust `execution-cli` convention, where the request's
outcome lives in a JSON/text field, never in the OS-level return code. A
non-zero process exit means the CLI could not answer at all (caught as
`SCLProtocolError` on the Python side).

### Request

```json
{
  "operation": "lj_pairwise_energy_forces",
  "backend": "cpu",
  "configuration_hex": "<48 hex chars: 3 little-endian float64 -- epsilon, sigma, cutoff>",
  "input_hex": "<N*48 hex chars: N particles of 3 little-endian float64 -- x, y, z>"
}
```

### Response (success)

```json
{
  "status": "completed",
  "exit_code": 0,
  "backend_used": "cpu",
  "backend_version": "scl-cli/0.1.0",
  "output_hex": "<16 + N*48 hex chars: total_energy f64, then N*(fx,fy,fz) f64>",
  "detail": null,
  "metrics": {"native_compute_seconds": 0.0037, "n_particles": 2000}
}
```

### Response (halted)

```json
{
  "status": "halted",
  "exit_code": 11,
  "backend_used": "cpu",
  "backend_version": "scl-cli/0.1.0",
  "output_hex": null,
  "detail": "sigma must be finite and > 0, got -1.000000",
  "metrics": {}
}
```

Byte layout assumption: x86_64, IEEE-754 double, little-endian — the same
representation `struct.pack("<d", ...)` produces in Python, decoded via a
raw `memcmp`-equivalent on the C++ side (`native/src/main.cpp::read_double_le`).
Documented, not portable; this substrate has not been built or tested on
any other architecture.

## 4. Fault vocabulary (never collapsed to one generic failure)

Single source of truth: `native/include/scl/protocol.hpp`, mirrored in
`python/scl/errors.py` and `python/scl/ste_adapter.py`'s `FAULT_*` constants.

| exit_code | name | meaning | raised as (Python, via `raise_for_result`) |
|---|---|---|---|
| 0 | — | completed | — |
| 10 | `PROTOCOL` | malformed/unreadable request envelope, or an unrecognised operation | `SCLProtocolError` |
| 11 | `VALIDATION` | structurally invalid parameters or input (bad sigma/epsilon/cutoff, wrong-length buffers, zero particles) | `SCLValidationError` |
| 12 | `BACKEND_UNAVAILABLE` | requested backend not usable in this build/host (e.g. `cuda` with no toolkit at build time, or no device at run time) | `SCLBackendUnavailableError` |
| 13 | `COMPUTATION` | the algorithm itself faulted (coincident particles, non-finite result) | `SCLComputationError` |
| 14 | `INTERNAL` | an unexpected exception was caught inside `scl_cli` | `SCLInternalError` |

Channel-level failures (missing binary, subprocess timeout, unparseable
stdout, non-zero **process** exit) are distinct again: `scl.client.run_scl_request`
raises `SCLProtocolError`/`SCLTimeoutError` directly rather than returning
a halted `SCLResult` — these are environment/channel problems, not
computational outcomes.

## 5. STE dimension mapping (mirrors `execution/gromacs.py`)

| STE field | Carries | SCL encoding |
|---|---|---|
| `program` | WHAT would be computed: kernel identity + build version + **backend** | `b"ste.scl.lj-pairwise-energy-forces.v1\n" + version_line + b"\n[backend]\n" + backend` |
| `configuration` | parameters GOVERNING the run | `epsilon, sigma, cutoff` as 3 little-endian float64 (24 bytes) |
| `input_payload` | the system the run is OVER | N particle positions as N×3 little-endian float64 |

Backend (`cpu`/`cuda`) is folded into `program`, not `configuration`,
because two backends are two *engines* — exactly as GROMACS folds its own
version line into `program` (`execution/gromacs.py:94-105`, pinned by
`test_engine_version_is_part_of_the_program`; SCL's analog is
`test_kernel_version_is_part_of_the_program_mirrors_gromacs_precedent` and
`test_backend_choice_is_part_of_the_program_identity` in this repo).

## 6. Identity (Task 6)

Two separate identity spaces, one shared scheme (see
`SCL_ARCHITECTURE.md` invariant G):

- **SCL's own request identity** (`python/scl/identity.py`,
  `scl.*` domain tags): `operation_identity`, `parameters_identity`,
  `input_identity`, `request_identity` on `SCLRequest`; `output_identity`,
  `computation_identity` on `SCLResult`. Meaningful only within SCL, for
  SCL-level testing/tracing; never asserted to STE.
- **STE's execution identity** (`execution.commitments`, `scout.execution.*`
  domain tags, imported for real in `python/scl/ste_adapter.py`):
  `program_identity`/`input_identity` computed from `spec` exactly as for
  any other backend; `output_identity`/`computation_identity` computed
  by the adapter **from bytes it holds**, never trusted from the child
  process — the same "checked, not trusted" posture GROMACS and the Rust
  engine both take.

## 7. Reproducibility — precisely scoped (Task 5)

| Kind | Claimed? | Evidence |
|---|---|---|
| **Bitwise** (same binary, same machine, same backend) | Yes | `test_repeat_determinism_same_binary_same_machine` (two full subprocess round trips, byte-identical `output`); native `test_bitwise_reproducible_same_process`. Enabled by `-fno-fast-math`, no SIMD flags, IEEE-754 strict double math (`native/CMakeLists.txt`). |
| **Numerical** (tolerance-based, cross-implementation) | Yes, for the force/energy relationship | `test_force_matches_finite_difference_gradient` (native, in-process) and `test_force_matches_finite_difference_energy_gradient` (through the full subprocess boundary) — an independent numerical check (finite-difference gradient of energy) against the analytically-derived force formula, within `1e-4`–`1e-3` relative tolerance. |
| **Physical** (matches a real experiment/measurement) | **Not claimed** | This is a model computation over an idealized system (a Lennard-Jones potential), never a measurement. `execution/gromacs.py`'s own "COMPUTATION != MEASUREMENT" posture applies with equal force here. |
| **Semantic** (same request ⇒ same identity) | Yes | `test_result_identity_is_content_addressed_across_two_runs`, `test_repeat_determinism_same_binary_same_machine` (STE-level). |
| **Cross-machine / cross-compiler bitwise** | **Not claimed, not tested** | No second machine or compiler was available in this environment. |
| **CPU vs. CUDA numerical agreement** | **Not claimed, not tested** | The CUDA backend has never been compiled or run (no CUDA toolkit/GPU in this sandbox) — see `PHASE1_AUDIT.md` §9. |

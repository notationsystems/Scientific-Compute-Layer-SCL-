# SCL Core Independence

SCL is a reusable **Scientific Compute Layer**, not "the CUDA/Lennard-Jones
computation layer inside Notations." The Notations/STE integration
(`python/scl/ste_adapter.py`) is one consumer and integration environment
for SCL — it is not SCL's definition. This document establishes that
property against the actual repository, not aspirationally.

## Reconnaissance: the actual dependency graph

Checked directly (`grep` over every real import statement, not filenames):

| Component | STE/Notations dependency |
|---|---|
| `native/` (C++17 core, CUDA backend, `scl_cli`) | **none** — pure C++/CUDA/CMake/nlohmann_json; two comment-only mentions of "Python"/"STE" (`main.cpp:69`, `version.hpp:6`), no code coupling |
| `python/scl/identity.py` | none — stdlib only (`hashlib`, `struct`) |
| `python/scl/errors.py` | none — stdlib only |
| `python/scl/client.py` | none — stdlib only (`json`, `pathlib`, `struct`, `subprocess`, `time`); imports only `.errors`/`.identity` |
| `python/scl/quantity.py` | none — stdlib only |
| `python/scl/method_block.py` | none — stdlib only |
| `python/scl/ste_adapter.py` | **the only file** importing STE: `execution.commitments`, `execution.engine`, `execution.specification` — plain data/identity types, the same two modules `execution/gromacs.py` depends on; never `evidence.*`, `materials.*`, or `core.canonical` |
| `python/scl/__init__.py` | re-exports only from `.client`/`.errors`/`.quantity`/`.method_block` — **never imports `.ste_adapter`** |

This was already true from Phase 1 onward (`ste_adapter.py`'s own module
docstring has said "SCL never imports evidence/, materials/,
core.canonical..." since it was written) — this document is the first
place it is stated as a formal boundary and checked mechanically rather
than left as a convention.

**Proof, not assertion**: `tests/test_standalone_boundary.py` —
1. Spawns a **separate Python subprocess** with `PYTHONPATH` set to
   *only* this repo's `python/` directory (no STE checkout on the path
   at all — not even importable, let alone imported), imports `scl.client`
   and `scl.quantity`, runs a real computation through the native
   `scl_cli` binary, decodes the result, and asserts zero
   `execution`/`evidence`/`materials`/`core.canonical`/`experiment`/
   `campaign` modules ever entered that subprocess's `sys.modules`. A
   separate process is used deliberately: checking `sys.modules`
   in-process would be contaminated by whichever other test files
   already ran and imported STE for their own (legitimate) reasons.
2. Statically greps `client.py`'s own source for a forbidden import line.
3. Asserts exactly one file under `python/scl/` — `ste_adapter.py` —
   imports STE's `execution` package, so a future accidental widening of
   the boundary fails a test immediately.

Five of SCL's eight pre-existing test files
(`test_client_subprocess.py`, `test_contract_identity.py`,
`test_failure_paths.py`, `test_numerical_validation.py`,
`test_performance_baseline.py`) already only ever imported
`scl.client`/`scl.errors`/`scl.quantity` — this was true before this
document existed; nothing about those files changed.

## The standalone contract

An external consumer — a researcher's script, another computational
system, anything that is not Notations/STE — needs only:

```python
from scl import SCLRequest, run_scl_request, encode_lj_configuration, encode_lj_positions, decode_lj_output

request = SCLRequest(
    operation="lj_pairwise_energy_forces",
    backend="cpu",  # or "cuda", where available
    parameters=encode_lj_configuration(epsilon=1.0, sigma=1.0, cutoff=5.0),
    input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
)
result = run_scl_request(request, cli_path="/path/to/scl_cli")
# result.status: "completed" | "halted"
# result.output: raw bytes (decode_lj_output) on success
# result.exit_code / result.detail: a specific, distinguishable fault on failure
# result.request_identity / result.output_identity / result.computation_identity
```

This is the actual, already-established public API
(`python/scl/client.py`; `python/scl/__init__.py` re-exports the same
names at the top level as of this document) — not a new one invented for
this phase. Per the smallest-stable-API principle: nothing was added to
make this "nicer" beyond re-exporting existing names at the package root
(`from scl import ...` instead of `from scl.client import ...`), which is
packaging polish, not new architecture.

**What a standalone consumer receives**: a computational result
(`SCLResult`) with its own request/operation/parameters/input identity
(`python/scl/identity.py`'s `scl.request.*`/`scl.result.*` scheme — see
below) and an explicit status/fault vocabulary (`SCLProtocolError`,
`SCLValidationError`, `SCLBackendUnavailableError`,
`SCLComputationError`, `SCLInternalError`, `SCLTimeoutError` —
`docs/SCL_CONTRACT.md` §4). **Not** evidence. **Not** a canonical-state
assertion. A standalone consumer that wants those things builds them
itself, or hands the result to `scl.ste_adapter` if it happens to be
Notations/STE.

## Identity separation, preserved

Four identity spaces already exist and remain distinct under the
standalone boundary (unchanged from Phase 1/2 — restated here because
this document is where the standalone/integration split is formalized):

| Identity | Lives in | Meaning |
|---|---|---|
| Scientific/request identity | `SCLRequest.identity()`/`operation_identity()`/`parameters_identity()`/`input_identity()` (`scl.request.*` tags) | "this exact ask" — independent of backend or execution |
| Backend identity | folded into `SCLRequest.operation_identity()` (`backend` field) and `SCLResult.backend_used`/`backend_version` | which physical engine ran — cpu vs. cuda are different engines, same discipline STE's own GROMACS integration uses for engine version |
| Execution/computation identity | `SCLResult.output_identity`/`computation_identity` (`scl.result.*` tags) | what running the request actually produced |
| Notations/STE evidence identity | `evidence.identity.content_hash`, reachable only through `scl.ste_adapter`, never through `scl.client` | STE's own, separate identity space (`docs/PHASE2_AUDIT.md` §6) |

A standalone consumer sees the first three. The fourth does not exist
until `scl.ste_adapter` is explicitly imported and used — it is not a
hidden dependency of the first three.

## Backend independence, preserved

`native/include/scl/backend.hpp`'s `Backend` enum and
`compute_lj_pairwise()` dispatch (unchanged, Phase 1/3) already model
"cpu / cuda / future backend" as implementations of one scientific
contract, not as SCL's definition. The scientific operation
(`lj_pairwise_energy_forces`) and its wire contract do not change based
on which backend answers it — confirmed directly by
`test_cpu_backend_is_unaffected_by_cuda_being_compiled_in`
(`tests/test_cpu_cuda_equivalence.py`). CUDA status is unchanged by this
document: compiled/linked/runtime-loaded verified, GPU execution pending
a GPU-accessible environment (`docs/PHASE4_AUDIT.md`).

## Packaging boundary

`import scl` (the package root) executes only `.client`, `.errors`,
`.quantity`, `.method_block` — confirmed by direct test
(`test_scl_client_imports_and_computes_in_a_fresh_interpreter_with_no_
notations_on_path`). `scl.ste_adapter` is a real Python module under the
same package, reachable by `import scl.ste_adapter` or
`from scl import ste_adapter`, and is simply never imported unless a
caller explicitly asks for it. This is a single-package superset, not a
fork or a second source tree: `pyproject.toml` still declares one
package (`scientific-compute-layer`, zero third-party dependencies);
nothing was split into a second distribution, because nothing required
it — the boundary already existed at the module level and needed only to
be made explicit and enforced by a test.

## Architectural invariant

**SCL Core Independence**: SCL's computational specifications,
representations, execution interfaces, and backend implementations
(`native/`, `python/scl/client.py`, `python/scl/identity.py`,
`python/scl/errors.py`, `python/scl/quantity.py`,
`python/scl/method_block.py`) remain usable independently of
Notations-specific acquisition, evidence, canonical-state, and epistemic
infrastructure. Notations integrations (`python/scl/ste_adapter.py`) may
add execution, provenance, verification, and evidence semantics on top,
but MUST NOT redefine the standalone computational contract those core
modules expose.

Enforced by `tests/test_standalone_boundary.py`, run as part of the
regular SCL suite — not merely documented.

## What this document does not change

- The existing STE adapter is unchanged in behavior (still the module
  STE's `SpecificationDispatcher` plugs into, exactly as
  `docs/PHASE1_AUDIT.md`/`docs/PHASE2_AUDIT.md` established).
- No new scientific capability, backend, or identity scheme was added.
- CUDA remains exactly where `docs/PHASE4_AUDIT.md` left it: compiled,
  linked, runtime-loaded, GPU execution pending hardware access.
- No package was split; no repository was forked; no source tree was
  duplicated.

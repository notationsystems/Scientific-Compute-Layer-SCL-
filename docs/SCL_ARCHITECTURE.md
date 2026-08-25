# SCL Architecture — Phase 1

## 0. SCL is reusable; Notations/STE is one integration, not SCL's definition

Read `docs/SCL_STANDALONE_BOUNDARY.md` first if you are asking "does SCL
require STE?" — the short, repository-grounded answer is no. `native/`
and every `python/scl/*.py` file except `ste_adapter.py` have zero
Notations/STE dependency (checked by grep and by
`tests/test_standalone_boundary.py`, which spawns a fresh interpreter
with no STE checkout even on `sys.path` and proves a full computation
still runs). The diagram immediately below (§1) describes the
**integrated** configuration — SCL attached beneath STE inside the
Notations stack — which remains real, verified, and unchanged. It is one
consumer of SCL, not SCL's boundary.

```
                     SCL
                      |
        +-------------+-------------+
        |                           |
Standalone consumers          ste_adapter.py
(scl.client, scl.quantity,          |
 scl.errors, scl.method_block)      v
                              Notations / STE
                              (execution/evidence/
                               canonical-state
                               integration)
```

## 1. What this repository is

`Notations-CUDA-Architecture-` is the **Scientific Compute Layer (SCL)**
repository: a new sibling to the existing `notationsystems` architecture
repos, not a fork or a replacement of any of them.

The diagram below shows SCL **integrated into Notations** — one real,
verified configuration, not the only one (§0):

```
Scientific Workbench / Projection Layer      (in scientific-transformer-engine, workbench/)
            |
Scientific Intelligence Layer (SIL)          (materials/, experiment/, retrieval/ -- STE repo)
            |
Scientific Transformer Engine (STE) — Rust   (execution/, crates/ -- scientific-transformer-engine repo)
            |
Scientific Compute Layer (SCL) — C/C++/CUDA  (native/, python/scl/ -- THIS repo)
            |
Physical / Numerical Computation
```

This repository started with **zero commits** (verified: empty git
history, no branches on the remote). The "existing architecture" the
Phase 1 brief refers to lives in sibling repos:

| System | Repo | What it is |
|---|---|---|
| STE (Rust + Python orchestration) | `notationsystems/scientific-transformer-engine` | deterministic execution substrate, canonical state, evidence/provenance, materials/experiment layers, workbench |
| DAF | `notationsystems/data-acquisition-channel-daq` | acquisition → normalization → evidence bridge |
| GROMACS / LAMMPS | `notationsystems/gromacs-molecular-simulation`, `notationsystems/lammps-md` | unmodified upstream forks (verified: plain vanilla READMEs, no notationsystems-specific commits — see §5) |

All reconnaissance in this document is grounded in reading the actual
code of the STE repo at the commit cloned for this work, not filenames or
docs alone.

## 2. Where SCL attaches to STE (repository-grounded)

STE's Rust execution substrate (`crates/execution-core`,
`execution-kernel`, `execution-native`, `execution-model`,
`execution-commitment`, `execution-trace`, `execution-verification`,
`execution-serialization`, `execution-cli`) is **closed by design**:

- Every crate's `Cargo.toml` declares only sibling-crate `path`
  dependencies — no `cc`, `bindgen`, `libc`, `cxx`, or `pyo3`, and no
  crate has a `build.rs`.
- `execution-native`/`execution-kernel` are `#![forbid(unsafe_code)]`,
  dependency-free Rust. "Native" there means "compiled Rust", not "calls
  into C/C++".
- `docs/RUST_EXECUTION_SEMANTIC_BOUNDARY.md:341` states this outright:
  *"No integration of any kind. No Python bindings, no FFI... The
  substrate does not know [SCOUT/DAF] exist, and a guard test keeps it
  that way."* `tests/test_phase127_rust_semantic_boundary.py` enforces
  this structurally, by parsing each crate's Cargo dependency table and
  failing if anything not on an allow-list appears.
- `execution-cli`'s registry of runnable kernels is a **fixed, hardcoded
  array of 4 in-process Rust functions** (`crates/execution-cli/src/main.rs`)
  — "no dynamic loading" by explicit design. There is no pluggable-backend
  abstraction inside the Rust workspace for SCL to register into.

**Conclusion: SCL must not touch the Rust workspace.** The correct, and
only structurally honest, attachment point is the Python orchestration
layer — specifically the exact seam `execution/gromacs.py` already
occupies:

```
execution/dispatcher.py
    SpecificationDispatcher.runner: Optional[Callable[[ExecutionSpecification], ExecutionResult]]
        default -> execution.engine.run_specification        (Rust CLI subprocess)
        alternative -> execution.gromacs.run_gromacs_specification  (external process, partially applied)
        NEW ALTERNATIVE -> scl.ste_adapter.run_scl_specification    (external process, partially applied)
```

This is not a guess: `execution/dispatcher.py`'s own docstring for the
`runner` field says so verbatim — *"Whatever runs, the SAME result shape
comes back and the SAME seam carries it onward — backends are
substitutable below this line without anything downstream knowing."*

`tests/test_ste_integration.py::test_scl_is_substitutable_at_the_real_specification_dispatcher_seam`
in this repo proves it: it imports STE's real `SpecificationDispatcher`,
plugs in `run_scl_specification` as `runner`, and calls `.dispatch()` —
no STE code changed, no mock objects.

### Why a process boundary, not FFI

The Phase 1 brief allows either an FFI surface or an external-executable
boundary, "whichever reflects the actual execution model." Given:

1. the Rust workspace is closed to FFI by explicit design and a guard
   test (above), and
2. the existing precedent for "a real external scientific engine" —
   `execution/gromacs.py` — is a subprocess wrapper, at the same Python
   layer, with the *same* dimension mapping and identity discipline SCL
   now reuses,

a process boundary is the boundary that "reflects the actual execution
model," not an aesthetic choice. `native/src/main.cpp` (`scl_cli`) is
architecturally the C++ sibling of `crates/execution-cli` and the
GROMACS `gmx` binary: one process per computation, JSON in on stdin, JSON
out on stdout, checked-not-trusted from the Python side.

## 3. Architectural invariants (Task 10 audit)

| # | Invariant | Status | Evidence |
|---|---|---|---|
| A | Canonical state remains authoritative | **Held** | SCL never imports `core.canonical`, `evidence.*`, or `materials.*`; `python/scl/ste_adapter.py` returns a plain `ExecutionResult`, nothing more. |
| B | DAF remains responsible for acquisition/normalization/evidence bridging | **Held (untouched)** | SCL never references DAF; DAF was read-only reconnaissance only. |
| C | STE remains the authoritative transformation/execution boundary | **Held** | SCL results only become STE `ExecutionResult` objects via `run_scl_specification`; STE's own dispatcher decides what happens next. SCL never calls `experiment.step.run_experiment_step` or any admission function. |
| D | SCL supplies computational capability | **Held** | `native/` is the entire computational surface: LJ pairwise energy/forces, CPU backend, CUDA backend stub. |
| E | SIL remains above STE/SCL | **Held (untouched)** | `materials/`, `experiment/`, `retrieval/` were read, never modified; SCL is invisible to them except through the substitutable `runner`. |
| F | Workbench remains a projection/interface layer | **Held (untouched)** | Not touched. |
| G | No duplicate identity system introduced | **Held, with a documented seam** | `python/scl/identity.py` implements the SAME canonicalization *pattern* STE's `execution.commitments` uses (length-prefixed tag+fields, SHA-256 hex) under SCL's own `scl.*` domain tags — for SCL's own request/job identity, which has no STE equivalent. Where SCL produces values that must live in STE's *own* execution-identity space (`ExecutionResult.output_identity`/`computation_identity`), `python/scl/ste_adapter.py` imports and calls STE's real `execution.commitments.commit_hex` with STE's real `OUTPUT_TAG`/`COMPUTATION_TAG` — exactly as `execution/gromacs.py` does. Two identity *spaces*, one identity *scheme*, zero duplication of STE's own space. |
| H | No duplicate provenance system introduced | **Held** | SCL emits no provenance object of any kind. `ExecutionResult`'s `specification`/`program_identity`/`input_identity`/`engine_occurrence` fields are STE's own type, populated by the adapter, never a parallel SCL-side record. |
| I | SCL cannot silently mutate canonical state | **Held structurally** | `python/scl/ste_adapter.py` has zero imports of, or references to, anything canonical/evidence/admission. It is architecturally incapable of writing to canonical state — there is no code path that does so, not just a policy against it. |
| J | Computational results remain distinguishable from verified scientific claims | **Held** | `interpret_lj_result()` is the Phase 112b firewall boundary: raw execution bookkeeping (`specification`, `computation_identity`, `engine_occurrence`) rides only in `ExecutionResult`/`DispatchedMeasurement.record_raw_content`; the semantic `content` dict carries only `total_energy_reduced_units`/`forces_reduced_units`. Verified by `test_interpret_result_is_the_evidence_firewall`. |
| K | Native computational failures remain observable | **Held** | Five distinct fault codes (10–14), never collapsed; see §7 of `SCL_CONTRACT.md` and `tests/test_failure_paths.py` (10 failure-path tests, all passing). |
| L | Existing verified invariants remain intact | **Held** | STE's full test suite: **1914 passed, 0 failed** on an unmodified checkout (see `docs/PHASE1_AUDIT.md` §6 for the one pre-existing test that is path-sensitive to clone location, unrelated to this work and independently reproduced). |

## 4. What SCL is not

- Not a second execution-control layer. It has no operation ledger, no
  campaign/policy logic, no retry/scheduling semantics beyond what
  `subprocess.run(..., timeout=...)` gives for free.
- Not a canonical-state writer. It cannot be, structurally (see invariant I).
- Not a provenance system. It computes identities for its own outputs so
  STE can fold them into STE's identities; it asserts nothing about trust,
  admission, or evidence.
- Not (yet) GROMACS or LAMMPS integration. See §5 for why.

## 5. Why GROMACS/LAMMPS were not the Phase 1 backend

`execution/gromacs.py` already exists in STE and already runs real GROMACS
subprocess workloads (`tests/test_execution_gromacs.py`, skipped only when
no `gmx` binary is on `PATH`). That precedent is exactly what SCL's own
adapter mirrors — but it is STE's existing integration, not a gap SCL
needs to fill.

The `notationsystems/gromacs-molecular-simulation` and
`notationsystems/lammps-md` repos were inspected directly (sparse/treeless
clones, no full checkout): both are **unmodified upstream forks** — plain
vanilla READMEs, no `notationsystems`-authored commits, no added
integration directories. Building either from source in this Phase 1
sandbox was evaluated against the task's own selection criteria ("does not
require premature integration complexity") and rejected for Phase 1
because:

- This sandbox has **no CUDA toolkit and no GPU** (`which nvcc` fails,
  `nvidia-smi` is not installed, no `/usr/local/cuda*`) — verified directly.
  A CUDA backend cannot be built or run here regardless of which
  scientific engine is chosen.
- This sandbox has **no MPI and no system BLAS/LAPACK** development
  packages — verified via `ldconfig -p` / `apt list --installed`. GROMACS
  and LAMMPS both build (optionally) against these; a from-source build
  here would either need hours of dependency bootstrapping or a
  degraded single-threaded, no-SIMD, reference build that would prove
  little more than a hand-written kernel does.
- Neither fork has any notationsystems-specific integration surface yet
  to attach to — building one from scratch (CMake toolchain wiring, a
  topology/parameter marshaling layer, output parsing) is exactly the
  "premature integration complexity" Task 3 says to avoid for a first
  backend.

A hand-written CPU kernel proves the *identical* architectural boundary
(SCL request → native process → STE `ExecutionResult`) with a real,
independently-verifiable computation, in an environment that can actually
build, run, and test it end to end. See `SCL_CONTRACT.md` §3 for the
selection rationale in full, and `PHASE1_AUDIT.md` §12 for why GROMACS/
LAMMPS/CUDA are named explicitly as Phase 2 candidates once real hardware
and a longer build budget are available.

## 6. Repository layout

```
native/                    C++17 computational core + CLI (the SCL substrate itself)
  include/scl/             lj_pairwise.hpp, backend.hpp, protocol.hpp, version.hpp
  src/                     lj_pairwise.cpp, backend.cpp, main.cpp (scl_cli)
  backends/cuda/           lj_pairwise_cuda.cu/.hpp -- compiled only with -DSCL_WITH_CUDA
  tests/                   native unit tests (CTest), 50 checks
  CMakeLists.txt
python/scl/                Python client + STE adapter
  identity.py               SCL's own commit-hash scheme
  errors.py                 typed exceptions, one per fault stage
  client.py                 SCLRequest/SCLResult, subprocess client (STE-agnostic)
  ste_adapter.py             ExecutionSpecification/ExecutionResult translation (STE-facing)
tests/                      pytest suite: 37 tests across identity, subprocess, failure-path,
                             numerical validation, performance baseline, and real STE integration
scripts/run_benchmark.py    Task 8 performance baseline sweep
docs/
  SCL_ARCHITECTURE.md        this file
  SCL_CONTRACT.md            the wire-level request/result contract
  PHASE1_AUDIT.md            numerical/failure/performance results, defects, open questions
```

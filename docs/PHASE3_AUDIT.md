# Phase 3 Audit — CPU ↔ CUDA Scientific Equivalence

Scope: advance SCL from verified CPU computation (Phase 1/2) to CPU↔CUDA
**correctness** equivalence for the existing Lennard-Jones workload.
Performance optimization is explicitly out of scope until correctness is
established. No Phase 2 architecture decision was reopened; no
`architecture/invariants.yaml`, evidence-class system, or canonical-state
system was created.

## Environment fact (unchanged since Phase 1, re-verified)

This development environment has **no GPU**: `lspci` shows no VGA/3D
device, there is no `/dev/nvidia*`. What changed this phase: the CUDA
toolkit is now installed (`apt-get install nvidia-cuda-toolkit-gcc
nvidia-cuda-dev`, nvcc 12.0.140), so the CUDA kernel written in Phase 1
but never built can now be **compiled and linked** for the first time.
It has never been, and cannot be, **executed on a device** in this
session. Every claim below keeps that distinction explicit.

## §1 — First inspect (Task 3)

Confirmed by direct inspection before any change:

| Item | State found |
|---|---|
| Existing CUDA code | `native/backends/cuda/lj_pairwise_cuda.cu`/`.hpp` — real, written in Phase 1, never previously compiled |
| Existing CPU reference | `native/src/lj_pairwise.cpp` — the numerical reference (Phase 1/2, 50/50 native checks, 1914 STE tests unaffected) |
| Build system | CMake, `native/CMakeLists.txt`, `SCL_WITH_CUDA` option already present (default OFF) but **had never successfully configured or built** — see §2's defects |
| Native-language boundary | `native/include/scl/backend.hpp`'s `Backend` enum + `compute_lj_pairwise()` dispatch — already the correct seam, no new abstraction needed |
| CUDA availability | Toolchain: yes, as of this phase (`nvidia-cuda-toolkit` installed). Hardware: no, and unchanged from Phase 1 |
| Previously attempted GPU implementation | None beyond the Phase 1 source; this is the first build attempt |

**No new CUDA abstraction was created.** `Backend`/`compute_lj_pairwise()`
(Phase 1) was already the correct, sufficient boundary; this phase only
made it actually build and fixed the defects that had made it
unbuildable and, in one case, silently less informative than intended.

## §2 — Implemented

1. **`native/CMakeLists.txt`** — `SCL_WITH_CUDA=ON` had never actually
   configured successfully before this phase (untested since Phase 1: no
   nvcc existed to try it against). Two real defects found and fixed:
   - Modern CMake (`CUDA_ARCHITECTURES`/policy CMP0104) requires an
     explicit target architecture list before enabling the CUDA
     language; none was set, so configuration failed with
     `CUDA_ARCHITECTURES is empty for target "scl_core"`. Fixed with an
     explicit, documented, overridable default (`SCL_CUDA_ARCHITECTURES`,
     defaulting to `"70;75;80;86;89"` — Volta through Ada; there is no
     real GPU in this environment to detect an architecture from, so this
     is a stated reasonable default, not a verified deployment target)
     set both as `CMAKE_CUDA_ARCHITECTURES` and as the `scl_core` target's
     `CUDA_ARCHITECTURES` property (the target is created before
     `enable_language(CUDA)` runs, so the global variable alone was not
     enough — confirmed by testing both ways).
   - `target_include_directories(scl_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})`
     added, scoped to the CUDA-enabled build only (see next item for why).
2. **`native/src/backend.cpp`, `native/backends/cuda/lj_pairwise_cuda.cu`**
   — both `#include "scl/backends/cuda/lj_pairwise_cuda.hpp"`, a path
   that **never matched the real file layout** (the file lives at
   `native/backends/cuda/lj_pairwise_cuda.hpp`, not under
   `native/include/scl/backends/cuda/`). This include had been silently
   broken since Phase 1 — undetectable without a compiler that would
   actually try to build the CUDA path, which did not exist until this
   phase. Fixed: both includes changed to `"backends/cuda/lj_pairwise_cuda.hpp"`
   (relative to `native/`, matching the real layout).
3. **`native/include/scl/backend.hpp`, `native/src/backend.cpp`** —
   a second real defect, found by writing (and failing) a Phase 3 test:
   `main.cpp`'s early `backend_available()` short-circuit (checked before
   request validation, by design — see §7) printed a generic `"backend
   'cuda' is not available in this build/host"` message, while the more
   specific reason (not compiled in, vs. compiled in but no device
   visible — genuinely different causes, both now reachable with a real
   toolchain) lived only in `BackendUnavailableError`'s message inside
   `compute_lj_pairwise()` — a code path the early short-circuit made
   unreachable in practice. Fixed by extracting a single new function,
   `backend_unavailable_reason(Backend) -> std::string` (empty when
   available), that both the early check and the exception now share as
   their **one** source of truth — no more possibility of the two
   messages drifting apart.
4. **`native/tests/test_lj_pairwise.cpp`** — new
   `test_cuda_reports_unavailable_when_built_with_cuda_but_no_device`,
   which only runs (and only means something) in a `SCL_WITH_CUDA`
   build: it exercises the REAL `cudaGetDeviceCount()` call for the first
   time, asserts it correctly reports unavailable, and asserts the
   message text proves the runtime check ran (not the compile-time
   stub).
5. **`tests/conftest.py`** — new `cuda_cli_path` session fixture: builds a
   *separate* `native/build_cuda` binary with `-DSCL_WITH_CUDA=ON`,
   independent of the default CPU-only `cli_path` build, returning
   `None` (callers skip) if `nvcc` is absent.
6. **`tests/test_cpu_cuda_equivalence.py`** — new, 9 tests (§9).

No STE code was modified. No files from Phase 1/2 were rewritten beyond
the three defect fixes above (backend.hpp/backend.cpp/main.cpp/CMakeLists.txt);
`lj_pairwise.cpp` (the CPU reference) was not touched at all, per this
phase's explicit instruction to preserve it.

## §3 — Verified (only what was actually executed)

| Check | Result |
|---|---|
| `cmake -S native -B native/build_cuda -DSCL_WITH_CUDA=ON` | Configures successfully (after §2 fix 1) |
| `cmake --build native/build_cuda` | **Compiles and links successfully** (after §2 fixes 1–2) — `scl_core` (static lib incl. the `.cu` translation unit), `scl_cli`, `scl_native_tests` all built, for 5 target architectures |
| Native tests, CUDA-enabled build | **51/51 checks pass** (`./build/tests/scl_native_tests`), including the new real `cudaGetDeviceCount()` exercise |
| Native tests, CPU-only build (regression) | **50/50 checks pass**, unchanged from Phase 1/2 — confirms the CUDA build changes did not affect the default build |
| `scl_cli --version` (CUDA-enabled binary) | Runs, prints a real version string |
| `scl_cli` with `backend=cpu` (CUDA-enabled binary) vs. `backend=cpu` (CPU-only binary), identical input | **Bit-identical output** (`test_cpu_backend_is_unaffected_by_cuda_being_compiled_in`) |
| `scl_cli` with `backend=cuda` (CUDA-enabled binary, no GPU) | Reports `status=halted, exit_code=12`, specific detail message, `backend_used=cuda` (never silently substituted) |
| Full SCL Python suite | **59 passed, 1 skipped** (the skip is the honest numerical-equivalence non-claim, §5) |
| STE's own full suite, unmodified | **1914 passed, 0 failed**, 109 skipped (STE's own environment-gap skips) |
| **GPU execution of the CUDA kernel itself** | **Never attempted, never claimed.** No device exists to run it on. |

**Compiled: yes. Linked: yes. Unit-tested: yes (the availability/failure
paths around it). GPU-executed: no.** These four are kept explicitly
distinct throughout this document, per this phase's own instruction.

## §4 — Preserved

The CPU implementation (`native/src/lj_pairwise.cpp`) was not touched.
Every Phase 1/2 invariant remains intact and re-verified: 50/50 native
checks (CPU-only build), the full Python suite (37 Phase 1 + 14 Phase 2 +
9 Phase 3 tests = 60 total, 59 passed / 1 honest skip), and STE's 1914
tests unmodified.

## §5 — Extended

New computational capability: a CUDA backend that **compiles, links, and
correctly reports its own unavailability** when no device is present —
not new architecture, a new backend under the existing `Backend`
enum/`compute_lj_pairwise()` seam (Phase 1's design, unchanged).

## §6 — Integrated

Same STE interfaces as Phase 1/2, unchanged: `execution.specification
.ExecutionSpecification`, `execution.engine.ExecutionResult`,
`execution.dispatcher.SpecificationDispatcher`, and the full real
evidence/derived-state loop (`evidence.pool.EvidencePool`,
`materials.results.*`, `experiment.step.run_experiment_step`,
`materials.model_state.update`) — now proven to also correctly refuse a
CUDA-selected specification when the backend is unavailable, admitting
nothing (§9).

## §7 — Qualified

The fault-priority ordering — `BACKEND_UNAVAILABLE` (12) is checked and
reported **before** parameter/input validation, so a request that is
*also* structurally invalid (e.g. negative sigma) still reports 12, not
11 — is an existing Phase 1 design decision (not new this phase), now
explicitly locked in as a test
(`test_backend_unavailability_is_checked_before_input_validation`) rather
than left implicit. This is a **Qualified** clarification of existing
behavior, not a new constraint and not a bend.

The default CUDA architecture list (`70;75;80;86;89`) is a **Qualified**,
stated, overridable choice — explicitly not a value verified against any
real deployment target, because none exists in this environment (§1).

## §8 — Bent

**Zero.** No STE code, no core schema, no Phase 1/2 evidence/identity
contract was changed.

## §9 — CPU reference state

`native/src/lj_pairwise.cpp`, unchanged since Phase 1:
- Potential: truncated Lennard-Jones 12-6, `V(r) = 4ε[(σ/r)^12 - (σ/r)^6]`
- Force: `F = 24ε/r² · [2(σ/r)^12 - (σ/r)^6] · r_vec` (derived, not
  independently re-verified this phase — Phase 1's finite-difference
  cross-check already established it, still passing)
- Cutoff: plain truncation (`V=F=0` beyond cutoff; not shifted to zero at
  the boundary)
- Units: reduced (energy in units of the caller's ε, force in units of
  ε/σ) — SCL performs no unit conversion of its own
- Precision: IEEE-754 `float64`, `-fno-fast-math`, no SIMD-reordering
  flags — the same determinism claim GROMACS's own module makes
- Boundary conditions: open (no periodic images)
- Input ordering: `forces[i]` corresponds to `positions[i]` by index; no
  reordering anywhere in the pipeline

## §10 — CUDA state

`native/backends/cuda/lj_pairwise_cuda.cu` — **compiled and linked for
the first time this phase**, for 5 target architectures
(`sm_70/75/80/86/89`), against real `libcudart` (CUDA 12.0.140). Its
algorithm mirrors the CPU reference term-for-term (one thread per
particle `i`, full `N-1` inner sweep, host-side halving of the doubled
energy sum — documented in the file itself since Phase 1). **Never
executed on a device.** The known, documented parity gap flagged in
Phase 1 remains open and unchanged this phase: the CUDA kernel does not
detect coincident particles (`r=0`) the way the CPU backend does (a
device-wide atomic would be needed, and there is still no hardware to
validate that against) — carried forward, not silently dropped.

## §11 — Numerical equivalence

**Not measured.** No GPU execution occurred, so there is no CPU-vs-CUDA
energy/force comparison to report — absolute error, relative error,
max elementwise error, and RMS error are all **N/A, not zero, not
passing, not claimed**. `test_numerical_equivalence_or_honest_absence`
is written to perform exactly that comparison (with `1e-9` tolerances,
appropriate to `float64` — a value chosen for when this test actually
runs, not fitted after the fact) the moment a real device is available,
and currently asserts the honest `BACKEND_UNAVAILABLE` outcome instead.
Determinism: the CPU path's own bitwise determinism (same binary, same
machine) remains verified (§3); CUDA-path determinism (relevant once a
device exists — thread-scheduling-dependent reduction order can affect
bitwise, though not necessarily numerical, reproducibility) is
**untested and unclaimed**.

## §12 — STE conformance

Both the CPU and the (halted) CUDA path were exercised through the real,
identical STE seam:

- `run_scl_specification` on a CUDA-selected spec returns STE's real
  `ExecutionResult(status="halted", exit_code=12, output=None,
  output_identity=None, computation_identity=None)` — same type, same
  shape discipline as any other halted backend (Phase 1's GROMACS
  precedent, Phase 2's coincident-particle precedent).
- The full real `run_experiment_step` loop, with the dispatcher
  requesting `backend="cuda"`, raises `RuntimeError` ("no output, no
  measurement") and leaves `EvidencePool.fingerprint()` byte-identical —
  nothing is admitted for an unavailable backend, exactly as Phase 2
  proved for a computation fault.

## §13 — Failure semantics (actually tested)

| Failure | Tested? | Behavior observed |
|---|---|---|
| Unknown backend string via the CUDA-capable binary | Yes | `PROTOCOL` (10), unchanged path, no CUDA-specific parsing leniency |
| `backend=cuda`, also-invalid parameters | Yes | `BACKEND_UNAVAILABLE` (12) reported first — the deliberate priority order (§7) |
| `backend=cuda`, no device present | Yes | `BACKEND_UNAVAILABLE` (12), specific detail text, `backend_used="cuda"` echoed (never silently substituted to "cpu") |
| Silent CPU fallback | Tested for absence | Never occurs — confirmed by `backend_used` always echoing what was requested, and by the STE-level test showing nothing is admitted |
| CUDA allocation/kernel/transfer failure | **Not testable** | No device to allocate on, launch a kernel on, or transfer to/from — N/A in this environment, not silently skipped: recorded as untestable, not "passing" |
| Unsupported device/runtime | Covered by the same `cudaGetDeviceCount()==0` path as "no device" — this environment cannot distinguish "no device" from "unsupported runtime" without one |

## §14 — Performance

**Not applicable this phase.** No GPU exists to measure kernel runtime,
host↔device transfer, or total GPU-path time against. The CPU baseline
established in Phase 1 (`docs/PHASE1_AUDIT.md` §7 — native compute time
scaling with N, STE-boundary/subprocess overhead measured separately) is
unchanged and not re-measured here, since nothing about the CPU path
changed this phase. **Measured bottleneck: not established — there is
nothing to bottleneck without a GPU-executed comparison.**

## §15 — Known dependencies (carried forward)

The Phase 2 nested-dictionary/hashability conflict
(`docs/PHASE2_AUDIT.md` §2/§3/§15: SCL's rich `Observation.content`
breaks `materials.analysis`'s comparison-context grouping) remains
**unresolved and untouched this phase** — this phase's CUDA-path
Observations carry the identical content shape as the CPU path
(`interpret_lj_result` does not branch on backend), so they would hit the
exact same `TypeError` if routed through `materials.analysis.analyze_program`.
Not re-tested here (already captured once, in Phase 2); not flattened,
not papered over, not re-owned by SCL.

## §16 — Unresolved

```text
materials.analysis rich-content incompatibility (Phase 2, unchanged -- see §15)
CUDA coincident-particle fault-parity gap (Phase 1, unchanged -- see §10)
CUDA kernel correctness (numerical equivalence) -- genuinely unknown; compiled
    and linked, never executed. This is the single largest open question this
    phase leaves.
CUDA-path determinism -- untested (no device)
evidence_class as a first-class STE schema field (Phase 2, unrelated to CUDA)
```

## §17 — Next executable frontier

**Exactly one concrete target**: run the existing, now-compiling CUDA
kernel on real GPU hardware for the first time, and execute
`test_numerical_equivalence_or_honest_absence` for real — that single
test run (not a new implementation) either confirms or refutes CPU↔CUDA
numerical equivalence for this workload, and is the one thing this phase
could not do without hardware. Everything else this phase built
(the build fix, the shared-reason-string fix, the failure-semantics
tests, the STE-conformance tests) is already in place and does not need
to change to receive that result — only `cuda_cli_path`'s environment
needs a real device.

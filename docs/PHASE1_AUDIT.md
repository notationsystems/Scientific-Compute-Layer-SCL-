# Phase 1 Audit — SCL Substrate

Workflow followed: **BUILD → RUN → OBSERVE → FIX → AUDIT**, per the Phase 1
brief. This document is the AUDIT step; §6 records what BUILD/RUN/OBSERVE/FIX
actually found.

## 1. Environment (verified, not assumed)

| Capability | Present? | Evidence |
|---|---|---|
| g++ / gcc | Yes | GNU 13.3.0 |
| CMake | Yes | 3.28.3 |
| Rust/Cargo | Yes (STE's own toolchain; unused by SCL) | 1.94.1 |
| nlohmann-json | Yes | `apt install nlohmann-json3-dev` (3.11.3), used only by `scl_cli`'s JSON marshaling |
| **CUDA toolkit (nvcc)** | **No** | `which nvcc` fails |
| **GPU / driver** | **No** | `nvidia-smi` not installed, no `/dev/nvidia*` |
| MPI | No | `which mpirun mpicc` fails |
| System BLAS/LAPACK dev packages | No | not installed; only CMake's `FindBLAS`/`FindLAPACK` modules exist, no libraries |

This directly drove the Task 3 backend choice (§3) and the Task 8 scope
(no GPU numbers).

## 2. Reconnaissance summary (Task 1)

Full detail in `SCL_ARCHITECTURE.md` §2. Key facts, each traced to actual
code rather than inferred from names:

1. STE's Rust execution substrate is closed to FFI/native code by explicit
   design, with a guard test (`test_phase127_rust_semantic_boundary.py`)
   that parses Cargo dependency tables to enforce it.
2. STE↔external-engine integration already has a live precedent:
   `execution/gromacs.py`, a real subprocess wrapper around `gmx`, plugged
   into `execution/dispatcher.py`'s `runner` field. This is the seam SCL
   uses.
3. `ExecutionSpecification`/`ExecutionResult` (execution/specification.py,
   execution/engine.py) are the actual "Operation"/result contract — not
   a class named `Operation`.
4. The Phase 112b "computational artifact firewall" is enforced by
   *identity absence*, not a runtime check: only `ModelState` carries an
   `id`, so nothing else can enter `EvidencePool`. The value-level
   discipline this repo actually enforces is `EXECUTION IDENTITY !=
   EVIDENCE IDENTITY` at `execution/dispatcher.py`'s seam — raw execution
   bookkeeping in `record_raw_content`, computed semantics in `content`.
5. DAF's own artifact identity (`daf/storage/identity.py`) is explicitly
   scoped to acquisition (`source_id, locator`) and explicitly documented
   as one of several *non-intersecting* identity spaces
   (`ARCHITECTURE_RECONNAISSANCE.md:371-383`) — reuse the pattern, never
   the namespace. SCL follows the same discipline (see `SCL_ARCHITECTURE.md`
   invariant G).
6. `gromacs-molecular-simulation` and `lammps-md` forks are unmodified
   upstream checkouts (verified directly: plain READMEs, latest commits
   are upstream merge commits, no notationsystems-authored changes).

## 3. First backend selection (Task 3)

**Chosen: a hand-written C++ CPU kernel for truncated Lennard-Jones
pairwise energy/forces**, `lj_pairwise_energy_forces`, with a CUDA backend
compiled-in-when-available but not yet buildable in this sandbox.

Against the brief's selection criteria:

1. *Real scientific computation* — yes: the LJ potential is the
   textbook non-bonded interaction term underlying MD codes including
   GROMACS/LAMMPS; STE's own Rust reference kernel
   (`execution-native::reference::PAIRWISE_ENERGY_DESCRIPTOR`, used for
   zk-provable execution) computes the *same class* of interaction on a
   fixed-point lattice — SCL now supplies the real-valued, HPC-shaped
   version of the same computational idea, which is exactly the division
   of labor Task 3 asks for (Rust = deterministic verifiable execution;
   SCL = high-performance computational capability).
2. *Measurable numerical output* — yes: total energy (scalar) + per-particle
   forces (vector field).
3. *Callable from STE* — proved directly: `test_ste_integration.py`,
   real STE types, real `SpecificationDispatcher`.
4. *Reproducible enough for testing* — yes, same-binary/same-machine
   bitwise, cross-checked numerically (see `SCL_CONTRACT.md` §7).
5. *Easy to validate* — yes: closed-form two-particle energy, finite-
   difference force cross-check, Newton's-third-law invariant, cutoff
   truncation — none requiring reference data from an external source.
6. *Demonstrates native computational capability* — yes: O(N²) double
   loop, the canonical target for the CUDA parallelization Phase 2 adds.
7. *No premature integration complexity* — yes: no GROMACS/LAMMPS build,
   no MPI, no external file formats; see `SCL_ARCHITECTURE.md` §5 for why
   those were explicitly rejected for Phase 1, not merely deferred by default.

## 4. Numerical validation results (Task 5)

All of the following are real, executed test results (not projected),
reproduced at both the native (in-process) layer and through the full
`scl_cli` subprocess boundary:

| Check | Layer(s) | Result |
|---|---|---|
| Two-particle energy matches closed form `4ε[(σ/r)^12-(σ/r)^6]` | native + subprocess | exact to `1e-12` relative |
| Force = −dE/dr, checked by central finite difference (δ=1e-6) | native + subprocess | agrees to `1e-4` (native) / `1e-3` (subprocess, looser due to two extra full runs' floating rounding) |
| Newton's third law: Σ forces = 0 for a 5-particle asymmetric configuration | native + subprocess | `< 1e-9` absolute |
| Cutoff truncation: pair beyond cutoff contributes exactly 0 | native + subprocess | exact (`== 0.0`, not merely small) |
| Sign check: attractive well at moderate separation, repulsive at close range | subprocess | correct sign in both regimes |
| Coincident particles (r=0) faults rather than producing ±∞ | native + subprocess | `ComputeFault::CoincidentParticles` / exit_code 13, no output fabricated |

Reproducibility scope is stated precisely, not oversold — see
`SCL_CONTRACT.md` §7 for the bitwise/numerical/physical/semantic
breakdown. Notably: **physical reproducibility is explicitly not
claimed** — this is a model computation, never a measurement, matching
`execution/gromacs.py`'s own stated posture.

## 5. Failure-path results (Task 7)

10 dedicated tests in `tests/test_failure_paths.py`, all passing, each
pinning a **distinct** exit code / exception type — see `SCL_CONTRACT.md`
§4 for the full table. Concretely exercised:

- invalid parameters (negative sigma, zero cutoff) → `exit_code 11`
- invalid input (empty particle set, malformed buffer lengths) → `exit_code 11`
- unknown operation → `exit_code 10` (distinct from validation)
- missing backend (`cuda` requested, not compiled in) → `exit_code 12`,
  **not** a silent fallback to CPU (`result.backend_used == "cuda"`, the
  thing that was actually asked for, is echoed back)
- backend/computation failure (coincident particles) → `exit_code 13`,
  distinct from validation (`11`)
- malformed native output (a fake CLI emitting non-JSON / incomplete
  JSON) → `SCLProtocolError`, channel-level, never silently accepted
- missing binary entirely → `SCLProtocolError` with a clear build hint
- timeout → `SCLTimeoutError`, distinct from every other failure type
  (forced deterministically with a 1-microsecond budget against a
  2000-particle request, so process-spawn latency alone guarantees the
  timeout regardless of machine speed)
- incompatible kernel version → covered at the STE layer
  (`test_kernel_version_is_part_of_the_program_mirrors_gromacs_precedent`):
  a version bump changes `program_identity`, so a stale/mismatched build
  cannot silently answer under an old identity

## 6. Defects found and corrected during BUILD/RUN/OBSERVE

One genuine defect surfaced by actually running the full picture, not by
inspection:

**STE's own test suite showed 1 failure** when run against the STE
checkout used for this work
(`tests/test_phase120_dispatch_default_witness.py::test_every_construction_site_is_enumerated_and_declares`).
Diagnosis: that test's `_construction_sites()` helper excludes any file
path containing the substring `"notationsystems"` from its scan — a
filter presumably aimed at something else in STE's own history, which
coincidentally matched this session's clone path
(`/home/user/notationsystems/scientific-transformer-engine/...`), causing
it to skip STE's *entire* source tree and find zero construction sites.

**This is not a defect in STE, and not something this SCL work
introduced** — it is an artifact of where the reconnaissance clone
happened to live. Verified independently: the identical STE commit,
cloned to `/home/user/ste-clean` (no `notationsystems` path component),
passes that test and STE's **full suite cleanly: 1914 passed, 0 failed,
108 skipped** (the skips are STE's own environment-gap skips — e.g. no
`gmx` binary — unrelated to SCL). `tests/conftest.py` in this repo now
points at the clean clone path, with a comment recording why, so this
integration suite never depends on a path-sensitive STE checkout.

No other defects were found: the native C++ build succeeded on the first
attempt with zero warnings under `-Wall -Wextra`; all 50 native checks,
all 37 SCL-side pytest tests, and the full STE suite passed without
requiring any fixes to this repo's own code.

## 7. Performance baseline (Task 8)

`scripts/run_benchmark.py`, median of 3 runs per N, CPU backend, this
sandbox's (unspecified/shared) CPU:

| N | wall_clock (ms) | native_compute (ms) | STE-boundary overhead (ms) | input (B) | output (B) |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.991 | 0.0010 | 1.990 | 240 | 248 |
| 50 | 1.892 | 0.0096 | 1.882 | 1,200 | 1,208 |
| 100 | 2.009 | 0.0277 | 1.981 | 2,400 | 2,408 |
| 250 | 2.425 | 0.1147 | 2.311 | 6,000 | 6,008 |
| 500 | 2.964 | 0.3212 | 2.643 | 12,000 | 12,008 |
| 1000 | 4.273 | 1.0456 | 3.228 | 24,000 | 24,008 |
| 2000 | 8.662 | 3.7306 | 4.932 | 48,000 | 48,008 |

Observations (descriptive, not an optimization exercise — matching the
brief's stated purpose for this task):

- **Native compute scales roughly quadratically**, as expected for the
  O(N²) pairwise loop (native time grows ~3700x from N=10 to N=2000,
  against a purely quadratic prediction of 40000x — the small-N numbers
  are dominated by timer-resolution noise around microsecond scale, not
  a true measurement of the algorithm at that size).
- **STE-boundary overhead (process spawn + JSON marshal/unmarshal) is
  roughly constant (~2 ms) at small N and grows slowly with payload size**
  at large N — it dominates wall-clock time up to roughly N≈1000, after
  which native compute starts to dominate. This is the "measurable but
  understandable overhead" the task asks for: it is entirely attributable
  to `subprocess.run` + JSON hex encoding of the payload, not to anything
  opaque.
- Memory was not separately profiled (RSS instrumentation was judged out
  of scope for a Phase 1 baseline); the working set is analytically
  bounded by `N * 3 * 8` bytes of positions plus the same for forces, i.e.
  under 1 MB even at N=2000.
- No GPU numbers: no CUDA toolkit/device in this sandbox (§1).

## 8. Test results (Task 9)

| Suite | Count | Result |
|---|---:|---|
| Native (`ctest` / `scl_native_tests`) | 50 checks (8 test functions) | **50/50 pass** |
| SCL Python (`pytest tests/`) | 37 tests | **37/37 pass** |
| — identity/provenance (`test_contract_identity.py`) | 6 | pass |
| — subprocess client (`test_client_subprocess.py`) | 3 | pass |
| — failure paths (`test_failure_paths.py`) | 13 | pass |
| — numerical validation (`test_numerical_validation.py`) | 5 | pass |
| — performance sanity (`test_performance_baseline.py`) | 1 | pass |
| — real STE integration (`test_ste_integration.py`) | 8 | pass, **none skipped** (real STE checkout was available) |
| STE's own full suite, unmodified | 2022 (1914 pass + 108 skip) | **0 failed** (see §6 for the one path-sensitivity finding, independently resolved) |

No existing STE test was edited, weakened, or deleted — this repo has no
push access to the STE repo and made none of its own tests skip that
should not (STE integration tests were skip-guarded only for the
environment-gap case of no STE checkout being present at all).

## 9. Architectural assumptions that remain unverified (Task 12)

1. **The CUDA backend has never compiled or run.** `native/backends/cuda/lj_pairwise_cuda.cu`
   is written against the CUDA 12 runtime API and mirrors the CPU math
   term-for-term, but this sandbox has no `nvcc` and no GPU. It carries
   none of the verification the CPU path has.
2. **CUDA's coincident-particle fault detection is incomplete.** The CPU
   backend detects `r==0` and reports `ComputeFault::CoincidentParticles`;
   the CUDA kernel currently just skips a zero-distance pair silently
   (documented in-line in the `.cu` file). This must be closed before the
   CUDA backend is trusted to have the same fault contract as CPU.
2b. **CUDA's energy reduction is host-side and unoptimized** (each thread
   writes its own doubled per-particle energy; the host halves the sum)
   — correct in principle, unverified in practice, and not the shape a
   production GPU reduction would take.
3. **x86_64/little-endian/IEEE-754 byte layout is assumed, not abstracted.**
   The wire protocol's raw `float64` encoding would need revisiting on
   a big-endian host.
4. **No cross-machine or cross-compiler bitwise reproducibility test exists** —
   only one machine and one compiler were available in this session.
5. **The `SpecificationDispatcher` integration test built a synthetic,
   minimal `ActionCandidate`** (via `make_action_candidate`/`make_referent`
   directly, bypassing STE's `EvidencePool`/`experiment.session` machinery)
   to prove substitutability at the real seam without the overhead of a
   full campaign fixture. This proves the `runner` contract genuinely
   works; it does not exercise SCL through a full `materials`-driven
   campaign end to end (STE's own `test_execution_dispatcher.py` shows what
   that looks like, at a scale not repeated here since it exercises
   dispatcher plumbing STE already owns and tests, not anything SCL adds).
6. **No load/concurrency testing.** Every test in this suite runs one
   `scl_cli` process at a time; nothing here characterizes behavior under
   many concurrent SCL invocations (STE's own operation ledger, not SCL,
   would be the natural place to serialize/parallelize that if needed).
7. **Timeout and resource limits are wall-clock only** (`subprocess.run(timeout=...)`);
   there is no memory ceiling, CPU affinity, or cgroup-based resource
   constraint enforced by SCL itself.

## 10. Phase 2 recommendation (Task 13)

In priority order, grounded in what this Phase 1 substrate proved and
what it explicitly left open (§9):

1. **Build and validate the CUDA backend on real hardware.** The code
   exists (`native/backends/cuda/lj_pairwise_cuda.cu`); Phase 2's job is
   to build it (`-DSCL_WITH_CUDA=ON`), fix whatever a real `nvcc`/GPU
   surfaces that this sandbox couldn't catch, close the coincident-
   particle fault-parity gap (§9.2), and add a CPU-vs-CUDA numerical
   agreement test (tolerance-based, not bitwise — see `SCL_CONTRACT.md` §7).
2. **A second, structurally different operation** (e.g. a small dense
   matrix operation via BLAS/LAPACK, or an FFT) to prove the
   `operation` dispatch in `native/src/main.cpp` generalizes beyond one
   kernel without new architectural surface — and to test whether the
   current one-operation-per-process-invocation JSON protocol still fits,
   or whether a second operation reveals a real need for the "artifact
   references" and "resource requirements" fields the Phase 1 brief
   scoped out as unjustified for a single kernel (Task 2's instruction not
   to over-generalize applies in reverse now: build the second backend,
   then decide).
3. **Wire SCL into the actual STE repository** — this Phase 1 work has
   read-only access to STE and proved the integration against a local
   clone; Phase 2 should land `execution/scl.py` (or equivalent) inside
   the real `scientific-transformer-engine` repo, alongside
   `execution/gromacs.py`, with STE's own maintainers/tests as the gate,
   not this repo's.
4. **Only then, GROMACS/LAMMPS**, if a real workload needs them: STE
   already has a working GROMACS integration pattern
   (`execution/gromacs.py`) that SCL's adapter deliberately mirrors: no
   new integration is needed until a specific computation exists that
   neither the existing GROMACS path nor a hand-written SCL kernel covers
   well.

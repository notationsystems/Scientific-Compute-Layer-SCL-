# Phase 4 Audit — Real GPU CUDA Equivalence (Hardware-Blocked)

Phase 4's sole objective, as scoped: execute the existing CPU↔CUDA
equivalence harness on real NVIDIA GPU hardware. This session's
environment has no such hardware. Per this phase's own explicit
instruction (§4, "If No GPU Is Available"), the empirical portion stops
here: no result is fabricated, no fallback is relabeled as CUDA, no
tolerance is weakened, and no new implementation work was done. This
document is the honest report that instruction calls for, plus
reconfirmation that everything Phase 3 established still holds.

## §0 — Closure addendum: hardware target vs. execution-environment visibility

Added after Phase 4's original report, once the specific target hardware
was identified. Nothing below changes Phase 4's conclusion — it sharpens
it with the actual target device and a second, independent runtime check.

**Physical development machine (as identified, not independently
inspectable from this session):**

| Device | Role |
|---|---|
| NVIDIA GeForce RTX 2080 Super with Max-Q Design, 8 GB VRAM | the CUDA target |
| Intel(R) UHD Graphics, 128 MB reported graphics memory | **not** the CUDA target — an integrated display adapter, not CUDA-capable at all |

**This remote/container execution environment does not expose that GPU.**
Checked two independent ways, both agreeing:

1. `cudaGetDeviceCount()`, via the actual compiled `scl_cli` binary — **0**.
2. `libnvidia-ml.so.1`'s `nvmlInit_v2()`, called directly (bypassing the
   CUDA runtime entirely) — returns **`NVML_ERROR_DRIVER_NOT_LOADED`
   (code 9)**; `nvmlDeviceGetCount_v2` correspondingly reports 0.

Supporting evidence: no `nvidia` entry in `lsmod`; `dmesg` contains zero
NVIDIA-related lines (the kernel has never even attempted to probe NVIDIA
hardware); `/dev/nvidia0` (the per-GPU device node) and
`/proc/driver/nvidia` (kernel driver state) are both absent;
`nvidia-smi` is not installed. `/dev/nvidiactl` **does** exist
(`crw-rw-rw-`, root:root) — but a control-device placeholder with no
working driver behind it (confirmed by the NVML/kernel-module checks
above) is not evidence of functional GPU access, and is not treated as
such here.

**The distinction this repository preserves**: physical hardware
availability (real, on the machine described) is a separate fact from
GPU visibility inside this execution environment (absent, checked
directly, twice, by independent APIs). Phase 4's conclusion holds
unchanged with the specific target hardware now on record.

**SCL software frontier closed; CUDA empirical validation pending
GPU-accessible execution environment.** No further CUDA implementation
work should be undertaken merely to advance the project — the harness
already exists, is already correct-by-inspection, and needs only to be
run in an environment where the RTX 2080 Super Max-Q is actually
reachable. See §"Next executable frontier" below.

## Hardware verification (§3)

Checked directly, fresh, this session — not assumed from Phase 3's
record:

| Item | Value |
|---|---|
| GPU vendor / model | **None present** |
| PCI VGA/3D device (`lspci`) | none found |
| `/dev/nvidia*` | does not exist |
| `/proc/driver/nvidia` | does not exist |
| `nvidia-smi` | not installed / not runnable |
| GPU compute capability | n/a — no device |
| CUDA driver version | n/a — no driver installed (this is a compiler-only toolchain) |
| CUDA runtime version | 12.0.140 (`libcudart`, installed, loadable) |
| `nvcc` version | 12.0.140, present, functional (same install as Phase 3) |
| Device count (`cudaGetDeviceCount()`) | 0 (confirmed live, §2 below) |
| Available device memory | n/a — no device to query |

**`nvcc available` is explicitly not treated as `CUDA execution
available`** — the distinction this phase requires. Compiling and
linking succeed; nothing beyond that was, or could honestly be, claimed.

## §1/§2 — Re-inspection and reconfirmation of Phase 3 state

Before writing anything, the existing harness was re-inspected
(`docs/PHASE3_AUDIT.md`, `native/CMakeLists.txt`, `native/backends/cuda/`,
`native/src/lj_pairwise.cpp`, `native/tests/test_lj_pairwise.cpp`,
`tests/test_cpu_cuda_equivalence.py`, `native/include/scl/backend.hpp`,
`python/scl/ste_adapter.py`) and confirmed, not assumed, to already do
exactly what §2 of this phase asks: `test_numerical_equivalence_or_honest_absence`
genuinely runs CPU and CUDA requests over **identical scientific
input**, decodes both outputs, and would compute absolute/relative/max-
elementwise/RMS error against a real `1e-9` tolerance — the comparison
code is written and ready, gated only on `cuda_result.status ==
"completed"` ever being true. **Nothing about the harness was replaced,
rewritten, or weakened to produce a passing result** — running it
unmodified against this hardware is exactly what was done.

Reconfirmed by actually rebuilding from a clean configure this session
(not reusing a stale artifact):

- `cmake -S native -B native/build_cuda -DCMAKE_BUILD_TYPE=Release -DSCL_WITH_CUDA=ON` — configures cleanly.
- `cmake --build native/build_cuda` — **compiles and links successfully** (all of `scl_core`, `scl_cli`, `scl_native_tests`, 5 target architectures).
- `native/build_cuda/tests/scl_native_tests` — **51/51 checks pass**, including the real `cudaGetDeviceCount()` exercise (genuinely called, genuinely returns 0 — this is the live confirmation behind the hardware table above).
- Full SCL Python suite: **59 passed, 1 skipped** — the one skip is `test_numerical_equivalence_or_honest_absence`, for the same reason as Phase 3: no device.
- STE's own full suite, unmodified: **1914 passed, 0 failed**, 109 skipped (STE's own environment-gap skips, unrelated to this phase).

No source file was changed this phase. `git status` in both repos is
clean before and after this session's work.

## Implemented

**None.** Per §4/§15 of this phase's instructions: no fabricated result,
no CPU-fallback-relabeled-as-CUDA, no weakened test, no new "artificial
CUDA implementation phase." There was nothing genuine to implement
without hardware — verification, not construction, is this phase's job,
and verification came back hardware-blocked.

## Verified

Exactly what §1/§2 above lists: a clean CUDA rebuild, native tests,
the full Python suite, and STE's full suite — all actually executed this
session, all results reported above verbatim, nothing else.

## Preserved

Every SCL and STE invariant from Phase 1–3: identity/provenance
separation, the Phase 112b firewall, evidence admission and
`ModelState.update()` write-path protection, the backend-unavailable
fault-priority ordering, `backend_unavailable_reason()`'s single source
of truth. Unchanged and re-confirmed passing, not re-derived.

## Extended

**None.** No new CUDA capability was added this phase — there was
nothing to extend without hardware to run it on.

## Integrated

No new STE/SCL boundary was exercised beyond what Phase 3 already
proved (`run_scl_specification`, `interpret_lj_result`,
`run_experiment_step`, `ModelState.update()` — all already exercised for
a halted CUDA request in Phase 3, re-confirmed passing this phase, not
re-integrated).

## Qualified

The CUDA architecture default (`70;75;80;86;89`, Phase 3) remains an
explicitly stated, unverified-against-real-hardware default — unchanged,
since no hardware exists this phase either to verify or correct it
against.

## Bent

Zero.

## Hardware state

See the table above — the authoritative answer for this phase.

## CUDA execution state

```text
compiled        YES  (native/build_cuda, this session, clean configure)
linked          YES  (scl_cli, scl_native_tests, all target architectures)
runtime-loaded  YES  (libcudart loads; cudaGetDeviceCount() call itself executes
                      and returns successfully with count=0 -- the CUDA runtime
                      IS reachable, it simply reports no device)
GPU-executed    NO   (no device exists to launch a kernel on)
```

## Numerical equivalence

**Unverified.** No energy/force comparison was performed — there is no
GPU-produced result to compare against the CPU reference. Absolute
error, relative error, maximum elementwise error, and RMS error are all
**not applicable**, not zero, not passing. Reporting them as anything
other than "not measured" would misrepresent what happened this phase.

## Determinism

**Untested for CUDA** (no device). CPU-side bitwise determinism (same
binary, same machine, repeated runs) remains verified from Phase 1–3 and
was not re-tested this phase (nothing about the CPU path changed).

## STE conformance

Unchanged from Phase 3's already-verified result: a CUDA-selected
`ExecutionSpecification`, run through the real `SpecificationDispatcher`/
`run_experiment_step` seam, correctly halts with `BACKEND_UNAVAILABLE`
(exit_code 12) and admits nothing (`EvidencePool.fingerprint()`
unchanged) — re-confirmed passing this session
(`test_cuda_selected_result_conforms_to_the_same_ste_execution_result_type`,
`test_cuda_selection_through_the_real_experiment_step_admits_nothing`).
No CUDA computation has ever been accepted as evidence, because none has
ever completed.

## Performance

**Not measured.** Per this phase's own instruction (§10: "Only after
numerical equivalence passes should performance measurement begin"),
and equivalence did not pass — it was never reachable. No CPU/GPU/
transfer timing, no workload sweep, no speedup number was collected or
estimated.

## Crossover

**Not determined.** Requires the performance measurement above, which
requires numerical equivalence, which requires a GPU. None of the
prerequisite steps occurred.

## Failure state

The CUDA failure path this environment CAN exercise — backend
unavailable, because no device exists — was exercised and re-confirmed
(§1/§2 above): explicit `BACKEND_UNAVAILABLE` state, no fabricated
result, no evidence promotion. Failure modes that require an actual
device to trigger (allocation failure, kernel launch failure, host↔device
transfer failure, unsupported device configuration) remain **untestable
in this environment**, exactly as reported in Phase 3 — not silently
passed, not fabricated.

## Measured bottleneck

**None established.** The only bottleneck this phase could measure is
"no GPU hardware in this environment" — which is an infrastructure fact,
not a computational one.

## Unresolved

```text
CPU/CUDA numerical equivalence      -- genuinely unknown; the single largest
                                        open question, unchanged since Phase 3,
                                        blocked on hardware access, not on any
                                        remaining implementation work
CUDA-path determinism                -- untestable without a device
CUDA performance / crossover point   -- untestable without a device
materials.analysis rich-content incompatibility (Phase 2, unrelated to CUDA, still unresolved)
CUDA coincident-particle fault-parity gap (Phase 1, unrelated to hardware access, still unresolved)
```

## Next executable frontier

Run the existing CUDA equivalence/determinism/performance harness when
the RTX 2080 Super Max-Q is exposed to the execution environment.

That is the only remaining step
(`tests/test_cpu_cuda_equivalence.py::test_numerical_equivalence_or_honest_absence`,
plus `native/build_cuda/tests/scl_native_tests`, plus the performance
sweep once equivalence passes). No code, test, or tolerance needs to
change to receive that result — every prerequisite is already built,
committed, and verified working up to the exact point hardware access is
required. This is not a design or implementation gap; it is an
infrastructure access gap, and no further "CUDA implementation phase"
against this same hardware-less environment would change that. Do not
create another SCL implementation phase to work around it.

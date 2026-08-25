# Scientific Compute Layer (SCL)

SCL is a **reusable native scientific-computation layer** (C/C++/CUDA):
a request/response contract, a process-boundary CLI, and typed
result/quantity shapes, usable by any external application. The
Scientific Transformer Engine (STE) is **one integration environment**
for SCL, not SCL's definition — see `docs/SCL_STANDALONE_BOUNDARY.md` for
the repository-grounded proof (a fresh-interpreter test with no STE
checkout even on `sys.path`) and the formal "SCL Core Independence"
invariant. `docs/SCL_ARCHITECTURE.md` covers the full boundary
description, including the integrated (Notations/STE) configuration;
`docs/SCL_CONTRACT.md` is the wire-level request/result contract;
`docs/PHASE1_AUDIT.md` covers numerical validation, failure-path, and CPU
performance results; `docs/PHASE2_AUDIT.md` covers how SCL's computed
results conform to STE's real evidence/derived-state machinery when
integrated; `docs/PHASE3_AUDIT.md`/`docs/PHASE4_AUDIT.md` cover CPU↔CUDA
build/correctness status.

## Two ways to use SCL

```python
# 1. Standalone -- any external application. No STE, no Notations.
from scl import SCLRequest, run_scl_request, encode_lj_configuration, encode_lj_positions

request = SCLRequest(
    operation="lj_pairwise_energy_forces", backend="cpu",
    parameters=encode_lj_configuration(epsilon=1.0, sigma=1.0, cutoff=5.0),
    input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
)
result = run_scl_request(request, cli_path="/path/to/scl_cli")
```

```python
# 2. Integrated into STE -- explicit opt-in, a separate import.
from scl.ste_adapter import build_lj_specification, run_scl_specification
# run_scl_specification returns STE's own ExecutionSpecification/ExecutionResult
# and plugs directly into execution.dispatcher.SpecificationDispatcher.
```

`import scl` alone (path 1) never imports `scl.ste_adapter` or anything
from STE — checked mechanically by `tests/test_standalone_boundary.py`,
not left as a convention.

## CUDA backend status

**SCL software frontier closed; CUDA empirical validation pending
GPU-accessible execution environment.**

The CUDA implementation is complete and verified up to the boundary that
requires physical GPU access:

| | Status |
|---|---|
| CUDA kernel implemented (`native/backends/cuda/lj_pairwise_cuda.cu`) | done |
| Real `nvcc` compilation | verified |
| CUDA linking | verified |
| `libcudart`/runtime loading | verified |
| CUDA availability detection (`cudaGetDeviceCount()`) | verified |
| Native CUDA tests | verified — 51/51 |
| SCL → STE conformance for a CUDA-selected request | verified |
| **Actual GPU kernel execution** | **not yet verified — no GPU-accessible environment** |
| **CPU ↔ CUDA numerical equivalence** | **not yet verified** |
| **CUDA determinism** | **not yet verified** |
| **CUDA performance / CPU-GPU crossover** | **not yet verified** |

**Hardware target**: the physical development machine has an NVIDIA
GeForce RTX 2080 Super with Max-Q Design (8 GB VRAM) — the CUDA target.
It also reports an Intel(R) UHD Graphics device (128 MB); that is a
display adapter, not CUDA-capable, and is not the target.

**This distinction is preserved deliberately**: physical hardware
availability and GPU visibility inside a given execution environment are
two different facts. The RTX 2080 Super Max-Q being real on the
development machine does not mean any particular session (including
remote/containerized ones) can see it. The current environment's fresh,
independently-confirmed state: `cudaGetDeviceCount() == 0`;
`nvmlInit_v2()` returns `NVML_ERROR_DRIVER_NOT_LOADED`; no `nvidia`
kernel module loaded; `/dev/nvidia0` and `/proc/driver/nvidia` both
absent; `nvidia-smi` not installed. (`/dev/nvidiactl` exists as a
placeholder control node only — not evidence of functional GPU access.)
Full detail: `docs/PHASE4_AUDIT.md` §0.

**No further CUDA implementation work should be undertaken merely to
advance this project.** The numerical-equivalence harness
(`tests/test_cpu_cuda_equivalence.py`) already exists, is already
correct by inspection, and is the one thing left to run — when a
GPU-accessible environment (with the RTX 2080 Super Max-Q reachable) is
available, run that harness directly rather than building another CUDA
implementation.

```
                              SCL
                               |
                 +-------------+-------------+
                 |                           |
        Standalone consumers            ste_adapter.py
     (any external application)               |
                                               v
                              Scientific Transformer Engine (STE) — Rust
                                               |
                              Scientific Intelligence Layer (SIL)
                                               |
                              Scientific Workbench / Projection Layer
```

The integrated configuration (SCL attached beneath STE inside the full
Notations stack) is real and verified — see `docs/SCL_ARCHITECTURE.md`
§1 for that diagram in its original stack form. Both are the same
repository; which path applies depends only on whether a caller imports
`scl.ste_adapter`.

## What's here

- `native/` — the C++17 computational substrate. Two scientific
  operations behind one fixed dispatch registry: `lj_pairwise_energy_forces`
  (truncated Lennard-Jones energy/forces) and `fourier_transform_1d` (the
  1-D discrete Fourier transform — the mathematical operation, not "FFT";
  CPU evaluates the defining O(N^2) sum, CUDA would use cuFFT). A CPU
  backend, a
  CUDA backend (compiles and links against a real CUDA 12 toolchain;
  never GPU-executed in any session so far — see "CUDA backend status"
  above and `docs/PHASE3_AUDIT.md`/`docs/PHASE4_AUDIT.md`), and `scl_cli`,
  the process-boundary entry point. No Notations/STE dependency anywhere
  in `native/`.
- `python/scl/` — `client.py`/`identity.py`/`errors.py`/`quantity.py`/
  `method_block.py` (the standalone core — zero STE dependency, all
  re-exported from `scl/__init__.py`) and `ste_adapter.py` (the *only*
  file that imports STE: translates STE's real
  `ExecutionSpecification`/`ExecutionResult` to/from SCL calls — a
  drop-in alternative to `execution.gromacs.run_gromacs_specification`,
  emitting an evidence-classed, method-blocked, typed-quantity content
  shape — see `docs/PHASE2_AUDIT.md`). See `docs/SCL_STANDALONE_BOUNDARY.md`.
- `tests/` — identity/provenance, subprocess client behavior, failure
  paths, numerical validation, a real integration suite against a local
  STE checkout (no mocks), full evidence/derived-state conformance
  (Phase 2), CPU↔CUDA build/failure-semantics coverage (Phase 3), and the
  standalone-boundary proof (`test_standalone_boundary.py`).
- `scripts/run_benchmark.py` — the Phase 1 CPU performance baseline sweep.

## Build and test

```sh
# Native substrate
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure

# Python + integration suite (builds native/ automatically if needed)
pip install pytest
python3 -m pytest tests/ -v
```

The real-STE integration tests (`tests/test_ste_integration.py`) look for
a local STE checkout via the `STE_REPO` environment variable (default
`/home/user/ste-clean` in this development environment) and skip cleanly,
rather than failing, if none is found.

To build the CUDA backend (requires the CUDA toolkit; compiles and links
successfully, never GPU-executed in any session so far — see "CUDA
backend status" above and `docs/PHASE4_AUDIT.md`):

```sh
cmake -S native -B native/build_cuda -DCMAKE_BUILD_TYPE=Release -DSCL_WITH_CUDA=ON
cmake --build native/build_cuda
# tests/test_cpu_cuda_equivalence.py picks this up automatically via the
# cuda_cli_path fixture (skips cleanly if nvcc is not on PATH)
```

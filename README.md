# Scientific Compute Layer (SCL)

The Scientific Compute Layer: native C/C++/CUDA computational capability
attached beneath the Scientific Transformer Engine (STE), as a strict
architectural superset extension. See `docs/SCL_ARCHITECTURE.md` for the
full boundary description and the repository-grounded map of where SCL
attaches to STE, `docs/SCL_CONTRACT.md` for the wire-level request/result
contract, `docs/PHASE1_AUDIT.md` for numerical validation, failure-path,
and CPU performance results, `docs/PHASE2_AUDIT.md` for how SCL's
computed results conform to STE's real evidence/derived-state machinery,
and `docs/PHASE3_AUDIT.md` for CPU↔CUDA build/correctness status (the
CUDA kernel now compiles and links against a real toolchain; it has never
been executed on a device — this development environment has no GPU).

```
Scientific Workbench / Projection Layer
            |
Scientific Intelligence Layer (SIL)
            |
Scientific Transformer Engine (STE) — Rust
            |
Scientific Compute Layer (SCL) — C/C++/CUDA        <-- this repository
            |
Physical / Numerical Computation
```

## What's here

- `native/` — the C++17 computational substrate: a Lennard-Jones pairwise
  energy/forces kernel (`lj_pairwise_energy_forces`), a CPU backend, a
  CUDA backend (compiles and links against a real CUDA 12 toolchain as of
  Phase 3; never GPU-executed — no device exists in this environment; see
  `docs/PHASE3_AUDIT.md`), and `scl_cli`, the process-boundary entry point.
- `python/scl/` — `client.py` (a standalone SCLRequest/SCLResult
  subprocess client), `ste_adapter.py` (translates STE's real
  `ExecutionSpecification`/`ExecutionResult` to/from SCL calls — a
  drop-in alternative to STE's `execution.gromacs.run_gromacs_specification`,
  now also emitting an evidence-classed, method-blocked, typed-quantity
  content shape — see `docs/PHASE2_AUDIT.md`), `quantity.py`, and
  `method_block.py`.
- `tests/` — identity/provenance, subprocess client behavior, failure
  paths, numerical validation, a real integration suite against a local
  STE checkout (no mocks), full evidence/derived-state conformance
  (Phase 2), and CPU↔CUDA build/failure-semantics coverage (Phase 3).
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
successfully as of Phase 3, never GPU-executed in this environment — see
`docs/PHASE3_AUDIT.md`):

```sh
cmake -S native -B native/build_cuda -DCMAKE_BUILD_TYPE=Release -DSCL_WITH_CUDA=ON
cmake --build native/build_cuda
# tests/test_cpu_cuda_equivalence.py picks this up automatically via the
# cuda_cli_path fixture (skips cleanly if nvcc is not on PATH)
```

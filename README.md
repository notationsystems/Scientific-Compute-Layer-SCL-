# Scientific Compute Layer (SCL) — Phase 1

The Scientific Compute Layer: native C/C++/CUDA computational capability
attached beneath the Scientific Transformer Engine (STE), as a strict
architectural superset extension. See `docs/SCL_ARCHITECTURE.md` for the
full boundary description and the repository-grounded map of where SCL
attaches to STE, `docs/SCL_CONTRACT.md` for the wire-level request/result
contract, and `docs/PHASE1_AUDIT.md` for numerical validation, failure-path,
and performance results.

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
  CUDA backend (compiled only when the toolkit is available — untested in
  this development environment; see the audit), and `scl_cli`, the
  process-boundary entry point.
- `python/scl/` — `client.py` (a standalone SCLRequest/SCLResult
  subprocess client) and `ste_adapter.py` (translates STE's real
  `ExecutionSpecification`/`ExecutionResult` to/from SCL calls — a
  drop-in alternative to STE's `execution.gromacs.run_gromacs_specification`).
- `tests/` — 37 tests: identity/provenance, subprocess client behavior,
  10 distinct failure-path tests, numerical validation (closed-form and
  finite-difference cross-checks), a performance sanity check, and a real
  integration suite against a local STE checkout (no mocks).
- `scripts/run_benchmark.py` — the Task 8 performance baseline sweep.

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

To build the CUDA backend (requires the CUDA toolkit; untested — see
`docs/PHASE1_AUDIT.md` §9):

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DSCL_WITH_CUDA=ON
cmake --build native/build
```

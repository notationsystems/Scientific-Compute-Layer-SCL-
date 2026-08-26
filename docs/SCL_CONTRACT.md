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

## 2. The operations this substrate implements

SCL is multi-operation. `scl_cli` dispatches on the request's `operation`
field through a fixed registry (`native/src/operation_registry.cpp`); each
operation owns its own configuration/input decoding, validation, backend
dispatch, output encoding and metrics behind `scl::Operation`
(`native/include/scl/operation.hpp`), while the envelope, hex framing,
backend-availability ordering, fault vocabulary and response shape are
shared and operation-agnostic.

### 2.1 `lj_pairwise_energy_forces`

Truncated Lennard-Jones pairwise potential energy and per-particle forces
for an N-particle system (see `SCL_ARCHITECTURE.md` §5 and
`PHASE1_AUDIT.md` §3 for why this operation was chosen first).

```
V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6],  r <= cutoff
V(r) = 0,                                        r >  cutoff   (plain truncation, not shifted)
```

### 2.2 `fourier_transform_1d`

The one-dimensional discrete Fourier transform — the MATHEMATICAL
operation, deliberately not named "FFT": FFT is an implementation
strategy, and the CPU backend here evaluates the defining O(N^2) sum while
the CUDA backend would use cuFFT. Same operation, same contract, different
algorithms; which one ran is recorded in the method block's `algorithm`
field, never conflated with the operation identity.

    forward  (direction=+1):  X_k = s * SUM_n x_n exp(-2*pi*i*k*n/N)
    inverse  (direction=-1):  X_k = s * SUM_n x_n exp(+2*pi*i*k*n/N)

| Aspect | Contract |
|---|---|
| input | N complex samples, `N*16` bytes, `(real f64, imag f64)` LE. A real signal is supplied with zero imaginary parts — complex-in/complex-out avoids all Hermitian-packing subtleties |
| output | N complex bins, same layout, `k = 0..N-1` ascending, no fftshift |
| direction | `+1` forward (negative exponent), `-1` inverse |
| normalization | `0` none (the bare sum above), `1` `1/N`, `2` `1/sqrt(N)` (unitary) — explicit, never implicit |
| precision | IEEE-754 float64 throughout; fixed by the contract, not a parameter |
| supported N | any `N >= 1`; no power-of-two restriction (validated at prime N=37) |
| sample spacing | OPTIONAL. Carried in configuration so it participates in `parameters_identity`, but **not used by the transform** |
| frequency axis | Present only when Δt was supplied: `f_k = k/(N*Δt)` for `k <= N/2`, `(k-N)/(N*Δt)` above. With no Δt, results are bin/index-only and the method block marks the axis not-applicable. **SCL never assumes Δt = 1** |

Configuration is exactly 24 bytes: `int32 direction | int32 normalization
| int32 has_sample_spacing | int32 reserved(=0) | float64
sample_spacing_seconds`.

**Measured identity property**: two requests differing only in Δt produce
byte-identical output, so they share an `output_identity` and
`computation_identity` while differing in `parameters_identity` and
request `identity()`. That is correct and deliberate — Δt does not change
what was computed, only how a consumer may interpret it — and it is
asserted directly in `tests/test_fourier_contract.py`.

**Validation posture**: the transform is validated primarily against
INDEPENDENT MATHEMATICS — impulse → flat spectrum, DC → single bin, pure
tone → exactly the predicted bin(s), Parseval's energy relation, unitary
normalization, and inverse reconstruction — because those hold regardless
of implementation. A hand-written stdlib O(N^2) DFT oracle is also present
but is deliberately the weakest evidence, not the primary argument: two
implementations of one spec reading can agree and both be wrong.

**Measured performance (CPU, this environment)**: cost per pair is flat at
~23 ns and doubling N consistently quadruples runtime — textbook O(N^2),
exactly as the direct-sum implementation intends. Native compute overtakes
the ~4–7 ms process/JSON boundary cost at N ≈ 512 (N=2048: 96 ms native,
103 ms wall). The measured bottleneck is therefore the ALGORITHM, not the
SCL boundary.

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

### 6.1 Annotating vs. participating parameters

Δt surfaced this as a Fourier curiosity — two requests differing only in
sample spacing produce byte-identical output, so they share a
`computation_identity` while differing in `request_identity`. It is not a
Fourier curiosity. It is a general shape, and it is stated here as a rule
because the same shape will recur every time a parameter annotates a
result without participating in it.

Every configuration field is exactly one of:

- **PARTICIPATING** — its value is read by the mathematics. Changing it
  changes the output bytes. (Fourier: `direction`, `normalization`.
  LJ: `epsilon`, `sigma`, `cutoff`.)
- **ANNOTATING** — its value is never read by the mathematics. Changing it
  cannot change the output bytes. It exists so a consumer can interpret
  the numbers. (Fourier: `sample_spacing_seconds`.)

The rule, in four clauses:

1. **Both kinds enter `parameters_identity` and `request_identity`.** An
   annotating field is part of what was *asked for*, so a request that
   supplies it is a different request. It is carried in the configuration
   bytes for exactly this reason — no side channel, no separate metadata
   identity.
2. **Only participating fields may enter `output_identity` and
   `computation_identity`.** Those identify what was *computed*, and an
   annotating field by definition did not affect that.
3. **Therefore equal `computation_identity` with differing
   `request_identity` is the SIGNATURE of an annotating field, not a
   defect.** A cache or dedup layer that treats identity collision as
   request equivalence is wrong at exactly this point: it may reuse the
   output bytes, and must not reuse the interpretation.
4. **An annotating field is never silently defaulted.** Absent and
   present-with-the-conventional-value are different facts and must remain
   distinguishable in `parameters_identity`. SCL does not assume Δt = 1;
   a result computed without Δt is bin-indexed, and the method block marks
   the frequency axis `applicable: false` rather than inventing an axis.
   This is the same absence discipline as `uncertainty_kind: absent` one
   layer up, and as clause 7 of the operation contract (an operation with
   no particles reports no `n_particles`, not `n_particles: 0`) one layer
   down. Three layers drawing the same distinction is why it is written as
   a rule rather than as three local conventions.

**Units are the next instance, and they are a sharper one.** A unit
annotates a quantity without participating in its arithmetic — the same
shape as Δt — but with a failure mode Δt does not have: unlike a
frequency axis, which is visibly absent when Δt is absent, a wrong unit is
invisibly present. When SCL grows unit-carrying quantities, clause 4 is
the clause that matters: an absent unit must not resolve to an assumed one.

The classification is mechanically enforced across the whole registry by
`tests/test_annotating_parameters.py`, which asserts each declared field
against its declared class rather than trusting the declaration.

**Unresolved downstream edge (recorded, not solved here).** Clause 3 says
two results with equal `computation_identity` may differ in
interpretation. That makes *comparability* weaker than identity: two
Fourier results over the same samples with different Δt are comparable as
bin-indexed vectors and NOT comparable as spectra, because bin `k` denotes
a different frequency in each. Nothing in SCL currently prevents a
consumer from comparing them as spectra — SCL does not own the comparison
layer, and there is no comparison layer yet to place the check in. When
one exists, the rule it needs is: results are spectrally comparable only
when their annotating fields agree, not merely when their
`computation_identity` agree. Written down now so it is a known edge
rather than a discovery.

### 6.2 One meaning, one encoding

> **A reader that normalizes is not a fix. It relocates the ambiguity to
> whoever opens the artifact.**

That is the repair, stated first because it is the sentence that stops the
next instance being fixed the wrong way. Every occurrence of this class so
far has had an available reader-side "fix" that looked cheaper and would
have moved the defect rather than closing it. The encoding is made
canonical **at the writer**, and the ambiguous form is refused.

Section 6.1 asks which parameters *participate*. This asks a prior
question: whether a parameter has one encoding at all.

**The rule.** Wherever a presence flag, a discriminant, or a reserved word
guards a payload, the guarded bytes must have **exactly one encoding for
each meaning**. A payload the guard renders unused must be *refused* when
it carries a value — never accepted and ignored.

**Why it is an identity rule and not a tidiness rule.** Two
byte-different configurations that mean the same thing produce different
`parameters_identity` values. That destroys the premise the whole identity
model rests on — that a parameter identity identifies the parameters — and
it does so silently, because both requests compute correctly.

**It was live in shipped, validated code.** `fourier_transform_1d`
accepted arbitrary bytes in `sample_spacing_seconds` whenever
`has_sample_spacing` was 0, and ignored them. So `no spacing` had 2^64
encodings. The operation had passed its full contract suite, its analytic
validation, and its identity tests; the defect was found by a mutation
check written for a *different* clause. That is worth recording as a fact
about coverage: a suite that passes completely can still leave a class
entirely unprobed.

**Third instance of one failure class.** "Two encodings, one meaning,
different digests" has now appeared at three layers:

| layer | the two encodings | what disagreed |
|---|---|---|
| exchange serializer | bare vs quoted YAML scalar | two parsers on the *type* |
| operation configuration | ignored payload behind a clear flag | two byte strings on `parameters_identity` |
| evidence content | absent vs sentinel vs omitted | absence with several spellings |

All three are closed by the move stated at the top of this section.

**Two independent arrivals, in one phase, by sessions that were not
coordinating.** The compute layer generalized this rule over
presence-flag-guarded payloads at the same time the acquisition layer
added `VALUE_AND_ABSENCE_BOTH_PRESENT` to its observation-table gate —
refusing a value and an absence-reason asserted together. Neither knew of
the other's work. Recorded as a candidate for **core** rather than for
SCL's contract or DAQ's, alongside the absent-is-not-zero candidate it
closely resembles.

**And a correction to how that arrival was first read.** This paragraph
originally called the second arrival *evidence about the class rather than
a coincidence about the phase*. That claim was later falsified directly.
Two sessions independently closed the canonical-YAML collection class,
independently reached the same refusal, by the same argument — and
independently wrote down the same **wrong reason** for it. Both recorded
that the dependency-free reader *refuses* a nested sequence. It does not:
the compact single-element form is silently mistyped, returning a string
where PyYAML returns a list, with no error on either side. Each half had
probed a form that does raise, and generalized from it.

So convergence is demoted here from evidence to **a prompt to
re-measure**. Two authors agreeing shows they share a method, a
vocabulary, or a blind spot — often all three; where both probed the same
shape, the second arrival added no coverage at all. What caught it was
running the two readers against each other on every shape, including the
ones neither half had probed.

One practice does survive intact, because it rests on coverage rather than
on agreement: when two independent efforts converge, **union their
fixtures** rather than choosing one. Neither set subsumed the other, and a
shape nobody pins is a shape nobody agreed.

Both the arrival and this correction are recorded in
`architecture/proof_integrity.yaml`, held byte-identical in both
repositories.

**Where it reaches next.** Any optional field, any union behind a kind
discriminant, any reserved word — and, one layer up, any representation of
*missing*. Absence is the same shape: if `missing` can be spelled as
`null`, as NaN, as an omitted key, or as a sentinel value, then one
meaning has four encodings and the identity of a record depends on which
was chosen.

Enforced by clause 2 of the operation contract
(`native/include/scl/operation.hpp`), checked for every registered
operation by `tests/test_operation_registry_contract.py`.

### 6.3 Supplied or asserted: the third axis

§6.1 splits a parameter by **what it changes** — participating fields
enter `output_identity` and `computation_identity`, annotating ones do
not. Kalman needs a second, orthogonal question, and the two must not be
conflated.

A linear filter carries two noise matrices. Both are parameters, both are
asserted by the caller, and **both participate** — change either and the
trajectory changes. On §6.1's axis they are indistinguishable. But:

- **R**, the measurement noise covariance, *may be measurement-derived*.
  An acquisition layer that characterises its instrument can supply it,
  and then it is traceable to something that was actually measured.
- **Q**, the process noise covariance, **never is**. There is no
  measurement of process noise. It states how much the modeller believes
  the state wanders between observations — a modelling assertion, always.

If both land in `parameters_identity` undifferentiated, the execution
record cannot answer the question an auditor will actually ask: *was this
filter's noise model supplied, or assumed?* Two runs with identical
`parameters_identity` may have one R characterised from a calibration and
another chosen to make the filter converge, and nothing separates them.

So each noise matrix carries a **provenance discriminant** beside its
value:

| discriminant | meaning | guarded payload |
|---|---|---|
| `asserted` | a modelling choice; no measurement stands behind it | none — a payload here is **refused** |
| `supplied` | derived from measurement | the identity of the input that supplied it |

Three consequences, none optional:

1. **`Q` may only ever be `asserted`.** `supplied` on Q is a validation
   fault, not a tolerated oddity — it would claim a measurement that
   cannot exist.
2. **The discriminant guards a payload, so clause 2 governs it.** This is
   exactly the presence-flag shape the operation contract states over: the
   source-identity bytes must have one encoding when absent, and a payload
   supplied under `asserted` must be refused rather than accepted and
   ignored. Not a new rule — the existing one arriving at its next
   instance, which is how §6.2 said the class would spread.
3. **The discriminant participates.** `asserted` and `supplied` describe
   different computations at identical numeric values, because they
   license different conclusions from the same trajectory. It is not an
   annotation.

The axes are independent and both are needed: §6.1 asks *does this change
the numbers*, §6.3 asks *does a measurement stand behind them*.

## 7. Reproducibility — precisely scoped (Task 5)

| Kind | Claimed? | Evidence |
|---|---|---|
| **Bitwise** (same binary, same machine, same backend) | Yes | `test_repeat_determinism_same_binary_same_machine` (two full subprocess round trips, byte-identical `output`); native `test_bitwise_reproducible_same_process`. Enabled by `-fno-fast-math`, no SIMD flags, IEEE-754 strict double math (`native/CMakeLists.txt`). |
| **Numerical** (tolerance-based, cross-implementation) | Yes, for the force/energy relationship | `test_force_matches_finite_difference_gradient` (native, in-process) and `test_force_matches_finite_difference_energy_gradient` (through the full subprocess boundary) — an independent numerical check (finite-difference gradient of energy) against the analytically-derived force formula, within `1e-4`–`1e-3` relative tolerance. |
| **Physical** (matches a real experiment/measurement) | **Not claimed** | This is a model computation over an idealized system (a Lennard-Jones potential), never a measurement. `execution/gromacs.py`'s own "COMPUTATION != MEASUREMENT" posture applies with equal force here. |
| **Semantic** (same request ⇒ same identity) | Yes | `test_result_identity_is_content_addressed_across_two_runs`, `test_repeat_determinism_same_binary_same_machine` (STE-level). |
| **Cross-machine / cross-compiler bitwise** | **Not claimed, not tested** | No second machine or compiler was available in this environment. |
| **CPU vs. CUDA numerical agreement** | **Not claimed, not tested** | The CUDA backend has never been compiled or run (no CUDA toolkit/GPU in this sandbox) — see `PHASE1_AUDIT.md` §9. |

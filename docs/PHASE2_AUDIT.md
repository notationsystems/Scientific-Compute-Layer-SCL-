# Phase 2 Audit — SCL Conformance to STE's Real Evidence/Canonical-State Substrate

Scope, as narrowed and re-confirmed across four follow-up messages in this
session (the last of which is authoritative — "SCL Scope Boundary"): make
Phase 1's Lennard-Jones computed result conform to STE's **real, existing**
evidence/identity/provenance/derived-state machinery, reusing what exists,
adding only what has a demonstrated consumer, creating **no** parallel
governance system (no local `invariants.yaml`/`evidence_class.yaml`/
`model_binding.yaml`, no doctrine generator, no chemistry ontology, no
CUDA work). Everything below is grounded in reading the real STE checkout
(`/home/user/ste-clean`, branch `claude/deterministic-state-architecture-q9kl5c`)
directly — not inferred from any prompt's vocabulary.

## §1 — Phase 0 reconnaissance: what actually exists

Verified by direct inspection (two parallel Explore agents plus direct
reads of `materials/model_state.py` and `experiment/step.py`), not
inferred:

| Referenced artifact | Actual state |
|---|---|
| `architecture/invariants.yaml`, `evidence_class.yaml`, `model_binding.yaml`, `functions.yaml` | **Do not exist.** No `architecture/` directory anywhere in the repo, on any branch checked (default branch and the active `claude/deterministic-state-architecture-q9kl5c` branch, fetched fresh). |
| Doctrine generator / CI regeneration gate | **Does not exist.** No `.github/workflows`, no generator script found. |
| Four-way evidence-class enum (measured/asserted/computed/derived) | **Does not exist anywhere** in `evidence/`, `materials/`, or `core/` — grep for the literal strings turns up only prose. |
| Typed quantity (`unit`/`uncertainty`/`uncertainty_kind`) | **Does not exist anywhere.** No `Quantity` class, no `unit` field on any dataclass, in the whole repository. |
| "Defeasible" canonical state | **Not found** as a word or concept anywhere in the repo. |
| A `Warrant` type gating a state transition | **Does not exist.** `campaign/warrant_cache.py`'s `WarrantRecord`/`CachedWarrant` is a zk-proof verification cache, structurally unrelated to `core.canonical` or `materials.decision`. |
| `core.canonical.CanonicalState`/`Version` | **Real and implemented** (`core/canonical/state.py`, `version.py`) — but **completely disconnected**: `materials/`, `experiment/`, `evidence/` never import `core.canonical`, and `core/` never imports them. Confirmed by grep and by the repo's own `README.md:34-45`: "`evidence/` is NOT `core.canonical`". No production code path constructs a `Version` from anything SCL or STE's execution layer produces. |
| The actual "canonical/derived state" STE's execution pipeline writes | `materials.model_state.ModelState`, reached ONLY via `experiment.step.run_experiment_step` → `materials.model_state.update()`. This is the real system the addendum's "Canonical State"/"Derived State" language corresponds to in this codebase — `core.canonical` is not it. |
| `materials.decision.Criterion`/`evaluate_program` (validation) | Real, and confirmed **purely advisory**: `evaluate_program`'s own docstring says "deterministic, side-effect-free, read-only"; it imports neither `evidence.pool` nor `materials.model_state`; there is no code path where its verdict gates admission or a state transition. |
| `EvidencePool` persistence / legacy records | **No persistent store exists at all.** `EvidencePool` is constructed fresh, in-memory, per test/session in every code path found. There is nothing to migrate — greenfield by construction, not by policy choice. |
| STE's own "backend already fully generic" claim | Confirmed for the identity layer (`execution.commitments`) and the dispatch seam (`SpecificationDispatcher.runner`) — both already worked, unmodified, for SCL in Phase 1. NOT confirmed for `materials.analysis`'s comparison-context mechanism — see §4's CONFLICT_REQUIRES_INVESTIGATION finding, discovered by actually running the pipeline. |

## §2 — SCL → Evidence → Validation → Derived-State mapping

Using the exact classification vocabulary specified: `REUSE` / `INTEGRATE`
/ `WAIT_FOR_UPSTREAM` / `NOT_REQUIRED` / `CONFLICT_REQUIRES_INVESTIGATION`.

| Boundary | Existing representation | Classification | What SCL actually does |
|---|---|---|---|
| Execution record | `execution.engine.ExecutionResult` / `execution.specification.ExecutionSpecification` | **REUSE** | Phase 1's `scl.ste_adapter.run_scl_specification` already returns STE's real type; zero Phase 2 changes needed here. |
| Raw bookkeeping (Record) | `evidence.types.Record`, `evidence.admission.admit_record` | **REUSE** | `SpecificationDispatcher.dispatch()` (unmodified STE code) already builds this from SCL's `DispatchedMeasurement.record_locator`/`record_raw_content`; SCL supplies correctly-shaped strings, nothing more. |
| Semantic evidence (Observation) | `evidence.types.Observation`, `materials.results.make_experimental_result`/`admit_experimental_result` | **INTEGRATE** | `Observation.content` is already an open `Mapping[str, object]` — no STE schema change needed. SCL's `interpret_lj_result()` now fills it with `property`, `value`, `evidence_class`, `quantities`, `method_block`, `parameters` (see §3). |
| First-class, schema-level `evidence_class` field + admission-time enforcement | none | **WAIT_FOR_UPSTREAM** | Genuinely missing at the STE level; adding a real field/enum to `evidence.types.Observation` and validating it in `evidence.admission.admit_observation` is an STE-repo change, outside SCL's remit and this session's push scope. SCL emits the classification as a `content` key today (see §2's INTEGRATE row above), ready to move into a first-class field if/when STE adds one. |
| Evidence-class immutability (`class_assigned_at_ingest`) | `EvidencePool`'s append-only, content-addressed design (no update/delete method anywhere; ids are `content_hash`) | **REUSE** for the structural guarantee | Real, already-enforced, and now tested for SCL's content specifically (`test_content_addressing_makes_the_pool_append_only_not_mutable`): "reclassifying" produces a new id, never mutates the original. A first-class schema-level enforcement remains WAIT_FOR_UPSTREAM (see row above). |
| `proposals_are_not_evidence` | n/a | **NOT_REQUIRED** | SCL has no optimizer/proposal code path. It performs one direct, deterministic computation; there is nothing in SCL that could ever generate a "proposal" for this invariant to guard against. |
| Numerical/method validation (Criterion, closed-form/finite-difference/Newton's-3rd-law checks) | `materials.decision.Criterion`/`evaluate_program` (advisory, over admitted results); Phase 1's own kernel-correctness tests (`tests/test_numerical_validation.py`) | **REUSE** | Two different, non-conflicting concerns, both already covered: SCL's tests prove the KERNEL is right (software-correctness, build-time); `materials.decision` is where STE would judge whether a PARTICULAR result satisfies a campaign's target (science-policy, run-time) — unmodified, already wired identically for every backend. |
| Derived state (`ModelState`) | `materials.model_state.ModelState`/`update()` | **INTEGRATE** | `update()` hard-requires `observation.content["value"]` to be numeric (confirmed by reading it directly) and `materials.results.make_experimental_result` hard-requires `content["property"] == entry.property` (confirmed the SAME way — by the real ValueError firing before the fix). Both are now supplied. Zero other STE code changed; the full real `run_experiment_step` path now completes end to end with SCL as the backend (`test_full_loop_admits_and_transitions_state`). |
| `core.canonical.CanonicalState`/`Version` | real but disconnected from the rest of the app (§1) | **NOT_REQUIRED** | Wiring SCL into a subsystem nothing else in STE is wired into would be inventing an integration STE itself hasn't built — clearly out of scope. |
| Typed quantity / unit / uncertainty | none (§1) | **INTEGRATE via new SCL-local shape**, not a competing system | `python/scl/quantity.py`'s `Quantity` — a plain dict-producing dataclass used ONLY to shape values placed inside STE's already-open `content` mapping. Adds no STE field, no STE type, no STE dependency. |
| Domain method block | none (§1) | **INTEGRATE via new SCL-local shape**, not a competing system | `python/scl/method_block.py`'s `LJMethodBlock` — same posture as `Quantity`: a shape for `content`, not a new schema. Every field is either a real applicable value or an explicit `{"applicable": False, "reason": ...}` marker (no MD trajectory fields fabricated for a static evaluation — see §3). |
| Cross-observation comparison/disagreement analysis | `materials.analysis._comparison_context`/`_group_by_comparison_context` | **CONFLICT_REQUIRES_INVESTIGATION** | Demonstrated, not guessed: this mechanism treats every `content` key except `property`/`value` as grouping "context" and requires that whole sub-mapping to be hashable (`tuple(sorted(context.items()))` used as a dict key). SCL's `quantities`/`method_block`/`parameters` are nested dicts — unhashable — so `materials.program.analyze_program` raises `TypeError: unhashable type: 'dict'` the moment it processes an SCL-produced Observation. Captured directly as a test (`test_rich_content_breaks_materials_analysis_comparison_grouping`), not silently avoided. **Not fixed here**: fixing it means changing STE's `materials/analysis.py` (outside SCL's remit) or flattening SCL's content (undoing Tasks 5/6's typed-quantity/method-block requirement) — both are STE-side or product decisions, not something SCL should decide unilaterally. Recorded as a genuine dependency. |
| Chemistry/substance identity | `evidence.types.Referent`, `materials.candidates.ActionCandidate.formulation` | **NOT_REQUIRED** | SCL's `content`/`interpret_lj_result` never reads or constructs a `Referent`/formulation identity of any kind — that's supplied entirely by the CALLER via `ActionCandidate.formulation`, unchanged from Phase 1. SCL structurally cannot corrupt substance identity because it never touches it. |
| Generator/CI/doctrine budget | none (§1) | **NOT_REQUIRED for SCL** | Per this session's own explicit scope boundary; out of SCL's remit entirely. |

## §3 — The computed-evidence contract, concretely

`python/scl/ste_adapter.py::interpret_lj_result(candidate, result)` now
returns:

```python
{
    "property": candidate.property,          # required by make_experimental_result
    "value": total_energy,                    # required by ModelState.update()
    "evidence_class": "computed",             # SCL's own declared classification
    "quantities": {
        "total_energy": {"value": ..., "unit": "epsilon", "uncertainty": None, "uncertainty_kind": "absent"},
        "forces": [{"fx": {...}, "fy": {...}, "fz": {...}}, ...],
    },
    "method_block": {
        "potential": {"applicable": True, "value": "lennard_jones_12_6"},
        "potential_version": {"applicable": True, "value": "<scl_cli --version output>"},
        "cutoff": {"applicable": True, "value": {"value": ..., "unit": "sigma"}},
        "boundary_conditions": {"applicable": True, "value": "open"},
        "numerical_precision": {"applicable": True, "value": "float64"},
        "system_definition": {"applicable": True, "value": {"n_particles": ...}},
        "backend": {"applicable": True, "value": "cpu"},
        "integration_configuration": {"applicable": False, "reason": "single-point evaluation: no integrator exists in this computation"},
        "initialization": {"applicable": False, "reason": "..."},
        "temperature": {"applicable": False, "reason": "..."},
        "timestep": {"applicable": False, "reason": "..."},
        "equilibration": {"applicable": False, "reason": "..."},
        "sampling_time": {"applicable": False, "reason": "..."},
        "thermostat": {"applicable": False, "reason": "..."},
        "barostat": {"applicable": False, "reason": "..."},
        "convergence_criteria": {"applicable": False, "reason": "..."},
    },
    "parameters": {"epsilon": ..., "sigma": ...},
}
```

**Explicitly NOT here** (the Phase 112b firewall, still held, now tested
against the richer shape by `test_interpret_result_is_the_evidence_firewall`
and `test_full_loop_admits_and_transitions_state`): `specification_identity`,
`program_identity`, `computation_identity`, `engine_occurrence` — those
ride only in `DispatchedMeasurement.record_raw_content`, admitted as a
`Record`, never in `Observation.content`.

**"computation occurred" ≠ "scientific claim validated" ≠ "canonical
assertion"**, held structurally, not just by convention:
`evidence_class: "computed"` is a value inside `content` (evidence-level);
`materials.decision.evaluate_program`'s PASS/FAIL verdict is a separate,
advisory, later computation over admitted evidence (validation-level);
`ModelState.update()` producing a new state is a third, distinct act
(derived-state-level) that only follows a successful admission — none of
the three implies or performs either of the others, and no SCL code calls
`update()` or any admission function itself (the caller — `experiment.step
.run_experiment_step`, unmodified STE code — always does).

## §4 — Method-block completeness (Task 5)

Every field genuinely applicable to a static pairwise LJ evaluation is
populated with a real value (potential, version, cutoff, boundary
conditions, precision, system size, backend). Every field that belongs
to an MD *trajectory* (ensemble, timestep, thermostat, barostat,
equilibration, sampling_time, integration_configuration, initialization,
convergence_criteria) is **explicitly marked `applicable: False` with a
stated reason** — none fabricated, none silently omitted. This is a
**Qualified** interpretation of the addendum's MD method block (a
narrower, domain-honest reading for a single-point evaluation), not a
bend of anything in `core/`.

## §5 — Quantity/uncertainty (Task 6)

`uncertainty_kind="absent"` on every quantity SCL emits is a **true
statement**: `native/src/lj_pairwise.cpp` is deterministic double-precision
arithmetic with no stochastic sampling and no implemented error
propagation — there is genuinely no uncertainty estimate to report, which
is different from uncertainty having been lost in transit. `Quantity`
(`python/scl/quantity.py`) rejects, at construction time, both a
fabricated uncertainty value under `uncertainty_kind="absent"` and a
missing uncertainty value under any other kind — tested directly
(`test_quantity_rejects_fabricated_and_contradictory_uncertainty`).

## §6 — Execution identity vs. evidence identity (Task 4/K/L)

Unchanged from Phase 1, now additionally tested explicitly
(`test_execution_identity_and_evidence_identity_are_distinct_namespaces`):
`execution.commitments`' raw-bytes SHA-256 scheme (program/input/output/
computation identities) and `evidence.identity.content_hash`'s
canonical-JSON scheme never collide and are never treated as
interchangeable — `result.computation_identity`, `result.output_identity`,
and `result.specification_identity` are each proven distinct from
`content_hash(content)` in the same test. SCL's own separate request-level
identity space (`python/scl/identity.py`, `scl.request.*` tags, Phase 1)
remains a THIRD, non-overlapping space, used only for SCL-internal
tracking — never asserted into STE's evidence or execution identity
spaces.

## §7 — Write-path protection (Task 8/N/O/P)

- `test_raw_scl_result_cannot_be_admitted_directly` — `EvidencePool.put_observation` raises `AttributeError` on a raw `ExecutionResult`; there is no code path by which SCL's own types satisfy `Observation`'s shape.
- `test_model_state_update_rejects_a_mismatched_candidate_and_result` — `ModelState.update()`'s own assertion fires on a genuinely different candidate.
- `test_a_halting_scl_execution_admits_nothing` — a halted SCL computation (coincident particles) leaves `pool.fingerprint()` byte-identical; `run_experiment_step` raises before any admission call.
- `test_content_addressing_makes_the_pool_append_only_not_mutable` — content-addressing is the concrete mechanism behind `class_assigned_at_ingest`-style immutability in this codebase today (§2's REUSE row): the original Observation is provably untouched after an attempted "reclassification".
- **Quarantine / rejection policy**: no such mechanism exists anywhere in STE (confirmed by inspection; not found in `evidence/`, `materials/`, or `experiment/`). `run_experiment_step` either succeeds (admits) or raises (admits nothing) — there is no third, quarantined outcome. This is recorded honestly as **absent from the upstream substrate**, not fabricated by SCL. SCL's own failure vocabulary (validation/backend-unavailable/computation/protocol/timeout faults, Phase 1) remains the actual mechanism by which an SCL-side failure is distinguished before it would ever reach STE's admission boundary.

## §8 — Implemented (files changed)

- `python/scl/quantity.py` — new. `Quantity`, `absent_uncertainty()`.
- `python/scl/method_block.py` — new. `LJMethodBlock`, `lj_method_block_for()`.
- `python/scl/client.py` — added `decode_lj_configuration()` (inverse of the existing encoder; needed to recover epsilon/sigma/cutoff from a real `ExecutionSpecification.configuration` for the method block).
- `python/scl/ste_adapter.py` — `interpret_lj_result()` signature changed from `(result)` to `(candidate, result)` (matching `SpecificationDispatcher`'s real `interpret` signature, and because `candidate.property` is now genuinely required, not just accepted); content shape extended per §3; added `_split_descriptor_full()` (recovers the kernel version line, not just backend, from a real `program` descriptor) and `LJ_EVIDENCE_CLASS`.
- `tests/test_ste_integration.py` — two Phase 1 tests updated to call `interpret_lj_result(candidate, result)` and assert the new content keys (their underlying invariant — the Phase 112b firewall, dispatcher substitutability — is unchanged; only the concrete key list they check changed, because that key list is exactly what Phase 2 intentionally enriched).
- `tests/test_phase2_evidence_conformance.py` — new, 14 tests (§9).
- `docs/PHASE2_AUDIT.md` — this file.

No native (`native/`) code changed. No STE (`ste-clean`) code changed —
zero write access used or needed there; every STE behavior this phase
depends on was verified by reading and running the real, unmodified
checkout.

## §9 — Verified

- Native: unaffected by this phase (not rebuilt/retested; no native change).
- SCL Python suite: **51/51 pass** (37 Phase 1 + 14 Phase 2), full pytest run, this session.
- STE's own full suite, unmodified: **1914 passed, 0 failed** (109 skipped — STE's own environment-gap skips, e.g. no `gmx` binary), re-run in full after every content-shape change in this phase.
- No lint/type-check/CI tooling was found configured in either repo to run additionally (§1) — none was invented.

## §10 — Preserved

Every Phase 1 invariant (identity/provenance separation, staged fault
codes, bitwise/numerical/semantic reproducibility claims exactly as
scoped, the Phase 112b firewall) remains intact and re-tested against the
richer content shape. Every STE invariant this session could observe
(evidence identity uncontaminated by execution history, the admission
firewall, append-only `EvidencePool`, `materials.decision`'s advisory-only
status) is unchanged and now additionally exercised by SCL-originated data.

## §11 — Extended

The `ExecutionResult` → `DispatchedMeasurement` → `Observation` path now
carries, for SCL results, a declared evidence class, method metadata, and
typed quantities — a strict superset of Phase 1's bare
`{total_energy, forces}` content, achieved with zero new STE types and
zero STE code changes (§2, §8).

## §12 — Integrated

`execution.specification.ExecutionSpecification`, `execution.engine
.ExecutionResult`, `execution.dispatcher.SpecificationDispatcher`,
`evidence.types.{Record,Observation}`, `evidence.admission.{admit_record,
admit_observation via admit_experimental_result}`, `evidence.pool
.EvidencePool`, `materials.results.{make_experimental_result,
admit_experimental_result}`, `materials.model_state.{update, ModelState,
resolve_model_state_key}`, `materials.decision.evaluate_program`,
`experiment.step.run_experiment_step` — all consumed exactly as STE
already defines them, through the real dispatcher seam Phase 1 already
proved substitutable.

## §13 — Generator state

Discovered: no. Canonical sources: none exist. Deterministic: n/a.
Generated doctrine: n/a. Regeneration test: n/a. CI diff gate: n/a.
Doctrine budget: n/a. Vendor-free enforcement: n/a. **Every item above
was checked by inspection this session, not assumed** — see §1's table.
Per this session's explicit scope boundary, none of this was built here;
it remains, if wanted, work for whatever repo/session actually owns
`architecture/` (nothing in this session's access does today).

## §14 — Execution-record retention state

The addendum's "agent execution retention" list (binding identity, vendor/
model snapshot, doctrine hash, effective prompt, ...) describes LLM/agent
executions — not applicable to SCL, which performs deterministic native
computation, not model inference. The analogous native-computation
retention (program/input/output/computation identity, backend, backend
version, exit code, timing) was already fully captured in Phase 1's
`ExecutionResult`/`SCLResult` and is unchanged here.

## §15 — Dependency state (upstream capabilities SCL requires but does not own)

1. A first-class, schema-level `evidence_class` field on `evidence.types
   .Observation` plus admission-time enforcement (§2's WAIT_FOR_UPSTREAM row).
2. `materials.analysis`'s comparison-context mechanism accepting
   non-hashable/nested content values, or a declared way to exclude rich
   sub-structures from grouping (§2/§3's CONFLICT_REQUIRES_INVESTIGATION row)
   — the one concrete blocker found by actually running the real pipeline.
3. Everything in §13 (architecture/*.yaml, doctrine generator, CI gate) —
   not SCL's to build, per this session's explicit scope boundary.

## §16 — Qualified / Bent

**Qualified**: the LJ method block (§4) — a narrower, single-point-
evaluation reading of the addendum's MD method block, core semantics
unchanged (nothing in `core/` was touched, because nothing in `core/`
defines a method block at all — see §1).

**Bent: none.** No core invariant was changed; none needed to be. The one
real conflict found (§2/§3, `materials.analysis`) is a dependency on
unmodified STE code, not a change to it.

## §17 — Migration state

No legacy `EvidencePool` records exist anywhere in the repository (§1) —
nothing to classify, nothing to quarantine. `unclassified` migration
semantics were therefore not implemented; there is nothing for them to do
yet.

## §18 — Identity decisions

None. SCL makes zero substance/chemistry identity decisions of any kind
(§2's NOT_REQUIRED row) — `formulation`/substance identity is supplied
entirely by the caller via `ActionCandidate.formulation`, exactly as in
Phase 1, unchanged.

## §19 — Unresolved (carried forward explicitly, not silently dropped)

```text
multi_writer.write_conflict        -- unrelated to this phase's changes; not touched
builder_check_lineage              -- unrelated (no enforcement-code authorship question exists in SCL)
attested_snapshot_identity         -- unrelated (SCL is native compute, not a hosted model binding)
capabilities 5-9                   -- unrelated (no acceptance-criteria framework exists to extend)
materials.analysis rich-content incompatibility (§2/§3, new this phase) -- a real, demonstrated
    upstream dependency: fixing it means either changing STE's materials/analysis.py (out of SCL's
    remit) or flattening SCL's content (undoing Tasks 5/6). Left exactly as found, captured by a
    passing test that documents the TypeError rather than hiding it.
evidence_class as a first-class STE schema field (§2/§15 item 1) -- SCL emits it inside content
    today; promoting it to a real field/enum on evidence.types.Observation is an STE-repo decision.
```

## §20 — Audit (A–U)

| # | Invariant | Status |
|---|---|---|
| A | Canonical structured architecture remains authoritative | N/A — none exists to be authoritative over this phase (§1); nothing here contradicts a canonical source because there isn't one yet. |
| B | Doctrine remains a generated projection | N/A — no doctrine, no generator, found or built (§13). |
| C | SCL does not become a doctrine source | **Held** — SCL wrote no doctrine of any kind. |
| D | SCL does not duplicate EvidencePool | **Held** — SCL holds no evidence store; every admitted object lives in STE's real, unmodified `EvidencePool`. |
| E | SCL does not duplicate canonical-state IR | **Held** — SCL constructs no `core.canonical.CanonicalState`/`Version` and no `ModelState` (only STE's `update()` does, called by STE's own `run_experiment_step`). |
| F | DAF remains the acquisition boundary | **Held (untouched)** — DAF was not read or referenced this phase. |
| G | Evidence class remains immutable after ingest | **Held**, at the structural level content-addressing already provides (§2, §7); no first-class enforcement exists yet upstream (§15). |
| H | Computed evidence remains distinct from validation and canonical assertion | **Held**, structurally (§3's three-act separation). |
| I | STE remains authoritative for scientific execution/transformation | **Held** — `run_scl_specification` produces STE's own `ExecutionResult` type; every subsequent step (admission, update) is unmodified STE code. |
| J | SCL remains computational capability | **Held** — no new evidence/canonical/validation system was added to SCL itself; only content-shaping helpers that feed STE's existing open fields. |
| K | Execution identity remains distinct from evidence identity | **Held and now directly tested** (§6). |
| L | Operation identity remains distinct from verification identity | **Held** — SCL has no verification/proof concept of its own; unchanged from Phase 1's posture (no zk/warrant system touched). |
| M | Provenance remains authoritative | **Held** — `test_provenance_traces_observation_back_through_record_to_specification` confirms the real chain (Observation → Record → specification/program/input/computation identities) is intact and traceable. |
| N | Derived state cannot directly write canonical state | **Held and now directly tested** (§7) — `ModelState.update()`'s own assertions were exercised, not assumed. |
| O | Rejected records enter quarantine | **N/A — no quarantine mechanism exists upstream** (§7); recorded honestly rather than fabricated. |
| P | No force/bypass path exists | **Held** — none was created; none was found upstream either. |
| Q | Mandatory execution-record retention remains intact | **N/A for SCL** (§14 — that requirement describes LLM/agent execution, not native computation); the analogous native retention (Phase 1) is unchanged. |
| R | Core schemas remain closed | **Held** — nothing in `core/` was touched or referenced by any SCL write path. |
| S | No vendor information is introduced into doctrine | **Held (vacuously)** — no doctrine exists for SCL to introduce anything into. |
| T | Existing Phase 1 SCL invariants remain intact | **Held** — 51/51 SCL tests pass, including every Phase 1 test (two updated in place to match the intentionally-enriched content shape, same invariant, new key list — §8). |
| U | Existing STE tests remain intact | **Held** — 1914/1914 pass, 0 failed, unmodified checkout. |

## §21 — Next executable frontier

The SCL boundary is now conformant for the write path this phase actually
exercised (evidence admission → derived state), with one concrete,
demonstrated, upstream-owned blocker remaining (§15 item 2: rich content
vs. `materials.analysis`'s comparison-context hashability). That blocker
sits outside SCL's own boundary — it does not block SCL from correctly
producing, classing, and getting its results admitted and folded into
derived state, which is what this phase set out to prove.

Given that: **recommend SCL Phase 3 — CPU/CUDA backend equivalence**, per
this session's own explicit sequencing ("once this phase establishes a
clean SCL boundary, CUDA becomes the next concrete computational
frontier"). Concretely, Phase 3 should (a) build and validate the
existing CUDA kernel (`native/backends/cuda/lj_pairwise_cuda.cu`) on real
hardware for the first time, closing the coincident-particle fault-parity
gap already flagged in `docs/PHASE1_AUDIT.md` §9, and (b) add a
tolerance-based CPU-vs-CUDA numerical agreement test — without touching
the evidence/derived-state boundary this phase just closed. Separately,
and outside SCL's own repo: §15's two dependency items (a first-class
`evidence_class` field, and `materials.analysis`'s hashability
requirement) are concrete, well-specified asks for whichever session owns
STE's `evidence/`/`materials/` code — not blocking, but real.

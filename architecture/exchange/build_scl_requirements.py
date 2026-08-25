"""Generates `scl_requirements.yaml` -- the SCL side of the Phase 2
requirement/capability exchange.

MEASURED FACTS AND REQUIREMENTS ONLY. This artifact deliberately contains
no ranking, no score, no recommendation and no workload selection: §2 of
the joint prompt forbids it, and the joint decision record (§3) is the
only thing entitled to select. Every "requires" entry below states what a
workload would need in order to run correctly, not whether it should be
built.

The substrate inventory is traced from executable code, not from
filenames or documentation:
    native/src/{lj_pairwise,fourier,backend,operation_registry,op_*}.cpp
    native/include/scl/*.hpp
    python/scl/*.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from canonical_yaml import canonical_bytes, canonical_sha256  # noqa: E402

# Classification vocabulary is fixed by the prompt:
#   EXISTING | REUSABLE | SMALL EXTENSION | MISSING | OUT OF SCOPE
SUBSTRATE = {
    "complex_arithmetic": {
        "classification": "EXISTING",
        "evidence": "std::complex<double> throughout native/src/fourier.cpp and the cuFFT backend; complex wire encoding in python/scl/fourier.py",
    },
    "decompositions": {
        "classification": "MISSING",
        "evidence": "no QR, Cholesky, LU, SVD or eigendecomposition exists in any traced source file",
    },
    "dynamic_programming": {
        "classification": "MISSING",
        "evidence": "no recurrence table, no argmax/backtracking structure anywhere in native/ or python/scl/",
    },
    "linear_solves": {
        "classification": "MISSING",
        "evidence": "no solver of any kind; the only division in native math is scalar",
    },
    "matrices": {
        "classification": "MISSING",
        "evidence": "no matrix type, no 2-D indexing convention, no row/column-major decision recorded anywhere",
    },
    "matrix_multiplication": {
        "classification": "MISSING",
        "evidence": "grep for matmul/gemm/transpose across native/src, native/include and python/scl returns nothing",
    },
    "numerical_differentiation": {
        "classification": "OUT OF SCOPE",
        "evidence": "a finite-difference check exists only inside tests as a validation device (native/tests/test_lj_pairwise.cpp); it is not a substrate capability and is not callable as an operation",
    },
    "numerical_integration": {
        "classification": "MISSING",
        "evidence": "no quadrature or accumulation-over-time primitive exists",
    },
    "optimization": {
        "classification": "MISSING",
        "evidence": "no objective/step/convergence machinery; scl_cli performs one deterministic evaluation per invocation",
    },
    "reductions": {
        "classification": "EXISTING",
        "evidence": "summation reductions in both operations: pairwise energy accumulation and the DFT inner sum",
    },
    "scalar_arithmetic": {
        "classification": "EXISTING",
        "evidence": "IEEE-754 float64 throughout; -fno-fast-math pinned in native/CMakeLists.txt so reductions are not reassociated",
    },
    "state_transitions": {
        "classification": "MISSING",
        "evidence": "every operation is a pure function of (configuration, input); no operation carries state across invocations and no recursive/streaming execution model exists",
    },
    "tensors": {
        "classification": "OUT OF SCOPE",
        "evidence": "no rank-N structure; deliberately not built -- the prompt forbids a universal tensor abstraction without a consuming workload",
    },
    "transforms": {
        "classification": "EXISTING",
        "evidence": "fourier_transform_1d, forward and inverse, three normalizations, validated against impulse/DC/pure-tone/Parseval/reconstruction",
    },
    "transpose": {
        "classification": "MISSING",
        "evidence": "no matrix type exists to transpose",
    },
    "vectors": {
        "classification": "REUSABLE",
        "evidence": "contiguous float64 sequences are the established wire form (N*24-byte particles, N*16-byte complex samples) with encode/decode helpers, but there is no general vector type with arithmetic",
    },
}

_ORDERED = "required_and_significant"
_UNORDERED = "not_required_rows_exchangeable"


def workload(*, modality, minimum_observation_fields, required_metadata, uncertainty,
             conditions, ordering, structured, observation_requirements, model_parameters,
             computational_parameters, primitives_required, primitives_missing, status,
             notes, blocking_requirements=()):
    """`blocking_requirements` is deliberately STRUCTURAL rather than prose
    in `notes`: a requirement buried in a paragraph arrives in the other
    repository as something to read, whereas a row arrives as something to
    answer. Each entry names a requirement that must be satisfied before
    the workload could be built at all -- a measured consequence of what
    the workload needs, never an argument for building it."""
    return {
        "blocking_requirements": list(blocking_requirements),
        "computational_parameters": computational_parameters,
        "condition_requirements": conditions,
        "minimum_observation_fields": minimum_observation_fields,
        "modality": modality,
        "model_parameters": model_parameters,
        "notes": notes,
        "observation_requirements": observation_requirements,
        "ordering_requirements": ordering,
        "primitives_missing": primitives_missing,
        "primitives_required": primitives_required,
        "required_metadata": required_metadata,
        "status_in_scl": status,
        "structured_data_requirements": structured,
        "uncertainty_requirements": uncertainty,
    }


WORKLOADS = {
    "fourier_transform_1d": workload(
        modality="ordered_1d_sequence",
        minimum_observation_fields=["ordered_values"],
        required_metadata=["units_of_sampled_quantity"],
        uncertainty="none",
        conditions="none_required_by_the_transform",
        ordering=_ORDERED,
        structured="none",
        observation_requirements=[
            "an ordered sequence of scalar values; sample order IS the signal",
            "sample_spacing OR timestamps -- OPTIONAL, required only if a physical frequency axis is wanted; absent means bin-index output",
        ],
        model_parameters=["none -- a transform asserts no model of the system"],
        computational_parameters=["direction", "normalization", "precision", "input_kind", "spectrum_convention"],
        primitives_required=["complex_arithmetic", "reductions", "transforms"],
        primitives_missing=[],
        status="IMPLEMENTED",
        blocking_requirements=[
            {
                "requirement": "ordered_scalar_sequence",
                "owner": "daq",
                "statement": "An ordered sequence of scalar values in which sample ORDER is significant and recoverable. This is the whole DAQ-side requirement of the transform family.",
                "measured_basis": "Built and validated against independent mathematics; the operation runs today over exactly this shape.",
                "consequence_if_unmet": "not applicable -- satisfied",
                "status": "SATISFIED",
            },
            {
                "requirement": "annotating_sample_spacing",
                "owner": "daq",
                "statement": "Sample spacing (or timestamps) is OPTIONAL and, when supplied, must be distinguishable from its own absence rather than defaulted. SCL never assumes dt=1: a result computed without a spacing is bin-indexed and says so.",
                "measured_basis": "Measured property of the built operation: two requests differing only in dt produce byte-identical output, so they share a computation_identity while differing in request_identity. Absence is carried by an explicit has_sample_spacing flag, not by a sentinel value.",
                "consequence_if_unmet": "A supplied-but-wrong or silently-defaulted spacing would place every frequency on a wrong axis while the numbers themselves stayed correct -- an error that is invisible in the output bytes.",
                "status": "SATISFIED",
                "generalizes_to": "This is the ANNOTATING PARAMETER shape, stated as a general identity rule in docs/SCL_CONTRACT.md section 6.1 rather than as a Fourier detail: a field that annotates a result without participating in it enters parameters_identity and request_identity but must never enter output_identity or computation_identity, and must never be silently defaulted. UNITS are the next instance and a sharper one, because a missing axis is visibly absent whereas a wrongly-assumed unit is invisibly present.",
            },
        ],
        notes="Implemented and validated against independent mathematics. input_kind is fixed at complex and spectrum_convention at two_sided by the contract (not backend defaults); making either selectable would require an explicit configuration field participating in parameters_identity.",
    ),
    "convolution_1d": workload(
        modality="ordered_1d_sequence",
        minimum_observation_fields=["ordered_values"],
        required_metadata=["units_of_sampled_quantity"],
        uncertainty="none",
        conditions="none_required",
        ordering=_ORDERED,
        structured="none",
        observation_requirements=["an ordered sequence of scalar values", "a kernel/second sequence, which may be asserted rather than observed"],
        model_parameters=["kernel, when the kernel is a modelling choice rather than a second observation"],
        computational_parameters=["mode_full_same_valid", "boundary_handling", "precision"],
        primitives_required=["reductions", "sliding_window"],
        primitives_missing=["sliding_window"],
        status="NOT_IMPLEMENTED",
        notes="Shares the ordered-1d modality with fourier_transform_1d, so its DAQ requirement is already satisfied wherever Fourier's is. Direct and FFT-based algorithms could share one mathematical operation identity.",
    ),
    "least_squares": workload(
        modality="multivariate_observation_table",
        minimum_observation_fields=["response_values", "predictor_values", "sample_identity", "variable_identity"],
        required_metadata=["units_per_variable", "missing_value_semantics"],
        uncertainty="scalar_per_observation_only_if_weighted",
        conditions="conditions_that_distinguish_samples_must_be_recoverable_as_predictors_or_strata",
        ordering=_UNORDERED,
        structured="aligned_rows -- response and predictors must be joinable per sample, which requires a stable sample identity",
        observation_requirements=[
            "a response value per sample",
            "one or more predictor values per sample",
            "sample identity sufficient to align response with predictors",
            "variable identity sufficient to know which column is which",
            "explicit missing-value semantics; silently dropped rows change the fit",
        ],
        model_parameters=["the choice of design matrix / basis functions is a modelling assertion, not an observation"],
        computational_parameters=["solver_qr_cholesky_svd_or_normal_equations", "weighting", "precision"],
        primitives_required=["matrices", "matrix_multiplication", "transpose", "decompositions", "linear_solves"],
        primitives_missing=["matrices", "matrix_multiplication", "transpose", "decompositions", "linear_solves"],
        status="NOT_IMPLEMENTED",
        blocking_requirements=[
            {
                "requirement": "stable_sample_and_variable_identity",
                "owner": "daq",
                "statement": "Each observation must carry a sample identity sufficient to align a response with its predictors, and a variable identity sufficient to know which column is which. Row position is NOT an acceptable identity here, because ordering is explicitly not required by this modality.",
                "measured_basis": "SCL's implemented operations take a flat byte payload with position-implied meaning, which is adequate for an ordered sequence and inadequate for a joinable table. No aligned-table input shape exists on either side.",
                "consequence_if_unmet": "Silently misaligned columns produce a well-formed fit of the wrong model; nothing in the numbers reveals it.",
                "status": "UNSATISFIED",
            },
            {
                "requirement": "explicit_missing_value_semantics",
                "owner": "daq",
                "statement": "Missing values must be explicitly represented and their semantics stated, not encoded as a sentinel number or elided by dropping rows.",
                "measured_basis": "No missing-value representation exists in the SCL input contract; a float payload has no way to say absent, which is the same absence problem the annotating-parameter rule solves one layer up.",
                "consequence_if_unmet": "Silently dropped rows change the fit and change it invisibly: the residuals of a fit over a quietly smaller sample look entirely healthy.",
                "status": "UNSATISFIED",
            },
            {
                "requirement": "linear_algebra_primitive_family",
                "owner": "scl",
                "statement": "Matrices, matrix multiplication, transpose, decompositions and linear solves must exist in SCL. None of them do.",
                "measured_basis": "substrate_inventory classifies all five as MISSING with traced evidence.",
                "consequence_if_unmet": "The workload cannot be built at all. This is SCL-owned work, recorded here so DAQ can see which gaps are not theirs to close.",
                "status": "UNSATISFIED",
            },
        ],
        notes="Requires the entire missing linear-algebra family. Ordering is NOT required, which is the sharpest modality contrast with the transform family. Its blocking_requirements split across BOTH repositories, which is why they name an owner.",
    ),
    "pca": workload(
        modality="multivariate_observation_table",
        minimum_observation_fields=["values_per_variable", "sample_identity", "variable_identity"],
        required_metadata=["units_per_variable", "missing_value_semantics"],
        uncertainty="none",
        conditions="none_required",
        ordering=_UNORDERED,
        structured="aligned_rows across variables",
        observation_requirements=[
            "a value per variable per sample",
            "sample and variable identity",
            "commensurable units, or an explicit scaling decision, since PCA on mixed units is scale-dependent",
        ],
        model_parameters=["centering and scaling choices are modelling assertions"],
        computational_parameters=["n_components", "centering", "scaling", "decomposition_method_svd_or_eigen", "precision"],
        primitives_required=["matrices", "matrix_multiplication", "transpose", "decompositions"],
        primitives_missing=["matrices", "matrix_multiplication", "transpose", "decompositions"],
        status="NOT_IMPLEMENTED",
        blocking_requirements=[
            {
                "requirement": "stable_sample_and_variable_identity",
                "owner": "daq",
                "statement": "Identical to the least_squares requirement of the same name, and satisfied by the same DAQ-side capability -- both are the multivariate_observation_table modality.",
                "measured_basis": "Same measured gap: no aligned-table input shape exists on either side.",
                "consequence_if_unmet": "Components computed over misaligned columns are well-formed and meaningless.",
                "status": "UNSATISFIED",
            },
            {
                "requirement": "commensurable_units_or_explicit_scaling",
                "owner": "daq",
                "statement": "Units per variable must be recorded, because PCA over mixed units is scale-dependent: the result changes if a variable is expressed in metres rather than millimetres. Either the units are commensurable or the scaling choice is asserted explicitly as a model parameter.",
                "measured_basis": "SCL carries no unit channel at all; a float64 payload is dimensionless as far as the substrate is concerned.",
                "consequence_if_unmet": "The principal components silently reflect the arbitrary choice of measurement units rather than the structure of the data.",
                "status": "UNSATISFIED",
                "relates_to": "This is the UNITS instance of the annotating-parameter shape recorded under fourier_transform_1d: a unit annotates a quantity without participating in its arithmetic, and an absent unit must not resolve to an assumed one.",
            },
            {
                "requirement": "decomposition_primitive",
                "owner": "scl",
                "statement": "Matrices, matrix multiplication, transpose and decompositions must exist in SCL.",
                "measured_basis": "substrate_inventory classifies all four as MISSING. This is a strict subset of the least_squares primitive set -- pca requires no linear solve.",
                "consequence_if_unmet": "The workload cannot be built at all.",
                "status": "UNSATISFIED",
            },
        ],
        notes="Strictly downstream of the decomposition primitive least_squares would establish. Its DAQ-side requirements are a superset of least_squares': the same table identity, plus a unit requirement least_squares does not have.",
    ),
    "kalman_filter_linear": workload(
        modality="ordered_multivariate_time_series",
        minimum_observation_fields=["ordered_measurement_vectors", "timestamps_or_uniform_interval"],
        required_metadata=["units_per_measurement_component"],
        uncertainty="structured -- a measurement covariance R, not a scalar, whenever R is measurement-derived",
        conditions="conditions affecting measurement noise must be recoverable if R varies with them",
        ordering=_ORDERED,
        structured="covariance matrices; a scalar uncertainty per observation is INSUFFICIENT unless the measurement is genuinely 1-D and uncorrelated",
        observation_requirements=[
            "an ordered stream of measurement vectors y_1..y_T",
            "timestamps or a stated uniform interval; the transition depends on elapsed time",
            "measurement uncertainty sufficient to construct or interpret R, IF R is to be measurement-derived rather than asserted",
            "stream identity, so a recursive trajectory can name the stream it consumed",
        ],
        model_parameters=[
            "Q -- process noise. ASSERTED BY THE MODELLER, NEVER SUPPLIED BY DAQ AND NEVER INFERRED FROM MEASUREMENTS. Q silently determines how much the filter trusts its own model versus the data, so deriving it from the data would make the estimate's confidence self-justifying.",
            "F -- state transition, asserted",
            "H -- observation model, asserted",
            "x0 -- initial state, asserted",
            "P0 -- initial covariance, asserted",
        ],
        computational_parameters=["precision", "gain_solve_method_cholesky_or_ldlt", "covariance_update_form_standard_or_joseph"],
        primitives_required=["matrices", "matrix_multiplication", "transpose", "linear_solves", "covariance_propagation", "psd_handling"],
        primitives_missing=["matrices", "matrix_multiplication", "transpose", "linear_solves", "covariance_propagation", "psd_handling"],
        status="NOT_IMPLEMENTED",
        blocking_requirements=[
            {
                "requirement": "structured_measurement_uncertainty",
                "owner": "daq",
                "statement": "DAQ must be able to express a measurement covariance R, not only a scalar uncertainty per observation. A scalar is sufficient ONLY when the measurement is genuinely 1-D and uncorrelated; for a measurement vector it discards the off-diagonal terms that determine how the filter weights components against each other.",
                "measured_basis": "SCL's implemented operations carry no uncertainty channel at all, and the evidence layer's uncertainty vocabulary is scalar-or-absent; neither side currently has a representation for a covariance.",
                "consequence_if_unmet": "R would have to be ASSERTED by the modeller rather than measurement-derived, which moves the estimate's confidence from measured to asserted without that being visible in the result.",
                "status": "UNSATISFIED",
            },
            {
                "requirement": "recursive_generation_depth",
                "owner": "daq",
                "statement": "generation_depth_bounded must acquire an operative semantic domain before a recursive operation can be admitted. It is currently declared with status vacuously_enforced and zero references; a Kalman trajectory is the first thing that would make it bite, so the rule must exist BEFORE the first recursive result, not be retrofitted around one.",
                "measured_basis": "architecture/invariants.yaml declares the invariant; its own recorded evidence is that no generative path exists to bound. Every SCL operation today is a pure function of (configuration, input) with no state carried between invocations, so SCL cannot exercise it either.",
                "consequence_if_unmet": "A recursive estimate would carry an unbounded, unrecorded provenance chain: step T's identity depends on step T-1's without any depth being tracked, so nothing would detect a trajectory that had drifted arbitrarily far from measured input.",
                "proposed_rule": "depth=0 when initialization_provenance=measured AND every input stream is class=measured; depth=prior_depth+1 when initialization_provenance=computed(prior_id). GUARD: if the measurement stream is ITSELF computed, depth inherits from the STREAM rather than from the initialization -- which closes composition, not merely recursion. Offered as a measured requirement for DAQ to accept, amend or reject; SCL does not own this invariant.",
                "status": "UNSATISFIED",
            },
        ],
        notes="The only candidate that is RECURSIVE, and therefore the only one that would make generation_depth_bounded operative rather than vacuous. Its two blocking_requirements are independent: structured R is a DATA-SHAPE gap and recursive depth is an INVARIANT gap, and satisfying either leaves the other untouched. See recursive_computation_analysis below.",
    ),
    "pid_controller": workload(
        modality="ordered_1d_sequence_with_setpoint",
        minimum_observation_fields=["ordered_measured_values", "timestamps_or_uniform_interval"],
        required_metadata=["units_of_measured_variable"],
        uncertainty="none",
        conditions="none_required_for_the_computation",
        ordering=_ORDERED,
        structured="none",
        observation_requirements=[
            "an ordered stream of measured process values",
            "timestamps or a stated interval -- MANDATORY, not optional: the integral and derivative terms are both defined in terms of dt, so an absent or wrong dt silently rescales two of the three gains",
        ],
        model_parameters=["Kp, Ki, Kd and the setpoint are asserted control choices, never observations"],
        computational_parameters=["anti_windup_strategy", "derivative_filtering", "precision"],
        primitives_required=["reductions", "numerical_integration", "numerical_differentiation", "state_transitions"],
        primitives_missing=["numerical_integration", "numerical_differentiation", "state_transitions"],
        status="NOT_IMPLEMENTED",
        notes="PURE COMPUTATION ONLY. Physical actuation is explicitly out of scope: connecting a controller output to equipment is physical intervention on the system under study, and no actuation-authority boundary exists in this architecture. Carried forward as unresolved.",
    ),
    "viterbi": workload(
        modality="ordered_discrete_symbol_sequence",
        minimum_observation_fields=["ordered_discrete_observations"],
        required_metadata=["observation_alphabet_identity"],
        uncertainty="none",
        conditions="none_required",
        ordering=_ORDERED,
        structured="none",
        observation_requirements=[
            "an ordered sequence of discrete symbols",
            "a stable mapping from observed symbol to alphabet index",
        ],
        model_parameters=["transition matrix A, emission matrix B and initial distribution pi are all asserted model choices, never observations"],
        computational_parameters=["log_space_or_linear", "tie_breaking_rule", "precision"],
        primitives_required=["dynamic_programming", "reductions", "argmax", "backtracking"],
        primitives_missing=["dynamic_programming", "argmax", "backtracking"],
        status="NOT_IMPLEMENTED",
        notes="Its primitives are shared with no other candidate here, so it would establish a third computational family rather than extend either existing one.",
    ),
}


DOCUMENT = {
    "artifact": "scl_requirements",
    "canonicalization": {
        "anchors_aliases": "forbidden",
        "encoding": "UTF-8, LF line endings, single trailing newline",
        "floats": "shortest round-trip repr; exponent form only when |x| < 1e-4 or |x| >= 1e16, and then always with a decimal point in the mantissa and an explicit exponent sign so the value round-trips as a float under YAML 1.1 as well as 1.2",
        "hash": "sha256 over the serialized bytes",
        "implementation": "architecture/exchange/canonical_yaml.py",
        "keys": "sorted lexicographically at every level",
        "reference_format": "sha256:<hex>",
        "serialization": "YAML 1.2, block style only; {} and [] are the one documented exception, since empty collections have no block form",
        "shared_fixture": "architecture/exchange/canonicalization_fixture.yaml",
        "strings": "double-quoted only where plain style would be unsafe or ambiguous",
    },
    "contains_workload_selection": False,
    "cuda_libraries_present": {
        "cublas": "present -- /usr/lib/x86_64-linux-gnu/libcublas.so",
        "cufft": "present and already linked by the fourier_transform_1d CUDA backend",
        "cusolver": "present -- /usr/lib/x86_64-linux-gnu/libcusolver.so",
        "note": "presence is a BUILD fact only. No CUDA device is visible in this environment (cudaGetDeviceCount()==0, nvmlInit_v2()==NVML_ERROR_DRIVER_NOT_LOADED), so no GPU numerical, determinism, performance or crossover claim is made for any backend.",
    },
    "cuda_state": {
        "compiled": True,
        "gpu_executed": False,
        "linked": True,
        "runtime_loaded": True,
    },
    "core_vocabulary_candidates": {
        "absent_is_not_zero": {
            "observation": "The same distinction has now been drawn independently at three layers: uncertainty_kind=absent in the evidence vocabulary; an operation that emits no n_particles metric rather than n_particles=0 (operation contract clause 7); and an absent sample spacing that is not dt=1 (identity contract section 6.1 clause 4). None of the three was derived from the others.",
            "reading": "Three independent arrivals at one distinction is mild evidence that ABSENT belongs in the shared core vocabulary as a first-class value rather than being re-specified per layer. Recorded as an observation for the joint decision to weigh; SCL does not own the core vocabulary and is not asserting the change.",
            "counter_consideration": "Three instances is not many, and each layer's absence means something slightly different (unmeasured, not-applicable, not-supplied). A premature unification could collapse distinctions that are worth keeping apart.",
            "status": "CANDIDATE_ONLY_NOT_PROPOSED",
        },
    },
    "daq_execution_record_finding": {
        "finding": "The joint prompt's section 16 specifies an ExecutionRecord carrying computation-shaped fields (backend, backend_version, hardware, seed, verification_status, computation_identity). DAQ's architecture/execution_record.yaml already defines a record of the same name with scope=daf_acquisition_only and owner=daf, keyed on operation_id=H(plan_id, source_id, parameters, mode), whose fields are acquisition-shaped (adapter_id, adapter_version, artifact_ids, version_ids, admission_failure_count, outcome, input/output_fingerprint, parent_execution_id).",
        "measured_basis": "Read directly from DAQ's committed architecture/execution_record.yaml. None of section 16's computation fields appear in it; the two field sets are disjoint apart from the record's name.",
        "why_it_matters": "The joint decision record is bound to these artifacts by hash. If the two repositories write it against two different contracts that share a name, the divergence is invisible in the hashes and surfaces at wiring time instead.",
        "daq_s_own_stated_position": "DAQ's file states its integration_dependency as: canonical for DAF acquisition only; a broader unified substrate that later grows its own execution-record contract should absorb this one, not sit beside it.",
        "consequence": "Section 16 as written would place a second ExecutionRecord BESIDE the existing one, which is the arrangement DAQ's own file rules out. Absorption and coexistence are different designs and the choice between them belongs to the joint decision, not to either repository alone.",
        "scl_position": "SCL raises this and does not resolve it. SCL has no ExecutionRecord of its own and is not proposing one; adding a second identity or record system is exactly what the architecture forbids.",
        "status": "RAISED_UNRESOLVED",
    },
    "extends": "core@1.0.0",
    "generated_by": "architecture/exchange/build_scl_requirements.py",
    "identity_model": {
        "annotating_vs_participating_parameters": "Every configuration field is either PARTICIPATING (its value is read by the mathematics, so changing it changes the output bytes) or ANNOTATING (never read by the mathematics; it exists so a consumer can interpret the numbers). BOTH enter parameters_identity and request_identity. ONLY participating fields may enter output_identity and computation_identity. Therefore equal computation_identity with differing request_identity is the SIGNATURE of an annotating field, not a defect -- a cache keyed on computation_identity may reuse the output bytes and must NOT reuse their interpretation. An annotating field is never silently defaulted: absent and present-with-the-conventional-value stay distinguishable. Stated as a general rule in docs/SCL_CONTRACT.md section 6.1 and enforced across the whole operation registry by tests/test_annotating_parameters.py. Measured on dt; units are the next instance.",
        "inputs_are_not_parameters": "observations and other bulk inputs are hashed into input_identity, never into parameters_identity; hashing a measurement stream into parameter identity would mint a new operation identity per timestep and make the model useless for recursive workloads",
        "input_identity": "H(input_payload)",
        "operation_identity": "H(operation || backend)",
        "parameters_identity": "H(configuration_bytes)",
        "request_identity": "H(operation || backend || parameters || input)",
        "status": "EXISTING and already general -- carried both implemented operations with no per-operation identity mechanism",
    },
    "recursive_computation_analysis": {
        "generation_depth_bounded_status_in_daq": "declared in architecture/invariants.yaml with status vacuously_enforced; zero Python references; its recorded evidence is that no generative path exists to bound",
        "scl_has_no_recursive_operation_today": "every SCL operation is a pure function of (configuration, input) and carries no state between invocations, so SCL cannot currently exercise the invariant either",
        "consequence": "kalman_filter_linear is the only candidate that would convert generation_depth_bounded from vacuous to operative. Neither branch of the prompt's IF-implemented/IF-not conditional applies cleanly: the invariant is declared but vacuous, a third state. Whether supplying it with a semantic domain for the first time constitutes a bend under bend_protocol is a DAQ-side judgement and is deliberately NOT decided here.",
        "depth_rule_if_adopted": "depth=0 when initialization_provenance=measured and every input stream is class=measured; depth=prior_depth+1 when initialization_provenance=computed(prior_id); GUARD: if the measurement stream is itself computed, depth inherits from the STREAM rather than the initialization, which closes composition and not merely recursion",
    },
    "substrate_inventory": SUBSTRATE,
    "unresolved_edges": {
        "comparability_is_weaker_than_identity": {
            "edge": "The annotating-parameter rule makes equal computation_identity compatible with different interpretations. Two Fourier results over the same samples with different dt are comparable as BIN-INDEXED VECTORS and are NOT comparable as SPECTRA, because bin k denotes a different physical frequency in each.",
            "measured_basis": "Directly measured: the two results are byte-identical and share a computation_identity, while their frequency axes differ by the ratio of their sample spacings.",
            "why_it_is_not_solved_here": "SCL does not own a comparison layer and there is no comparison layer yet in which to place the check. Solving it inside SCL would mean inventing a second identity notion for comparability, which is precisely the parallel-architecture failure the design forbids.",
            "rule_needed_when_a_comparison_layer_exists": "results are spectrally comparable only when their ANNOTATING fields agree, not merely when their computation_identity agree",
            "generalizes": "Any consumer that treats identity collision as request equivalence is wrong at exactly this point, for any annotating field -- units included.",
            "status": "RECORDED_UNRESOLVED",
        },
        "physical_actuation_boundary": {
            "edge": "pid_controller is recorded as PURE COMPUTATION ONLY. Connecting a controller output to equipment is physical intervention on the system under study, and no actuation-authority boundary exists anywhere in this architecture.",
            "why_it_is_not_solved_here": "An actuation boundary is a safety and authority question, not a numerical one, and neither repository currently has a place to state it.",
            "status": "RECORDED_UNRESOLVED",
        },
    },
    "validation_requirements": {
        "cross_backend": "tolerance-based only, never bitwise, and only after real GPU execution; different summation order legitimately changes the last bits",
        "determinism": "same binary, same machine, same backend is bitwise reproducible and is asserted for both implemented operations",
        "principle": "independent mathematical properties are preferred over an independently coded replica; a reference written by the same author from the same reading of a spec reproduces the misreading and then agrees with itself",
        "worked_example_fourier": "impulse -> flat spectrum; DC -> single bin; pure tone -> predicted bin; real cosine -> conjugate pair; Parseval; unitary normalization; inverse reconstruction",
        "worked_example_kalman": "innovation mean, covariance and whiteness with sample count and tolerance STATED BEFORE running, plus analytic scalar cases with known steady-state gain; comparing an estimate to simulated ground truth conflates filter, model and simulation correctness",
        "worked_example_least_squares": "known beta from constructed y=X*beta; residual orthogonality X^T r ~ 0; rank-deficient cases; deliberately ill-conditioned systems such as a Hilbert matrix",
    },
    "workloads": WORKLOADS,
}


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    target = here / "scl_requirements.yaml"
    target.write_bytes(canonical_bytes(DOCUMENT))
    digest = canonical_sha256(DOCUMENT)
    (here / "scl_requirements.sha256").write_text(digest + "\n")
    print(f"wrote {target.name}")
    print(f"requirements_artifact_hash: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

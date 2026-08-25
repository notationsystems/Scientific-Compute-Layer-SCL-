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
             notes):
    return {
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
        notes="Requires the entire missing linear-algebra family. Ordering is NOT required, which is the sharpest modality contrast with the transform family.",
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
        notes="Strictly downstream of the decomposition primitive least_squares would establish.",
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
        notes="The only candidate that is RECURSIVE, and therefore the only one that would make generation_depth_bounded operative rather than vacuous. See recursive_computation_analysis below.",
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
    "extends": "core@1.0.0",
    "generated_by": "architecture/exchange/build_scl_requirements.py",
    "identity_model": {
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

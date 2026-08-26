"""Every registered operation must satisfy the operation contract stated
in native/include/scl/operation.hpp -- checked mechanically, not by
imitation.

The design point: the operation list is ENUMERATED FROM THE BINARY (via
the unknown-operation fault, which names what the build supports) rather
than hardcoded here. A third operation is therefore held to clauses 2-9
the moment it is registered, without anyone remembering to extend this
file. Clause 7's metrics check needs one known-good request per operation,
so `KNOWN_GOOD` is asserted to cover the registry exactly -- adding an
operation without adding a sample request fails loudly rather than
silently skipping the only clause that needs domain knowledge.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from scl.client import (
    SCLRequest,
    encode_lj_configuration,
    encode_lj_positions,
    run_scl_request,
)
from scl.fourier import encode_fourier_configuration, encode_real_signal
from scl.kalman import (
    encode_kalman_configuration,
    encode_kalman_input,
)
from scl.least_squares import (
    encode_least_squares_configuration,
    encode_least_squares_input,
)

#: scl/protocol.hpp's complete vocabulary. Clause 8: an operation may use
#: these and may not mint others.
FAULT_VOCABULARY = {0, 10, 11, 12, 13, 14}

#: Clause 7: the one metric every operation must emit.
REQUIRED_METRIC = "native_compute_seconds"

#: Per-operation known-good request (the only thing here that needs
#: domain knowledge). Asserted below to cover the registry exactly.
KNOWN_GOOD = {
    "lj_pairwise_energy_forces": (
        encode_lj_configuration(1.0, 1.0, 5.0),
        encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
    ),
    "fourier_transform_1d": (
        encode_fourier_configuration(),
        encode_real_signal([1.0, 0.0, 0.0, 0.0]),
    ),
    "least_squares": (
        encode_least_squares_configuration(),
        encode_least_squares_input([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], [2.0, 3.0, 4.0]),
    ),
    "kalman_filter_linear": (
        encode_kalman_configuration(
            transition=[[1.0, 0.1], [0.0, 1.0]],
            observation=[[1.0, 0.0], [0.0, 1.0]],
            process_noise=[[0.01, 0.0], [0.0, 0.04]],
            measurement_noise=[[0.25, 0.0], [0.0, 0.16]],
        ),
        encode_kalman_input(
            initial_state=[0.0, 0.0],
            initial_covariance=[[1.0, 0.0], [0.0, 1.0]],
            measurements=[[0.1, 0.9], [0.2, 1.1], [0.35, 0.95]],
        ),
    ),
}

#: Byte offsets of RESERVED int32 fields in each operation's
#: configuration. Clause 2 requires these to be zero so the layout can
#: grow compatibly; an empty tuple means the operation declares none.
RESERVED_OFFSETS = {
    "lj_pairwise_energy_forces": (),      # three float64s, no reserved words
    "fourier_transform_1d": (12,),
    "least_squares": (4, 8, 12),
    "kalman_filter_linear": (16, 20),
}

#: Operation-specific metric keys, so clause 7's "must not emit another
#: operation's keys" can be checked concretely.
OPERATION_SPECIFIC_METRICS = {
    "lj_pairwise_energy_forces": {"n_particles"},
    "fourier_transform_1d": {"n_samples", "transform_size"},
    "least_squares": {"n_rows", "n_cols", "condition_number", "effective_rank",
                       "jacobi_sweeps", "smallest_singular_value",
                       "largest_singular_value"},
    "kalman_filter_linear": {"state_dimension", "measurement_dimension", "steps",
                             "smallest_posterior_eigenvalue",
                             "measurement_noise_is_supplied"},
}


def registered_operations(cli_path) -> list:
    """Enumerate the registry from the binary itself."""
    result = run_scl_request(
        SCLRequest("__not_a_real_operation__", "cpu", b"", b""), cli_path=cli_path)
    assert result.exit_code == 10, "unknown operation must be a PROTOCOL fault"
    names = re.findall(r"'([a-z0-9_]+)'", result.detail)
    names = [n for n in names if n != "__not_a_real_operation__"]
    assert names, f"could not enumerate the registry from: {result.detail!r}"
    return sorted(names)


def test_the_registry_is_enumerable_and_known_good_covers_it(cli_path):
    """If this fails after adding an operation, add its known-good request
    -- do not delete the assertion."""
    assert registered_operations(cli_path) == sorted(KNOWN_GOOD)
    assert sorted(OPERATION_SPECIFIC_METRICS) == sorted(KNOWN_GOOD)


def _each_operation(cli_path):
    return registered_operations(cli_path)


# --- clause 7: metrics ---------------------------------------------------

def test_clause7_every_operation_emits_the_required_metric(cli_path):
    for operation in _each_operation(cli_path):
        params, payload = KNOWN_GOOD[operation]
        result = run_scl_request(SCLRequest(operation, "cpu", params, payload), cli_path=cli_path)
        assert result.status == "completed", f"{operation}: {result.detail}"
        assert REQUIRED_METRIC in result.metrics, f"{operation} omits {REQUIRED_METRIC}"
        assert result.metrics[REQUIRED_METRIC] >= 0.0


def test_clause7_no_operation_emits_another_operations_metrics(cli_path):
    """Absent and zero are different facts: an operation with no particles
    reports no `n_particles` at all, rather than zero."""
    for operation in _each_operation(cli_path):
        params, payload = KNOWN_GOOD[operation]
        result = run_scl_request(SCLRequest(operation, "cpu", params, payload), cli_path=cli_path)
        foreign = set().union(*(keys for name, keys in OPERATION_SPECIFIC_METRICS.items()
                                 if name != operation))
        leaked = foreign.intersection(result.metrics) - OPERATION_SPECIFIC_METRICS[operation]
        assert not leaked, f"{operation} emitted foreign metrics {leaked}"


# --- clause 2: configuration decoder ------------------------------------

def test_clause2_every_reserved_field_is_declared(cli_path):
    """If an operation is added without declaring its reserved offsets,
    say so rather than skipping the only clause-2 check that matters."""
    assert sorted(RESERVED_OFFSETS) == sorted(KNOWN_GOOD)


def test_clause2_a_non_zero_reserved_field_is_refused(cli_path):
    """Reserved fields must be ZERO so the layout can grow compatibly. A
    build that silently ignores them cannot later give them meaning
    without changing what old requests mean.

    Previously this clause had no dedicated check -- it was exercised only
    through clause 8's malformed-configuration cases, which assert the
    fault VOCABULARY is respected rather than that a non-zero reserved
    field is refused at all."""
    for operation in _each_operation(cli_path):
        params, payload = KNOWN_GOOD[operation]
        for offset in RESERVED_OFFSETS[operation]:
            mutated = bytearray(params)
            mutated[offset] = 0x01
            result = run_scl_request(
                SCLRequest(operation, "cpu", bytes(mutated), payload), cli_path=cli_path)
            assert result.status == "halted", (
                f"{operation} accepted a non-zero reserved field at offset {offset}")
            assert result.exit_code == 11
            assert "reserved" in result.detail.lower(), (
                f"{operation}'s refusal must name the offending field: {result.detail!r}")


def test_clause2_an_ignored_configuration_field_is_refused_not_tolerated(cli_path):
    """The CANONICAL-ENCODING half of clause 2, and it found a real defect
    the first time it ran.

    `fourier_transform_1d` accepted arbitrary bytes in
    sample_spacing_seconds whenever has_sample_spacing was 0, and ignored
    them. Two byte-different configurations therefore meant exactly the
    same thing -- "no sample spacing" -- while producing DIFFERENT
    parameters_identity values, which breaks the premise that a parameter
    identity identifies the parameters. A conditionally-unused field must
    be refused when it carries a value, not quietly dropped."""
    from scl.fourier import encode_fourier_configuration
    import struct

    params = bytearray(encode_fourier_configuration())          # no spacing
    assert struct.unpack_from("<i", params, 8)[0] == 0, "has_sample_spacing is 0 here"
    struct.pack_into("<d", params, 16, 0.5)                     # ...but a spacing is present

    _, payload = KNOWN_GOOD["fourier_transform_1d"]
    result = run_scl_request(
        SCLRequest("fourier_transform_1d", "cpu", bytes(params), payload), cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 11
    assert "sample_spacing_seconds must be 0" in result.detail


def test_clause2_each_operation_rejects_every_wrong_configuration_length(cli_path):
    for operation in _each_operation(cli_path):
        params, payload = KNOWN_GOOD[operation]
        for length in (0, len(params) - 1, len(params) + 1, len(params) * 2):
            mutated = (params * 3)[:length]
            result = run_scl_request(
                SCLRequest(operation, "cpu", mutated, payload), cli_path=cli_path)
            if length == len(params):
                continue
            assert result.status == "halted", (
                f"{operation} accepted a {length}-byte configuration")
            assert result.exit_code == 11


# --- clause 3: input decoder --------------------------------------------

def test_clause3_empty_input_is_a_validation_fault_not_an_empty_success(cli_path):
    """"An empty input is a VALIDATION fault, never a silently-empty
    success." Previously exercised only through clause 6, which asserts a
    halted outcome carries no output -- not that an empty input halts."""
    for operation in _each_operation(cli_path):
        params, _ = KNOWN_GOOD[operation]
        result = run_scl_request(SCLRequest(operation, "cpu", params, b""), cli_path=cli_path)
        assert result.status == "halted", f"{operation} accepted an empty input"
        assert result.exit_code == 11, f"{operation} used {result.exit_code} for empty input"
        assert result.output is None


def test_clause3_a_partial_element_is_refused(cli_path):
    """A payload that is not a whole number of elements. Truncating the
    known-good input by one byte can never be a valid request."""
    for operation in _each_operation(cli_path):
        params, payload = KNOWN_GOOD[operation]
        result = run_scl_request(
            SCLRequest(operation, "cpu", params, payload[:-1]), cli_path=cli_path)
        assert result.status == "halted", f"{operation} accepted a truncated input"
        assert result.exit_code == 11


# --- clause 8: fault vocabulary -----------------------------------------

@pytest.mark.parametrize("malformation", ["empty_config", "empty_input", "garbage_config",
                                           "odd_input", "huge_config"])
def test_clause8_malformed_requests_only_use_the_shared_vocabulary(cli_path, malformation):
    """Generic across operations: no operation may mint a fault code."""
    payloads = {
        "empty_config": (b"", b"\x00" * 24),
        "empty_input": (b"\x00" * 24, b""),
        "garbage_config": (b"\xff" * 24, b"\x00" * 24),
        "odd_input": (b"\x00" * 24, b"\x00" * 7),
        "huge_config": (b"\x00" * 4096, b"\x00" * 24),
    }
    params, payload = payloads[malformation]
    for operation in _each_operation(cli_path):
        result = run_scl_request(SCLRequest(operation, "cpu", params, payload), cli_path=cli_path)
        assert result.exit_code in FAULT_VOCABULARY, (
            f"{operation} minted fault {result.exit_code} for {malformation}")
        assert result.status in ("completed", "halted")


# --- clause 9: totality --------------------------------------------------

@pytest.mark.parametrize("hostile", [b"", b"{", b"[]", b'{"operation":1}', b"\x00\xff\x00",
                                      b'{"operation":"x","backend":"y"}'])
def test_clause9_the_cli_always_answers_and_never_crashes(cli_path, hostile):
    """`run()` is total: a well-formed JSON answer and process exit 0, for
    every input including deliberately hostile ones."""
    proc = subprocess.run([str(cli_path)], input=hostile, capture_output=True, timeout=30)
    assert proc.returncode == 0, f"process exited {proc.returncode} for {hostile!r}"
    response = json.loads(proc.stdout)
    assert response["status"] == "halted"
    assert response["exit_code"] in FAULT_VOCABULARY


def test_clause9_totality_holds_for_each_registered_operation(cli_path):
    for operation in _each_operation(cli_path):
        for params, payload in ((b"", b""), (b"\xff" * 3, b"\xff" * 5)):
            envelope = json.dumps({
                "operation": operation, "backend": "cpu",
                "configuration_hex": params.hex(), "input_hex": payload.hex()}).encode()
            proc = subprocess.run([str(cli_path)], input=envelope, capture_output=True, timeout=30)
            assert proc.returncode == 0
            assert json.loads(proc.stdout)["exit_code"] in FAULT_VOCABULARY


# --- clause 5: one availability source of truth --------------------------

def test_clause5_backend_unavailability_is_identical_across_operations(cli_path):
    """Every operation must route through backend_unavailable_reason(), so
    the message is byte-identical -- they drifted once already."""
    details = set()
    for operation in _each_operation(cli_path):
        params, payload = KNOWN_GOOD[operation]
        result = run_scl_request(SCLRequest(operation, "cuda", params, payload), cli_path=cli_path)
        assert result.exit_code == 12
        assert result.backend_used == "cuda"
        details.add(result.detail)
    assert len(details) == 1, f"operations disagree on the availability message: {details}"


# --- clause 6: absent output is absent ------------------------------------

def test_clause6_halted_operations_carry_no_output_or_identities(cli_path):
    for operation in _each_operation(cli_path):
        result = run_scl_request(SCLRequest(operation, "cpu", b"", b""), cli_path=cli_path)
        assert result.status == "halted"
        assert result.output is None
        assert result.output_identity is None
        assert result.computation_identity is None


def test_clause6_the_wire_response_omits_output_entirely_when_halted(cli_path):
    """Clause 6 constrains what the CLI EMITS, and the previous test
    cannot see that: `run_scl_request` sets output=None on the halted
    branch before it ever reads `output_hex`, so an empty-but-present
    output on the wire is invisible through the client.

    Found by mutation -- making OperationOutcome::halted set
    has_output=true survived the whole suite. Asserted here against the
    raw JSON, which is the layer the clause is about: absent, not ""."""
    for operation in _each_operation(cli_path):
        envelope = json.dumps({"operation": operation, "backend": "cpu",
                               "configuration_hex": "", "input_hex": ""}).encode()
        proc = subprocess.run([str(cli_path)], input=envelope, capture_output=True, timeout=30)
        response = json.loads(proc.stdout)
        assert response["status"] == "halted"
        assert response.get("output_hex", None) is None, (
            f"{operation} emitted output_hex={response.get('output_hex')!r} on a halted "
            "outcome; absent and empty-but-present are different facts")


# --- clause 4: faults name the offending field ---------------------------

def test_clause4_validation_faults_are_actionable(cli_path):
    """A fault a caller cannot act on is barely better than a crash."""
    for operation in _each_operation(cli_path):
        result = run_scl_request(SCLRequest(operation, "cpu", b"\x00" * 3, b""), cli_path=cli_path)
        assert result.detail, f"{operation} produced a fault with no detail"
        assert len(result.detail) > 20, f"{operation} detail is too terse: {result.detail!r}"


# --- clause 1 & 10: naming and per-operation descriptor -------------------

def test_clause1_operation_names_are_snake_case_and_not_algorithm_names(cli_path):
    for operation in _each_operation(cli_path):
        assert re.fullmatch(r"[a-z][a-z0-9_]*", operation), operation
        assert operation != "fft", "name the mathematical operation, not the algorithm"


def test_clause11_every_operation_has_its_own_program_identity(cli_path):
    ste_adapter = pytest.importorskip(
        "scl.ste_adapter", reason="STE checkout absent; environment gap, not a contract failure")
    from execution.specification import ExecutionSpecification

    identities = {}
    for operation in _each_operation(cli_path):
        program = ste_adapter.scl_program_descriptor("scl-cli/0.1.0", "cpu", operation)
        assert ste_adapter.operation_from_descriptor_header(
            ste_adapter.descriptor_header(operation)) == operation
        identities[operation] = ExecutionSpecification(program, b"", b"").program_identity()
    assert len(set(identities.values())) == len(identities), (
        f"operations share a program identity: {identities}")

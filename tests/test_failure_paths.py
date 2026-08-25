"""Failure-path coverage (Task 7 of the SCL Phase 1 brief). Each failure
stage must stay distinguishable -- never collapsed into one generic
"execution failed" -- so every test here pins BOTH the outcome class
(exception type, or SCLResult.status/exit_code) and enough of the detail
message to prove the failure was diagnosed, not just detected."""

from __future__ import annotations

import pathlib
import stat

import pytest

from scl.client import (
    SCLRequest,
    default_cli_path,
    encode_lj_configuration,
    encode_lj_positions,
    raise_for_result,
    run_scl_request,
)
from scl.errors import (
    SCLBackendUnavailableError,
    SCLComputationError,
    SCLProtocolError,
    SCLTimeoutError,
    SCLValidationError,
)

VALID_POSITIONS = encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])


def _request(backend="cpu", parameters=None, input_payload=None, operation="lj_pairwise_energy_forces"):
    return SCLRequest(
        operation=operation,
        backend=backend,
        parameters=parameters if parameters is not None else encode_lj_configuration(1.0, 1.0, 5.0),
        input_payload=input_payload if input_payload is not None else VALID_POSITIONS,
    )


# --- invalid parameters -------------------------------------------------

def test_negative_sigma_halts_with_validation_fault(cli_path):
    request = _request(parameters=encode_lj_configuration(1.0, -1.0, 5.0))
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 11
    assert "sigma" in result.detail
    with pytest.raises(SCLValidationError):
        raise_for_result(result)


def test_zero_cutoff_halts_with_validation_fault(cli_path):
    request = _request(parameters=encode_lj_configuration(1.0, 1.0, 0.0))
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 11
    assert "cutoff" in result.detail


# --- invalid input --------------------------------------------------------

def test_empty_particle_set_halts_with_validation_fault(cli_path):
    request = _request(input_payload=b"")
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 11
    assert "at least one particle" in result.detail


def test_malformed_input_length_halts_with_validation_fault(cli_path):
    # 24 bytes must divide evenly into 24-byte particles; 10 bytes cannot.
    request = _request(input_payload=b"\x00" * 10)
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 11
    assert "24-byte" in result.detail


def test_malformed_configuration_length_halts_with_validation_fault(cli_path):
    request = _request(parameters=b"\x00" * 5)
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 11
    assert "24 bytes" in result.detail


def test_unknown_operation_halts_at_the_protocol_layer_not_a_crash(cli_path):
    request = _request(operation="not_a_real_operation")
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 10
    assert "unknown operation" in result.detail


# --- missing / unavailable backend ----------------------------------------

def test_cuda_backend_is_distinguishable_unavailable_not_a_silent_cpu_fallback(cli_path):
    request = _request(backend="cuda")
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 12
    assert result.backend_used == "cuda"  # echoes what was ASKED for, not a silent substitution
    with pytest.raises(SCLBackendUnavailableError):
        raise_for_result(result)


def test_missing_binary_raises_protocol_error_not_a_computation_result(tmp_path):
    request = _request()
    with pytest.raises(SCLProtocolError, match="not found"):
        run_scl_request(request, cli_path=tmp_path / "does_not_exist")


# --- computation-level fault (distinct from validation) --------------------

def test_coincident_particles_is_a_computation_fault_not_a_validation_fault(cli_path):
    request = _request(input_payload=encode_lj_positions([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]))
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 13  # computation, not 11 (validation) -- distinct fault stages
    assert "singular" in result.detail or "separation" in result.detail
    with pytest.raises(SCLComputationError):
        raise_for_result(result)


# --- malformed native output (protocol-layer, not swallowed) --------------

def test_malformed_cli_output_raises_protocol_error(tmp_path):
    fake_cli = tmp_path / "fake_scl_cli"
    fake_cli.write_text("#!/bin/sh\necho 'not json at all'\n")
    fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SCLProtocolError, match="non-JSON"):
        run_scl_request(_request(), cli_path=fake_cli)


def test_incomplete_cli_response_raises_protocol_error(tmp_path):
    fake_cli = tmp_path / "fake_scl_cli"
    fake_cli.write_text('#!/bin/sh\necho \'{"status": "completed"}\'\n')
    fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SCLProtocolError, match="missing field"):
        run_scl_request(_request(), cli_path=fake_cli)


def test_catastrophic_process_exit_raises_protocol_error(tmp_path):
    fake_cli = tmp_path / "fake_scl_cli"
    fake_cli.write_text("#!/bin/sh\nexit 1\n")
    fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(SCLProtocolError, match="catastrophically"):
        run_scl_request(_request(), cli_path=fake_cli)


# --- timeout / resource failure --------------------------------------------

def test_timeout_is_a_distinct_exception_not_a_hang_or_generic_error(cli_path):
    # A tiny timeout budget cannot be met even by process startup alone,
    # regardless of workload size -- deterministic without depending on
    # machine speed for a large N.
    big_positions = encode_lj_positions([(float(i), 0.0, 0.0) for i in range(2000)])
    request = _request(input_payload=big_positions)
    with pytest.raises(SCLTimeoutError):
        run_scl_request(request, cli_path=cli_path, timeout=1e-6)


def test_default_cli_path_points_under_native_build():
    assert default_cli_path().parts[-3:] == ("native", "build", "scl_cli")

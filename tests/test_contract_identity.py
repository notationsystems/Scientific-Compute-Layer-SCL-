"""SCLRequest/SCLResult identity behaviour (Task 6 of the SCL Phase 1
brief: result identity, without duplicating any existing namespace)."""

from __future__ import annotations

from scl.client import SCLRequest, encode_lj_configuration, encode_lj_positions, run_scl_request


def _request(epsilon=1.0, sigma=1.0, cutoff=5.0, positions=((0, 0, 0), (1.5, 0, 0)), backend="cpu"):
    return SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend=backend,
        parameters=encode_lj_configuration(epsilon, sigma, cutoff),
        input_payload=encode_lj_positions(positions),
    )


def test_identical_requests_share_every_identity_field():
    a = _request()
    b = _request()
    assert a.identity() == b.identity()
    assert a.operation_identity() == b.operation_identity()
    assert a.parameters_identity() == b.parameters_identity()
    assert a.input_identity() == b.input_identity()


def test_changing_parameters_changes_only_parameters_and_request_identity():
    a = _request(sigma=1.0)
    b = _request(sigma=1.2)
    assert a.parameters_identity() != b.parameters_identity()
    assert a.identity() != b.identity()
    # input and operation identity are independent of parameters
    assert a.input_identity() == b.input_identity()
    assert a.operation_identity() == b.operation_identity()


def test_changing_input_changes_only_input_and_request_identity():
    a = _request(positions=((0, 0, 0), (1.5, 0, 0)))
    b = _request(positions=((0, 0, 0), (1.8, 0, 0)))
    assert a.input_identity() != b.input_identity()
    assert a.identity() != b.identity()
    assert a.parameters_identity() == b.parameters_identity()
    assert a.operation_identity() == b.operation_identity()


def test_changing_backend_changes_operation_and_request_identity():
    # backend is folded into operation_identity (Task 2's "execution
    # constraints" -- which physical engine ran is part of WHAT ran).
    cpu = _request(backend="cpu")
    cuda = _request(backend="cuda")
    assert cpu.operation_identity() != cuda.operation_identity()
    assert cpu.identity() != cuda.identity()
    assert cpu.input_identity() == cuda.input_identity()


def test_result_identity_is_content_addressed_across_two_runs(cli_path):
    request = _request()
    first = run_scl_request(request, cli_path=cli_path)
    second = run_scl_request(request, cli_path=cli_path)
    assert first.status == second.status == "completed"
    assert first.request_identity == second.request_identity
    assert first.output_identity == second.output_identity
    assert first.computation_identity == second.computation_identity
    # re-running does not fabricate a new identity for the same request
    assert first.output == second.output


def test_different_input_yields_different_output_and_computation_identity(cli_path):
    near = run_scl_request(_request(positions=((0, 0, 0), (1.3, 0, 0))), cli_path=cli_path)
    far = run_scl_request(_request(positions=((0, 0, 0), (3.0, 0, 0))), cli_path=cli_path)
    assert near.request_identity != far.request_identity
    assert near.output_identity != far.output_identity
    assert near.computation_identity != far.computation_identity

"""scl.client happy-path behaviour: the subprocess round trip, encode/
decode symmetry, timing metadata, and same-machine reproducibility
(Task 5's bitwise-reproducibility claim, exercised through the full
process boundary this time, not just the in-process native test)."""

from __future__ import annotations

import math

from scl.client import (
    SCLRequest,
    decode_lj_output,
    encode_lj_configuration,
    encode_lj_positions,
    run_scl_request,
)


def test_two_particle_energy_matches_closed_form(cli_path):
    epsilon, sigma, r = 1.0, 1.0, 1.5
    request = SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend="cpu",
        parameters=encode_lj_configuration(epsilon, sigma, 5.0),
        input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (r, 0.0, 0.0)]),
    )
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "completed"
    total_energy, forces = decode_lj_output(result.output)

    sr6 = (sigma / r) ** 6
    expected = 4.0 * epsilon * (sr6 * sr6 - sr6)
    assert math.isclose(total_energy, expected, rel_tol=1e-12)
    assert len(forces) == 2
    # equal and opposite (Newton's third law) for a two-body system
    assert math.isclose(forces[0][0], -forces[1][0], rel_tol=1e-12)


def test_metrics_and_timing_are_populated(cli_path):
    request = SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend="cpu",
        parameters=encode_lj_configuration(1.0, 1.0, 5.0),
        input_payload=encode_lj_positions([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
    )
    result = run_scl_request(request, cli_path=cli_path)
    assert result.wall_clock_seconds > 0.0
    assert result.native_compute_seconds is not None
    assert result.native_compute_seconds >= 0.0
    # STE overhead (subprocess spawn/JSON) must be observable and,
    # ordinarily, larger than the native compute time itself for a
    # workload this small -- Task 8's baseline claim.
    assert result.wall_clock_seconds >= result.native_compute_seconds
    assert result.n_particles == 2


def test_repeat_run_is_bitwise_reproducible_same_binary_same_machine(cli_path):
    request = SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend="cpu",
        parameters=encode_lj_configuration(0.9, 1.05, 6.0),
        input_payload=encode_lj_positions([(0.1, 0.2, 0.3), (1.4, -0.5, 0.2), (-0.6, 0.9, -1.1)]),
    )
    first = run_scl_request(request, cli_path=cli_path)
    second = run_scl_request(request, cli_path=cli_path)
    assert first.output == second.output  # bit-identical bytes, not just close
    assert first.computation_identity == second.computation_identity

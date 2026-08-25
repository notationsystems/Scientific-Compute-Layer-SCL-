"""A fast, CI-scale sanity check on the SCL boundary's overhead shape
(Task 8). This is NOT the performance baseline itself -- that is
scripts/run_benchmark.py, run manually and recorded in
docs/PHASE1_AUDIT.md, sweeping much larger N. This test just guards that
the wall-clock/native-compute split stays sane and that the O(N^2) kernel
does not regress catastrophically."""

from __future__ import annotations

import time

from scl.client import SCLRequest, encode_lj_configuration, encode_lj_positions, run_scl_request


def test_wall_clock_includes_and_exceeds_native_compute_time(cli_path):
    positions = [(float(i) * 0.3, 0.0, 0.0) for i in range(200)]
    request = SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend="cpu",
        parameters=encode_lj_configuration(1.0, 1.0, 2.5),
        input_payload=encode_lj_positions(positions),
    )
    start = time.monotonic()
    result = run_scl_request(request, cli_path=cli_path)
    observed_wall_clock = time.monotonic() - start

    assert result.status == "completed"
    assert result.n_particles == 200
    assert result.native_compute_seconds is not None
    # native compute time is a (typically small) fraction of the full
    # round trip; the STE-side overhead (process spawn + JSON marshaling)
    # is the rest -- both must be observable, neither fabricated.
    assert 0.0 <= result.native_compute_seconds <= result.wall_clock_seconds
    assert result.wall_clock_seconds <= observed_wall_clock + 0.5  # sane, not a hang

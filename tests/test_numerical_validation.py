"""Numerical validation through the FULL subprocess boundary (Task 5 of
the SCL Phase 1 brief): input -> computation -> expected numerical
property -> observed result -> verification, tolerance-aware where exact
equality is inappropriate.

Reproducibility claims made explicit and NOT overstated:
  * BITWISE reproducibility: same binary, same machine, same backend --
    claimed and tested (test_client_subprocess.py's repeat-run test).
  * NUMERICAL reproducibility (tolerance-based, e.g. cross the CPU/CUDA
    backend boundary): the finite-difference cross-check below IS a
    numerical (tolerance-based), not bitwise, comparison.
  * PHYSICAL reproducibility (does it match a real experiment): NOT
    APPLICABLE -- this is a model computation (a Lennard-Jones potential
    of an idealized system), never a measurement. Nothing here is
    compared against laboratory data, mirroring execution/gromacs.py's
    own "COMPUTATION != MEASUREMENT" posture.
  * SEMANTIC reproducibility (same request => same computation_identity):
    claimed and tested in test_contract_identity.py.
"""

from __future__ import annotations

import math

from scl.client import SCLRequest, decode_lj_output, encode_lj_configuration, encode_lj_positions, run_scl_request


def _run(positions, epsilon=1.0, sigma=1.0, cutoff=6.0, cli_path=None):
    request = SCLRequest(
        operation="lj_pairwise_energy_forces",
        backend="cpu",
        parameters=encode_lj_configuration(epsilon, sigma, cutoff),
        input_payload=encode_lj_positions(positions),
    )
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "completed", result.detail
    return decode_lj_output(result.output)


def test_newtons_third_law_total_force_is_zero(cli_path):
    positions = [
        (0.0, 0.0, 0.0), (1.3, 0.2, -0.4), (-0.7, 1.1, 0.3),
        (2.1, -0.9, 0.6), (0.4, 0.4, 1.8),
    ]
    _, forces = _run(positions, cli_path=cli_path)
    total = [sum(axis) for axis in zip(*forces)]
    for component in total:
        assert math.isclose(component, 0.0, abs_tol=1e-9)


def test_force_matches_finite_difference_energy_gradient(cli_path):
    """Independent numerical cross-check: perturb one particle by +-delta
    along one axis and compare the resulting energy difference to the
    analytically-reported force -- exercised THROUGH the subprocess
    boundary this time (native/tests/test_lj_pairwise.cpp already checks
    the same invariant in-process; this proves the CLI's JSON/hex framing
    does not corrupt the values in transit)."""
    base = [(0.0, 0.0, 0.0), (1.6, 0.3, -0.2), (-1.1, 0.9, 0.5)]
    epsilon, sigma, cutoff = 0.8, 1.1, 6.0
    energy0, forces0 = _run(base, epsilon, sigma, cutoff, cli_path=cli_path)

    delta = 1e-6
    particle, axis = 1, 0  # perturb particle 1's x coordinate
    plus = [list(p) for p in base]
    minus = [list(p) for p in base]
    plus[particle][axis] += delta
    minus[particle][axis] -= delta

    e_plus, _ = _run([tuple(p) for p in plus], epsilon, sigma, cutoff, cli_path=cli_path)
    e_minus, _ = _run([tuple(p) for p in minus], epsilon, sigma, cutoff, cli_path=cli_path)

    numeric_dE = (e_plus - e_minus) / (2 * delta)
    analytic_force = forces0[particle][axis]
    assert math.isclose(analytic_force, -numeric_dE, rel_tol=1e-3, abs_tol=1e-6)


def test_cutoff_truncation_zeroes_distant_pairs(cli_path):
    energy, forces = _run([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)], cutoff=2.0, cli_path=cli_path)
    assert energy == 0.0
    assert all(component == 0.0 for force in forces for component in force)


def test_attractive_well_sign_at_moderate_separation(cli_path):
    """A semantic check on the WORKLOAD (mirrors
    tests/test_execution_gromacs.py's argon-pair sign check in STE): at a
    separation somewhat beyond sigma, the LJ potential is in its
    attractive well and must be negative."""
    energy, _ = _run([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)], epsilon=1.0, sigma=1.0, cutoff=5.0, cli_path=cli_path)
    assert energy < 0.0


def test_repulsive_at_close_separation(cli_path):
    energy, _ = _run([(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)], epsilon=1.0, sigma=1.0, cutoff=5.0, cli_path=cli_path)
    assert energy > 0.0

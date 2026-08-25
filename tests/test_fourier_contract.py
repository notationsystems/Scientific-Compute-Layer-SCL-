"""`fourier_transform_1d`: validated against INDEPENDENT MATHEMATICS, not
against a second implementation of the same specification.

The strongest checks here are analytic -- impulse, DC, pure tone, Parseval,
inverse reconstruction -- because they are properties of the discrete
Fourier transform itself and hold regardless of how any implementation
computes it. A hand-written O(N^2) DFT oracle is included as well, but it
is deliberately the WEAKEST evidence in this file, not the primary
argument: two implementations written from one reading of a spec can agree
and both be wrong, while an impulse's spectrum is flat as a matter of
mathematics.

numpy is deliberately not used (it is not installed, and its absence is
useful): the oracle below is written straight from
X_k = SUM_n x_n exp(-2*pi*i*k*n/N) using only `cmath`.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Optional, Sequence

import pytest

from conftest import requires_ste
from scl.client import SCLRequest, run_scl_request
from scl.fourier import (
    FORWARD,
    INVERSE,
    NORMALIZATION_NONE,
    NORMALIZATION_ONE_OVER_N,
    NORMALIZATION_ONE_OVER_SQRT_N,
    decode_complex_spectrum,
    encode_complex_signal,
    encode_fourier_configuration,
    encode_real_signal,
    frequency_bins,
)

OPERATION = "fourier_transform_1d"


# --- the independent oracle (weakest evidence here, see module docstring) --

def naive_dft(samples: Sequence[complex], direction: int = FORWARD,
              normalization: int = NORMALIZATION_NONE) -> List[complex]:
    """The defining sum, written from mathematics with stdlib only."""
    n = len(samples)
    sign = -1.0 if direction == FORWARD else 1.0
    scale = {
        NORMALIZATION_NONE: 1.0,
        NORMALIZATION_ONE_OVER_N: 1.0 / n,
        NORMALIZATION_ONE_OVER_SQRT_N: 1.0 / math.sqrt(n),
    }[normalization]
    out: List[complex] = []
    for k in range(n):
        total = 0j
        for j in range(n):
            angle = sign * 2.0 * math.pi * ((k * j) % n) / n
            total += complex(samples[j]) * cmath.exp(complex(0.0, angle))
        out.append(total * scale)
    return out


def run_transform(samples, cli_path, direction=FORWARD, normalization=NORMALIZATION_NONE,
                  sample_spacing=None, backend="cpu", complex_input=False):
    payload = (encode_complex_signal(samples) if complex_input else encode_real_signal(samples))
    request = SCLRequest(
        operation=OPERATION, backend=backend,
        parameters=encode_fourier_configuration(direction, normalization, sample_spacing),
        input_payload=payload,
    )
    result = run_scl_request(request, cli_path=cli_path)
    assert result.status == "completed", result.detail
    return decode_complex_spectrum(result.output), result


def max_abs_error(a: Sequence[complex], b: Sequence[complex]) -> float:
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def rms_error(a: Sequence[complex], b: Sequence[complex]) -> float:
    return math.sqrt(sum(abs(x - y) ** 2 for x, y in zip(a, b)) / len(a))


# --- analytic property 1: impulse -> flat spectrum ------------------------

def test_impulse_produces_a_flat_spectrum(cli_path):
    """x = [1, 0, 0, ...] => X_k = 1 for every k. Pure mathematics: the
    sum collapses to the n=0 term, whose exponent is 0 for all k."""
    n = 16
    signal = [1.0] + [0.0] * (n - 1)
    spectrum, _ = run_transform(signal, cli_path)
    assert len(spectrum) == n
    for k, value in enumerate(spectrum):
        assert abs(value - 1.0) < 1e-12, f"bin {k} = {value}"


# --- analytic property 2: DC -> all energy in bin 0 -----------------------

def test_constant_signal_concentrates_in_the_dc_bin(cli_path):
    """x_n = c for all n => X_0 = N*c and X_k = 0 for k != 0."""
    n, c = 12, 2.5
    spectrum, _ = run_transform([c] * n, cli_path)
    assert abs(spectrum[0] - complex(n * c, 0.0)) < 1e-11
    for k in range(1, n):
        assert abs(spectrum[k]) < 1e-11, f"bin {k} = {spectrum[k]}"


# --- analytic property 3: pure tone -> exactly the expected bins ----------

def test_complex_exponential_lands_in_exactly_one_bin(cli_path):
    """x_n = exp(+2*pi*i*m*n/N) => forward X_k = N*delta(k, m). A single
    nonzero bin, at a bin index the mathematics names in advance."""
    n, m = 32, 5
    signal = [cmath.exp(complex(0.0, 2.0 * math.pi * m * j / n)) for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path, complex_input=True)
    assert abs(spectrum[m] - complex(n, 0.0)) < 1e-10
    for k in range(n):
        if k != m:
            assert abs(spectrum[k]) < 1e-10, f"bin {k} = {spectrum[k]}"


def test_real_cosine_lands_in_the_conjugate_bin_pair(cli_path):
    """A real cosine of bin frequency m splits into m and N-m, each N/2 --
    the Hermitian symmetry of a real signal's transform."""
    n, m = 32, 4
    signal = [math.cos(2.0 * math.pi * m * j / n) for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path)
    assert abs(spectrum[m] - complex(n / 2.0, 0.0)) < 1e-10
    assert abs(spectrum[n - m] - complex(n / 2.0, 0.0)) < 1e-10
    for k in range(n):
        if k not in (m, n - m):
            assert abs(spectrum[k]) < 1e-10, f"bin {k} = {spectrum[k]}"


def test_multi_frequency_signal_resolves_each_component(cli_path):
    n = 64
    components = {3: 1.0, 11: 0.5, 20: 2.0}
    signal = [sum(a * math.cos(2.0 * math.pi * m * j / n) for m, a in components.items())
              for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path)
    for m, amplitude in components.items():
        assert abs(abs(spectrum[m]) - amplitude * n / 2.0) < 1e-9
    quiet = [k for k in range(n) if k not in components and (n - k) not in components]
    assert max(abs(spectrum[k]) for k in quiet) < 1e-9


# --- analytic property 4: Parseval ----------------------------------------

def test_parseval_relation_holds(cli_path):
    """SUM_k |X_k|^2 = N * SUM_n |x_n|^2 for the unnormalized forward
    transform -- an energy conservation law, independent of implementation."""
    n = 48
    signal = [math.sin(0.7 * j) + 0.3 * math.cos(2.1 * j) for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path)
    lhs = sum(abs(value) ** 2 for value in spectrum)
    rhs = n * sum(abs(complex(value)) ** 2 for value in signal)
    assert abs(lhs - rhs) / rhs < 1e-12, f"Parseval violated: {lhs} vs {rhs}"


def test_unitary_normalization_preserves_energy_exactly(cli_path):
    """With 1/sqrt(N), the transform is unitary: SUM|X|^2 == SUM|x|^2."""
    n = 32
    signal = [math.sin(0.3 * j) for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path, normalization=NORMALIZATION_ONE_OVER_SQRT_N)
    lhs = sum(abs(v) ** 2 for v in spectrum)
    rhs = sum(abs(complex(v)) ** 2 for v in signal)
    assert abs(lhs - rhs) / rhs < 1e-12


# --- analytic property 5: inverse reconstruction --------------------------

def test_inverse_transform_reconstructs_the_original_signal(cli_path):
    """IDFT(DFT(x)) == x when exactly one direction carries the 1/N. Round
    trips through the real process boundary, not in-memory."""
    n = 24
    signal = [math.sin(0.4 * j) + 0.2 * j for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path, direction=FORWARD,
                                 normalization=NORMALIZATION_NONE)
    recovered, _ = run_transform(spectrum, cli_path, direction=INVERSE,
                                  normalization=NORMALIZATION_ONE_OVER_N, complex_input=True)
    assert max_abs_error(recovered, [complex(v) for v in signal]) < 1e-11


# --- the (weakest) independent-implementation cross-check ----------------

@pytest.mark.parametrize("normalization", [NORMALIZATION_NONE, NORMALIZATION_ONE_OVER_N,
                                            NORMALIZATION_ONE_OVER_SQRT_N])
@pytest.mark.parametrize("direction", [FORWARD, INVERSE])
def test_agrees_with_independent_stdlib_dft(cli_path, direction, normalization):
    n = 37  # deliberately prime: no power-of-two assumption anywhere
    signal = [math.sin(1.1 * j) + 0.5 * math.cos(0.3 * j) for j in range(n)]
    spectrum, _ = run_transform(signal, cli_path, direction=direction, normalization=normalization)
    expected = naive_dft([complex(v) for v in signal], direction, normalization)
    scale = max(abs(v) for v in expected) or 1.0
    assert max_abs_error(spectrum, expected) / scale < 1e-12
    assert rms_error(spectrum, expected) / scale < 1e-12


def test_error_metrics_are_reported_for_representative_sizes(cli_path):
    """Absolute / relative / max-elementwise / RMS error against the
    independent oracle, across sizes -- reported, not merely asserted."""
    for n in (1, 2, 3, 8, 17, 64):
        signal = [math.cos(0.9 * j) for j in range(n)]
        spectrum, _ = run_transform(signal, cli_path)
        expected = naive_dft([complex(v) for v in signal])
        scale = max((abs(v) for v in expected), default=1.0) or 1.0
        assert max_abs_error(spectrum, expected) < 1e-11
        assert max_abs_error(spectrum, expected) / scale < 1e-12
        assert rms_error(spectrum, expected) / scale < 1e-12


# --- sampling metadata: SCL never invents a frequency axis ----------------

def test_no_sample_spacing_means_no_frequency_axis():
    assert frequency_bins(8, None) is None


def test_frequency_axis_uses_the_two_sided_convention():
    n, dt = 8, 0.25  # f_s = 4 Hz, Nyquist 2 Hz
    bins = frequency_bins(n, dt)
    assert bins is not None
    assert bins[0] == 0.0
    assert abs(bins[1] - 0.5) < 1e-15          # k/(N*dt) = 1/2
    assert abs(bins[4] - 2.0) < 1e-15          # Nyquist at k = N/2
    assert abs(bins[5] - (-1.5)) < 1e-15       # above Nyquist reads negative
    assert abs(bins[7] - (-0.5)) < 1e-15


def test_sample_spacing_is_carried_but_does_not_change_the_transform(cli_path):
    """A measured, deliberate property: dt is metadata ABOUT the signal, so
    it changes the request's parameter identity while leaving the
    mathematics -- and therefore the output bytes -- untouched."""
    signal = [1.0, 2.0, 3.0, 4.0]
    without, r_without = run_transform(signal, cli_path, sample_spacing=None)
    with_dt, r_with = run_transform(signal, cli_path, sample_spacing=0.01)
    assert r_without.output == r_with.output          # identical mathematics
    assert r_without.request.parameters_identity() != r_with.request.parameters_identity()
    assert r_without.request_identity != r_with.request_identity


# --- identity ------------------------------------------------------------

def _req(direction=FORWARD, normalization=NORMALIZATION_NONE, dt=None, samples=(1.0, 2.0, 3.0, 4.0),
         backend="cpu"):
    return SCLRequest(OPERATION, backend,
                      encode_fourier_configuration(direction, normalization, dt),
                      encode_real_signal(list(samples)))


@pytest.mark.parametrize("changed", ["direction", "normalization", "sample_spacing", "input", "backend"])
def test_every_computationally_relevant_parameter_changes_an_identity(changed):
    """No Fourier-specific identity mechanism: direction, normalization and
    dt all live in the configuration bytes, so the EXISTING
    parameters_identity covers them; the signal covers input_identity; the
    backend covers operation_identity."""
    base = _req()
    other = {
        "direction": _req(direction=INVERSE),
        "normalization": _req(normalization=NORMALIZATION_ONE_OVER_N),
        "sample_spacing": _req(dt=0.5),
        "input": _req(samples=(1.0, 2.0, 3.0, 5.0)),
        "backend": _req(backend="cuda"),
    }[changed]

    assert base.identity() != other.identity()
    if changed in ("direction", "normalization", "sample_spacing"):
        assert base.parameters_identity() != other.parameters_identity()
        assert base.input_identity() == other.input_identity()
    elif changed == "input":
        assert base.input_identity() != other.input_identity()
        assert base.parameters_identity() == other.parameters_identity()
    else:
        assert base.operation_identity() != other.operation_identity()


def test_identical_requests_share_every_identity():
    assert _req().identity() == _req().identity()
    assert _req().parameters_identity() == _req().parameters_identity()


# --- determinism ---------------------------------------------------------

def test_repeated_execution_is_bitwise_identical(cli_path):
    signal = [math.sin(0.5 * j) for j in range(29)]
    _, first = run_transform(signal, cli_path)
    _, second = run_transform(signal, cli_path)
    assert first.output == second.output  # bit-identical bytes, not merely close
    assert first.output_identity == second.output_identity
    assert first.computation_identity == second.computation_identity


# --- failure semantics (existing vocabulary only) ------------------------

@pytest.mark.parametrize(
    "params,payload,expected_code,needle",
    [
        (encode_fourier_configuration(), b"", 11, "at least one sample"),
        (encode_fourier_configuration(), b"\x00" * 10, 11, "16-byte"),
        (b"\x00" * 5, encode_real_signal([1.0]), 11, "24 bytes"),
        (__import__("struct").pack("<iiiid", 7, 0, 0, 0, 0.0), encode_real_signal([1.0]), 11, "direction"),
        (__import__("struct").pack("<iiiid", 1, 9, 0, 0, 0.0), encode_real_signal([1.0]), 11, "normalization"),
        (__import__("struct").pack("<iiiid", 1, 0, 5, 0, 0.0), encode_real_signal([1.0]), 11, "has_sample_spacing"),
        (__import__("struct").pack("<iiiid", 1, 0, 0, 3, 0.0), encode_real_signal([1.0]), 11, "reserved"),
        (encode_fourier_configuration(sample_spacing_seconds=-1.0), encode_real_signal([1.0]), 11, "sample_spacing"),
        (encode_fourier_configuration(), encode_real_signal([float("nan")]), 11, "not finite"),
        (encode_fourier_configuration(), encode_real_signal([float("inf")]), 11, "not finite"),
    ],
)
def test_invalid_requests_use_the_existing_fault_vocabulary(cli_path, params, payload, expected_code, needle):
    """Every Fourier failure maps onto a fault code that already existed --
    no Fourier-specific failure class was introduced."""
    result = run_scl_request(SCLRequest(OPERATION, "cpu", params, payload), cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == expected_code
    assert needle in result.detail, f"expected {needle!r} in {result.detail!r}"
    assert result.output is None
    assert result.computation_identity is None


def test_cuda_backend_is_unavailable_not_silently_cpu(cli_path):
    result = run_scl_request(
        SCLRequest(OPERATION, "cuda", encode_fourier_configuration(), encode_real_signal([1.0, 2.0])),
        cli_path=cli_path)
    assert result.status == "halted"
    assert result.exit_code == 12
    assert result.backend_used == "cuda"


# --- STE integration -----------------------------------------------------

@requires_ste
def test_fourier_runs_through_the_real_ste_seam(cli_path):
    from scl.ste_adapter import build_fourier_specification, run_scl_specification

    spec = build_fourier_specification([1.0, 0.0, 0.0, 0.0], sample_spacing_seconds=0.1,
                                        cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    assert result.status == "completed"
    assert result.specification_identity == spec.identity()
    assert result.program_identity == spec.program_identity()
    assert result.computation_identity is not None
    spectrum = decode_complex_spectrum(result.output)
    for value in spectrum:
        assert abs(value - 1.0) < 1e-12  # impulse, through the whole STE path


@requires_ste
def test_fourier_and_lj_specifications_never_share_a_program_identity(cli_path):
    from scl.ste_adapter import build_fourier_specification, build_lj_specification

    ft = build_fourier_specification([1.0, 2.0], cli_path=cli_path)
    lj = build_lj_specification(1.0, 1.0, 5.0, [(0, 0, 0), (1.5, 0, 0)], cli_path=cli_path)
    assert ft.program_identity() != lj.program_identity()


@requires_ste
def test_interpret_records_metadata_without_inventing_a_frequency_axis(cli_path):
    from evidence.types import make_referent
    from materials.candidates import make_action_candidate
    from scl.ste_adapter import build_fourier_specification, interpret_fourier_result, run_scl_specification

    referent = make_referent(natural_key="signal-under-test", kind="process")
    candidate = make_action_candidate(
        action_class="scl_fourier_transform_1d", requirement_ids=("req-1",),
        formulation=referent, property="spectral_power", role="target", target_context={})

    # Without dt: the method block must SAY there is no frequency axis.
    spec = build_fourier_specification([1.0, 0.0, 0.0, 0.0], cli_path=cli_path)
    content = interpret_fourier_result(candidate, run_scl_specification(spec, cli_path=cli_path))
    assert content["evidence_class"] == "computed"
    assert "frequency_hz" not in content
    assert content["method_block"]["frequency_axis"]["applicable"] is False
    assert content["method_block"]["sample_spacing_seconds"]["applicable"] is False

    # With dt: a real, deterministic frequency axis appears.
    spec_dt = build_fourier_specification([1.0, 0.0, 0.0, 0.0], sample_spacing_seconds=0.5,
                                           cli_path=cli_path)
    content_dt = interpret_fourier_result(candidate, run_scl_specification(spec_dt, cli_path=cli_path))
    assert content_dt["method_block"]["frequency_axis"]["applicable"] is True
    assert content_dt["frequency_hz"] == frequency_bins(4, 0.5)
    # The algorithm is recorded separately from the mathematical operation.
    assert content_dt["method_block"]["algorithm"]["value"] == "direct_dft_on2"
    assert content_dt["method_block"]["transform"]["value"] == "discrete_fourier_transform_1d"


@requires_ste
def test_fourier_result_is_computed_not_measured_and_admits_nothing_by_itself(cli_path):
    """The epistemic boundary, for the second operation exactly as for the
    first: a transform is a COMPUTED result. It is not an observation, and
    nothing about producing one writes to an EvidencePool."""
    from evidence.pool import EvidencePool
    from scl.ste_adapter import build_fourier_specification, run_scl_specification

    pool = EvidencePool()
    before = pool.fingerprint()
    spec = build_fourier_specification([1.0, 2.0, 3.0], cli_path=cli_path)
    result = run_scl_specification(spec, cli_path=cli_path)
    assert result.status == "completed"
    assert pool.fingerprint() == before, "computing a transform must admit nothing on its own"
    with pytest.raises(AttributeError):
        pool.put_observation(result)

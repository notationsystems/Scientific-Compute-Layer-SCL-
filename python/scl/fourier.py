"""The `fourier_transform_1d` operation's wire encoding and frequency-axis
helpers -- the Python half of the contract implemented in
`native/src/fourier.cpp` / `native/src/op_fourier.cpp`.

Mathematical contract (see native/include/scl/fourier.hpp for the
authoritative statement, and docs/SCL_CONTRACT.md for the wire form):

    forward  (direction=+1):  X_k = s * SUM_n x_n exp(-2*pi*i*k*n/N)
    inverse  (direction=-1):  X_k = s * SUM_n x_n exp(+2*pi*i*k*n/N)

    k = 0..N-1 ascending, no fftshift; s is the normalization scale
    (NONE=1, ONE_OVER_N=1/N, ONE_OVER_SQRT_N=1/sqrt(N)); float64
    throughout; any N >= 1; complex in, complex out.

THE TRANSFORM IS NOT A FREQUENCY AXIS. The mathematics above is defined
purely on the sample SEQUENCE -- it never needs to know how fast the
signal was sampled. A physical frequency axis exists only when a caller
supplies the sampling interval, and SCL never invents one: with no Δt the
result is interpretable in bin/index terms only, and `frequency_bins`
returns None rather than silently assuming Δt = 1. See
`FourierMethodBlock` (scl/method_block.py) for how that presence/absence
is recorded explicitly rather than implied.
"""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

from .errors import SCLProtocolError

#: Transform direction. FORWARD carries the negative exponent.
FORWARD = 1
INVERSE = -1

#: Normalization conventions. NONE reproduces the bare summation in the
#: module docstring; ONE_OVER_SQRT_N is the unitary convention.
NORMALIZATION_NONE = 0
NORMALIZATION_ONE_OVER_N = 1
NORMALIZATION_ONE_OVER_SQRT_N = 2

_NORMALIZATION_NAMES = {
    NORMALIZATION_NONE: "none",
    NORMALIZATION_ONE_OVER_N: "one_over_n",
    NORMALIZATION_ONE_OVER_SQRT_N: "one_over_sqrt_n",
}
_DIRECTION_NAMES = {FORWARD: "forward", INVERSE: "inverse"}


def encode_fourier_configuration(
    direction: int = FORWARD,
    normalization: int = NORMALIZATION_NONE,
    sample_spacing_seconds: Optional[float] = None,
) -> bytes:
    """Canonical 24-byte configuration. Matches
    native/src/op_fourier.cpp::decode_configuration exactly:
    `int32 direction | int32 normalization | int32 has_sample_spacing |
     int32 reserved | float64 sample_spacing_seconds`.

    `sample_spacing_seconds=None` means genuinely absent -- the encoded
    has_sample_spacing flag is 0 and the float64 slot is 0.0 and ignored.
    Note this is still part of the configuration bytes, so Δt (or its
    absence) participates in `parameters_identity` exactly like every
    other parameter."""
    has_spacing = 0 if sample_spacing_seconds is None else 1
    spacing_value = 0.0 if sample_spacing_seconds is None else float(sample_spacing_seconds)
    return struct.pack("<iiiid", direction, normalization, has_spacing, 0, spacing_value)


def decode_fourier_configuration(parameters: bytes) -> Tuple[int, int, Optional[float]]:
    """Inverse of `encode_fourier_configuration`:
    `(direction, normalization, sample_spacing_seconds_or_None)`."""
    if len(parameters) != 24:
        raise SCLProtocolError(f"malformed fourier_transform_1d configuration: {len(parameters)} bytes")
    direction, normalization, has_spacing, reserved, spacing = struct.unpack("<iiiid", parameters)
    if reserved != 0:
        raise SCLProtocolError(f"fourier_transform_1d configuration reserved field must be 0, got {reserved}")
    return direction, normalization, (spacing if has_spacing == 1 else None)


def encode_complex_signal(samples: Sequence[complex]) -> bytes:
    """Canonical input encoding: N * 16 bytes, each `(real f64, imag f64)`
    little-endian. A real-valued signal is supplied with zero imaginary
    parts -- `encode_real_signal` below is the convenience for that."""
    out = b""
    for sample in samples:
        value = complex(sample)
        out += struct.pack("<dd", value.real, value.imag)
    return out


def encode_real_signal(samples: Sequence[float]) -> bytes:
    """Convenience for the common real-valued case."""
    return encode_complex_signal([complex(float(s), 0.0) for s in samples])


def decode_complex_spectrum(output: bytes) -> List[complex]:
    """Inverse of native/src/op_fourier.cpp::encode_spectrum."""
    if len(output) % 16 != 0:
        raise SCLProtocolError(f"malformed fourier_transform_1d output: {len(output)} bytes")
    spectrum: List[complex] = []
    for offset in range(0, len(output), 16):
        real, imag = struct.unpack_from("<dd", output, offset)
        spectrum.append(complex(real, imag))
    return spectrum


def frequency_bins(n: int, sample_spacing_seconds: Optional[float]) -> Optional[List[float]]:
    """The physical frequency coordinate of each bin, or **None** when no
    sampling interval was supplied.

    Returning None is the point: with no Δt there is no physical frequency
    axis, and inventing one (by assuming Δt = 1) would silently fabricate
    scientific metadata. Callers that get None have bin indices 0..N-1 and
    nothing more -- which is exactly what the mathematics gives them.

    When Δt IS supplied, the standard two-sided convention:

        f_k = k / (N*Δt)          for k <= N/2
        f_k = (k - N) / (N*Δt)    for k >  N/2   (negative frequencies)

    so bins above Nyquist read as the negative frequencies they represent
    rather than as aliased positive ones."""
    if sample_spacing_seconds is None:
        return None
    if n <= 0:
        raise SCLProtocolError(f"frequency_bins requires n >= 1, got {n}")
    if not (sample_spacing_seconds > 0.0):
        raise SCLProtocolError(
            f"frequency_bins requires a positive sample spacing, got {sample_spacing_seconds}"
        )
    span = n * sample_spacing_seconds
    return [((k if k * 2 <= n else k - n) / span) for k in range(n)]


def normalization_name(normalization: int) -> str:
    return _NORMALIZATION_NAMES.get(normalization, f"unknown({normalization})")


def direction_name(direction: int) -> str:
    return _DIRECTION_NAMES.get(direction, f"unknown({direction})")


def spectral_power_total(spectrum: Sequence[complex]) -> float:
    """SUM_k |X_k|^2 -- a derived scalar summary of a spectrum, used where a
    single number is required (see scl.ste_adapter.interpret_fourier_result).
    It is a FUNCTION OF the transform, never a substitute for it: the full
    spectrum remains the actual result."""
    return sum((value.real * value.real + value.imag * value.imag) for value in spectrum)

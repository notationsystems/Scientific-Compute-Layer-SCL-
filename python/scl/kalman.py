"""Encoders for the `kalman_filter_linear` operation.

THE THREE-WAY SPLIT, restated here because the third axis is new and is
the one most easily collapsed at a call site:

  * F, H, Q, R and both tolerances are PARAMETERS -- how to filter.
  * the measurement stream, x0 and P0 are INPUTS -- what was measured, and
    what this run started from.
  * the PROVENANCE of each noise matrix is a third question, orthogonal to
    both: does a measurement stand behind it? Q's answer is always
    `asserted`, because there is no measurement of process noise. R's may
    be `supplied`, and then it carries a source identity.

The discriminant lives in the CONFIGURATION so it reaches
`parameters_identity`; the source identity it guards lives in the INPUT,
because an identity of a supplied artifact is a reference to data.

ORDER IS SIGNIFICANT. Unlike `least_squares`, whose modality explicitly
does not require ordering, two identical measurement multisets in
different orders are DIFFERENT inputs here.
"""

from __future__ import annotations

import struct
from typing import Optional, Sequence

ASSERTED = 0
SUPPLIED = 1

#: Both change which inputs are accepted, so both change the answer and
#: both participate. Explicit at every call site rather than defaulted
#: silently, for the same reason as the least-squares rank tolerance.
DEFAULT_SYMMETRY_TOLERANCE = 1e-10
DEFAULT_PSD_TOLERANCE = 1e-10


def _flatten(matrix: Sequence[Sequence[float]], rows: int, cols: int, name: str):
    if len(matrix) != rows:
        raise ValueError(f"{name} must have {rows} rows, got {len(matrix)}")
    out = []
    for row in matrix:
        if len(row) != cols:
            raise ValueError(f"{name} rows must have {cols} entries, got {len(row)}")
        out.extend(float(v) for v in row)
    return out


def encode_kalman_configuration(transition: Sequence[Sequence[float]],
                                observation: Sequence[Sequence[float]],
                                process_noise: Sequence[Sequence[float]],
                                measurement_noise: Sequence[Sequence[float]],
                                measurement_noise_provenance: int = ASSERTED,
                                symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
                                psd_tolerance: float = DEFAULT_PSD_TOLERANCE) -> bytes:
    """Note there is no `process_noise_provenance` argument.

    Not an omission and not a default: Q is always `asserted`, so offering
    the choice at the call site would offer a value the operation refuses.
    A parameter whose only legal value is one value is not a parameter."""
    n = len(transition)
    m = len(observation)
    header = struct.pack("<iiiiii", n, m, ASSERTED, int(measurement_noise_provenance),
                         0, 0)
    tolerances = struct.pack("<dd", float(symmetry_tolerance), float(psd_tolerance))
    body = (_flatten(transition, n, n, "transition F")
            + _flatten(observation, m, n, "observation H")
            + _flatten(process_noise, n, n, "process noise Q")
            + _flatten(measurement_noise, m, m, "measurement noise R"))
    return header + tolerances + struct.pack(f"<{len(body)}d", *body)


def encode_kalman_input(initial_state: Sequence[float],
                        initial_covariance: Sequence[Sequence[float]],
                        measurements: Sequence[Sequence[float]],
                        source_identity: Optional[str] = None) -> bytes:
    n = len(initial_state)
    steps = len(measurements)
    if steps == 0:
        raise ValueError("kalman_filter_linear requires at least one measurement step")
    m = len(measurements[0])
    identity = (source_identity or "").encode("utf-8")
    values = ([float(v) for v in initial_state]
              + _flatten(initial_covariance, n, n, "initial covariance P0")
              + _flatten(measurements, steps, m, "measurements"))
    return (struct.pack("<ii", steps, len(identity))
            + struct.pack(f"<{len(values)}d", *values)
            + identity)

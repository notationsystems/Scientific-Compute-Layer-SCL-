"""Encoders and decoders for the `least_squares` operation.

THE PARAMETER / INPUT SPLIT, restated here because it is the thing most
easily got wrong at a call site:

  * `rank_tolerance` and `weighted` are PARAMETERS -- how to fit.
  * `X`, `y` and the WEIGHTS are INPUTS -- what was measured.

Weights are per-observation data with one entry per row; they change when
the data changes. Carrying them in the configuration would mint a new
`parameters_identity` for every dataset and make fits incomparable across
data, which is the failure the state-estimation observations hit once.
"""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence

#: Relative singular-value cutoff. Explicit at every call site rather than
#: defaulted silently: it changes the answer for a rank-deficient system.
DEFAULT_RANK_TOLERANCE = 1e-12


def encode_least_squares_configuration(rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
                                        weighted: bool = False) -> bytes:
    return struct.pack("<iiiid", 1 if weighted else 0, 0, 0, 0, float(rank_tolerance))


def encode_least_squares_input(design: Sequence[Sequence[float]],
                               response: Sequence[float],
                               weights: Optional[Sequence[float]] = None) -> bytes:
    n_rows = len(design)
    if n_rows == 0:
        raise ValueError("design matrix has no rows")
    n_cols = len(design[0])
    if any(len(row) != n_cols for row in design):
        raise ValueError("design matrix rows have differing lengths")
    if len(response) != n_rows:
        raise ValueError("response length does not match the number of rows")
    if weights is not None and len(weights) != n_rows:
        raise ValueError("weights length does not match the number of rows")

    out = bytearray(struct.pack("<ii", n_rows, n_cols))
    for row in design:
        for value in row:
            out += struct.pack("<d", float(value))
    for value in response:
        out += struct.pack("<d", float(value))
    if weights is not None:
        for value in weights:
            out += struct.pack("<d", float(value))
    return bytes(out)


def decode_least_squares_output(payload: bytes) -> List[float]:
    if len(payload) % 8 != 0:
        raise ValueError("coefficient payload is not a whole number of float64 values")
    return [struct.unpack_from("<d", payload, offset)[0]
            for offset in range(0, len(payload), 8)]


def residuals(design: Sequence[Sequence[float]], response: Sequence[float],
              coefficients: Sequence[float]) -> List[float]:
    """r = y - X b, computed here rather than returned by the operation:
    a caller that wants to check the fit should be able to derive the
    check from the coefficients, not be handed the answer."""
    return [y - sum(x * b for x, b in zip(row, coefficients))
            for row, y in zip(design, response)]


def normal_equation_residual(design: Sequence[Sequence[float]],
                             response: Sequence[float],
                             coefficients: Sequence[float]) -> List[float]:
    """X^T r -- zero (to tolerance) exactly when the fit is the least-squares
    solution. This is the orthogonality condition, and it is the strongest
    check available that does not need a second implementation."""
    r = residuals(design, response, coefficients)
    n_cols = len(design[0])
    return [sum(design[i][j] * r[i] for i in range(len(design))) for j in range(n_cols)]

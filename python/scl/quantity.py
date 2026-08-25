"""A typed numeric quantity, for shaping values inside STE's existing
`evidence.types.Observation.content: Mapping[str, object]` field.

Repository fact this module responds to (verified by direct inspection of
the real STE checkout, not inferred): there is no `Quantity`/`unit`/
`uncertainty`/`uncertainty_kind` representation anywhere in that
repository today (`evidence/`, `materials/`, `core/`) -- grepping the
whole tree for those tokens turns up only prose. A numeric scientific
result therefore has nowhere upstream to declare its unit or uncertainty
honestly; `Observation.content` is a plain open `Mapping[str, object]`,
so nothing stops a caller from putting a bare float there.

This is NOT a second evidence/quantity system. `Quantity` is a plain,
JSON-serializable shape SCL uses to fill values *inside* STE's own,
already-open `content` mapping -- it changes no STE type, adds no STE
field, and is not admitted, stored, or identified by anything of its own.
If STE's own schema later grows a first-class typed-quantity field, this
shape is deliberately simple enough to drop straight into it; until then
it lives here, as SCL's own boundary-shaping convention, documented as
exactly that in docs/SCL_ARCHITECTURE.md's Phase 2 addendum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: The four uncertainty kinds the Phase 2 addendum specifies. `absent` is
#: a REQUIRED explicit value, distinct from simply omitting uncertainty --
#: it distinguishes "the source/computation genuinely carries no
#: uncertainty estimate" from "uncertainty was lost/never wired up."
UNCERTAINTY_KINDS = ("stated", "estimated", "propagated", "absent")


@dataclass(frozen=True)
class Quantity:
    """One numeric value with its unit and uncertainty semantics made
    explicit. Never construct one with a fabricated uncertainty: if the
    producing computation did not estimate uncertainty, uncertainty_kind
    must be "absent" and uncertainty must be None -- the two are
    validated together below, not independently.
    """

    value: float
    unit: str
    uncertainty: Optional[float]
    uncertainty_kind: str

    def __post_init__(self) -> None:
        if self.uncertainty_kind not in UNCERTAINTY_KINDS:
            raise ValueError(
                f"uncertainty_kind must be one of {UNCERTAINTY_KINDS}, got {self.uncertainty_kind!r}"
            )
        if self.uncertainty_kind == "absent" and self.uncertainty is not None:
            raise ValueError(
                "uncertainty_kind='absent' requires uncertainty=None -- "
                "a numeric uncertainty with kind 'absent' is a contradiction, not a fabrication guard"
            )
        if self.uncertainty_kind != "absent" and self.uncertainty is None:
            raise ValueError(
                f"uncertainty_kind={self.uncertainty_kind!r} requires a numeric uncertainty value; "
                "if none is genuinely available, use uncertainty_kind='absent' instead of inventing one"
            )
        if not self.unit:
            raise ValueError("unit must be a non-empty string (use an explicit reduced-units label, never '')")

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "uncertainty_kind": self.uncertainty_kind,
        }


def absent_uncertainty(value: float, unit: str) -> Quantity:
    """Convenience for the common honest case: a deterministic numerical
    computation that never estimated uncertainty at all (Phase 1's LJ
    kernel: double-precision arithmetic, no stochastic sampling, no
    error propagation implemented -- 'absent' is the true state, not a
    default filled in for convenience)."""
    return Quantity(value=value, unit=unit, uncertainty=None, uncertainty_kind="absent")

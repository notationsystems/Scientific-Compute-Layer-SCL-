"""Domain method metadata for SCL's Lennard-Jones workload.

Repository fact this responds to: no domain-method-block schema exists
anywhere in STE today (grep for "force_field"/"ensemble"/"ambiguity
between potential and model" etc. across evidence/materials/core: no
hits). The Phase 2 addendum lists an MD method block (force_field,
ensemble, timestep, thermostat, barostat, ...) as the PROJECT-WIDE
convention for molecular-dynamics *trajectory* workloads -- Phase 1's LJ
kernel is a single-point, static pairwise energy/force EVALUATION, not an
MD trajectory: it has no integrator, no ensemble, no thermostat, and no
timestep, and inventing values for them would misrepresent what was
computed. Per the addendum's own instruction ("required when
scientifically applicable; explicitly absent when not applicable"),
every field below is either a real, applicable value or an explicit
`applicable: False` marker with a stated reason -- never omitted
silently and never fabricated.

Like quantity.py, this is a shape for values placed inside STE's already-
open `Observation.content`, not a new STE schema or a competing method-
block registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _applicable(value: object) -> Mapping[str, object]:
    return {"applicable": True, "value": value}


def _not_applicable(reason: str) -> Mapping[str, object]:
    return {"applicable": False, "reason": reason}


@dataclass(frozen=True)
class LJMethodBlock:
    """The complete method-metadata block for one
    `lj_pairwise_energy_forces` computation. Field-by-field applicability
    determined against the actual computation `native/src/lj_pairwise.cpp`
    performs -- see each `_not_applicable` reason for why."""

    potential: str  # e.g. "lennard_jones_12_6"
    potential_version: str  # scl_cli's own version string (native/include/scl/version.hpp)
    cutoff: float
    cutoff_unit: str
    boundary_conditions: str  # "open" (no periodic images -- see native/src/lj_pairwise.cpp)
    numerical_precision: str  # "float64"
    n_particles: int
    backend: str  # "cpu" | "cuda"

    def to_dict(self) -> Mapping[str, object]:
        return {
            "potential": _applicable(self.potential),
            "potential_version": _applicable(self.potential_version),
            "cutoff": _applicable({"value": self.cutoff, "unit": self.cutoff_unit}),
            "boundary_conditions": _applicable(self.boundary_conditions),
            "numerical_precision": _applicable(self.numerical_precision),
            "system_definition": _applicable({"n_particles": self.n_particles}),
            "backend": _applicable(self.backend),
            # Genuinely not applicable to a static pairwise evaluation --
            # this is not an MD trajectory (native/src/lj_pairwise.cpp has
            # no time-stepping loop, no velocity state, no coupling to a
            # thermostat/barostat at all).
            "integration_configuration": _not_applicable(
                "single-point evaluation: no integrator exists in this computation"
            ),
            "initialization": _not_applicable(
                "positions are caller-supplied directly; no initialization procedure runs"
            ),
            "temperature": _not_applicable(
                "no thermostat or canonical-ensemble coupling exists in this computation"
            ),
            "timestep": _not_applicable(
                "no time integration occurs; this is one static energy/force evaluation"
            ),
            "equilibration": _not_applicable("not applicable: no dynamics are run"),
            "sampling_time": _not_applicable("not applicable: no trajectory is sampled"),
            "thermostat": _not_applicable("not applicable: no ensemble is simulated"),
            "barostat": _not_applicable("not applicable: no pressure coupling exists"),
            "convergence_criteria": _not_applicable(
                "not applicable: this is a closed-form pairwise sum, not an iterative solver"
            ),
        }


def lj_method_block_for(
    cutoff: float,
    n_particles: int,
    backend: str,
    kernel_version: str,
) -> LJMethodBlock:
    return LJMethodBlock(
        potential="lennard_jones_12_6",
        potential_version=kernel_version,
        cutoff=cutoff,
        cutoff_unit="sigma",
        boundary_conditions="open",
        numerical_precision="float64",
        n_particles=n_particles,
        backend=backend,
    )

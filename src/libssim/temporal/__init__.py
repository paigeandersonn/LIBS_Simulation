"""
libssim.temporal
================
Temporal evolution layer (Phase 4): time-dependent plasma parameters
and detector-gate integration.

Implements the t_delay / t_gate observation model of Herrera (2008),
pp. 46-47: plasma parameters evolve after the laser pulse and the
recorded signal is the emission integrated over the detector gate.
Default closed-form decays are documented simplifications of the
thesis ODE model (Eq. 5-35 / App. B-C) following the time-resolved
LIBS literature (Aguilera & Aragon 2004; Cristoforetti et al. 2004 =
thesis ref [171]); any user callable t -> value can replace them.

Provides:
- `TimeProfile` contract + `PlasmaEvolution` ABC (`base`)
- `Constant`, `ExponentialDecay`, `PowerLawDecay` profiles and the
  `UniformPlasmaEvolution` (primary), `CustomEvolution` (escape hatch)
  and `ExpandingOnionEvolution` (advanced, App. B self-similar)
  evolutions (`decay_models`)
- `GateIntegrator` — time-resolved snapshots and gate-integrated
  spectra (`integrator`)

Strict SI units throughout (development_rules.md).
"""

from .base import PlasmaEvolution, TimeProfile
from .decay_models import (
    Constant,
    CustomEvolution,
    ExpandingOnionEvolution,
    ExponentialDecay,
    PowerLawDecay,
    UniformPlasmaEvolution,
)
from .integrator import GateIntegrator

__all__ = [
    "TimeProfile",
    "PlasmaEvolution",
    "Constant",
    "ExponentialDecay",
    "PowerLawDecay",
    "UniformPlasmaEvolution",
    "CustomEvolution",
    "ExpandingOnionEvolution",
    "GateIntegrator",
]

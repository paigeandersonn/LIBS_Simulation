"""
libssim.temporal.base
=====================
Contracts for time-dependent plasma descriptions (Phase 4).

Physical Context (Herrera 2008)
-------------------------------
Time-resolved LIBS detection is parameterized by the delay time
t_delay — "the time between the initiation of the laser pulse and the
beginning of the gate width" — and the gate width t_gate, "the
integration window during which the plasma emission is recorded"
(pp. 46-47; symbols p. 23). During that window the plasma cools and
expands: the thesis evolves it with an energy-balance ODE for T
(Eq. 5-35, p. 115; Appendix C) and a self-similar expansion for the
densities (Appendix B, Eqs. B-1..B-9, pp. 267-269).

This module defines the two contracts the rest of Phase 4 builds on:

- `TimeProfile` — any callable t[s] -> value describing one scalar
  parameter's evolution (temperature, density, radius, ...). Simple
  closed-form defaults live in decay_models.py; a numerical solution
  of the thesis ODE system can be plugged in through the same type.
- `PlasmaEvolution` — maps a time to a `PlasmaGeometry` snapshot, so
  the Phase 3 transport layer (and through it all of Phase 2) is
  reused unchanged at every instant: evolution produces geometries,
  transport consumes them.

Quasi-static assumption (documented)
------------------------------------
Evaluating the emission at fixed times treats the plasma as frozen at
each instant — valid when radiative and collisional relaxation are
fast against the hydrodynamic evolution, the same stationarity
argument used for the transfer equation itself ("the rates of gas
dynamic processes are much slower than the rates of radiative
processes", p. 117).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. pp. 46-47
(t_delay/t_gate); Eq. 5-35 p. 115; App. B pp. 267-269; App. C
pp. 270-273.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..transport.base import PlasmaGeometry

#: A scalar plasma parameter as a function of time: t [s] -> value.
#: Implementations must accept any t >= 0 they are evaluated at and
#: return a finite float; see decay_models for frozen defaults.
TimeProfile = Callable[[float], float]


def validate_time(time_s: float) -> float:
    """Return the time as float, enforcing finite and >= 0 s."""
    t = float(time_s)
    if not (np.isfinite(t) and t >= 0.0):
        raise ValueError("time_s must be finite and >= 0")
    return t


def evaluate_profile(profile: TimeProfile, time_s: float, name: str) -> float:
    """
    Evaluate a `TimeProfile`, enforcing a finite scalar result.

    Centralizes the error message so user-supplied callables that
    return NaN/inf (or arrays) fail loudly at the evaluation site.
    """
    value = float(profile(time_s))
    if not np.isfinite(value):
        raise ValueError(
            f"{name} profile returned a non-finite value at t = {time_s:.6g} s"
        )
    return value


class PlasmaEvolution(ABC):
    """
    Abstract time-dependent plasma: a `PlasmaGeometry` per instant.

    The single contract keeps Phases 2-3 untouched: whatever the
    temporal model (closed-form decays, the thesis App. B/C system, or
    user code), each evaluated instant is an ordinary geometry that the
    existing transfer solver renders (quasi-static assumption, module
    docstring).
    """

    @abstractmethod
    def geometry_at(self, time_s: float) -> PlasmaGeometry:
        """
        Plasma snapshot at the given time since the laser pulse.

        Parameters
        ----------
        time_s : float
            Time since plasma initiation (s, >= 0); the t_delay /
            t_gate clock of pp. 46-47 (Herrera 2008).

        Returns
        -------
        PlasmaGeometry
            Frozen geometry describing the plasma at that instant.
        """

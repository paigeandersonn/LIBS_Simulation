r"""Immutable plasma state container (Phase 0 core).

This module defines the fundamental, frozen representation of a laser-induced
plasma (LIP) at a given instant and spatial scale. It is deliberately minimal
and physics-agnostic so that downstream modules (Saha ionization, Boltzmann
populations, radiative transfer, instrumental convolution) can derive all
derived quantities.

Physical context (aligned with Herrera 2008 PhD thesis)
-------------------------------------------------------
The fields directly mirror the key variables used in:

- CF-LIBS (Ch. 5): single $T$ (LTE), $n_e$ (Stark or Saha-Boltzmann),
  relative concentrations $C^{s}$, assumption of stoichiometric ablation
  and optically thin plasma.
- MC-LIBS Monte Carlo simulated annealing (Ch. 5 pp. 112–121, App. B–D):
  initial total number density $n_j(r,t)$ per constituent $j$, plasma
  temperature $T$, outer radius $R$, time $t$, and composition (relative
  fractions normalized to 1). The thesis solves the inverse problem by
  optimizing these initials so that synthetic spectra (including
  Doppler + Stark + self-absorption line profiles) match experimental
  ones.

All quantities are stored in strict SI units. Composition fractions are
automatically normalized to sum exactly to 1.0 (with floating-point tolerance
in tests). The dataclass is immutable (frozen + slots) to guarantee
thread-safety and reproducibility during Monte Carlo sampling or optimization
loops (Phase 5).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy... University of Florida. Chapters 3 (line profiles, self-absorption),
5 (CF-LIBS & MC-LIBS hypotheses and equations), 6–7 (experimental validation on
Al alloys, brass, soils), and Appendices B–D (explicit formulas for n_tot, T,
n_atom, n_ion, n_e).

See also:
- n_j(r,t) total number density of constituent j (p. 22, App. B)
- Saha-Boltzmann plots and LTE validation (Figs. 5-4, 6-5, 6-19)
- Relative concentration outputs vs. certified values (Tables 6-9..6-35, 7-7..7-33)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class PlasmaState:
    r"""
    Frozen container for the minimal set of plasma parameters needed to
    initialize a forward LIBS simulation or MC-LIBS inverse problem.

    All fields are validated in ``__post_init__``. Composition is
    normalized in place using ``object.__setattr__`` (required for a
    frozen dataclass).

    Attributes
    ----------
    temperature_K : float
        Plasma temperature in Kelvin. Under the LTE hypothesis central
        to both CF-LIBS and MC-LIBS in Herrera (2008), this single
        value represents $T_e = T_{\mathrm{exc}} = T_{\mathrm{ion}}$.
        Used for Boltzmann level populations, the Saha ionization
        balance, the Planck blackbody continuum and Doppler width
        calculations.
    electron_density_m3 : float
        Electron number density $n_e$ (m$^{-3}$). Determined
        experimentally via Stark broadening of H$\alpha$ (Ch. 3, 5,
        6) or calculated via the Saha solver in MC-LIBS (Ch. 5,
        App. D). Must satisfy
        $0 \le n_e \le$ `total_density_m3`.
    total_density_m3 : float
        Total particle number density (atoms + singly-charged ions +
        electrons) in m$^{-3}$. Corresponds to $n_{\mathrm{tot}}$ or
        $\sum_j n_j(r,t)$ in the thesis radiative plasma expansion
        model (Ch. 5, App. B); depends on radial coordinate and time.
    radius_m : float
        Characteristic / outer plasma radius $R$ (m). Used for the
        spherical onion geometry (Phase 3) and the radial expansion
        velocity model $u(r,t)$ (App. A).
    time_s : float
        Time since laser ablation (reference epoch) in seconds.
        Enables temporal integration over experimental gate width
        $t_{\mathrm{gate}}$ and delay times $t_{\mathrm{delay}}$
        (Ch. 6–7 temporally-resolved studies).
    composition : Mapping[str, float]
        Relative abundance fractions for each species (element or
        isotope label). Keys: ``"Ce"``, ``"Al"``, ``"Pu-239"``, etc.
        Values are automatically normalized so $\sum$ fractions = 1.0
        exactly (within float precision). This matches the "relative
        concentration values" computed and validated against
        certified standards throughout Chapters 6 and 7.

    Raises
    ------
    ValueError
        If any physical invariant is violated (negative temperature,
        $n_e > n_{\mathrm{tot}}$, non-positive composition sum,
        etc.).
    """

    temperature_K: float
    electron_density_m3: float
    total_density_m3: float
    radius_m: float
    time_s: float
    composition: Mapping[str, float]

    def __post_init__(self) -> None:
        # --- Scalar physical validations (SI units, physical bounds) ---
        if not (self.temperature_K > 0.0):
            raise ValueError(
                "temperature_K must be > 0 K (LTE plasma temperature)"
            )
        if not (self.electron_density_m3 >= 0.0):
            raise ValueError("electron_density_m3 must be >= 0")
        if not (self.total_density_m3 >= 0.0):
            raise ValueError("total_density_m3 must be >= 0")
        if self.electron_density_m3 > self.total_density_m3:
            raise ValueError(
                "electron_density_m3 cannot exceed total_density_m3 "
                "(total particle density includes atoms + ions + electrons; "
                "see Herrera App. D mass balance)"
            )
        if not (self.radius_m > 0.0):
            raise ValueError("radius_m must be > 0 (outer plasma boundary R)")
        if not (self.time_s >= 0.0):
            raise ValueError("time_s must be >= 0")

        # --- Composition normalization (stoichiometric ablation) ---
        comp = dict(self.composition)  # shallow copy
        if len(comp) == 0:
            object.__setattr__(self, "composition", {})
            return

        total = float(sum(comp.values()))
        if total <= 0.0:
            raise ValueError(
                "composition fractions must sum to a positive value "
                "(relative concentrations C^s in CF/MC-LIBS)"
            )
        for k, v in list(comp.items()):
            if v < 0.0:
                raise ValueError(f"composition[{k!r}] must be >= 0")

        comp_norm = {k: float(v) / total for k, v in comp.items()}
        object.__setattr__(self, "composition", comp_norm)

    # --- Convenience read-only properties ---
    @property
    def n_e(self) -> float:
        """Electron number density (alias, matches thesis n_e)."""
        return self.electron_density_m3

    @property
    def n_tot(self) -> float:
        """Total particle number density (alias, matches thesis n_tot)."""
        return self.total_density_m3

    @property
    def species(self) -> list[str]:
        """List of species labels present in the plasma."""
        return list(self.composition.keys())

    def __repr__(self) -> str:
        comp_str = ", ".join(f"{k}={v:.4f}" for k, v in self.composition.items())
        return (
            f"PlasmaState(T={self.temperature_K:.1f} K, "
            f"n_e={self.n_e:.2e} m^{-3}, n_tot={self.n_tot:.2e} m^{-3}, "
            f"R={self.radius_m*1000:.2f} mm, t={self.time_s*1e6:.2f} µs, "
            f"comp=[{comp_str}])"
        )
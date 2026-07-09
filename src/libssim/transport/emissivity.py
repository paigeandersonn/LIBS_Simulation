"""
libssim.transport.emissivity
============================
Per-zone spectral emission and absorption from Phase 2 physics
(Phase 3 adapter layer).

Physical Context (Herrera 2008)
-------------------------------
The radiative transfer solution (Eq. 5-48, p. 119) needs, at every
point of the plasma, the total absorption coefficient of Eq. 5-49,
p. 119,

    kappa'_nu = kappa_ff + kappa_fb + kappa_bb,

and — under LTE — the matching source term kappa'_nu * I_nu^b of the
RTE (Eq. 5-44, p. 117), i.e. the emission coefficient. Because the
onion zones are locally uniform, both are exactly the Phase 2
single-point quantities, evaluated once per zone:

- Saha stage densities per element: Eqs. 5-38/5-39/5-40, p. 116
  (`SahaSolver.balance_from_state`);
- level populations: Eq. 5-1, p. 98 (`boltzmann`);
- line profiles: Voigt with Doppler FWHM (Eq. 3-1, p. 50) and
  quadratic-Stark FWHM (Eq. 3-8, p. 53 / Eq. 5-17, p. 106)
  (`line_profiles`);
- line emission and bound-bound absorption: Eq. 5-8, pp. 103-104 and
  Eq. 5-52, p. 120 (`emission`);
- free-free + free-bound continuum: Eqs. 5-50/5-51, pp. 119-120
  (`continuum`).

`LTESpectralModel` composes those calls and returns, per zone, the
arrays (epsilon_nu, kappa'_nu) on a fixed wavelength grid — transport
adds no new local physics (architecture.md layering).

Units and Conventions
---------------------
- Wavelength grid in meters (SI), strictly increasing.
- epsilon_nu: W m^-3 Hz^-1 sr^-1 (per-frequency emission coefficient,
  matching Phase 2); kappa'_nu: m^-1. Radiance is converted to
  per-wavelength only when packing a `Spectrum` (radiative.py).
- Line profiles are evaluated in the wavelength domain (widths from
  Ch. 3) and converted to per-Hz with the local Jacobian
  P_nu = P_lambda * lambda^2 / c (narrow-line, consistent with
  line_profiles.fwhm conversions).

Implementation Decisions (documented per development_rules.md)
--------------------------------------------------------------
- Two-stage model: transitions must be stage I or II (matching
  `SahaSolver`); stage III lines are rejected at construction.
- `Transition.stark_width` is interpreted as the electron-impact
  half-width w in meters at the reference density 1e22 m^-3 (the SI
  convention of the atomic layer and of Eq. 3-8, p. 53). Lines without
  Stark data get Doppler-only (Gaussian) profiles; empirical
  broadening fallbacks are Phase 6 (implementation_plan.md).
- Ion broadening (alpha) is neglected — "normally neglected due to the
  negligible contribution of ion-broadening under typical LIBS
  conditions" (p. 106, Eq. 5-17).
- Transitions of elements absent from a zone's composition contribute
  nothing to that zone and are skipped.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-1 p. 98;
Eq. 5-8 pp. 103-104; Eqs. 5-36..5-40 p. 116; Eq. 5-44 p. 117;
Eqs. 5-49..5-52 pp. 119-120; Eq. 3-1 p. 50; Eq. 3-8 p. 53.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

from ..atomic.transition import Transition
from ..core.constants import C
from ..core.state import PlasmaState
from ..physics.boltzmann import boltzmann_population_fraction, upper_level_density
from ..physics.continuum import (
    continuum_absorption_coefficient,
    continuum_emission_coefficient,
)
from ..physics.emission import (
    line_absorption_coefficient,
    line_emission_coefficient,
)
from ..physics.line_profiles import (
    doppler_fwhm_m,
    stark_fwhm_m,
    voigt_profile_wavelength_m,
)
from ..physics.saha import SahaSolver
from .base import PlasmaGeometry


@dataclass(frozen=True, eq=False)
class LTESpectralModel:
    """
    Evaluator of per-zone (epsilon_nu, kappa'_nu) on a fixed wavelength
    grid, composing the Phase 2 physics (module docstring).

    Parameters
    ----------
    saha_solver : SahaSolver
        Phase 2 solver; must carry partition data (stages I and II) and
        ionization energies for every element that appears in the
        transitions and zone compositions.
    wavelength_m : array_like of float
        Strictly increasing wavelength grid (m, > 0). Copied and locked
        read-only.
    transitions : tuple of Transition, optional
        Lines to include (may be empty for a continuum-only model).
        Stage I/II only (two-stage Saha model).
    atomic_masses_kg : Mapping[str, float], optional
        Emitter masses per element (kg) for the Doppler width,
        Eq. 3-1, p. 50. Required for every element that has transitions.
    include_continuum : bool, optional
        Add free-free + free-bound continuum (Eqs. 5-50/5-51). Default
        True.
    gaunt_factor, bound_free_correction : float, optional
        G and xi of the continuum formulas (~1, thesis p. 118/120).

    Notes
    -----
    Immutable (frozen); evaluation methods are pure. eq=False: identity
    comparison, consistent with the project's array-holding containers.
    """

    saha_solver: SahaSolver
    wavelength_m: NDArray[np.float64]
    transitions: Tuple[Transition, ...] = ()
    atomic_masses_kg: Mapping[str, float] = field(default_factory=dict)
    include_continuum: bool = True
    gaunt_factor: float = 1.0
    bound_free_correction: float = 1.0

    def __post_init__(self) -> None:
        grid = np.array(self.wavelength_m, dtype=np.float64, copy=True)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("wavelength_m must be a 1-D grid with >= 2 points")
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
            raise ValueError("wavelength_m must be finite and > 0")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("wavelength_m must be strictly increasing")
        grid.setflags(write=False)
        object.__setattr__(self, "wavelength_m", grid)

        transitions = tuple(self.transitions)
        masses = {
            str(el).strip().upper(): float(m)
            for el, m in dict(self.atomic_masses_kg).items()
        }
        for el, m in masses.items():
            if not (m > 0.0 and np.isfinite(m)):
                raise ValueError(f"atomic mass for {el!r} must be > 0 kg")
        for tr in transitions:
            if not isinstance(tr, Transition):
                raise TypeError("transitions must contain Transition objects")
            if tr.ion_stage not in (1, 2):
                raise ValueError(
                    f"{tr!r}: only stage I/II lines are supported (two-stage "
                    "Saha model, Eqs. D-1..D-10 / saha.py docs)"
                )
            if tr.element.strip().upper() not in masses:
                raise ValueError(
                    f"no atomic mass registered for {tr.element!r} "
                    "(needed for the Doppler width, Eq. 3-1)"
                )
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "atomic_masses_kg", masses)
        if not (self.gaunt_factor > 0 and np.isfinite(self.gaunt_factor)):
            raise ValueError("gaunt_factor must be finite and > 0")
        if not (
            self.bound_free_correction > 0
            and np.isfinite(self.bound_free_correction)
        ):
            raise ValueError("bound_free_correction must be finite and > 0")

    # ------------------------------------------------------------------
    @property
    def n_wavelengths(self) -> int:
        """Number of grid points."""
        return int(self.wavelength_m.size)

    # ------------------------------------------------------------------
    def zone_properties(
        self, state: PlasmaState
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        (epsilon_nu, kappa'_nu) for one locally uniform zone.

        Assembles Eq. 5-49, p. 119 (Herrera 2008) at the zone's
        conditions: Saha stage densities -> Boltzmann level populations
        -> Voigt profiles -> line emission/absorption (Eqs. 5-8, 5-52)
        plus the ff+fb continuum (Eqs. 5-50/5-51).

        Parameters
        ----------
        state : PlasmaState
            Zone conditions (T, n_e, composition). `radius_m` is not
            used (geometry owns radii — geometry.py docs).

        Returns
        -------
        (ndarray, ndarray)
            epsilon_nu (W m^-3 Hz^-1 sr^-1) and kappa'_nu (m^-1), each
            of shape (n_wavelengths,).
        """
        T = state.temperature_K
        grid = self.wavelength_m
        epsilon = np.zeros(grid.shape, dtype=np.float64)
        kappa = np.zeros(grid.shape, dtype=np.float64)

        balance = self.saha_solver.balance_from_state(state)
        provider = self.saha_solver.partition_provider

        for tr in self.transitions:
            element = tr.element.strip().upper()
            if element not in {el.upper() for el in balance.species}:
                continue  # element absent from this zone: no contribution
            zone_element = next(
                el for el in balance.species if el.upper() == element
            )
            if tr.ion_stage == 1:
                stage_density = balance.neutral_density_m3[zone_element]
            else:
                stage_density = balance.ion_density_m3[zone_element]
            if stage_density == 0.0:
                continue

            U = provider.partition_function(element, tr.ion_stage, T)
            n_upper = upper_level_density(tr, stage_density, U, T)
            n_lower = stage_density * boltzmann_population_fraction(
                tr.g_lower, tr.energy_lower_ev, U, T
            )

            # Line widths: Doppler (Eq. 3-1) always; Stark (Eq. 3-8,
            # electron-impact term) only when the line carries data.
            fwhm_gauss = float(
                np.asarray(
                    doppler_fwhm_m(
                        tr.wavelength_m, T, self.atomic_masses_kg[element]
                    )
                )
            )
            if tr.stark_width is not None:
                fwhm_lorentz = float(
                    np.asarray(stark_fwhm_m(tr.stark_width, state.n_e))
                )
            else:
                fwhm_lorentz = 0.0  # Doppler-only (documented decision)

            profile_lambda = np.asarray(
                voigt_profile_wavelength_m(
                    grid, tr.wavelength_m, fwhm_gauss, fwhm_lorentz
                )
            )
            # Per-Hz profile via the narrow-line Jacobian |dlambda/dnu|:
            # P_nu = P_lambda * lambda^2 / c.
            profile_nu = profile_lambda * grid**2 / C

            epsilon += np.asarray(
                line_emission_coefficient(tr, n_upper, profile_nu)
            )
            kappa += np.asarray(
                line_absorption_coefficient(tr, n_lower, T, profile_nu)
            )

        if self.include_continuum and state.electron_density_m3 > 0.0:
            frequency = C / grid
            n_ion_total = balance.total_ion_density_m3
            kappa += np.asarray(
                continuum_absorption_coefficient(
                    frequency,
                    T,
                    state.electron_density_m3,
                    n_ion_total,
                    1.0,
                    self.gaunt_factor,
                    self.bound_free_correction,
                )
            )
            epsilon += np.asarray(
                continuum_emission_coefficient(
                    frequency,
                    T,
                    state.electron_density_m3,
                    n_ion_total,
                    1.0,
                    self.gaunt_factor,
                    self.bound_free_correction,
                )
            )

        return epsilon, kappa

    # ------------------------------------------------------------------
    def geometry_properties(
        self, geometry: PlasmaGeometry
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Stacked per-zone properties for a whole geometry.

        Each zone is evaluated once and reused for every line of sight
        (the broadcasting strategy of the Phase 3 plan gap-check).

        Returns
        -------
        (ndarray, ndarray)
            epsilon_nu and kappa'_nu, each of shape
            (n_zones, n_wavelengths), row k belonging to
            geometry.zones[k].
        """
        pairs = [self.zone_properties(zone) for zone in geometry.zones]
        epsilon = np.stack([p[0] for p in pairs])
        kappa = np.stack([p[1] for p in pairs])
        return epsilon, kappa

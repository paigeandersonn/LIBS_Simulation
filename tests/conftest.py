"""
Shared fixtures for the whole libssim test suite.

Toy atomic data only — the element-agnostic physics is exercised with
simple two-level species whose partition functions and populations can
be computed by hand. All fixture objects are frozen/immutable, so
session scope is safe. Constants are exposed through fixtures (never
imported from this module): pytest caches at most one `conftest`
module per basename, so cross-importing per-package conftests is
unreliable.
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.atomic.transition import Transition
from libssim.core.constants import C, EV, H
from libssim.core.state import PlasmaState
from libssim.physics.partition import (
    PartitionFunctionProvider,
    PartitionFunctionTable,
)
from libssim.physics.saha import SahaSolver

#: Temperature grid covering the full test range (K).
_GRID = np.linspace(500.0, 30000.0, 600)

#: Toy bound levels per species: (statistical weights, energies in eV).
_LEVELS = {
    ("Fe", 1): ([9.0, 11.0], [0.0, 1.0]),
    ("Fe", 2): ([10.0, 12.0], [0.0, 1.5]),
    ("Al", 1): ([2.0, 4.0], [0.0, 0.014]),
    ("Al", 2): ([1.0, 5.0], [0.0, 4.6]),
}

#: NIST first ionization potentials (eV).
_IONIZATION_EV = {"Fe": 7.9024, "Al": 5.9858}

#: Atomic masses in kg (u values x 1 u).
_ATOMIC_MASSES_KG = {
    "Fe": 55.845 * 1.66053906660e-27,
    "Al": 26.9815 * 1.66053906660e-27,
}

#: Toy resonance line: lower level at 0 eV, upper at 3 eV, wavelength
#: exactly energy-consistent so Kirchhoff closure holds to machine
#: precision in integration tests.
_RESONANCE_UPPER_EV = 3.0
_RESONANCE_WAVELENGTH_M = H * C / (_RESONANCE_UPPER_EV * EV)


@pytest.fixture(scope="session")
def partition_provider() -> PartitionFunctionProvider:
    """Provider with direct-summation tables for Fe I/II and Al I/II."""
    tables = {
        key: PartitionFunctionTable.from_levels(g, e_ev, _GRID)
        for key, (g, e_ev) in _LEVELS.items()
    }
    return PartitionFunctionProvider(tables=tables)


@pytest.fixture(scope="session")
def saha_solver(partition_provider: PartitionFunctionProvider) -> SahaSolver:
    """Two-stage Saha solver over the toy Fe/Al plasma."""
    return SahaSolver(
        partition_provider=partition_provider,
        ionization_energies_ev=_IONIZATION_EV,
    )


@pytest.fixture(scope="session")
def atomic_masses_kg() -> dict[str, float]:
    """Emitter masses (kg) for Doppler widths, Eq. 3-1."""
    return dict(_ATOMIC_MASSES_KG)


@pytest.fixture(scope="session")
def resonance_wavelength_m() -> float:
    """Center wavelength of the energy-consistent toy resonance line."""
    return _RESONANCE_WAVELENGTH_M


@pytest.fixture(scope="session")
def fe_transition() -> Transition:
    """Fe I 404.58 nm-like line with round-number upper-level data."""
    return Transition(
        element="Fe",
        ion_stage=1,
        wavelength_m=404.58e-9,
        energy_lower_ev=1.485,
        energy_upper_ev=4.549,
        a_ki=8.5e6,
        g_lower=9,
        g_upper=9,
    )


@pytest.fixture(scope="session")
def resonance_transition() -> Transition:
    """Energy-consistent Fe I resonance line (0 -> 3 eV, ~413.3 nm)."""
    return Transition(
        element="Fe",
        ion_stage=1,
        wavelength_m=_RESONANCE_WAVELENGTH_M,
        energy_lower_ev=0.0,
        energy_upper_ev=_RESONANCE_UPPER_EV,
        a_ki=5.0e7,
        g_lower=9,
        g_upper=11,
    )


@pytest.fixture(scope="session")
def make_state():
    """Factory fixture: single-element zone state with given heavy density."""

    def _make(
        temperature_K: float,
        heavy_density_m3: float,
        electron_density_m3: float = 1.0e20,
        element: str = "Fe",
    ) -> PlasmaState:
        return PlasmaState(
            temperature_K=temperature_K,
            electron_density_m3=electron_density_m3,
            total_density_m3=heavy_density_m3 + electron_density_m3,
            radius_m=1.0e-3,  # ignored by transport (geometry owns radii)
            time_s=0.0,
            composition={element: 1.0},
        )

    return _make

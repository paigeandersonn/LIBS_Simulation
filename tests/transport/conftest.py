"""
Shared fixtures for the Phase 3 transport tests.

Mirrors the toy Fe/Al atomic data of tests/physics/conftest.py (test
packages are independent, so the small fixture block is duplicated
rather than imported across packages). The toy transition is built
energy-consistent (E_upper - E_lower = h*c/lambda0) so that Kirchhoff
closure tests hold exactly.
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

GRID = np.linspace(500.0, 30000.0, 600)

LEVELS = {
    ("Fe", 1): ([9.0, 11.0], [0.0, 1.0]),
    ("Fe", 2): ([10.0, 12.0], [0.0, 1.5]),
    ("Al", 1): ([2.0, 4.0], [0.0, 0.014]),
    ("Al", 2): ([1.0, 5.0], [0.0, 4.6]),
}

IONIZATION_EV = {"Fe": 7.9024, "Al": 5.9858}

#: Atomic masses in kg (CODATA-consistent u values x 1 u).
ATOMIC_MASSES_KG = {
    "Fe": 55.845 * 1.66053906660e-27,
    "Al": 26.9815 * 1.66053906660e-27,
}

#: Toy resonance line: lower level at 0 eV, upper at 3 eV, wavelength
#: chosen exactly energy-consistent so eps/kappa = B_nu holds to
#: machine precision in integration tests.
RESONANCE_UPPER_EV = 3.0
RESONANCE_WAVELENGTH_M = H * C / (RESONANCE_UPPER_EV * EV)


@pytest.fixture(scope="session")
def partition_provider() -> PartitionFunctionProvider:
    tables = {
        key: PartitionFunctionTable.from_levels(g, e_ev, GRID)
        for key, (g, e_ev) in LEVELS.items()
    }
    return PartitionFunctionProvider(tables=tables)


@pytest.fixture(scope="session")
def saha_solver(partition_provider: PartitionFunctionProvider) -> SahaSolver:
    return SahaSolver(
        partition_provider=partition_provider,
        ionization_energies_ev=IONIZATION_EV,
    )


@pytest.fixture(scope="session")
def resonance_transition() -> Transition:
    """Energy-consistent Fe I resonance line (0 -> 3 eV, ~413.3 nm)."""
    return Transition(
        element="Fe",
        ion_stage=1,
        wavelength_m=RESONANCE_WAVELENGTH_M,
        energy_lower_ev=0.0,
        energy_upper_ev=RESONANCE_UPPER_EV,
        a_ki=5.0e7,
        g_lower=9,
        g_upper=11,
    )


@pytest.fixture(scope="session")
def make_state():
    """Factory fixture: single-element zone state with given heavy density.

    A fixture (rather than an importable helper) because the two test
    packages each have a `conftest.py`; importing `conftest` by module
    name would be ambiguous under pytest's prepend import mode.
    """

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

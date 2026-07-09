"""
Shared fixtures for the Phase 2 physics tests.

Toy atomic data only — element-agnostic physics is exercised with
simple two-level species whose partition functions and populations can
be computed by hand. All fixture objects are frozen/immutable, so
session scope is safe.
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.atomic.transition import Transition
from libssim.physics.partition import (
    PartitionFunctionProvider,
    PartitionFunctionTable,
)
from libssim.physics.saha import SahaSolver

#: Temperature grid covering the full test range (K).
GRID = np.linspace(500.0, 30000.0, 600)

#: Toy bound levels per species: (statistical weights, energies in eV).
LEVELS = {
    ("Fe", 1): ([9.0, 11.0], [0.0, 1.0]),
    ("Fe", 2): ([10.0, 12.0], [0.0, 1.5]),
    ("Al", 1): ([2.0, 4.0], [0.0, 0.014]),
    ("Al", 2): ([1.0, 5.0], [0.0, 4.6]),
}

#: NIST first ionization potentials (eV).
IONIZATION_EV = {"Fe": 7.9024, "Al": 5.9858}


@pytest.fixture(scope="session")
def partition_provider() -> PartitionFunctionProvider:
    """Provider with direct-summation tables for Fe I/II and Al I/II."""
    tables = {
        key: PartitionFunctionTable.from_levels(g, e_ev, GRID)
        for key, (g, e_ev) in LEVELS.items()
    }
    return PartitionFunctionProvider(tables=tables)


@pytest.fixture(scope="session")
def saha_solver(partition_provider: PartitionFunctionProvider) -> SahaSolver:
    """Two-stage Saha solver over the toy Fe/Al plasma."""
    return SahaSolver(
        partition_provider=partition_provider,
        ionization_energies_ev=IONIZATION_EV,
    )


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

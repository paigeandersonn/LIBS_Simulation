"""
Tests for libssim.physics.boltzmann (Herrera 2008: Eq. 5-1, p. 98).

Covers the Phase 2 acceptance criterion (validation_strategy.md):
- Population fractions must sum to 1.0
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.constants import KB_EV
from libssim.physics.boltzmann import (
    boltzmann_population_fraction,
    level_population_fractions,
    upper_level_density,
)

T0 = 10000.0
G = np.array([2.0, 4.0, 6.0, 8.0])
E_EV = np.array([0.0, 1.2, 2.9, 4.4])


class TestPopulationFraction:
    def test_matches_hand_computation(self):
        # Also validates the eV -> J conversion (KB_EV = KB/EV).
        f = boltzmann_population_fraction(9.0, 3.0, 25.0, T0)
        expected = 9.0 * np.exp(-3.0 / (KB_EV * T0)) / 25.0
        assert f == pytest.approx(expected, rel=1e-12)

    def test_vectorized_over_levels(self):
        f = boltzmann_population_fraction(G, E_EV, 25.0, T0)
        assert f.shape == G.shape
        assert np.all(f > 0)

    def test_scalar_input_returns_float(self):
        assert isinstance(
            boltzmann_population_fraction(9.0, 3.0, 25.0, T0), float
        )

    @pytest.mark.parametrize(
        "g, e_ev, U, T",
        [
            (0.0, 3.0, 25.0, T0),    # non-positive weight
            (9.0, -0.1, 25.0, T0),   # negative energy
            (9.0, 3.0, 0.0, T0),     # non-positive partition function
            (9.0, 3.0, 25.0, -1.0),  # non-positive temperature
        ],
    )
    def test_invalid_inputs_rejected(self, g, e_ev, U, T):
        with pytest.raises(ValueError):
            boltzmann_population_fraction(g, e_ev, U, T)


class TestLevelPopulationFractions:
    def test_fractions_sum_to_one(self):
        # Phase 2 acceptance criterion. Holds by construction: U(T) is
        # the sum of the same Boltzmann weights over the same level list,
        # so only float rounding (~n*eps) can move the sum off 1.
        fractions = level_population_fractions(G, E_EV, 12000.0)
        assert fractions.sum() == pytest.approx(1.0, abs=1e-12)

    def test_ground_state_dominates_when_cold(self):
        fractions = level_population_fractions(G, E_EV, 300.0)
        assert fractions[0] > 1.0 - 1e-12
        assert fractions[0] == fractions.max()

    def test_higher_temperature_shifts_population_upward(self):
        cold = level_population_fractions(G, E_EV, 5000.0)
        hot = level_population_fractions(G, E_EV, 20000.0)
        assert hot[-1] > cold[-1]
        assert hot[0] < cold[0]

    def test_array_temperature_rejected(self):
        with pytest.raises(ValueError, match="scalar"):
            level_population_fractions(G, E_EV, np.array([1e4, 2e4]))


class TestUpperLevelDensity:
    def test_matches_eq_5_1(self, fe_transition):
        n_s, U = 1.0e22, 25.0
        n_k = upper_level_density(fe_transition, n_s, U, T0)
        expected = (
            n_s
            * fe_transition.g_upper
            * np.exp(-fe_transition.energy_upper_ev / (KB_EV * T0))
            / U
        )
        assert n_k == pytest.approx(expected, rel=1e-12)

    def test_scales_linearly_with_species_density(self, fe_transition):
        n1 = upper_level_density(fe_transition, 1.0e22, 25.0, T0)
        n2 = upper_level_density(fe_transition, 3.0e22, 25.0, T0)
        assert n2 == pytest.approx(3.0 * n1, rel=1e-12)

    def test_negative_density_rejected(self, fe_transition):
        with pytest.raises(ValueError):
            upper_level_density(fe_transition, -1.0, 25.0, T0)

"""
Tests for libssim.physics.saha (Herrera 2008: Eq. 5-2 p. 98; Eq. 5-3
p. 99; Eq. 5-16 p. 106; Eqs. D-1..D-10 pp. 274-276).

Covers the Phase 2 acceptance criteria (validation_strategy.md):
- Total elemental abundance conserved to <= 1e-10 relative error
- Neutral fraction approaches 1 at low temperature
- Ionization increases monotonically with temperature
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.constants import H, KB, KB_EV, ME
from libssim.core.state import PlasmaState
from libssim.physics.partition import (
    PartitionFunctionProvider,
    PartitionFunctionTable,
)
from libssim.physics.saha import (
    IonizationBalance,
    SahaSolver,
    debye_length_m,
    debye_sphere_particle_count,
    ionization_potential_lowering_ev,
    mcwhirter_minimum_electron_density_m3,
    saha_factor,
)

T0 = 10000.0
CHI_FE = 7.9024
DENSITIES = {"Fe": 7.0e22, "Al": 3.0e22}


class TestSahaFactor:
    def test_matches_hand_computation(self):
        # Eq. D-1, p. 274 with U_II/U_I = 1
        expected = (
            2.0
            * (2.0 * np.pi * ME * KB * T0 / H**2) ** 1.5
            * np.exp(-CHI_FE / (KB_EV * T0))
        )
        assert saha_factor(T0, 1.0, 1.0, CHI_FE) == pytest.approx(
            expected, rel=1e-12
        )

    def test_lowering_increases_ionization(self):
        s0 = saha_factor(T0, 1.0, 1.0, CHI_FE)
        s1 = saha_factor(T0, 1.0, 1.0, CHI_FE, lowering_ev=0.1)
        assert s1 > s0

    def test_lowering_must_stay_below_chi(self):
        with pytest.raises(ValueError, match="lowering_ev"):
            saha_factor(T0, 1.0, 1.0, CHI_FE, lowering_ev=CHI_FE)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"u_neutral": 0.0, "u_ion": 1.0, "ionization_energy_ev": CHI_FE},
            {"u_neutral": 1.0, "u_ion": 1.0, "ionization_energy_ev": -1.0},
        ],
    )
    def test_invalid_inputs_rejected(self, kwargs):
        with pytest.raises(ValueError):
            saha_factor(T0, **kwargs)


class TestBalance:
    def test_mass_conservation_exact(self, saha_solver):
        # Acceptance criterion: <= 1e-10; construction makes it ~exact.
        bal = saha_solver.balance(T0, 1.0e22, DENSITIES)
        for element, n_j in DENSITIES.items():
            rel = abs(bal.elemental_density_m3(element) - n_j) / n_j
            assert rel <= 1e-12

    def test_neutral_fraction_approaches_one_at_low_temperature(
        self, saha_solver
    ):
        bal = saha_solver.balance(2000.0, 1.0e22, {"Fe": 1.0e23})
        assert bal.neutral_fraction("Fe") > 1.0 - 1e-6

    def test_ionization_monotonic_in_temperature(self, saha_solver):
        # At fixed n_e, f_ion = s/(s+n_e) inherits the strict monotonic
        # growth of s(T) (Eq. D-1) — the acceptance criterion.
        temps = np.linspace(4000.0, 25000.0, 40)
        fracs = [
            saha_solver.balance(t, 1.0e22, {"Fe": 1.0e23}).ionization_fraction(
                "Fe"
            )
            for t in temps
        ]
        assert np.all(np.diff(fracs) > 0)

    def test_underflow_gives_neutral_limit_without_nan(self, saha_solver):
        # s(T) underflows to exactly 0 around T ~ 100 K for chi ~ 8 eV;
        # covered by a table reaching 50 K.
        grid = np.geomspace(50.0, 30000.0, 400)
        provider = PartitionFunctionProvider(
            tables={
                ("Fe", 1): PartitionFunctionTable.from_levels(
                    [9.0, 11.0], [0.0, 1.0], grid
                ),
                ("Fe", 2): PartitionFunctionTable.from_levels(
                    [10.0, 12.0], [0.0, 1.5], grid
                ),
            }
        )
        cold = SahaSolver(
            partition_provider=provider,
            ionization_energies_ev={"Fe": CHI_FE},
        )
        bal = cold.balance(100.0, 0.0, {"Fe": 1.0e23})
        assert bal.ionization_fraction("Fe") == 0.0
        assert np.isfinite(bal.neutral_density_m3["Fe"])
        assert cold.solve_electron_density(100.0, {"Fe": 1.0e23}) == 0.0

    def test_unknown_element_raises_keyerror(self, saha_solver):
        # The partition provider (queried first) normalizes keys to
        # upper case, so match case-insensitively.
        with pytest.raises(KeyError, match="(?i)cu"):
            saha_solver.saha_factor("Cu", T0)

    @pytest.mark.parametrize(
        "T, n_e, dens",
        [
            (-1.0, 1e22, DENSITIES),
            (T0, -1.0, DENSITIES),
            (T0, 1e22, {"Fe": -1.0}),
        ],
    )
    def test_invalid_inputs_rejected(self, saha_solver, T, n_e, dens):
        with pytest.raises(ValueError):
            saha_solver.balance(T, n_e, dens)


class TestSelfConsistentElectronDensity:
    def test_charge_equilibrium_satisfied(self, saha_solver):
        # Eq. D-9, pp. 275-276: n_e = sum_j n_i^j at the root.
        n_e = saha_solver.solve_electron_density(T0, DENSITIES)
        bal = saha_solver.balance(T0, n_e, DENSITIES)
        assert bal.total_ion_density_m3 == pytest.approx(n_e, rel=1e-10)

    def test_root_bounded_by_heavy_density(self, saha_solver):
        n_e = saha_solver.solve_electron_density(T0, DENSITIES)
        assert 0.0 < n_e < sum(DENSITIES.values())

    def test_empty_plasma_returns_zero(self, saha_solver):
        assert saha_solver.solve_electron_density(T0, {}) == 0.0


class TestBalanceFromState:
    def test_distributes_heavy_particle_density(self, saha_solver):
        # n^j = C_j * (n_tot - n_e), documented reading of App. B/D.
        state = PlasmaState(
            temperature_K=T0,
            electron_density_m3=1.0e22,
            total_density_m3=1.1e23,
            radius_m=1e-3,
            time_s=1e-6,
            composition={"Fe": 0.7, "Al": 0.3},
        )
        bal = saha_solver.balance_from_state(state)
        n_heavy = 1.1e23 - 1.0e22
        for element, fraction in (("Fe", 0.7), ("Al", 0.3)):
            assert bal.elemental_density_m3(element) == pytest.approx(
                fraction * n_heavy, rel=1e-12
            )


class TestIonizationBalanceContainer:
    def test_mismatched_keys_rejected(self):
        with pytest.raises(ValueError, match="same keys"):
            IonizationBalance(T0, 1e22, {"Fe": 1.0}, {"Al": 1.0})

    def test_negative_density_rejected(self):
        with pytest.raises(ValueError):
            IonizationBalance(T0, 1e22, {"Fe": -1.0}, {"Fe": 1.0})

    def test_fraction_of_vanishing_element(self):
        bal = IonizationBalance(T0, 0.0, {"Fe": 0.0}, {"Fe": 0.0})
        assert bal.ionization_fraction("Fe") == 0.0
        assert bal.neutral_fraction("Fe") == 1.0


class TestLTEDiagnostics:
    def test_mcwhirter_value(self):
        # Eq. 5-3, p. 99 in SI: 1.6e18 * sqrt(T) * dE^3 at T=1e4, dE=3 eV.
        n_min = mcwhirter_minimum_electron_density_m3(1.0e4, 3.0)
        assert n_min == pytest.approx(1.6e18 * 100.0 * 27.0, rel=1e-12)

    def test_debye_count_matches_thesis_cgs_form(self):
        # Eq. 5-16, p. 106: n_D = 1.72e9 * T_eV^1.5 / sqrt(n_e[cm^-3]).
        T, n_e = 11604.518, 1.0e23
        n_d_si = debye_sphere_particle_count(T, n_e)
        n_d_cgs = 1.72e9 * (KB_EV * T) ** 1.5 / np.sqrt(n_e * 1e-6)
        assert n_d_si == pytest.approx(n_d_cgs, rel=5e-3)

    def test_lowering_magnitude_at_libs_conditions(self):
        dchi = ionization_potential_lowering_ev(1.0e4, 1.0e23)
        assert 0.01 < dchi < 0.3  # ~0.07 eV, small against chi ~ 6-9 eV

    def test_debye_length_positive_and_decreasing_with_density(self):
        lam1 = debye_length_m(1.0e4, 1.0e22)
        lam2 = debye_length_m(1.0e4, 1.0e24)
        assert lam1 > lam2 > 0.0

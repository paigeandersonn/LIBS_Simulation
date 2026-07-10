"""
Tests for libssim.transport.emissivity (Herrera 2008: Eq. 5-49 p. 119
assembly from Eqs. 5-1, 5-8, 5-50/5-51/5-52, 3-1, 3-8).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from libssim.core.constants import C
from libssim.physics.boltzmann import upper_level_density
from libssim.physics.continuum import (
    continuum_absorption_coefficient,
    continuum_emission_coefficient,
)
from libssim.physics.emission import blackbody_spectral_radiance_hz
from libssim.transport.emissivity import LTESpectralModel
from libssim.transport.geometry import SphericalOnion


@pytest.fixture(scope="module")
def line_grid(resonance_wavelength_m) -> np.ndarray:
    """Fine wavelength grid spanning +/- 10 Doppler FWHM of the line."""
    fwhm = 4.0e-12  # ~ Doppler FWHM of Fe at 1e4 K near 413 nm
    return np.linspace(
        resonance_wavelength_m - 10 * fwhm,
        resonance_wavelength_m + 10 * fwhm,
        3001,
    )


class TestConstruction:
    def test_stage_three_transition_rejected(
        self, saha_solver, resonance_transition, line_grid, atomic_masses_kg
    ):
        from dataclasses import replace

        bad = replace(resonance_transition, ion_stage=3)
        with pytest.raises(ValueError, match="stage I/II"):
            LTESpectralModel(
                saha_solver=saha_solver,
                wavelength_m=line_grid,
                transitions=(bad,),
                atomic_masses_kg=atomic_masses_kg,
            )

    def test_missing_mass_rejected(
        self, saha_solver, resonance_transition, line_grid
    ):
        with pytest.raises(ValueError, match="atomic mass"):
            LTESpectralModel(
                saha_solver=saha_solver,
                wavelength_m=line_grid,
                transitions=(resonance_transition,),
                atomic_masses_kg={},  # Fe missing
            )

    @pytest.mark.parametrize(
        "grid",
        [
            np.array([5.0e-7]),                    # single point
            np.array([5.0e-7, 4.0e-7]),            # decreasing
            np.array([5.0e-7, -1.0e-7]),           # non-positive
        ],
    )
    def test_bad_grid_rejected(self, saha_solver, grid):
        with pytest.raises(ValueError):
            LTESpectralModel(saha_solver=saha_solver, wavelength_m=grid)

    def test_grid_locked_read_only(self, saha_solver, line_grid):
        model = LTESpectralModel(saha_solver=saha_solver, wavelength_m=line_grid)
        with pytest.raises(ValueError):
            model.wavelength_m[0] = 0.0


class TestContinuumOnly:
    def test_matches_physics_layer_directly(
        self, saha_solver, make_state, line_grid
    ):
        # With no transitions, zone properties must equal the Phase 2
        # continuum functions evaluated at the zone's Saha ion density.
        model = LTESpectralModel(
            saha_solver=saha_solver, wavelength_m=line_grid
        )
        state = make_state(1.0e4, 1.0e22, electron_density_m3=1.0e21)
        epsilon, kappa = model.zone_properties(state)

        balance = saha_solver.balance_from_state(state)
        frequency = C / line_grid
        kappa_direct = continuum_absorption_coefficient(
            frequency, 1.0e4, 1.0e21, balance.total_ion_density_m3
        )
        epsilon_direct = continuum_emission_coefficient(
            frequency, 1.0e4, 1.0e21, balance.total_ion_density_m3
        )
        assert np.allclose(kappa, kappa_direct, rtol=1e-13)
        assert np.allclose(epsilon, epsilon_direct, rtol=1e-13)

    def test_continuum_can_be_disabled(
        self, saha_solver, make_state, line_grid
    ):
        model = LTESpectralModel(
            saha_solver=saha_solver,
            wavelength_m=line_grid,
            include_continuum=False,
        )
        epsilon, kappa = model.zone_properties(make_state(1.0e4, 1.0e22))
        assert np.all(epsilon == 0.0) and np.all(kappa == 0.0)


@pytest.fixture(scope="module")
def line_model(saha_solver, resonance_transition, line_grid, atomic_masses_kg):
    return LTESpectralModel(
        saha_solver=saha_solver,
        wavelength_m=line_grid,
        transitions=(resonance_transition,),
        atomic_masses_kg=atomic_masses_kg,
        include_continuum=False,  # isolate the line
    )


class TestSingleLine:
    def test_kirchhoff_closure_pointwise(
        self, line_model, make_state, line_grid, resonance_wavelength_m
    ):
        # LTE consistency across the whole adapter: eps/kappa must equal
        # B_nu(nu0, T) wherever the line absorbs (narrow-line Kirchhoff,
        # emission.py docs) — populations, profiles and unit conversions
        # all cancel correctly or this fails.
        T = 1.0e4
        epsilon, kappa = line_model.zone_properties(make_state(T, 1.0e22))
        mask = kappa > kappa.max() * 1e-6
        planck = blackbody_spectral_radiance_hz(
            C / resonance_wavelength_m, T
        )
        assert np.allclose(epsilon[mask] / kappa[mask], planck, rtol=1e-10)

    def test_emission_integral_recovers_total(
        self, line_model, saha_solver, resonance_transition,
        make_state, line_grid, resonance_wavelength_m,
    ):
        # integral(eps_nu dnu) = h*nu0 * n_upper * A / (4*pi), with
        # n_upper reproduced independently through Saha + Boltzmann.
        from libssim.core.constants import H

        T = 1.0e4
        state = make_state(T, 1.0e22)
        epsilon, _ = line_model.zone_properties(state)

        balance = saha_solver.balance_from_state(state)
        U = saha_solver.partition_provider.partition_function("Fe", 1, T)
        n_upper = upper_level_density(
            resonance_transition, balance.neutral_density_m3["Fe"], U, T
        )
        expected = (
            H * (C / resonance_wavelength_m) * n_upper
            * resonance_transition.a_ki / (4 * np.pi)
        )
        # integral over nu via the wavelength grid: dnu = c/lambda^2 dlambda
        integral = trapezoid(epsilon * C / line_grid**2, line_grid)
        assert integral == pytest.approx(expected, rel=1e-3)

    def test_absent_element_contributes_nothing(
        self, line_model, make_state
    ):
        # A pure-Al zone sees no Fe line (documented skip).
        state = make_state(1.0e4, 1.0e22, element="Al")
        epsilon, kappa = line_model.zone_properties(state)
        assert np.all(epsilon == 0.0) and np.all(kappa == 0.0)

    def test_stark_width_broadens_the_line(
        self, saha_solver, resonance_transition, line_grid, make_state,
        atomic_masses_kg, resonance_wavelength_m,
    ):
        from dataclasses import replace

        with_stark = replace(resonance_transition, stark_width=2.0e-11)
        model = LTESpectralModel(
            saha_solver=saha_solver,
            wavelength_m=line_grid,
            transitions=(with_stark,),
            atomic_masses_kg=atomic_masses_kg,
            include_continuum=False,
        )
        base = LTESpectralModel(
            saha_solver=saha_solver,
            wavelength_m=line_grid,
            transitions=(resonance_transition,),
            atomic_masses_kg=atomic_masses_kg,
            include_continuum=False,
        )
        state = make_state(1.0e4, 1.0e22, electron_density_m3=1.0e23)
        _, kappa_stark = model.zone_properties(state)
        _, kappa_doppler = base.zone_properties(state)
        center = np.argmin(np.abs(line_grid - resonance_wavelength_m))
        # Lorentzian wings raise absorption far from center and lower
        # the (normalized) peak.
        assert kappa_stark[0] > kappa_doppler[0]
        assert kappa_stark[center] < kappa_doppler[center]


class TestGeometryProperties:
    def test_stacked_shapes_and_rows(
        self, saha_solver, resonance_transition, line_grid, make_state,
        atomic_masses_kg,
    ):
        model = LTESpectralModel(
            saha_solver=saha_solver,
            wavelength_m=line_grid,
            transitions=(resonance_transition,),
            atomic_masses_kg=atomic_masses_kg,
        )
        core = make_state(12000.0, 1.0e22)
        shell = make_state(6000.0, 1.0e21)
        onion = SphericalOnion(
            zones=(core, shell), boundaries_m=(1.0e-3, 2.0e-3)
        )
        epsilon, kappa = model.geometry_properties(onion)
        assert epsilon.shape == (2, line_grid.size)
        assert kappa.shape == (2, line_grid.size)
        eps_core, kap_core = model.zone_properties(core)
        assert np.array_equal(epsilon[0], eps_core)
        assert np.array_equal(kappa[0], kap_core)

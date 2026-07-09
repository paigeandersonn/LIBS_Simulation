"""
Tests for libssim.physics.emission (Herrera 2008: Eq. 5-8 pp. 103-104;
Eq. 5-44 p. 117; Eq. 5-52 p. 120; Eq. 3-11 p. 55).

Covers the Phase 2 acceptance criterion (implementation_plan.md):
- Emission coefficient scales correctly with upper-state population,
  A_ki, and photon energy.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from libssim.core.constants import C, H, KB
from libssim.physics.emission import (
    blackbody_spectral_radiance_hz,
    blackbody_spectral_radiance_wavelength,
    einstein_b_lu,
    line_absorption_coefficient,
    line_emission_coefficient,
    line_photon_emission_rate,
    line_power_density,
    transition_frequency_hz,
)

T0 = 10000.0


class TestLineEmission:
    def test_photon_rate_is_density_times_aki(self, fe_transition):
        # Eq. 5-8, pp. 103-104: I_ki = n_k * A_ki.
        n_k = 2.0e18
        rate = line_photon_emission_rate(fe_transition, n_k)
        assert rate == pytest.approx(n_k * fe_transition.a_ki, rel=1e-12)

    def test_power_density_weights_by_photon_energy(self, fe_transition):
        n_k = 2.0e18
        nu0 = transition_frequency_hz(fe_transition)
        power = line_power_density(fe_transition, n_k)
        assert power == pytest.approx(
            H * nu0 * n_k * fe_transition.a_ki, rel=1e-12
        )

    def test_emission_scales_with_population_aki_and_energy(
        self, fe_transition
    ):
        # Acceptance criterion: linear in n_k (A_ki and h*nu0 fixed per
        # line and already verified explicitly above).
        eps1 = line_emission_coefficient(fe_transition, 1.0e18, 1e-10)
        eps2 = line_emission_coefficient(fe_transition, 5.0e18, 1e-10)
        assert eps2 == pytest.approx(5.0 * eps1, rel=1e-12)

    def test_frequency_integral_recovers_total_over_4pi(self, fe_transition):
        # integral(eps dnu) = h*nu0*n_k*A_ki/(4*pi) for a unit-area profile.
        n_k = 2.0e18
        nu0 = transition_frequency_hz(fe_transition)
        sigma = 5.0e9
        nu = np.linspace(nu0 - 8 * sigma, nu0 + 8 * sigma, 4001)
        profile = np.exp(-0.5 * ((nu - nu0) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
        integral = trapezoid(
            line_emission_coefficient(fe_transition, n_k, profile), nu
        )
        expected = H * nu0 * n_k * fe_transition.a_ki / (4 * np.pi)
        # rel=1e-6 covers the trapezoid discretization plus the Gaussian
        # tail mass beyond the +/-8 sigma window (~1e-15).
        assert integral == pytest.approx(expected, rel=1e-6)

    def test_negative_profile_rejected(self, fe_transition):
        with pytest.raises(ValueError):
            line_emission_coefficient(fe_transition, 1e18, -1e-10)


class TestKirchhoffClosure:
    def test_emission_over_absorption_is_planck(self, fe_transition):
        # LTE detailed balance: eps_nu / kappa_bb == B_nu(nu0, T)
        # (Eq. 5-44 source term; validates the Einstein A -> B relation).
        nu0 = transition_frequency_hz(fe_transition)
        x0 = H * nu0 / (KB * T0)
        n_l = 5.0e19
        n_u = (
            n_l
            * (fe_transition.g_upper / fe_transition.g_lower)
            * np.exp(-x0)
        )
        profile = 3.3e-11
        eps = line_emission_coefficient(fe_transition, n_u, profile)
        kappa = line_absorption_coefficient(fe_transition, n_l, T0, profile)
        assert eps / kappa == pytest.approx(
            blackbody_spectral_radiance_hz(nu0, T0), rel=1e-12
        )

    def test_stimulated_emission_factor(self, fe_transition):
        # Eq. 5-52 stimulated factor: kappa_on/kappa_off = 1 - e^-x0.
        nu0 = transition_frequency_hz(fe_transition)
        x0 = H * nu0 / (KB * T0)
        on = line_absorption_coefficient(fe_transition, 1e19, T0, 1e-10)
        off = line_absorption_coefficient(
            fe_transition, 1e19, T0, 1e-10, include_stimulated_emission=False
        )
        assert on / off == pytest.approx(1.0 - np.exp(-x0), rel=1e-12)

    def test_einstein_relation(self, fe_transition):
        # A_ul / B_ul = 8*pi*h*nu0^3/c^3 with g_l*B_lu = g_u*B_ul.
        nu0 = transition_frequency_hz(fe_transition)
        b_ul = (
            einstein_b_lu(fe_transition)
            * fe_transition.g_lower
            / fe_transition.g_upper
        )
        assert fe_transition.a_ki / b_ul == pytest.approx(
            8 * np.pi * H * nu0**3 / C**3, rel=1e-12
        )


class TestBlackbody:
    def test_rayleigh_jeans_limit(self):
        # B_nu -> 2 nu^2 kB T / c^2; the RJ formula itself deviates at
        # order x/2 ~ 2.4e-6 at this frequency.
        nu = 1.0e9
        b = blackbody_spectral_radiance_hz(nu, T0)
        assert b == pytest.approx(2 * nu**2 * KB * T0 / C**2, rel=5e-6)

    def test_wien_tail_underflows_to_zero(self):
        assert blackbody_spectral_radiance_hz(1.0e22, 300.0) == 0.0

    def test_wavelength_and_frequency_forms_consistent(self):
        # B_lambda = B_nu * c / lambda^2 (Eq. 3-11, p. 55).
        lam = 500e-9
        b_lam = blackbody_spectral_radiance_wavelength(lam, T0)
        b_nu = blackbody_spectral_radiance_hz(C / lam, T0)
        assert b_lam == pytest.approx(b_nu * C / lam**2, rel=1e-12)

    def test_hotter_is_brighter_everywhere(self):
        lam = np.linspace(200e-9, 800e-9, 20)
        cold = blackbody_spectral_radiance_wavelength(lam, 8000.0)
        hot = blackbody_spectral_radiance_wavelength(lam, 12000.0)
        assert np.all(hot > cold)

    @pytest.mark.parametrize("nu, T", [(-1.0, T0), (1e15, 0.0)])
    def test_invalid_inputs_rejected(self, nu, T):
        with pytest.raises(ValueError):
            blackbody_spectral_radiance_hz(nu, T)

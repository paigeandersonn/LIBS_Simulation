"""
Tests for libssim.physics.continuum (Herrera 2008: Eq. 5-7 p. 101;
Eqs. 5-46/5-47 p. 118; Eqs. 5-49/5-50 p. 119; Eq. 5-51 p. 120).

The SI implementations are validated against the thesis' printed CGS
forms evaluated with Gaussian-unit constants. Expected agreement is
~2e-9, not machine precision: the esu charge uses the pre-2019 exact
mu0 relation while the SI code uses the CODATA-2018 measured EPSILON0
(the conventions differ at ~5e-10, cubed in e^6).
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.constants import C, H, KB
from libssim.physics.continuum import (
    _KRAMERS_SI,
    continuum_absorption_coefficient,
    continuum_emission_coefficient,
    free_bound_absorption_coefficient,
    free_free_absorption_coefficient,
    planck_mean_absorption_coefficient,
)
from libssim.physics.emission import blackbody_spectral_radiance_hz

T0 = 10000.0
NU = C / 500e-9  # 500 nm
X = H * NU / (KB * T0)
N_E, N_I = 1.0e23, 8.0e22          # m^-3
N_E_CGS, N_I_CGS = N_E * 1e-6, N_I * 1e-6  # cm^-3
TOL_UNITSYS = 5e-9

# Gaussian-CGS constants for cross-validation.
E_CGS = 1.602176634e-19 * 2.99792458e9  # esu
M_CGS = 9.1093837015e-28                # g
H_CGS = 6.62607015e-27                  # erg s
C_CGS = 2.99792458e10                   # cm/s
K_CGS = 1.380649e-16                    # erg/K
SIGMA_CGS = 5.670374419e-5              # erg cm^-2 s^-1 K^-4

KRAMERS_CGS = (
    8
    * np.pi
    * E_CGS**6
    / (3 * M_CGS * H_CGS * C_CGS * np.sqrt(6 * np.pi * M_CGS * K_CGS))
)


class TestAgainstPrintedCGSForms:
    def test_kramers_prefactor_is_the_printed_3_7e8(self):
        # Eq. 5-50, p. 119 prints "3.7e8 cgs"; exact value 3.692e8.
        assert 3.65e8 < KRAMERS_CGS < 3.72e8

    def test_free_free_si_vs_cgs(self):
        si = free_free_absorption_coefficient(NU, T0, N_E, N_I)
        cgs = (
            KRAMERS_CGS
            * N_E_CGS
            * N_I_CGS
            / (np.sqrt(T0) * NU**3)
            * (1 - np.exp(-X))
        )
        assert si / 100.0 == pytest.approx(cgs, rel=TOL_UNITSYS)

    def test_free_bound_si_vs_cgs(self):
        # Eq. 5-51, p. 120 exactly as printed: e^x (1 - e^-x)^2.
        si = free_bound_absorption_coefficient(NU, T0, N_E, N_I)
        cgs = (
            KRAMERS_CGS
            * N_E_CGS
            * N_I_CGS
            / (np.sqrt(T0) * NU**3)
            * np.exp(X)
            * (1 - np.exp(-X)) ** 2
        )
        assert si / 100.0 == pytest.approx(cgs, rel=TOL_UNITSYS)

    def test_planck_mean_si_vs_cgs(self):
        # Eq. 5-46, p. 118.
        si = planck_mean_absorption_coefficient(T0, N_E, N_I)
        prefactor_cgs = (
            np.sqrt(128 * K_CGS / 27)
            * (np.pi / M_CGS) ** 1.5
            * E_CGS**6
            / (H_CGS * SIGMA_CGS * C_CGS**3)
        )
        cgs = prefactor_cgs * N_E_CGS * N_I_CGS / T0**3.5
        assert si / 100.0 == pytest.approx(cgs, rel=TOL_UNITSYS)


class TestKirchhoffConsistency:
    def test_emission_matches_eq_5_7_bracket(self):
        # (kappa_ff + kappa_fb) * B_nu == C_K*(2h/c^2)*n_e*n_i/sqrt(T)
        #   * [G e^-x + xi (1 - e^-x)]  — the Eq. 5-7, p. 101 structure.
        gaunt, xi = 1.3, 0.8
        eps = continuum_emission_coefficient(
            NU, T0, N_E, N_I, 1.0, gaunt, xi
        )
        bracket = gaunt * np.exp(-X) + xi * (1 - np.exp(-X))
        analytic = (
            _KRAMERS_SI * (2 * H / C**2) * N_E * N_I / np.sqrt(T0) * bracket
        )
        assert eps == pytest.approx(analytic, rel=1e-10)

    def test_emission_is_absorption_times_planck(self):
        kappa = continuum_absorption_coefficient(NU, T0, N_E, N_I)
        eps = continuum_emission_coefficient(NU, T0, N_E, N_I)
        assert eps == pytest.approx(
            kappa * blackbody_spectral_radiance_hz(NU, T0), rel=1e-12
        )


class TestSpectralBehaviour:
    NUS = np.linspace(C / 800e-9, C / 200e-9, 50)

    def test_free_free_emission_decays_with_frequency(self):
        # epsilon_ff ~ G e^-x (Wien-like decay).
        eps = continuum_emission_coefficient(
            self.NUS, T0, N_E, N_I, 1.0, 1.0, 1e-300
        )
        assert np.all(np.diff(eps) < 0)

    def test_free_bound_emission_saturates_with_frequency(self):
        # epsilon_fb ~ xi (1 - e^-x), increasing toward a plateau.
        eps = continuum_emission_coefficient(
            self.NUS, T0, N_E, N_I, 1.0, 1e-300, 1.0
        )
        assert np.all(np.diff(eps) > 0)

    def test_total_is_sum_of_members(self):
        # Continuum part of Eq. 5-49, p. 119.
        total = continuum_absorption_coefficient(self.NUS, T0, N_E, N_I)
        parts = free_free_absorption_coefficient(
            self.NUS, T0, N_E, N_I
        ) + free_bound_absorption_coefficient(self.NUS, T0, N_E, N_I)
        assert np.allclose(total, parts, rtol=1e-13)

    def test_scales_with_electron_and_ion_density(self):
        k1 = free_free_absorption_coefficient(NU, T0, N_E, N_I)
        k2 = free_free_absorption_coefficient(NU, T0, 2 * N_E, 3 * N_I)
        assert k2 == pytest.approx(6.0 * k1, rel=1e-12)


class TestValidation:
    @pytest.mark.parametrize(
        "bad_call",
        [
            lambda: free_free_absorption_coefficient(-1.0, T0, N_E, N_I),
            lambda: free_free_absorption_coefficient(NU, -5.0, N_E, N_I),
            lambda: free_bound_absorption_coefficient(NU, T0, -1.0, N_I),
            lambda: planck_mean_absorption_coefficient(
                T0, N_E, N_I, charge_number=0.0
            ),
            lambda: free_free_absorption_coefficient(
                NU, T0, N_E, N_I, gaunt_factor=-1.0
            ),
        ],
    )
    def test_invalid_inputs_rejected(self, bad_call):
        with pytest.raises(ValueError):
            bad_call()

    def test_scalar_returns_float(self):
        assert isinstance(
            free_free_absorption_coefficient(NU, T0, N_E, N_I), float
        )

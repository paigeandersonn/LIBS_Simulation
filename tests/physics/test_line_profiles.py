"""
Tests for libssim.physics.line_profiles (Herrera 2008: Eqs. 3-1/3-2
pp. 50-51; Eqs. 3-8/3-9 p. 53; Eqs. 5-15/5-16/5-17 p. 106; Eq. 5-18
p. 107; Eqs. 5-53/5-54 p. 120; Eq. 5-55 p. 121).

Covers the Phase 2 acceptance criterion (validation_strategy.md):
- Area under the Voigt profile must integrate to 1.0 (+/- 1e-10)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from libssim.core.constants import C, KB_EV
from libssim.physics.line_profiles import (
    damping_parameter,
    doppler_fwhm_m,
    doppler_fwhm_practical_m,
    fwhm_frequency_to_wavelength,
    fwhm_wavelength_to_frequency,
    gaussian_profile_hz,
    lorentzian_fwhm_from_voigt,
    lorentzian_profile_hz,
    stark_fwhm_m,
    stark_shift_m,
    stark_shifted_voigt_reduced,
    voigt_fwhm_estimate,
    voigt_profile_hz,
    voigt_profile_wavelength_m,
)

LAM0 = 404.58e-9
NU0 = C / LAM0
DG = 7.0e9   # Gaussian FWHM, Hz
DL = 2.0e10  # Lorentzian FWHM, Hz


def area(profile_of_detuning, scale):
    """Adaptive quadrature of a centered profile, width-scaled so the
    feature has O(1) extent in the integration variable."""
    value, _ = quad(
        lambda t: profile_of_detuning(t * scale) * scale,
        -np.inf,
        np.inf,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=400,
    )
    return value


def measured_fwhm(x, y):
    """Numerical FWHM on a fine grid via linear interpolation."""
    half = y.max() / 2.0
    above = np.where(y >= half)[0]
    i0, i1 = above[0], above[-1]
    left = np.interp(half, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    right = np.interp(half, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])
    return right - left


class TestUnitArea:
    """Phase 2 acceptance criterion: area = 1.0 +/- 1e-10."""

    @pytest.mark.parametrize(
        "name, fn, scale",
        [
            ("gaussian", lambda x: gaussian_profile_hz(NU0 + x, NU0, DG), DG),
            (
                "lorentzian",
                lambda x: lorentzian_profile_hz(NU0 + x, NU0, DL),
                DL,
            ),
            ("voigt", lambda x: voigt_profile_hz(NU0 + x, NU0, DG, DL), DL),
            (
                "voigt_shifted",
                lambda x: voigt_profile_hz(
                    NU0 + x, NU0, DG, DL, stark_shift_hz=5e9
                ),
                DL,
            ),
            (
                "voigt_wavelength",
                lambda x: voigt_profile_wavelength_m(
                    LAM0 + x, LAM0, 4e-12, 20e-12, stark_shift_m=3e-12
                ),
                20e-12,
            ),
            (
                "reduced_eq_5_53",
                lambda x: stark_shifted_voigt_reduced(0.7, x),
                1.0,
            ),
        ],
    )
    def test_profile_integrates_to_one(self, name, fn, scale):
        assert abs(area(fn, scale) - 1.0) <= 1e-10


class TestFWHMSemantics:
    def test_gaussian_fwhm_exact(self):
        x = np.linspace(-6 * DG, 6 * DG, 200001)
        y = gaussian_profile_hz(NU0 + x, NU0, DG)
        assert measured_fwhm(x, y) == pytest.approx(DG, rel=1e-6)

    def test_lorentzian_fwhm_exact(self):
        x = np.linspace(-30 * DL, 30 * DL, 400001)
        y = lorentzian_profile_hz(NU0 + x, NU0, DL)
        assert measured_fwhm(x, y) == pytest.approx(DL, rel=1e-6)

    def test_voigt_fwhm_matches_eq_5_18_inverse(self):
        # Whiting-type estimate documented accurate to ~1-2%.
        x = np.linspace(-30 * DL, 30 * DL, 400001)
        y = voigt_profile_hz(NU0 + x, NU0, DG, DL)
        assert measured_fwhm(x, y) == pytest.approx(
            voigt_fwhm_estimate(DG, DL), rel=0.02
        )

    def test_eq_5_18_recovers_lorentzian_width(self):
        x = np.linspace(-30 * DL, 30 * DL, 400001)
        fwhm_v = measured_fwhm(x, voigt_profile_hz(NU0 + x, NU0, DG, DL))
        assert lorentzian_fwhm_from_voigt(fwhm_v, DG) == pytest.approx(
            DL, rel=0.02
        )


class TestDopplerWidth:
    M_FE_KG = 55.845 * 1.66053906660e-27

    def test_eq_3_1_matches_practical_eq_3_2(self):
        # 7.16e-7 is Eq. 3-2's 3-digit rounding of sqrt(8*ln2*R/1e-3)/c
        # = 7.162e-7, so agreement is limited to ~3e-4.
        exact = doppler_fwhm_m(LAM0, 1.0e4, self.M_FE_KG)
        practical = doppler_fwhm_practical_m(LAM0, 1.0e4, 55.845)
        assert practical == pytest.approx(exact, rel=5e-4)

    def test_fe_line_magnitude(self):
        # ~3.88 pm for Fe I 404.58 nm at 10^4 K.
        width = doppler_fwhm_m(LAM0, 1.0e4, self.M_FE_KG)
        assert 3.7e-12 < width < 4.0e-12

    def test_scales_as_sqrt_temperature(self):
        w1 = doppler_fwhm_m(LAM0, 5.0e3, self.M_FE_KG)
        w2 = doppler_fwhm_m(LAM0, 2.0e4, self.M_FE_KG)
        assert w2 == pytest.approx(2.0 * w1, rel=1e-12)

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            doppler_fwhm_m(LAM0, -5.0, self.M_FE_KG)


class TestStarkWidthAndShift:
    W = 2.0e-11   # electron-impact half-width, m
    N_E = 1.0e23  # m^-3 (= 1e17 cm^-3)

    def test_alpha_zero_reduces_to_eq_5_17(self):
        # Eq. 3-8 with alpha=0 == 2*w*(n_e/1e16 cm^-3) == 2e-22*w*n_e.
        si = stark_fwhm_m(self.W, self.N_E)
        cgs = 2 * self.W * ((self.N_E * 1e-6) / 1e16)
        assert si == pytest.approx(cgs, rel=1e-12)

    def test_full_bracket_matches_eq_5_15_debye_form(self):
        # Eq. 3-8's 0.0068 folds beta=0.75 and the n_D conversion of
        # Eqs. 5-15/5-16 (module docs); agreement to coefficient rounding.
        T, alpha = 1.0e4, 0.05
        si = stark_fwhm_m(
            self.W, self.N_E, ion_broadening_alpha=alpha, temperature_K=T
        )
        n_e_cgs = self.N_E * 1e-6
        n_d = 1.72e9 * (KB_EV * T) ** 1.5 / np.sqrt(n_e_cgs)  # Eq. 5-16
        bracket = 1 + 1.75 * alpha * (n_e_cgs / 1e16) ** 0.25 * (
            1 - 0.75 * n_d ** (-1.0 / 3.0)
        )
        cgs = 2 * self.W * (n_e_cgs / 1e16) * bracket
        assert si == pytest.approx(cgs, rel=0.02)

    def test_shift_alpha_zero_is_signed_linear_form(self):
        # Eq. 3-9 with alpha=0: dlambda = (d/w) * w * n_e * 1e-22.
        shift = stark_shift_m(self.W, -0.4, self.N_E)
        assert shift == pytest.approx(
            -0.4 * self.W * self.N_E * 1e-22, rel=1e-12
        )

    def test_ion_term_requires_temperature(self):
        with pytest.raises(ValueError, match="temperature_K is required"):
            stark_fwhm_m(self.W, self.N_E, ion_broadening_alpha=0.1)

    def test_width_scales_linearly_with_density(self):
        w1 = stark_fwhm_m(self.W, 1.0e22)
        w2 = stark_fwhm_m(self.W, 5.0e22)
        assert w2 == pytest.approx(5.0 * w1, rel=1e-12)


class TestVoigtTraceability:
    def test_eq_5_53_matches_direct_double_integral(self):
        # Verbatim check of P = (a/(pi*sqrt(pi))) * integral.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            for a_par, lam in [(0.3, 0.0), (0.7, 1.3), (2.0, -2.5)]:
                inner, _ = quad(
                    lambda y: np.exp(-(y**2))
                    / ((lam - y) ** 2 + a_par**2),
                    -np.inf,
                    np.inf,
                    epsabs=1e-14,
                    epsrel=1e-14,
                    limit=400,
                )
                direct = a_par / (np.pi * np.sqrt(np.pi)) * inner
                assert stark_shifted_voigt_reduced(a_par, lam) == pytest.approx(
                    direct, abs=1e-12
                )

    def test_physical_profile_equals_reduced_mapping(self):
        # Physical scipy Voigt == Eq. 5-53 with a from Eq. 5-55 and the
        # sqrt(ln 2)-restored Eq. 5-54 (documented ambiguity).
        shift = 5.0e9
        nu = NU0 + np.linspace(-8e10, 8e10, 11)
        a_par = damping_parameter(DL, DG)
        u = 2 * np.sqrt(np.log(2)) * (nu - NU0 + shift) / DG
        reduced = (
            stark_shifted_voigt_reduced(a_par, u)
            * 2
            * np.sqrt(np.log(2))
            / DG
        )
        physical = voigt_profile_hz(nu, NU0, DG, DL, stark_shift_hz=shift)
        assert np.allclose(reduced, physical, rtol=1e-12, atol=0.0)

    def test_damping_parameter_eq_5_55(self):
        assert damping_parameter(DL, DG) == pytest.approx(
            DL * np.sqrt(np.log(2)) / DG, rel=1e-14
        )


class TestShiftConventions:
    def test_frequency_peak_at_nu0_minus_shift(self):
        # Eq. 5-54 "+" convention: positive shift -> lower frequency.
        x = np.linspace(-6e10, 6e10, 60001)
        y = voigt_profile_hz(NU0 + x, NU0, DG, DL, stark_shift_hz=1e10)
        assert x[np.argmax(y)] == pytest.approx(-1e10, abs=5e7)

    def test_wavelength_peak_at_lam0_plus_shift(self):
        # Red shift -> larger wavelength (Eq. 3-9 sign).
        x = np.linspace(-60e-12, 60e-12, 60001)
        y = voigt_profile_wavelength_m(
            LAM0 + x, LAM0, 4e-12, 20e-12, stark_shift_m=10e-12
        )
        assert x[np.argmax(y)] == pytest.approx(10e-12, abs=5e-14)


class TestConversionsAndValidation:
    def test_wavelength_frequency_roundtrip(self):
        dnu = fwhm_wavelength_to_frequency(3.88e-12, LAM0)
        assert fwhm_frequency_to_wavelength(dnu, LAM0) == pytest.approx(
            3.88e-12, rel=1e-14
        )

    def test_both_widths_zero_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            voigt_profile_hz(NU0, NU0, 0.0, 0.0)

    def test_eq_5_18_domain_enforced(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            lorentzian_fwhm_from_voigt(1.0, 2.0)

    def test_vectorization_and_scalar_types(self):
        x = np.linspace(-6e10, 6e10, 101)
        assert voigt_profile_hz(NU0 + x, NU0, DG, DL).shape == x.shape
        assert isinstance(voigt_profile_hz(NU0, NU0, DG, DL), float)

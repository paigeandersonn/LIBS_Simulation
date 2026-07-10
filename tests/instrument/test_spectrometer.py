"""
Tests for libssim.instrument.spectrometer (Herrera 2008: Eq. 3-19
p. 57; Eq. 3-20 p. 59; Eq. 3-22 p. 60; Fig. 3-5 p. 64).

Covers the Phase 4 acceptance criterion (implementation_plan.md):
- Gaussian convolution produces the expected FWHM for a given slit
  width.
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.spectrum import Spectrum
from libssim.instrument.spectrometer import (
    InstrumentalProfile,
    diffraction_limited_bandpass_m,
)

R_D = 1.6         # nm/mm — typical 0.5 m Czerny-Turner
SLIT_UM = 100.0   # um
LAMBDA0 = 400e-9  # m


def gaussian_line(fwhm_m: float, step_m: float = 1.0e-12) -> Spectrum:
    """Unit-peak Gaussian line on a uniform grid, wide margins."""
    sigma = fwhm_m / (2 * np.sqrt(2 * np.log(2)))
    grid = LAMBDA0 + np.arange(-2000, 2001) * step_m
    intensity = np.exp(-0.5 * ((grid - LAMBDA0) / sigma) ** 2)
    return Spectrum(wavelength_m=grid, intensity=intensity)


def measured_fwhm(spectrum: Spectrum) -> float:
    x, y = spectrum.wavelength_m, spectrum.intensity
    half = y.max() / 2
    above = np.where(y >= half)[0]
    i0, i1 = above[0], above[-1]
    left = np.interp(half, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    right = np.interp(half, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])
    return right - left


class TestBandpass:
    def test_slit_bandpass_eq_3_19(self):
        profile = InstrumentalProfile(R_D, SLIT_UM)
        # 1.6 nm/mm x 100 um = 0.16 nm.
        assert profile.slit_bandpass_m == pytest.approx(1.6e-10, rel=1e-12)
        assert profile.fwhm_m == profile.slit_bandpass_m  # no aberrations

    def test_aberration_added_in_quadrature(self):
        # Eq. 3-22 quadrature combination.
        profile = InstrumentalProfile(R_D, SLIT_UM, aberration_fwhm_m=1.2e-10)
        assert profile.fwhm_m == pytest.approx(
            np.hypot(1.6e-10, 1.2e-10), rel=1e-12
        )

    def test_diffraction_limited_bandpass_eq_3_20(self):
        # w_d = 2*f*lambda/a; dlambda_d = R_d * w_d.
        value = diffraction_limited_bandpass_m(R_D, 0.5, 400e-9, 0.05)
        w_d = 2 * 0.5 * 400e-9 / 0.05
        assert value == pytest.approx(R_D * 1e-6 * w_d, rel=1e-12)


class TestConvolution:
    def test_gaussian_fwhm_matches_eq_3_22(self):
        # Acceptance criterion: line FWHM_D (x) LSF FWHM_I ->
        # sqrt(FWHM_D^2 + FWHM_I^2).
        fwhm_line = 5.0e-11
        profile = InstrumentalProfile(R_D, SLIT_UM)  # FWHM_I = 1.6e-10
        convolved = profile.convolve(gaussian_line(fwhm_line))
        expected = np.hypot(fwhm_line, profile.fwhm_m)
        assert measured_fwhm(convolved) == pytest.approx(expected, rel=1e-3)

    def test_flux_conserved(self):
        line = gaussian_line(5.0e-11)
        convolved = InstrumentalProfile(R_D, SLIT_UM).convolve(line)
        assert convolved.intensity.sum() == pytest.approx(
            line.intensity.sum(), rel=1e-12
        )

    def test_triangular_shape_broadens_and_conserves(self):
        line = gaussian_line(5.0e-11)
        profile = InstrumentalProfile(R_D, SLIT_UM, shape="triangular")
        convolved = profile.convolve(line)
        assert convolved.intensity.sum() == pytest.approx(
            line.intensity.sum(), rel=1e-12
        )
        assert convolved.intensity.max() < line.intensity.max()
        assert measured_fwhm(convolved) > measured_fwhm(line)

    def test_kernel_unit_sum(self):
        profile = InstrumentalProfile(R_D, SLIT_UM)
        for shape in ("gaussian", "triangular"):
            kernel = InstrumentalProfile(R_D, SLIT_UM, shape=shape).kernel(
                1.0e-12
            )
            assert kernel.sum() == pytest.approx(1.0, abs=1e-14)
            assert np.all(kernel >= 0)
        assert profile.kernel(1e-12).size % 2 == 1  # centered

    def test_nonuniform_grid_rejected(self):
        grid = LAMBDA0 + np.array([0, 1e-12, 3e-12, 6e-12, 1e-11])
        spectrum = Spectrum(wavelength_m=grid, intensity=np.ones(5))
        with pytest.raises(ValueError, match="uniform"):
            InstrumentalProfile(R_D, SLIT_UM).convolve(spectrum)

    def test_unresolved_kernel_rejected(self):
        # FWHM = 0.16 nm but grid step 0.1 nm -> < 3 samples.
        grid = LAMBDA0 + np.arange(0, 50) * 1.0e-10
        spectrum = Spectrum(wavelength_m=grid, intensity=np.ones(50))
        with pytest.raises(ValueError, match="fewer than 3 grid steps"):
            InstrumentalProfile(R_D, SLIT_UM).convolve(spectrum)


class TestPixelSampling:
    def test_bin_average_preserves_mean(self):
        line = gaussian_line(5.0e-11)
        profile = InstrumentalProfile(R_D, SLIT_UM)
        binned = profile.sample_to_pixels(line, 400)
        assert binned.wavelength_m.size == 400
        assert binned.intensity.mean() == pytest.approx(
            line.intensity.mean(), rel=5e-3
        )
        assert binned.metadata["n_pixels"] == 400

    def test_invalid_pixel_count_rejected(self):
        line = gaussian_line(5.0e-11)
        profile = InstrumentalProfile(R_D, SLIT_UM)
        with pytest.raises(ValueError):
            profile.sample_to_pixels(line, 0)
        with pytest.raises(ValueError):
            profile.sample_to_pixels(line, line.wavelength_m.size + 1)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"reciprocal_dispersion_nm_per_mm": 0.0, "slit_width_um": 100.0},
            {"reciprocal_dispersion_nm_per_mm": R_D, "slit_width_um": -1.0},
            {
                "reciprocal_dispersion_nm_per_mm": R_D,
                "slit_width_um": 100.0,
                "aberration_fwhm_m": -1e-10,
            },
            {
                "reciprocal_dispersion_nm_per_mm": R_D,
                "slit_width_um": 100.0,
                "shape": "boxcar",
            },
        ],
    )
    def test_invalid_construction_rejected(self, kwargs):
        with pytest.raises(ValueError):
            InstrumentalProfile(**kwargs)

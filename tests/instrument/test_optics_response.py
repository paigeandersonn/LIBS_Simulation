"""
Tests for libssim.instrument.optics (Eqs. 5-9..5-11, p. 104) and the
composite libssim.instrument.response pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.spectrum import Spectrum
from libssim.instrument.noise import NoiseModel
from libssim.instrument.optics import CollectionOptics, tabulated_efficiency
from libssim.instrument.response import InstrumentResponse
from libssim.instrument.spectrometer import InstrumentalProfile


def line_spectrum() -> Spectrum:
    grid = 400e-9 + np.arange(-1000, 1001) * 1e-12
    sigma = 2.0e-11
    intensity = 1000.0 * np.exp(-0.5 * ((grid - 400e-9) / sigma) ** 2)
    return Spectrum(wavelength_m=grid, intensity=intensity)


class TestCollectionOptics:
    def test_flat_default_scales_by_absolute_factor(self):
        spectrum = line_spectrum()
        scaled = CollectionOptics(absolute_factor=2.5).apply(spectrum)
        assert np.allclose(scaled.intensity, 2.5 * spectrum.intensity)
        assert scaled.metadata["collection_absolute_factor"] == 2.5
        assert scaled.metadata["collection_relative_curve"] is False

    def test_tabulated_relative_efficiency_applied(self):
        # Linear ramp across the window (Eq. 5-9's F_rel(lambda)).
        spectrum = line_spectrum()
        lam = spectrum.wavelength_m
        curve = tabulated_efficiency(
            np.array([lam[0], lam[-1]]), np.array([0.5, 1.0])
        )
        scaled = CollectionOptics(relative_efficiency=curve).apply(spectrum)
        expected = spectrum.intensity * np.interp(
            lam, [lam[0], lam[-1]], [0.5, 1.0]
        )
        assert np.allclose(scaled.intensity, expected, rtol=1e-12)

    def test_tabulated_efficiency_validation(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            tabulated_efficiency([2.0, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError):
            tabulated_efficiency([1.0, 2.0], [1.0, -0.2])

    def test_invalid_absolute_factor_rejected(self):
        with pytest.raises(ValueError):
            CollectionOptics(absolute_factor=0.0)


class TestInstrumentResponse:
    PROFILE = InstrumentalProfile(1.6, 100.0)

    def test_resolution_only_is_pure_convolution(self):
        spectrum = line_spectrum()
        response = InstrumentResponse(
            instrumental_profile=self.PROFILE,
            collection_optics=CollectionOptics(absolute_factor=3.0),
            noise_model=NoiseModel(),
        )
        resolution = response.resolution_only(spectrum)
        assert np.allclose(
            resolution.intensity,
            self.PROFILE.convolve(spectrum).intensity,
            rtol=1e-14,
        )
        # Intensity scale untouched (no optics factor applied).
        assert resolution.intensity.sum() == pytest.approx(
            spectrum.intensity.sum(), rel=1e-12
        )

    def test_noise_free_is_deterministic_full_pipeline(self):
        spectrum = line_spectrum()
        response = InstrumentResponse(
            instrumental_profile=self.PROFILE,
            collection_optics=CollectionOptics(absolute_factor=3.0),
            noise_model=NoiseModel(read_noise_rms_counts=5.0),
        )
        clean_a = response.noise_free(spectrum)
        clean_b = response.noise_free(spectrum)
        assert np.array_equal(clean_a.intensity, clean_b.intensity)
        expected = (
            self.PROFILE.convolve(spectrum).intensity * 3.0
        )
        assert np.allclose(clean_a.intensity, expected, rtol=1e-12)
        assert "noise_seed" not in clean_a.metadata

    def test_apply_scatters_around_noise_free_mean(self):
        spectrum = line_spectrum()
        response = InstrumentResponse(
            instrumental_profile=self.PROFILE,
            collection_optics=CollectionOptics(absolute_factor=3.0),
            noise_model=NoiseModel(),
        )
        clean = response.noise_free(spectrum)
        noisy = response.apply(spectrum, seed=11)
        assert noisy.metadata["noise_seed"] == 11
        # Total counts agree within Poisson counting statistics.
        total = clean.intensity.sum()
        assert noisy.intensity.sum() == pytest.approx(
            total, abs=5 * np.sqrt(total)
        )

    def test_seed_required_when_noise_configured(self):
        response = InstrumentResponse(noise_model=NoiseModel())
        with pytest.raises(ValueError, match="seed is required"):
            response.apply(line_spectrum())

    def test_stages_are_individually_optional(self):
        spectrum = line_spectrum()
        empty = InstrumentResponse()
        assert empty.apply(spectrum) is spectrum  # no-op pipeline
        no_noise = InstrumentResponse(instrumental_profile=self.PROFILE)
        assert np.allclose(
            no_noise.apply(spectrum).intensity,
            self.PROFILE.convolve(spectrum).intensity,
        )

    def test_pixel_stage_and_metadata_chain(self):
        spectrum = line_spectrum()
        response = InstrumentResponse(
            instrumental_profile=self.PROFILE,
            collection_optics=CollectionOptics(absolute_factor=2.0),
            n_pixels=200,
        )
        out = response.noise_free(spectrum)
        assert out.wavelength_m.size == 200
        assert out.metadata["n_pixels"] == 200
        assert out.metadata["instrumental_fwhm_m"] == self.PROFILE.fwhm_m
        assert out.metadata["collection_absolute_factor"] == 2.0

    def test_pixels_without_profile_rejected(self):
        response = InstrumentResponse(n_pixels=100)
        with pytest.raises(ValueError, match="pixel sampling requires"):
            response.noise_free(line_spectrum())

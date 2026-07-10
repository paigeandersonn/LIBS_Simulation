"""
Tests for libssim.instrument.noise.

Covers the Phase 4 acceptance criteria (implementation_plan.md):
- Noise statistics behave correctly (Poisson variance ~ mean, Gaussian
  read noise, dark current offset).
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.spectrum import Spectrum
from libssim.instrument.noise import NoiseModel

N = 20000  # samples for statistical assertions


def flat_spectrum(level: float, n: int = N) -> Spectrum:
    grid = 400e-9 + np.arange(n) * 1e-12
    return Spectrum(wavelength_m=grid, intensity=np.full(n, level))


class TestStatistics:
    def test_poisson_variance_matches_mean(self):
        # Acceptance: shot noise variance ~ mean.
        noisy = NoiseModel().apply(flat_spectrum(1000.0), seed=42)
        assert noisy.intensity.mean() == pytest.approx(1000.0, rel=5e-3)
        assert noisy.intensity.var() == pytest.approx(1000.0, rel=0.05)

    def test_gaussian_read_noise_recovered(self):
        # Acceptance: additive Gaussian read noise of the configured rms.
        model = NoiseModel(
            read_noise_rms_counts=12.0, include_shot_noise=False
        )
        noisy = model.apply(flat_spectrum(500.0), seed=7)
        residual = noisy.intensity - 500.0
        assert residual.mean() == pytest.approx(0.0, abs=12.0 * 3 / np.sqrt(N))
        assert residual.std() == pytest.approx(12.0, rel=0.03)

    def test_dark_current_offset_recovered(self):
        # Acceptance: dark level appears as a mean offset (with shot
        # statistics) even for zero signal.
        model = NoiseModel(dark_mean_counts=300.0)
        noisy = model.apply(flat_spectrum(0.0), seed=3)
        assert noisy.intensity.mean() == pytest.approx(300.0, rel=1e-2)
        assert noisy.intensity.var() == pytest.approx(300.0, rel=0.05)

    def test_background_adds_to_expectation(self):
        model = NoiseModel(
            background_mean_counts=250.0, include_shot_noise=False
        )
        noisy = model.apply(flat_spectrum(100.0), seed=1)
        assert np.allclose(noisy.intensity, 350.0)


class TestReproducibility:
    def test_same_seed_identical_output(self):
        model = NoiseModel(read_noise_rms_counts=5.0, dark_mean_counts=50.0)
        spectrum = flat_spectrum(800.0, n=512)
        a = model.apply(spectrum, seed=1234)
        b = model.apply(spectrum, seed=1234)
        assert np.array_equal(a.intensity, b.intensity)

    def test_different_seeds_differ(self):
        model = NoiseModel()
        spectrum = flat_spectrum(800.0, n=512)
        a = model.apply(spectrum, seed=1)
        b = model.apply(spectrum, seed=2)
        assert not np.array_equal(a.intensity, b.intensity)

    def test_metadata_records_seed_and_parameters(self):
        model = NoiseModel(read_noise_rms_counts=5.0)
        noisy = model.apply(flat_spectrum(10.0, n=16), seed=99)
        assert noisy.metadata["noise_seed"] == 99
        assert noisy.metadata["noise_model"]["read_noise_rms_counts"] == 5.0
        assert noisy.metadata["intensity_units"] == "counts (noisy)"


class TestValidation:
    def test_negative_intensity_rejected(self):
        grid = 400e-9 + np.arange(4) * 1e-12
        spectrum = Spectrum(
            wavelength_m=grid, intensity=np.array([1.0, -0.5, 2.0, 3.0])
        )
        with pytest.raises(ValueError, match="non-negative"):
            NoiseModel().apply(spectrum, seed=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"read_noise_rms_counts": -1.0},
            {"dark_mean_counts": np.inf},
            {"background_mean_counts": -0.1},
        ],
    )
    def test_invalid_construction_rejected(self, kwargs):
        with pytest.raises(ValueError):
            NoiseModel(**kwargs)

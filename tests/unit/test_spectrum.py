"""
tests/unit/test_spectrum.py
===========================
Comprehensive unit tests for libssim.core.spectrum.Spectrum

These tests verify:
- Immutability (frozen dataclass)
- Input validation and automatic numpy conversion
- Convenience properties (wavelength_nm, n_points)
- Methods: to_dataframe(), summary()
- Metadata handling
- Behavior expected for MC-LIBS style spectrum comparison (Phase 5)

Run with: PYTHONPATH=. python -m pytest tests/unit/test_spectrum.py -v
"""

import pytest
import numpy as np

from libssim.core.spectrum import Spectrum


def make_spectrum(**overrides):
    """Helper to create a basic valid Spectrum for testing."""
    base = dict(
        wavelength_m=np.linspace(200e-9, 800e-9, 1000),
        intensity=np.random.random(1000) * 1000,
        metadata={"source": "test", "R_correlation": 0.987}
    )
    base.update(overrides)
    return Spectrum(**base)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_spectrum_is_immutable():
    s = make_spectrum()
    with pytest.raises(AttributeError):
        s.wavelength_m = np.array([1, 2, 3])
    with pytest.raises(AttributeError):
        s.intensity = np.array([10, 20])


# ---------------------------------------------------------------------------
# Input validation & automatic conversion
# ---------------------------------------------------------------------------

def test_accepts_lists_and_converts_to_float64():
    s = Spectrum(
        wavelength_m=[200e-9, 300e-9, 400e-9],
        intensity=[100, 200, 150]
    )
    assert isinstance(s.wavelength_m, np.ndarray)
    assert isinstance(s.intensity, np.ndarray)
    assert s.wavelength_m.dtype == np.float64
    assert s.intensity.dtype == np.float64


def test_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="must have same shape"):
        Spectrum(
            wavelength_m=np.array([200e-9, 300e-9]),
            intensity=np.array([100, 200, 150])
        )


def test_raises_on_non_1d_arrays():
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        Spectrum(
            wavelength_m=np.array([[200e-9], [300e-9]]),
            intensity=np.array([[100], [200]])
        )


def test_empty_arrays_allowed():
    s = Spectrum(wavelength_m=np.array([]), intensity=np.array([]))
    assert s.n_points == 0
    assert len(s.wavelength_nm) == 0


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------

def test_wavelength_nm_property():
    s = make_spectrum()
    expected = s.wavelength_m * 1e9
    np.testing.assert_allclose(s.wavelength_nm, expected)


def test_n_points_property():
    s = make_spectrum()
    assert s.n_points == len(s.wavelength_m)
    assert s.n_points == 1000


# ---------------------------------------------------------------------------
# Metadata handling
# ---------------------------------------------------------------------------

def test_metadata_is_dict():
    s = make_spectrum(metadata={"experiment_id": "EXP-001"})
    assert isinstance(s.metadata, dict)
    assert s.metadata["experiment_id"] == "EXP-001"


def test_metadata_defaults_to_empty_dict():
    s = Spectrum(wavelength_m=np.array([1e-9]), intensity=np.array([100]))
    assert s.metadata == {}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def test_to_dataframe():
    s = make_spectrum()
    df = s.to_dataframe()
    assert "wavelength_m" in df.columns
    assert "wavelength_nm" in df.columns
    assert "intensity" in df.columns
    assert len(df) == s.n_points


def test_summary_contains_key_info():
    s = make_spectrum()
    summary = s.summary()
    assert "Spectrum with" in summary
    assert "Wavelength range:" in summary
    assert "Intensity range:" in summary
    assert "R_correlation" in summary  # because we set it in make_spectrum


def test_summary_handles_missing_r_correlation():
    s = make_spectrum(metadata={})  # no R_correlation
    summary = s.summary()
    assert "Correlation R with experiment" not in summary


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

def test_repr_is_readable():
    s = make_spectrum()
    r = repr(s)
    assert "Spectrum(n_points=" in r
    assert "λ_range=" in r
    assert "metadata_keys=" in r
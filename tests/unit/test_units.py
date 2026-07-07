"""
tests/unit/test_units.py
========================
Unit tests for libssim.core.units

Focuses on verifying that all unit conversion functions work correctly,
including round-trip conversions and support for both scalars and numpy arrays.
"""

import numpy as np
import pytest

from libssim.core import units as u


class TestLengthConversions:
    """Test wavelength / length unit conversions."""

    def test_m_to_nm_and_back(self):
        """Round-trip conversion meters ↔ nanometers should recover original value."""
        values_m = np.array([1e-9, 500e-9, 656.3e-9, 1e-6])
        nm = u.m_to_nm(values_m)
        back_to_m = u.nm_to_m(nm)
        np.testing.assert_allclose(back_to_m, values_m)

    def test_nm_to_m_scalar(self):
        assert u.nm_to_m(500.0) == pytest.approx(5e-7)

    def test_m_to_nm_scalar(self):
        assert u.m_to_nm(5e-7) == pytest.approx(500.0)

    def test_angstrom_conversions(self):
        """Test Ångstrom ↔ meter conversions."""
        ang = 6562.72  # H-alpha in Å
        meters = u.angstrom_to_m(ang)
        back = u.m_to_angstrom(meters)
        assert meters == pytest.approx(6.56272e-7)
        assert back == pytest.approx(ang)


class TestEnergyTemperatureConversions:
    """Test energy (eV) and temperature (K) conversions."""

    def test_ev_to_k_and_back(self):
        """Round-trip eV ↔ K conversion."""
        values_ev = np.array([0.5, 1.0, 2.0, 10.0])
        k = u.ev_to_k(values_ev)
        back = u.k_to_ev(k)
        np.testing.assert_allclose(back, values_ev, rtol=1e-10)

    def test_k_to_ev_known_value(self):
        """1 eV should correspond to approximately 11604.5 K."""
        assert u.k_to_ev(11604.518) == pytest.approx(1.0, abs=1e-3)

    def test_ev_to_k_scalar(self):
        assert u.ev_to_k(1.0) == pytest.approx(11604.518, abs=0.1)


class TestDensityConversions:
    """Test number density conversions (cm⁻³ ↔ m⁻³)."""

    def test_cm3_to_m3_and_back(self):
        values_cm3 = np.array([1e10, 1e15, 1e23])
        m3 = u.cm3_to_m3(values_cm3)
        back = u.m3_to_cm3(m3)
        np.testing.assert_allclose(back, values_cm3)

    def test_density_scalar(self):
        assert u.cm3_to_m3(1e16) == pytest.approx(1e22)
        assert u.m3_to_cm3(1e22) == pytest.approx(1e16)


class TestWavenumberConversions:
    """Test wavenumber (cm⁻¹) conversions."""

    def test_wavenumber_round_trip(self):
        """Convert wavenumber → meters → wavenumber."""
        wn = np.array([1000.0, 15000.0, 25000.0])  # cm⁻¹
        meters = u.wavenumber_to_m(wn)
        back = u.m_to_wavenumber(meters)
        np.testing.assert_allclose(back, wn)


class TestFrequencyEnergyConversions:
    """Test frequency ↔ energy conversions."""

    def test_frequency_to_ev_round_trip(self):
        freq = np.array([1e14, 5e14, 1e15])  # Hz
        ev = u.frequency_to_ev(freq)
        back = u.ev_to_frequency(ev)
        np.testing.assert_allclose(back, freq, rtol=1e-10)

    def test_ev_to_frequency_known(self):
        # 1 eV should correspond to frequency = E / h
        freq = u.ev_to_frequency(1.0)
        assert freq == pytest.approx(2.417989e14, rel=1e-6)


class TestDopplerWidth:
    """Basic sanity check for Doppler width function."""

    def test_doppler_width_positive(self):
        width = u.doppler_width_nm(
            wavelength_nm=500.0,
            temperature_k=10000,
            atomic_mass_u=27.0,  # Aluminum
        )
        assert width > 0
        assert width < 1.0  # Should be much less than 1 nm at these conditions


# Optional: Test that functions accept both float and array
def test_functions_accept_array_and_scalar():
    scalar_result = u.m_to_nm(5e-7)
    array_result = u.m_to_nm(np.array([5e-7, 6e-7]))

    assert isinstance(scalar_result, (float, np.floating))
    assert isinstance(array_result, np.ndarray)
    assert len(array_result) == 2
"""SI unit conversion utilities for LIBS plasma modeling.

All functions are pure, side-effect free, and vectorized where it makes
sense (accept float or numpy array).

These conversions bridge common spectroscopy units (nm, eV, cm$^{-3}$)
to the strict SI units used internally by PlasmaState, Saha solver,
line profile calculations, etc.

Physical motivation (from Herrera 2008)
---------------------------------------
- Wavelengths are almost always reported in nm in LIBS spectra (220-700 nm range
  in the thesis experimental work).
- Energies in eV for atomic levels (NIST/Blaise databases).
- Number densities in cm$^{-3}$ in older literature and many tables;
  SI uses m$^{-3}$.
- Temperature conversions between eV and K appear in Saha-Boltzmann plots
  and LTE validation.

Examples
--------
>>> from libssim.core.units import nm_to_m, ev_to_k
>>> nm_to_m(656.272)          # H-alpha
2.99792458e-07
>>> ev_to_k(1.0)
11604.518...

References
----------
Herrera (2008) Ch. 3 (line broadening), Ch. 5 (Saha-Boltzmann), pp. 19-25 symbols.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .constants import C, KB, EV, KB_EV


def ev_to_k(energy_ev: ArrayLike) -> np.ndarray:
    """Convert energy from electron-volts to Kelvin."""
    return np.asarray(energy_ev) / KB_EV


def k_to_ev(temperature_k: ArrayLike) -> np.ndarray:
    """Convert temperature in Kelvin to energy in eV."""
    return np.asarray(temperature_k) * KB_EV


def nm_to_m(wavelength_nm: ArrayLike) -> np.ndarray:
    """Convert wavelength from nanometers to meters (SI)."""
    return np.asarray(wavelength_nm) * 1e-9


def m_to_nm(wavelength_m: ArrayLike) -> np.ndarray:
    """Convert wavelength from meters to nanometers."""
    return np.asarray(wavelength_m) * 1e9


def angstrom_to_m(wavelength_ang: ArrayLike) -> np.ndarray:
    """Convert wavelength from Ångstroms to meters."""
    return np.asarray(wavelength_ang) * 1e-10


def m_to_angstrom(wavelength_m: ArrayLike) -> np.ndarray:
    """Convert wavelength from meters to Ångstroms."""
    return np.asarray(wavelength_m) * 1e10


def cm3_to_m3(density_cm3: ArrayLike) -> np.ndarray:
    """
    Convert number density from cm^{-3} to m^{-3}.
    Common in atomic physics tables and older LIBS papers
    (including Herrera 2008 tables and MC-LIBS number densities).
    """
    return np.asarray(density_cm3) * 1e6


def m3_to_cm3(density_m3: ArrayLike) -> np.ndarray:
    """Convert number density from m^{-3} to cm^{-3}."""
    return np.asarray(density_m3) * 1e-6


def wavenumber_to_m(wavenumber_cm1: ArrayLike) -> np.ndarray:
    """Convert wavenumber (cm^{-1}) to wavelength in meters."""
    return 1.0 / (np.asarray(wavenumber_cm1) * 100.0)


def m_to_wavenumber(wavelength_m: ArrayLike) -> np.ndarray:
    """Convert wavelength in meters to wavenumber in cm^{-1}."""
    return 1.0 / (np.asarray(wavelength_m) * 100.0)


def frequency_to_ev(frequency_hz: ArrayLike) -> np.ndarray:
    r"""Convert frequency (Hz) to energy (eV) using $E = h\nu$."""
    from .constants import H
    return np.asarray(frequency_hz) * H / EV


def ev_to_frequency(energy_ev: ArrayLike) -> np.ndarray:
    """Convert energy (eV) to frequency (Hz)."""
    from .constants import H
    return np.asarray(energy_ev) * EV / H


def doppler_width_nm(
    wavelength_nm: float,
    temperature_k: float,
    atomic_mass_u: float,
) -> float:
    """
    Calculate thermal Doppler FWHM in nm (approximate formula).
    Used in Phase 2 line_profiles.py. Formula from standard spectroscopy
    (Herrera Ch. 3 Doppler broadening section).
    """
    from .constants import C, KB
    v_mp = np.sqrt(2 * KB * temperature_k / (atomic_mass_u * 1.66053906660e-27))
    delta_lambda = (wavelength_nm * 1e-9) * (v_mp / C) * 1e9
    return 2.35482 * delta_lambda / 2.0
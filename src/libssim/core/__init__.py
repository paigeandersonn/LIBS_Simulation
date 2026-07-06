"""
libssim.core
============
Immutable core infrastructure for the LIBS simulation framework.

Exposes:
- Physical constants (SI)
- Unit conversion utilities
- PlasmaState dataclass (frozen, validated)
- Spectrum dataclass (for wavelength/intensity pairs + metadata)

All quantities use strict SI units unless explicitly documented otherwise.
"""

from .constants import *
from .units import *
from .state import PlasmaState
from .spectrum import Spectrum

__all__ = [
    "C", "H", "KB", "ME", "E", "EPSILON0", "NA", "SIGMA",
    "ev_to_k", "k_to_ev", "nm_to_m", "m_to_nm", "cm3_to_m3", "m3_to_cm3",
    "angstrom_to_m", "m_to_angstrom",
    "PlasmaState",
    "Spectrum",
]
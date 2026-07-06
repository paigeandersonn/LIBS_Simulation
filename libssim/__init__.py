"""
libssim
=======
Open-source forward-modeling framework for Laser-Induced Breakdown Spectroscopy (LIBS)
plasma emission, inspired by and extending the modeling approaches in
Herrera (2008) PhD thesis.

Phase 0 provides the immutable core: physical constants, SI unit conversions,
PlasmaState, and Spectrum containers.
"""

__version__ = "0.1.0-phase0"
__author__ = "libssim contributors (building on Herrera 2008)"

from .core.state import PlasmaState
from .core.spectrum import Spectrum

__all__ = ["PlasmaState", "Spectrum"]
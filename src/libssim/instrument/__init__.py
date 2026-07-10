"""
libssim.instrument
==================
Instrument response layer (Phase 4): spectral resolution, collection
efficiency and detector noise.

Implements the detection chain of Herrera (2008): the instrumental
line-spread function controlled by the slit width (Eqs. 3-19/3-20/3-22,
pp. 57-60; Fig. 3-5 p. 64), the spectral efficiency of the collection
system (Eqs. 5-9..5-11, p. 104), and standard detector noise
statistics (shot + readout + dark, reproducible via mandatory seeds).

Provides:
- `InstrumentalProfile`, `diffraction_limited_bandpass_m`
  (`spectrometer`)
- `CollectionOptics`, `tabulated_efficiency` (`optics`)
- `NoiseModel` (`noise`)
- `InstrumentResponse` — the composite pipeline with `resolution_only`
  and `noise_free` validation paths (`response`)
"""

from .noise import NoiseModel
from .optics import CollectionOptics, tabulated_efficiency
from .response import InstrumentResponse
from .spectrometer import InstrumentalProfile, diffraction_limited_bandpass_m

__all__ = [
    "InstrumentalProfile",
    "diffraction_limited_bandpass_m",
    "CollectionOptics",
    "tabulated_efficiency",
    "NoiseModel",
    "InstrumentResponse",
]

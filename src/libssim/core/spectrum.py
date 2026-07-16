"""Spectrum dataclass for storing wavelength-intensity pairs plus
rich metadata.

This is the primary output container of the forward model (after Phase 4
instrumental effects) and the target for comparison in the MC-LIBS style
optimizer (Phase 5).

Design goals
------------
- Immutable (frozen) for safety during optimization / Monte Carlo loops.
- Supports both high-resolution physics arrays and binned experimental-style data.
- Rich metadata for provenance (config, random seed, plasma state snapshot,
  instrumental parameters, correlation coefficient R with experiment, etc.).
- Easy conversion to pandas DataFrame or plotting (matplotlib / plotly).

Physical context (Herrera 2008)
-------------------------------
LIBS spectra in the thesis span ~220–700 nm. Experimental data are acquired
in multiple overlapping windows and spliced (Ch. 4). Synthetic spectra must
match this format after convolution with instrumental function (slit width,
diffraction, aberrations — Ch. 3 & 4) and addition of noise.

The Spectrum object is what gets compared to the "experimental spectrum"
in the correlation coefficient R maximization of MC-LIBS (Ch. 5, Fig. 5-6).

Examples
--------
After the full pipeline:

>>> spectrum = simulator.simulate(state)  # doctest: +SKIP
>>> print(spectrum.metadata["R_correlation"])  # doctest: +SKIP
>>> spectrum.to_dataframe().plot(x="wavelength_nm", y="intensity")  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    """
    Immutable container for a (wavelength, intensity) spectrum plus metadata.

    Attributes
    ----------
    wavelength_m : np.ndarray
        Wavelength array in **meters** (SI). Shape (N,).
        Use the `.wavelength_nm` property for nanometers.

    intensity : np.ndarray
        Intensity (or radiance) array. Shape must match `wavelength_m`.
        Units are typically W cm^{-2} sr^{-1} nm^{-1} or counts (after
        instrumental model), but are documented in metadata["intensity_units"].

    metadata : Dict[str, Any]
        Arbitrary provenance and calculation information. Recommended keys:
        - "plasma_state" : dict or PlasmaState summary
        - "config" : YAML or dict of simulation parameters
        - "timestamp" : ISO time
        - "random_seed"
        - "instrument" : slit_width_um, gate_width_s, etc.
        - "R_correlation" : float (with experimental reference)
        - "species_contributions" : dict of per-element synthetic spectra
        - "calculation_notes"

    Notes
    -----
    Both arrays are stored as float64 numpy arrays. The dataclass does **not**
    copy the input arrays (for memory efficiency with large spectra); callers
    should treat them as read-only.
    """

    wavelength_m: np.ndarray
    intensity: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.wavelength_m, np.ndarray):
            object.__setattr__(self, "wavelength_m", np.asarray(self.wavelength_m, dtype=np.float64))
        if not isinstance(self.intensity, np.ndarray):
            object.__setattr__(self, "intensity", np.asarray(self.intensity, dtype=np.float64))

        if self.wavelength_m.shape != self.intensity.shape:
            raise ValueError(
                f"wavelength_m and intensity must have same shape. "
                f"Got {self.wavelength_m.shape} vs {self.intensity.shape}"
            )
        if self.wavelength_m.ndim != 1:
            raise ValueError("wavelength_m and intensity must be 1-dimensional arrays")

        # Ensure metadata is a dict (defensive)
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def wavelength_nm(self) -> np.ndarray:
        """Convenience property: wavelength in nanometers."""
        from .units import m_to_nm
        return m_to_nm(self.wavelength_m)

    @property
    def n_points(self) -> int:
        """Number of spectral points."""
        return len(self.wavelength_m)

    def to_dataframe(self):
        """Return as pandas DataFrame (requires pandas)."""
        import pandas as pd
        df = pd.DataFrame({
            "wavelength_m": self.wavelength_m,
            "wavelength_nm": self.wavelength_nm,
            "intensity": self.intensity,
        })
        for k, v in self.metadata.items():
            if not isinstance(v, (list, dict, np.ndarray)):
                df[k] = v
        return df

    def __repr__(self) -> str:
        meta_keys = list(self.metadata.keys())[:5]
        return (
            f"Spectrum(n_points={self.n_points}, "
            f"λ_range=[{self.wavelength_nm.min():.2f}–{self.wavelength_nm.max():.2f}] nm, "
            f"metadata_keys={meta_keys}...)"
        )

    def summary(self) -> str:
        """Human-readable multi-line summary (useful for logging)."""
        lines = [
            f"Spectrum with {self.n_points} points",
            f"  Wavelength range: {self.wavelength_nm.min():.3f} – {self.wavelength_nm.max():.3f} nm",
            f"  Intensity range: {self.intensity.min():.3e} – {self.intensity.max():.3e}",
            f"  Metadata keys: {list(self.metadata.keys())}",
        ]
        if "R_correlation" in self.metadata:
            lines.append(f"  Correlation R with experiment: {self.metadata['R_correlation']:.4f}")
        return "\n".join(lines)
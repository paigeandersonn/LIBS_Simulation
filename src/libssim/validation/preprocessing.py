"""Experimental-spectrum preprocessing for model comparison (Phase 4
validation workflow).

Mirrors the preparation steps the thesis applies before CF-LIBS
analysis — "after the raw spectrum has been corrected for background
and detector spectral efficiency" (p. 103): background removal,
optional normalization, and cropping to the modeled window. Wavelength
resampling lives in `libssim.analysis.resample` (reused, not
duplicated).

All functions are pure Spectrum -> Spectrum; each records what it did
in the metadata so a processed spectrum carries its preprocessing
provenance.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid

from ..core.spectrum import Spectrum

#: Background-fit window: (low, high) wavelengths in meters.
Window = Tuple[float, float]


def crop(spectrum: Spectrum, low_m: float, high_m: float) -> Spectrum:
    """
    Restrict a spectrum to [low_m, high_m] (inclusive).

    Use before resampling so endpoint clamping (analysis.resample docs)
    never manufactures data outside the measured range.
    """
    low, high = float(low_m), float(high_m)
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError("require finite low_m < high_m")
    mask = (spectrum.wavelength_m >= low) & (spectrum.wavelength_m <= high)
    if mask.sum() < 2:
        raise ValueError(
            "fewer than 2 samples inside the crop window "
            f"[{low:.4g}, {high:.4g}] m"
        )
    metadata = dict(spectrum.metadata)
    metadata["crop_window_m"] = (low, high)
    return Spectrum(
        wavelength_m=spectrum.wavelength_m[mask],
        intensity=spectrum.intensity[mask],
        metadata=metadata,
    )


def subtract_background(
    spectrum: Spectrum,
    windows: Sequence[Window],
    fit: str = "constant",
) -> Spectrum:
    """
    Remove a baseline estimated from line-free spectral windows.

    The thesis' operational background correction (p. 103) in its two
    simplest forms:

    - ``fit="constant"``: subtract the mean intensity over the windows
      (dark/continuum pedestal).
    - ``fit="linear"``: least-squares straight line over the window
      samples (sloped continuum, e.g. recombination background).

    Parameters
    ----------
    spectrum : Spectrum
        Input (typically experimental counts).
    windows : sequence of (low_m, high_m)
        Line-free regions used to estimate the baseline; at least one
        window containing at least 2 samples in total.
    fit : {"constant", "linear"}
        Baseline model.

    Returns
    -------
    Spectrum
        Baseline-subtracted spectrum (negative residuals are kept —
        clipping would bias noise statistics); metadata records the
        fit and windows.
    """
    if fit not in ("constant", "linear"):
        raise ValueError("fit must be 'constant' or 'linear'")
    if not windows:
        raise ValueError("provide at least one background window")
    mask = np.zeros(spectrum.wavelength_m.shape, dtype=bool)
    for low, high in windows:
        if not (float(low) < float(high)):
            raise ValueError("each window needs low < high")
        mask |= (spectrum.wavelength_m >= float(low)) & (
            spectrum.wavelength_m <= float(high)
        )
    if mask.sum() < 2:
        raise ValueError("background windows contain fewer than 2 samples")

    x = spectrum.wavelength_m[mask]
    y = spectrum.intensity[mask]
    if fit == "constant":
        baseline = np.full(
            spectrum.wavelength_m.shape, float(np.mean(y))
        )
    else:
        # Center the abscissa for a well-conditioned linear fit.
        x0 = x.mean()
        slope, intercept = np.polyfit(x - x0, y, deg=1)
        baseline = slope * (spectrum.wavelength_m - x0) + intercept

    metadata = dict(spectrum.metadata)
    metadata["background_fit"] = fit
    metadata["background_windows_m"] = [
        (float(lo), float(hi)) for lo, hi in windows
    ]
    return Spectrum(
        wavelength_m=spectrum.wavelength_m,
        intensity=spectrum.intensity - baseline,
        metadata=metadata,
    )


def normalize(spectrum: Spectrum, mode: str = "peak") -> Spectrum:
    """
    Scale a spectrum for shape comparison.

    - ``mode="peak"``: peak value -> 1 (the usual overlay convention).
    - ``mode="area"``: unit trapezoidal integral over wavelength.

    R of Eq. 5-56 is scale-invariant, but the rms residual and overlay
    plots need a common scale — normalize both spectra the same way.
    """
    if mode not in ("peak", "area"):
        raise ValueError("mode must be 'peak' or 'area'")
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    if mode == "peak":
        scale = float(np.max(np.abs(intensity)))
    else:
        scale = float(abs(trapezoid(intensity, spectrum.wavelength_m)))
    if scale == 0.0 or not np.isfinite(scale):
        raise ValueError(f"cannot normalize: {mode} scale is {scale}")
    metadata = dict(spectrum.metadata)
    metadata["normalized"] = mode
    return Spectrum(
        wavelength_m=spectrum.wavelength_m,
        intensity=intensity / scale,
        metadata=metadata,
    )

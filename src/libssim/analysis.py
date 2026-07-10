"""
libssim.analysis
================
Plotting, comparison and sweep helpers for synthetic vs experimental
spectra (Phase 4).

Physical Context (Herrera 2008)
-------------------------------
The MC-LIBS inverse method judges a synthetic spectrum against the
experimental one with the linear correlation coefficient, Eq. 5-56,
p. 122:

    R = sum_i (x_i - <x>)(y_i - <y>)
        / sqrt( sum_i (x_i - <x>)^2 * sum_i (y_i - <y>)^2 )

"the value of the correlation coefficient R determines the similarity
between the experimental and synthetic spectrum" (p. 122; example
R = 0.9913, Fig. 5-6). `correlation_coefficient` implements exactly
this metric; `load_spectrum_csv` and `resample` bring experimental
data onto the model grid so the comparison is well-posed.

Plotting requires matplotlib — an optional dependency (install with
``pip install libssim[viz]``); everything else in this module is
matplotlib-free.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-56 p. 122;
Fig. 5-6.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from .core.spectrum import Spectrum

if TYPE_CHECKING:  # circular-import-free type hint for sweep_gate_delay
    from .temporal.integrator import GateIntegrator

#: Wavelength-unit converters to meters for CSV import.
_UNIT_TO_M = {
    "m": 1.0,
    "nm": 1.0e-9,
    "angstrom": 1.0e-10,
    "um": 1.0e-6,
}


def load_spectrum_csv(
    path: Union[str, Path],
    wavelength_column: int = 0,
    intensity_column: int = 1,
    wavelength_unit: str = "nm",
    delimiter: str = ",",
    skip_header: int = 0,
    intensity_units: str = "arbitrary (experimental)",
) -> Spectrum:
    """
    Load an experimental spectrum from a delimited text file.

    Parameters
    ----------
    path : str or Path
        File to read (columns of numbers; comment lines starting with
        '#' are ignored by numpy).
    wavelength_column, intensity_column : int, optional
        Zero-based column indices (defaults 0 and 1).
    wavelength_unit : {"nm", "angstrom", "um", "m"}, optional
        Unit of the wavelength column (default nm, the LIBS
        convention; converted to SI meters).
    delimiter : str, optional
        Column delimiter (default comma).
    skip_header : int, optional
        Leading lines to skip (default 0).
    intensity_units : str, optional
        Recorded in metadata (experimental units are instrument
        counts unless calibrated).

    Returns
    -------
    Spectrum
        Wavelengths in meters (sorted ascending; duplicates kept),
        metadata recording provenance.
    """
    unit = wavelength_unit.strip().lower()
    if unit not in _UNIT_TO_M:
        raise ValueError(
            f"unknown wavelength_unit {wavelength_unit!r}; "
            f"use one of {sorted(_UNIT_TO_M)}"
        )
    data = np.genfromtxt(
        Path(path), delimiter=delimiter, skip_header=skip_header, comments="#"
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_columns = data.shape[1]
    if max(wavelength_column, intensity_column) >= n_columns:
        raise ValueError(
            f"file has {n_columns} columns; requested columns "
            f"{wavelength_column} and {intensity_column}"
        )
    wavelength = data[:, wavelength_column] * _UNIT_TO_M[unit]
    intensity = data[:, intensity_column]
    valid = np.isfinite(wavelength) & np.isfinite(intensity)
    if not np.any(valid):
        raise ValueError(f"no valid (wavelength, intensity) rows in {path}")
    wavelength, intensity = wavelength[valid], intensity[valid]
    order = np.argsort(wavelength)
    return Spectrum(
        wavelength_m=wavelength[order],
        intensity=intensity[order],
        metadata={
            "source": str(path),
            "wavelength_unit_original": wavelength_unit,
            "intensity_units": intensity_units,
        },
    )


def resample(spectrum: Spectrum, wavelength_m: ArrayLike) -> Spectrum:
    """
    Linearly interpolate a spectrum onto a new wavelength grid.

    Points outside the original range are clamped to the endpoint
    values (np.interp behaviour) — compare only over the overlapping
    window. Needed before `correlation_coefficient` or instrument
    convolution when grids differ.
    """
    grid = np.asarray(wavelength_m, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("wavelength_m must be a 1-D increasing grid")
    intensity = np.interp(grid, spectrum.wavelength_m, spectrum.intensity)
    metadata = dict(spectrum.metadata)
    metadata["resampled"] = True
    return Spectrum(wavelength_m=grid, intensity=intensity, metadata=metadata)


def correlation_coefficient(
    synthetic: Spectrum, experimental: Spectrum
) -> float:
    """
    Linear correlation coefficient R between two spectra —
    Herrera (2008), Eq. 5-56, p. 122 (the MC-LIBS cost function).

    Both spectra must share the same wavelength grid (use `resample`);
    R is scale- and offset-invariant, so uncalibrated experimental
    counts compare directly against physical synthetic units.

    Returns
    -------
    float
        R in [-1, 1]; 1 means identical shapes (thesis example
        R = 0.9913, Fig. 5-6).
    """
    x = np.asarray(synthetic.intensity, dtype=np.float64)
    y = np.asarray(experimental.intensity, dtype=np.float64)
    if synthetic.wavelength_m.shape != experimental.wavelength_m.shape or not (
        np.allclose(
            synthetic.wavelength_m, experimental.wavelength_m, rtol=1e-12
        )
    ):
        raise ValueError(
            "spectra must share the same wavelength grid; use "
            "analysis.resample first"
        )
    dx = x - x.mean()
    dy = y - y.mean()
    denominator = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    if denominator == 0.0:
        raise ValueError(
            "correlation undefined for a constant spectrum (zero variance)"
        )
    return float(np.sum(dx * dy) / denominator)


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------
def sweep(
    values: Sequence[Any],
    build: Callable[[Any], Spectrum],
    label: str = "sweep_value",
) -> List[Spectrum]:
    """
    Evaluate `build(value)` for each value, tagging results in metadata.

    The generic validation-study helper: any parameter that can be
    closed over (gate delay, slit width, temperature, density, ...)
    can be swept.

    Example
    -------
    >>> spectra = sweep([50e-6, 100e-6, 200e-6],
    ...                 lambda w: response.noise_free(base),  # doctest: +SKIP
    ...                 label="slit_width_um")
    """
    results: List[Spectrum] = []
    for value in values:
        spectrum = build(value)
        if not isinstance(spectrum, Spectrum):
            raise TypeError("build must return a Spectrum")
        metadata = dict(spectrum.metadata)
        metadata[label] = value
        results.append(
            Spectrum(
                wavelength_m=spectrum.wavelength_m,
                intensity=spectrum.intensity,
                metadata=metadata,
            )
        )
    return results


def sweep_gate_delay(
    integrator: "GateIntegrator",
    delays_s: Sequence[float],
    gate_width_s: float,
    **gate_kwargs: Any,
) -> List[Spectrum]:
    """
    Gate-integrated spectra at several delays with a fixed width —
    the temporally-resolved study pattern of thesis Ch. 6-7 (spectra
    "at 5 different delay times", Tables 6-9ff).
    """
    return sweep(
        list(delays_s),
        lambda delay: integrator.gate_integrated(
            delay, gate_width_s, **gate_kwargs
        ),
        label="gate_delay_s",
    )


# ---------------------------------------------------------------------------
# Plotting (optional matplotlib)
# ---------------------------------------------------------------------------
def plot_spectra(
    *spectra: Spectrum,
    labels: Optional[Sequence[str]] = None,
    normalize: bool = False,
    wavelength_unit: str = "nm",
    ax: Any = None,
    title: Optional[str] = None,
):
    """
    Overlay spectra on one axes (matplotlib required:
    ``pip install libssim[viz]``).

    Parameters
    ----------
    *spectra : Spectrum
        One or more spectra to plot.
    labels : sequence of str, optional
        Legend labels; defaults to metadata hints (gate delay, time,
        or index).
    normalize : bool, optional
        Peak-normalize each spectrum (shape comparison; R of Eq. 5-56
        is scale-invariant anyway). Default False.
    wavelength_unit : {"nm", "angstrom", "um", "m"}, optional
        Axis unit (default nm).
    ax : matplotlib Axes, optional
        Target axes; a new figure is created when omitted.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised without viz
        raise ImportError(
            "plotting requires matplotlib: pip install libssim[viz]"
        ) from exc

    unit = wavelength_unit.strip().lower()
    if unit not in _UNIT_TO_M:
        raise ValueError(
            f"unknown wavelength_unit {wavelength_unit!r}; "
            f"use one of {sorted(_UNIT_TO_M)}"
        )
    if not spectra:
        raise ValueError("provide at least one Spectrum")
    if labels is not None and len(labels) != len(spectra):
        raise ValueError("labels must match the number of spectra")

    if ax is None:
        _, ax = plt.subplots()
    for index, spectrum in enumerate(spectra):
        if labels is not None:
            label = labels[index]
        elif "gate_delay_s" in spectrum.metadata:
            label = f"t_d = {spectrum.metadata['gate_delay_s']:.3g} s"
        elif "time_s" in spectrum.metadata:
            label = f"t = {spectrum.metadata['time_s']:.3g} s"
        else:
            label = f"spectrum {index}"
        intensity = spectrum.intensity
        if normalize:
            peak = np.max(np.abs(intensity))
            if peak > 0:
                intensity = intensity / peak
        ax.plot(
            spectrum.wavelength_m / _UNIT_TO_M[unit], intensity, label=label
        )
    ax.set_xlabel(f"wavelength ({wavelength_unit})")
    first_units = spectra[0].metadata.get("intensity_units", "intensity")
    ax.set_ylabel("normalized intensity" if normalize else first_units)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax

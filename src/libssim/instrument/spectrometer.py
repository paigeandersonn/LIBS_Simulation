"""
libssim.instrument.spectrometer
===============================
Instrumental line-spread function and spectral-resolution effects
(Phase 4).

Physical Context (Herrera 2008)
-------------------------------
A real spectrometer broadens every monochromatic input into the
instrumental line profile (Ch. 3, pp. 56-60). Its width is set by the
slits through the geometric spectral bandpass, Eq. 3-19, p. 57:

    dlambda_s = R_d * w_slit

with R_d the reciprocal linear dispersion (nm mm^-1) and w_slit the
slit width; validity is bounded below by the diffraction-limited
bandpass, Eq. 3-20, p. 59 (dlambda_d = R_d * 2*f*lambda / a). The
shape is triangular for equal entrance/exit slits (Fig. 3-5, p. 64),
but "Doppler and instrumental line profiles usually have a Gaussian
distribution" (p. 59) — the Gaussian is the thesis' working treatment
and the default here (also the Phase 4 acceptance criterion). Gaussian
widths combine in quadrature (Eq. 3-22, p. 60):

    dlambda_G = sqrt(dlambda_D^2 + dlambda_I^2)

which is the validation identity for the convolution implemented here.

Implementation Decisions (documented per development_rules.md)
--------------------------------------------------------------
- Convolution is discrete on the spectrum's own grid, which must be
  uniform (checked); the kernel is normalized to unit *sum*, making
  the convolution flux-conserving on the grid.
- Edges use zero padding (numpy 'same' mode): within half a kernel of
  the grid ends the response is dimmed. Pad the wavelength grid beyond
  the region of interest — the same discipline as for Lorentzian line
  wings (line_profiles docs).
- The instrumental FWHM must be resolved by >= 3 grid steps, else the
  discrete kernel misrepresents the profile and the call raises.
- Aberrations enter as an optional extra Gaussian FWHM combined in
  quadrature (Eq. 3-22) — the thesis measures the combined function
  rather than modeling each aberration (Ch. 4, pp. 84-88).

Units: SI (m); slit width and dispersion accepted in their
conventional units (um, nm/mm) and converted internally.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 3-19 p. 57;
Eq. 3-20 p. 59; Eq. 3-22 p. 60; Fig. 3-5 p. 64; Ch. 4 pp. 84-88.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.spectrum import Spectrum

_GAUSS_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def diffraction_limited_bandpass_m(
    reciprocal_dispersion_nm_per_mm: float,
    focal_length_m: float,
    wavelength_m: float,
    beam_width_m: float,
) -> float:
    """
    Diffraction-limited spectral bandpass, Eq. 3-20, p. 59
    (Herrera 2008): dlambda_d = R_d * w_d with w_d = 2*f*lambda/a.

    The slit model of Eq. 3-19 is meaningful only for slit widths above
    w_d; `InstrumentalProfile` warns against configurations below it.
    """
    for name, value in (
        ("reciprocal_dispersion_nm_per_mm", reciprocal_dispersion_nm_per_mm),
        ("focal_length_m", focal_length_m),
        ("wavelength_m", wavelength_m),
        ("beam_width_m", beam_width_m),
    ):
        if not (value > 0 and np.isfinite(value)):
            raise ValueError(f"{name} must be finite and > 0")
    slit_d = 2.0 * focal_length_m * wavelength_m / beam_width_m
    # R_d in nm/mm = 1e-6 (m of wavelength per m of focal plane).
    return reciprocal_dispersion_nm_per_mm * 1.0e-6 * slit_d


@dataclass(frozen=True, eq=False)
class InstrumentalProfile:
    """
    Slit-controlled instrumental line-spread function (LSF).

    Parameters
    ----------
    reciprocal_dispersion_nm_per_mm : float
        R_d of Eq. 3-19 (> 0), a property of the spectrograph
        (thesis Ch. 4: 2nd-order polynomial in wavelength, Fig. 4-10;
        supply the value for your working window).
    slit_width_um : float
        Entrance-slit width (> 0).
    aberration_fwhm_m : float, optional
        Extra Gaussian broadening combined in quadrature (Eq. 3-22),
        default 0.
    shape : {"gaussian", "triangular"}, optional
        LSF shape: Gaussian (default; p. 59) or triangular (equal
        entrance/exit slits, Fig. 3-5, p. 64). The triangular option
        ignores `aberration_fwhm_m` cross-combination subtleties and
        applies quadrature to its FWHM as an approximation
        (documented).

    Notes
    -----
    FWHM (Eq. 3-19): dlambda_I = R_d[nm/mm] * 1e-6 * w_slit[m].
    """

    reciprocal_dispersion_nm_per_mm: float
    slit_width_um: float
    aberration_fwhm_m: float = 0.0
    shape: str = "gaussian"

    def __post_init__(self) -> None:
        if not (
            self.reciprocal_dispersion_nm_per_mm > 0
            and np.isfinite(self.reciprocal_dispersion_nm_per_mm)
        ):
            raise ValueError(
                "reciprocal_dispersion_nm_per_mm must be finite and > 0"
            )
        if not (self.slit_width_um > 0 and np.isfinite(self.slit_width_um)):
            raise ValueError("slit_width_um must be finite and > 0")
        if not (
            self.aberration_fwhm_m >= 0 and np.isfinite(self.aberration_fwhm_m)
        ):
            raise ValueError("aberration_fwhm_m must be finite and >= 0")
        if self.shape not in ("gaussian", "triangular"):
            raise ValueError("shape must be 'gaussian' or 'triangular'")

    # ------------------------------------------------------------------
    @property
    def slit_bandpass_m(self) -> float:
        """Geometric spectral bandpass dlambda_s = R_d * w_slit (Eq. 3-19)."""
        return (
            self.reciprocal_dispersion_nm_per_mm
            * 1.0e-6
            * self.slit_width_um
            * 1.0e-6
        )

    @property
    def fwhm_m(self) -> float:
        """Total instrumental FWHM: slit bandpass (+) aberrations in
        quadrature (Eq. 3-22)."""
        return float(
            np.hypot(self.slit_bandpass_m, self.aberration_fwhm_m)
        )

    # ------------------------------------------------------------------
    def kernel(self, grid_step_m: float) -> NDArray[np.float64]:
        """
        Discrete unit-sum LSF kernel for a uniform grid step.

        Gaussian: sampled to +/- 5 sigma. Triangular: half-base equal
        to the FWHM (a triangle's FWHM is half its base, Fig. 3-5).

        Raises
        ------
        ValueError
            If the FWHM spans fewer than 3 grid steps (unresolved
            kernel — refine the wavelength grid).
        """
        step = float(grid_step_m)
        if not (step > 0 and np.isfinite(step)):
            raise ValueError("grid_step_m must be finite and > 0")
        fwhm = self.fwhm_m
        if fwhm < 3.0 * step:
            raise ValueError(
                f"instrumental FWHM {fwhm:.3e} m spans fewer than 3 grid "
                f"steps ({step:.3e} m); refine the wavelength grid so the "
                "LSF is resolved"
            )
        if self.shape == "gaussian":
            sigma = fwhm / _GAUSS_FWHM_PER_SIGMA
            half_width = int(np.ceil(5.0 * sigma / step))
            offsets = np.arange(-half_width, half_width + 1) * step
            kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        else:  # triangular
            half_width = int(np.ceil(fwhm / step))
            offsets = np.arange(-half_width, half_width + 1) * step
            kernel = np.maximum(1.0 - np.abs(offsets) / fwhm, 0.0)
        return kernel / kernel.sum()  # unit sum: flux-conserving

    def convolve(self, spectrum: Spectrum) -> Spectrum:
        """
        Convolve a spectrum with the LSF (finite spectral resolution).

        Requires a uniform wavelength grid (relative step deviation
        < 1e-6). Zero-padded edges (module notes).
        """
        wavelength = np.asarray(spectrum.wavelength_m, dtype=np.float64)
        steps = np.diff(wavelength)
        if wavelength.size < 2 or np.any(steps <= 0):
            raise ValueError("spectrum grid must be increasing with >= 2 points")
        step = float(steps.mean())
        if np.max(np.abs(steps - step)) > 1e-6 * step:
            raise ValueError(
                "convolution requires a uniform wavelength grid "
                "(resample first — analysis.resample)"
            )
        kernel = self.kernel(step)
        convolved = np.convolve(spectrum.intensity, kernel, mode="same")
        metadata = dict(spectrum.metadata)
        metadata["instrumental_fwhm_m"] = self.fwhm_m
        metadata["instrumental_shape"] = self.shape
        metadata["slit_width_um"] = self.slit_width_um
        metadata["reciprocal_dispersion_nm_per_mm"] = (
            self.reciprocal_dispersion_nm_per_mm
        )
        return Spectrum(
            wavelength_m=spectrum.wavelength_m,
            intensity=convolved,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def sample_to_pixels(self, spectrum: Spectrum, n_pixels: int) -> Spectrum:
        """
        Bin-average the spectrum onto `n_pixels` equal wavelength bins
        (multi-channel detector sampling; exit "slit" = pixel, Ch. 3
        p. 57 bullet list).

        Mean-preserving within each bin; bin centers become the new
        wavelength axis.
        """
        pixels = int(n_pixels)
        if pixels < 1 or pixels > spectrum.wavelength_m.size:
            raise ValueError(
                "n_pixels must be between 1 and the number of grid points"
            )
        edges = np.linspace(
            spectrum.wavelength_m[0], spectrum.wavelength_m[-1], pixels + 1
        )
        indices = np.clip(
            np.searchsorted(edges, spectrum.wavelength_m, side="right") - 1,
            0,
            pixels - 1,
        )
        sums = np.bincount(
            indices, weights=spectrum.intensity, minlength=pixels
        )
        counts = np.bincount(indices, minlength=pixels)
        intensity = sums / np.maximum(counts, 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        metadata = dict(spectrum.metadata)
        metadata["n_pixels"] = pixels
        return Spectrum(
            wavelength_m=centers, intensity=intensity, metadata=metadata
        )

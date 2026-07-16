r"""Instrumental line-spread function and spectral-resolution effects
(Phase 4).

Physical context (Herrera 2008)
-------------------------------
A real spectrometer broadens every monochromatic input into the
instrumental line profile (Ch. 3, pp. 56-60). Its width is set by the
slits through the geometric spectral bandpass, Eq. 3-19, p. 57:

$$
\Delta\lambda_s \;=\; R_d\, W_{\mathrm{slit}}
$$

with $R_d$ the reciprocal linear dispersion (nm mm$^{-1}$) and
$W_{\mathrm{slit}}$ the slit width; validity is bounded below by the
diffraction-limited bandpass, Eq. 3-20, p. 59
($\Delta\lambda_d = R_d\, w_d$ with $w_d = 2 f \lambda / a$). The
shape is triangular for equal entrance/exit slits (Fig. 3-5, p. 64),
but "Doppler and instrumental line profiles usually have a Gaussian
distribution" (p. 59) — the Gaussian is the thesis' working treatment
and the default here (also the Phase 4 acceptance criterion). Gaussian
widths combine in quadrature (Eq. 3-22, p. 60):

$$
\Delta\lambda_G \;=\;
\sqrt{\Delta\lambda_D^{2} + \Delta\lambda_I^{2}}
$$

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
- The instrumental FWHM must be resolved by $\ge 3$ grid steps, else
  the discrete kernel misrepresents the profile and the call raises.
- Aberrations enter as an optional extra Gaussian FWHM combined in
  quadrature (Eq. 3-22) — the thesis measures the combined function
  rather than modeling each aberration (Ch. 4, pp. 84-88).
- Optional "voigt" shape: measured instrument functions of real
  (especially echelle) spectrographs show a non-negligible Lorentzian
  component on top of the Gaussian core (e.g. Hansen, DTU PhD thesis,
  Figs. 3.3.3-3.3.4: Hg-lamp Voigt fits with
  $\mathrm{FWHM}_\gamma \approx \tfrac{1}{2}\,\mathrm{FWHM}_\sigma$).
  The "voigt" LSF keeps Eq. 3-19 (+ aberrations in quadrature) as the
  *Gaussian* part and adds a user-supplied Lorentzian FWHM
  (`lorentzian_fwhm_m`, from an instrument-function calibration).
  Kernel wings extend to $\max(5\sigma,\, 40\,\mathrm{FWHM}_L)$ so
  the truncated Lorentzian area loss stays below ~1%; unit-sum
  normalization then makes it flux-conserving on the grid.

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
from ..physics.line_profiles import (
    voigt_fwhm_estimate,
    voigt_profile_wavelength_m,
)

_GAUSS_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def diffraction_limited_bandpass_m(
    reciprocal_dispersion_nm_per_mm: float,
    focal_length_m: float,
    wavelength_m: float,
    beam_width_m: float,
) -> float:
    r"""
    Diffraction-limited spectral bandpass, Eq. 3-20, p. 59
    (Herrera 2008): $\Delta\lambda_d = R_d\, w_d$ with
    $w_d = 2 f \lambda / a$.

    The slit model of Eq. 3-19 is meaningful only for slit widths above
    $w_d$; `InstrumentalProfile` warns against configurations below it.
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
    r"""
    Slit-controlled instrumental line-spread function (LSF).

    Parameters
    ----------
    reciprocal_dispersion_nm_per_mm : float
        $R_d$ of Eq. 3-19 (> 0), a property of the spectrograph
        (thesis Ch. 4: 2nd-order polynomial in wavelength, Fig. 4-10;
        supply the value for your working window).
    slit_width_um : float
        Entrance-slit width (> 0).
    aberration_fwhm_m : float, optional
        Extra Gaussian broadening combined in quadrature (Eq. 3-22),
        default 0.
    shape : {"gaussian", "triangular", "voigt"}, optional
        LSF shape: Gaussian (default; p. 59), triangular (equal
        entrance/exit slits, Fig. 3-5, p. 64), or Voigt (Gaussian core
        of Eq. 3-19 + measured Lorentzian component; module notes).
        The triangular option ignores `aberration_fwhm_m`
        cross-combination subtleties and applies quadrature to its
        FWHM as an approximation (documented).
    lorentzian_fwhm_m : float, optional
        Lorentzian FWHM of the "voigt" LSF (m, >= 0; from an
        instrument-function calibration). Only valid with
        shape="voigt"; zero degenerates to the Gaussian LSF.

    Notes
    -----
    FWHM (Eq. 3-19): $\Delta\lambda_I = 10^{-6}\, R_d\, W_{\mathrm{slit}}$
    with $R_d$ in nm/mm and $W_{\mathrm{slit}}$ in m.
    """

    reciprocal_dispersion_nm_per_mm: float
    slit_width_um: float
    aberration_fwhm_m: float = 0.0
    shape: str = "gaussian"
    lorentzian_fwhm_m: float = 0.0

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
        if self.shape not in ("gaussian", "triangular", "voigt"):
            raise ValueError(
                "shape must be 'gaussian', 'triangular' or 'voigt'"
            )
        if not (
            self.lorentzian_fwhm_m >= 0
            and np.isfinite(self.lorentzian_fwhm_m)
        ):
            raise ValueError("lorentzian_fwhm_m must be finite and >= 0")
        if self.lorentzian_fwhm_m > 0 and self.shape != "voigt":
            raise ValueError(
                "lorentzian_fwhm_m > 0 requires shape='voigt'"
            )

    # ------------------------------------------------------------------
    @property
    def slit_bandpass_m(self) -> float:
        r"""Geometric spectral bandpass
        $\Delta\lambda_s = R_d\, W_{\mathrm{slit}}$ (Eq. 3-19)."""
        return (
            self.reciprocal_dispersion_nm_per_mm
            * 1.0e-6
            * self.slit_width_um
            * 1.0e-6
        )

    @property
    def gaussian_fwhm_m(self) -> float:
        """Gaussian part: slit bandpass (+) aberrations in quadrature
        (Eq. 3-22)."""
        return float(
            np.hypot(self.slit_bandpass_m, self.aberration_fwhm_m)
        )

    @property
    def fwhm_m(self) -> float:
        """
        Total instrumental FWHM. Gaussian/triangular: the quadrature
        width (Eq. 3-22). Voigt: Whiting-type estimate (inverse
        Eq. 5-18) of the Gaussian part combined with
        `lorentzian_fwhm_m`.
        """
        if self.shape == "voigt" and self.lorentzian_fwhm_m > 0:
            return float(
                voigt_fwhm_estimate(
                    self.gaussian_fwhm_m, self.lorentzian_fwhm_m
                )
            )
        return self.gaussian_fwhm_m

    # ------------------------------------------------------------------
    def kernel(self, grid_step_m: float) -> NDArray[np.float64]:
        r"""
        Discrete unit-sum LSF kernel for a uniform grid step.

        Gaussian: sampled to $\pm 5\sigma$. Triangular: half-base equal
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
        elif self.shape == "voigt":
            sigma = self.gaussian_fwhm_m / _GAUSS_FWHM_PER_SIGMA
            # Lorentzian wings need a much wider support than 5 sigma:
            # beyond +/- 40 FWHM_L the truncated area is below ~1%.
            extent = max(5.0 * sigma, 40.0 * self.lorentzian_fwhm_m)
            half_width = int(np.ceil(extent / step))
            # The Voigt is even in detuning: evaluate one half via the
            # shared normalized profile (at a um-scale anchor so the
            # detunings survive the round-trip to ~1e-12 relative) and
            # mirror it, making the kernel symmetric by construction.
            anchor = 1.0e-6
            positive = np.arange(0, half_width + 1) * step
            half_kernel = np.asarray(voigt_profile_wavelength_m(
                anchor + positive, anchor,
                self.gaussian_fwhm_m, self.lorentzian_fwhm_m,
            ))
            kernel = np.concatenate([half_kernel[:0:-1], half_kernel])
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
        # 'same'-with-respect-to-the-spectrum regardless of which of
        # the two is longer (numpy's mode="same" returns the LONGER
        # length, which breaks when a wide-winged Voigt kernel exceeds
        # the grid span).
        full = np.convolve(spectrum.intensity, kernel, mode="full")
        start = (kernel.size - 1) // 2
        convolved = full[start:start + spectrum.intensity.size]
        metadata = dict(spectrum.metadata)
        metadata["instrumental_fwhm_m"] = self.fwhm_m
        metadata["instrumental_shape"] = self.shape
        if self.shape == "voigt":
            metadata["instrumental_lorentzian_fwhm_m"] = (
                self.lorentzian_fwhm_m
            )
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

"""Quantitative synthetic-vs-experimental comparison metrics (Phase 4
validation workflow).

The primary metric is the linear correlation coefficient R of
Eq. 5-56, p. 122 (Herrera 2008), reused from `libssim.analysis` — the
MC-LIBS cost function (thesis example R = 0.9913, Fig. 5-6).
Supporting metrics quantify the qualitative checks the validation plan
asks for: peak positions (wavelength calibration + line data), peak
intensity ratios (population/transition-probability physics), and the
normalized rms residual (overall shape).

All metrics assume both spectra share one wavelength grid (use
`analysis.resample`) and a common normalization (use
`preprocessing.normalize`) where scale matters; R itself is
scale-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from ..analysis import correlation_coefficient
from ..core.spectrum import Spectrum


def find_peak_positions(
    spectrum: Spectrum,
    min_relative_height: float = 0.05,
) -> Tuple[float, ...]:
    """
    Wavelengths (m) of local maxima above a relative height threshold.

    Thin wrapper over `scipy.signal.find_peaks` with prominence set to
    ``min_relative_height * max(intensity)`` — robust for isolated LIBS
    lines; heavily blended spectra deserve dedicated fitting (thesis
    fits Voigt functions per line, p. 103).
    """
    if not (0.0 < min_relative_height < 1.0):
        raise ValueError("min_relative_height must be in (0, 1)")
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    if intensity.size < 3:
        return ()
    scale = float(intensity.max())
    if scale <= 0:
        return ()
    indices, _ = find_peaks(
        intensity, prominence=min_relative_height * scale
    )
    return tuple(float(w) for w in spectrum.wavelength_m[indices])


@dataclass(frozen=True)
class PeakMatch:
    """One expected line vs the nearest found peak (if any)."""

    expected_m: float
    found_m: Optional[float]

    @property
    def matched(self) -> bool:
        return self.found_m is not None

    @property
    def offset_m(self) -> Optional[float]:
        """found - expected (m); positive = red of the expected line."""
        if self.found_m is None:
            return None
        return self.found_m - self.expected_m


def match_peaks(
    found_m: Sequence[float],
    expected_m: Sequence[float],
    tolerance_m: float,
) -> Tuple[PeakMatch, ...]:
    """
    Associate each expected line with the nearest found peak within a
    tolerance (one peak may serve several expected lines if they are
    closer than the tolerance — blends are reported, not hidden).
    """
    if not (tolerance_m > 0 and np.isfinite(tolerance_m)):
        raise ValueError("tolerance_m must be finite and > 0")
    found = np.asarray(sorted(found_m), dtype=np.float64)
    matches = []
    for expected in expected_m:
        if found.size == 0:
            matches.append(PeakMatch(float(expected), None))
            continue
        nearest = float(found[np.argmin(np.abs(found - expected))])
        if abs(nearest - expected) <= tolerance_m:
            matches.append(PeakMatch(float(expected), nearest))
        else:
            matches.append(PeakMatch(float(expected), None))
    return tuple(matches)


def intensity_ratio(
    spectrum: Spectrum,
    line_a_m: float,
    line_b_m: float,
    half_window_m: float,
) -> float:
    """
    Peak-height ratio I(line_a)/I(line_b), each taken as the maximum
    within +/- half_window_m of the nominal wavelength.

    The classic doublet diagnostic: e.g. Na D2/D1 -> g_u ratio 2.0 in
    the optically thin limit, decreasing toward 1 as self-absorption
    saturates the stronger line (thesis Ch. 3, Fig. 3-2).
    """
    if not (half_window_m > 0 and np.isfinite(half_window_m)):
        raise ValueError("half_window_m must be finite and > 0")

    def window_max(center: float) -> float:
        mask = np.abs(spectrum.wavelength_m - center) <= half_window_m
        if not np.any(mask):
            raise ValueError(
                f"no samples within {half_window_m:.3g} m of {center:.6g} m"
            )
        return float(spectrum.intensity[mask].max())

    denominator = window_max(line_b_m)
    if denominator == 0.0:
        raise ValueError("line_b window has zero peak intensity")
    return window_max(line_a_m) / denominator


def rms_residual(a: Spectrum, b: Spectrum) -> float:
    """
    Root-mean-square of (a - b), meaningful when both spectra share a
    grid and normalization (e.g. peak-normalized: rms in units of the
    peak).
    """
    if a.wavelength_m.shape != b.wavelength_m.shape or not np.allclose(
        a.wavelength_m, b.wavelength_m, rtol=1e-12
    ):
        raise ValueError("spectra must share the same wavelength grid")
    return float(np.sqrt(np.mean((a.intensity - b.intensity) ** 2)))


@dataclass(frozen=True)
class ValidationMetrics:
    """
    Bundle of comparison metrics for one validation run.

    Attributes
    ----------
    r_correlation : float
        Eq. 5-56, p. 122 (Herrera 2008) — the primary metric.
    rms_residual : float
        RMS difference of the peak-normalized spectra (units of the
        peak).
    peak_matches : tuple of PeakMatch
        Expected lines vs peaks found in the *experimental* spectrum.
    """

    r_correlation: float
    rms_residual: float
    peak_matches: Tuple[PeakMatch, ...]

    @property
    def n_expected(self) -> int:
        return len(self.peak_matches)

    @property
    def n_matched(self) -> int:
        return sum(1 for m in self.peak_matches if m.matched)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"R (Eq. 5-56)        : {self.r_correlation:.4f}",
            f"rms residual (peak) : {self.rms_residual:.4f}",
            f"peaks matched       : {self.n_matched}/{self.n_expected}",
        ]
        for match in self.peak_matches:
            expected_nm = match.expected_m * 1e9
            if match.matched:
                offset = match.offset_m
                assert offset is not None
                lines.append(
                    f"  {expected_nm:9.3f} nm -> found "
                    f"{match.found_m * 1e9:9.3f} nm "
                    f"(offset {offset * 1e12:+7.1f} pm)"
                )
            else:
                lines.append(f"  {expected_nm:9.3f} nm -> NOT FOUND")
        return "\n".join(lines)


def compute_metrics(
    synthetic: Spectrum,
    experimental: Spectrum,
    expected_lines_m: Sequence[float],
    peak_tolerance_m: float = 5.0e-11,
    min_relative_height: float = 0.05,
) -> ValidationMetrics:
    """
    Standard metric bundle: R (Eq. 5-56) + normalized rms + peak
    matching against the experimental spectrum.

    Both spectra must already share a grid and (for the rms) a common
    normalization — the `ValidationCase` workflow guarantees this.
    """
    r_value = correlation_coefficient(synthetic, experimental)
    residual = rms_residual(synthetic, experimental)
    found = find_peak_positions(experimental, min_relative_height)
    matches = match_peaks(found, expected_lines_m, peak_tolerance_m)
    return ValidationMetrics(
        r_correlation=r_value,
        rms_residual=residual,
        peak_matches=matches,
    )

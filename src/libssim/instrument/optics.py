r"""Collection-optics spectral efficiency and radiometric scaling
(Phase 4).

Physical context (Herrera 2008)
-------------------------------
The measured line intensity is the physical emission scaled by the
optical efficiency of the detection system, Eq. 5-9, p. 104:

$$
I_{ki}^{\mathrm{meas}} \;=\; F_{\mathrm{det}}(\lambda)\, I_{ki},
\qquad
F_{\mathrm{det}}(\lambda) \;=\;
F_{\mathrm{rel}}(\lambda)\, F_{\mathrm{abs}}
\quad \text{(Eqs. 5-9..5-11)}
$$

with $F_{\mathrm{rel}}$ the wavelength-dependent *relative* efficiency
(measured with calibrated lamps, Ch. 4, pp. 84-88) and
$F_{\mathrm{abs}}$ a wavelength-independent absolute factor. CF-LIBS
divides this response out of measured spectra (Eq. 5-10); the forward
model applies it to synthetic ones — same physics, opposite direction.

`CollectionOptics` implements that scaling. The absolute factor is the
user's calibration burden: it lumps collection etendue, transmission,
detector quantum efficiency, gain and any exposure-to-counts
conversion into one number with units [output counts per input
intensity unit]. Leave it at 1.0 to stay in physical radiometric
units.

Units and Conventions
---------------------
Pure per-wavelength scaling:
$\mathrm{output}(\lambda) = \mathrm{input}(\lambda)\,
F_{\mathrm{rel}}(\lambda)\, F_{\mathrm{abs}}$.
The relative curve is dimensionless (peak normalization is the
caller's convention); metadata records what was applied.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eqs. 5-9, 5-10,
5-11 p. 104; Ch. 4 pp. 84-88 (efficiency calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.spectrum import Spectrum

#: Relative-efficiency curve: wavelength array (m) -> efficiency array.
EfficiencyCurve = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def tabulated_efficiency(
    wavelength_m: ArrayLike,
    efficiency: ArrayLike,
) -> EfficiencyCurve:
    """
    Build an `EfficiencyCurve` by linear interpolation of a measured
    table (the Ch. 4 calibration product). Outside the tabulated range
    the endpoint values are held (np.interp clamping) — extrapolating a
    measured response would be meaningless.
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    eff = np.asarray(efficiency, dtype=np.float64)
    if lam.ndim != 1 or lam.shape != eff.shape or lam.size < 2:
        raise ValueError(
            "wavelength_m and efficiency must be equal-length 1-D arrays "
            "with >= 2 points"
        )
    if np.any(np.diff(lam) <= 0):
        raise ValueError("wavelength_m must be strictly increasing")
    if np.any(eff < 0) or not np.all(np.isfinite(eff)):
        raise ValueError("efficiency must be finite and >= 0")
    lam = lam.copy()
    eff = eff.copy()

    def curve(grid: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(grid, lam, eff)

    return curve


@dataclass(frozen=True, eq=False)
class CollectionOptics:
    r"""
    Detection-system response
    $F_{\mathrm{det}}(\lambda) = F_{\mathrm{rel}}(\lambda)\,
    F_{\mathrm{abs}}$ (Eqs. 5-9..5-11, p. 104).

    Parameters
    ----------
    relative_efficiency : EfficiencyCurve, optional
        $F_{\mathrm{rel}}(\lambda)$, dimensionless (default:
        flat 1.0). Use `tabulated_efficiency` for measured curves.
    absolute_factor : float, optional
        $F_{\mathrm{abs}}$ plus any radiometric conversion (module
        docstring), > 0, default 1.0.
    """

    relative_efficiency: Optional[EfficiencyCurve] = None
    absolute_factor: float = 1.0

    def __post_init__(self) -> None:
        if not (self.absolute_factor > 0 and np.isfinite(self.absolute_factor)):
            raise ValueError("absolute_factor must be finite and > 0")

    def efficiency(
        self, wavelength_m: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""$F_{\mathrm{det}}(\lambda)$ on the given grid (the
        dimensionless relative curve times $F_{\mathrm{abs}}$)."""
        grid = np.asarray(wavelength_m, dtype=np.float64)
        if self.relative_efficiency is None:
            relative = np.ones(grid.shape)
        else:
            relative = np.asarray(
                self.relative_efficiency(grid), dtype=np.float64
            )
            if relative.shape != grid.shape:
                raise ValueError(
                    "relative_efficiency must return an array matching the "
                    "wavelength grid"
                )
            if np.any(relative < 0) or not np.all(np.isfinite(relative)):
                raise ValueError(
                    "relative_efficiency must be finite and >= 0"
                )
        return relative * self.absolute_factor

    def apply(self, spectrum: Spectrum) -> Spectrum:
        r"""
        Scale a spectrum by $F_{\mathrm{det}}(\lambda)$ (Eq. 5-9
        applied forward).

        Returns a new `Spectrum`; metadata gains the applied absolute
        factor and whether a relative curve was used.
        """
        response = self.efficiency(spectrum.wavelength_m)
        metadata = dict(spectrum.metadata)
        metadata["collection_absolute_factor"] = self.absolute_factor
        metadata["collection_relative_curve"] = (
            self.relative_efficiency is not None
        )
        return Spectrum(
            wavelength_m=spectrum.wavelength_m,
            intensity=spectrum.intensity * response,
            metadata=metadata,
        )

"""
libssim.instrument.noise
========================
Detector noise models (Phase 4).

Physical Context
----------------
The thesis records spectra with an intensified CCD (Ch. 4); like any
photon-counting chain the signal carries:

- **shot noise** — Poisson statistics of the detected photoelectrons
  (variance = mean);
- **readout noise** — additive Gaussian electronics noise, rms
  sigma_read counts, independent of signal;
- **dark / background offset** — a mean additive level (thermal dark
  counts, ambient background) that itself carries shot noise.

The thesis treats these operationally (background subtraction before
CF-LIBS analysis, p. 103) rather than with printed formulas; the
standard statistical models implemented here follow general detector
practice (e.g. Janesick, Scientific Charge-Coupled Devices, 2001) —
documented extension per ai_instructions.md. They satisfy the Phase 4
acceptance criteria (implementation_plan.md): Poisson variance equal
to the mean, Gaussian read noise, dark-current offset.

Reproducibility (cross-cutting requirement)
-------------------------------------------
`apply` takes a mandatory integer seed and uses a dedicated
`numpy.random.Generator`: the same spectrum + same seed always yields
the identical noisy spectrum. No global random state is touched.

Units: counts. Apply noise *after* the collection optics have scaled
the spectrum to expected counts — Poisson statistics are meaningless
on radiometric W/J units (enforced only as a documentation contract;
the code requires non-negative intensity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.spectrum import Spectrum


@dataclass(frozen=True)
class NoiseModel:
    """
    Shot + readout + dark/background detector noise.

    Parameters
    ----------
    read_noise_rms_counts : float, optional
        Gaussian readout rms sigma_read (counts, >= 0; default 0).
    dark_mean_counts : float, optional
        Mean dark level added to every sample (counts, >= 0;
        default 0). Carries shot noise.
    background_mean_counts : float, optional
        Mean ambient/continuum background level (counts, >= 0;
        default 0). Carries shot noise.
    include_shot_noise : bool, optional
        Poisson-sample the (signal + dark + background) expectation
        (default True). Disable to study additive noise alone.

    Notes
    -----
    Output = Poisson(signal + dark + background) + Normal(0, sigma_read)
    per wavelength sample, mutually independent.
    """

    read_noise_rms_counts: float = 0.0
    dark_mean_counts: float = 0.0
    background_mean_counts: float = 0.0
    include_shot_noise: bool = True

    def __post_init__(self) -> None:
        for name in (
            "read_noise_rms_counts",
            "dark_mean_counts",
            "background_mean_counts",
        ):
            value = getattr(self, name)
            if not (value >= 0.0 and np.isfinite(value)):
                raise ValueError(f"{name} must be finite and >= 0")

    def apply(self, spectrum: Spectrum, seed: int) -> Spectrum:
        """
        Return a noisy copy of the spectrum (counts).

        Parameters
        ----------
        spectrum : Spectrum
            Expected-counts spectrum (intensity >= 0 required — scale
            with `CollectionOptics` first).
        seed : int
            RNG seed; identical seeds reproduce identical noise
            (reproducibility requirement).
        """
        intensity = np.asarray(spectrum.intensity, dtype=np.float64)
        if np.any(intensity < 0) or not np.all(np.isfinite(intensity)):
            raise ValueError(
                "noise requires finite, non-negative expected counts; "
                "apply CollectionOptics scaling first"
            )
        rng = np.random.default_rng(int(seed))
        expectation = (
            intensity + self.dark_mean_counts + self.background_mean_counts
        )
        if self.include_shot_noise:
            counts = rng.poisson(expectation).astype(np.float64)
        else:
            counts = expectation.copy()
        if self.read_noise_rms_counts > 0.0:
            counts = counts + rng.normal(
                0.0, self.read_noise_rms_counts, size=counts.shape
            )
        metadata = dict(spectrum.metadata)
        metadata["noise_seed"] = int(seed)
        metadata["noise_model"] = {
            "read_noise_rms_counts": self.read_noise_rms_counts,
            "dark_mean_counts": self.dark_mean_counts,
            "background_mean_counts": self.background_mean_counts,
            "include_shot_noise": self.include_shot_noise,
        }
        metadata["intensity_units"] = "counts (noisy)"
        return Spectrum(
            wavelength_m=spectrum.wavelength_m,
            intensity=counts,
            metadata=metadata,
        )

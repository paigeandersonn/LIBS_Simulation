"""Composite instrument response pipeline (Phase 4).

Chains the individual effects in their physical order:

    physics spectrum
      -> LSF convolution        (finite resolution, spectrometer.py,
                                 Eqs. 3-19/3-22)
      -> collection efficiency  (F_det(lambda), optics.py,
                                 Eqs. 5-9..5-11 p. 104)
      -> pixel sampling         (optional; multi-channel detector)
      -> detector noise         (optional; shot/read/dark, noise.py)

Every stage is optional, and `resolution_only` / `noise_free` give the
partially-processed spectra directly — the validation and debugging
paths requested for comparing against physics-level results.

Each stage appends its parameters to `Spectrum.metadata`, so a fully
processed spectrum carries its complete instrument provenance next to
the plasma/gate provenance added by the temporal layer.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Ch. 3 pp. 56-60
(instrumental function); Eqs. 5-9..5-11 p. 104 (detection efficiency);
Ch. 4 pp. 84-88 (calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.spectrum import Spectrum
from .noise import NoiseModel
from .optics import CollectionOptics
from .spectrometer import InstrumentalProfile


@dataclass(frozen=True, eq=False)
class InstrumentResponse:
    """
    Composable end-to-end instrument model.

    Parameters
    ----------
    instrumental_profile : InstrumentalProfile, optional
        Finite spectral resolution (LSF convolution). None = perfect
        resolution.
    collection_optics : CollectionOptics, optional
        Spectral efficiency and radiometric scaling. None = unit
        response.
    n_pixels : int, optional
        If given, bin-average onto this many detector pixels after
        convolution/scaling.
    noise_model : NoiseModel, optional
        Detector noise; requires a seed at `apply` time. None =
        noiseless output.
    """

    instrumental_profile: Optional[InstrumentalProfile] = None
    collection_optics: Optional[CollectionOptics] = None
    n_pixels: Optional[int] = None
    noise_model: Optional[NoiseModel] = None

    # ------------------------------------------------------------------
    def resolution_only(self, spectrum: Spectrum) -> Spectrum:
        """
        Apply only the finite spectral resolution (LSF convolution).

        The debugging/validation path: line shapes as the spectrometer
        sees them, with physical intensity units untouched.
        """
        if self.instrumental_profile is None:
            return spectrum
        return self.instrumental_profile.convolve(spectrum)

    def noise_free(self, spectrum: Spectrum) -> Spectrum:
        """
        Full pipeline except the noise stage: convolution, efficiency
        scaling and pixel sampling with deterministic output.

        This is the expected-counts spectrum — the mean around which
        `apply` scatters.
        """
        result = self.resolution_only(spectrum)
        if self.collection_optics is not None:
            result = self.collection_optics.apply(result)
        if self.n_pixels is not None:
            if self.instrumental_profile is not None:
                result = self.instrumental_profile.sample_to_pixels(
                    result, self.n_pixels
                )
            else:
                raise ValueError(
                    "pixel sampling requires an instrumental_profile "
                    "(it provides sample_to_pixels)"
                )
        return result

    def apply(self, spectrum: Spectrum, seed: Optional[int] = None) -> Spectrum:
        """
        Full pipeline including noise.

        Parameters
        ----------
        spectrum : Spectrum
            Physics-level spectrum (from the temporal integrator or the
            transport layer).
        seed : int, optional
            Required when a `noise_model` is configured (reproducible
            noise); ignored otherwise.
        """
        result = self.noise_free(spectrum)
        if self.noise_model is None:
            return result
        if seed is None:
            raise ValueError(
                "a seed is required when noise_model is set "
                "(reproducibility requirement); use noise_free() for the "
                "deterministic spectrum"
            )
        return self.noise_model.apply(result, seed)

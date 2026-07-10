"""
libssim.temporal.integrator
===========================
Gate-delay / gate-width temporal integration of the emission (Phase 4).

Physical Context (Herrera 2008)
-------------------------------
Time-resolved LIBS detection records the plasma emission through an
intensified detector gate: the delay time t_delay is "the time between
the initiation of the laser pulse and the beginning of the gate width"
and the gate width t_gate is "the integration window during which the
plasma emission is recorded" (pp. 46-47; symbols p. 23). The recorded
signal is therefore the time integral of the instantaneous spectral
radiance,

    E_lambda = Integral_{t_delay}^{t_delay + t_gate} I_lambda(t) dt,

with units of spectral radiant exposure (J m^-2 m^-1 sr^-1). All the
temporally-resolved studies of Chapters 6-7 vary exactly these two
parameters.

`GateIntegrator` evaluates I_lambda(t) by rendering the
`PlasmaEvolution` snapshot at each quadrature node with the Phase 3
transfer solver (quasi-static assumption, temporal/base.py) and sums
with Gauss-Legendre (default) or trapezoid weights.

Numerical Assumptions and Limitations
-------------------------------------
- Quadrature accuracy is controlled by `n_time_nodes`; Gauss-Legendre
  is effectively exact for the smooth closed-form decays of
  decay_models.py (validated against an analytic optically-thin
  exponential case in the tests). For a *constant* plasma the
  quadrature is exact for any node count, so the gate-width scaling
  acceptance criterion (integrated intensity proportional to t_gate,
  implementation_plan.md Phase 4) holds identically.
- The detector gate is treated as a perfect top-hat in time;
  intensifier rise/fall shapes are an instrument-layer refinement.

Units: SI. Instantaneous spectra W m^-2 m^-1 sr^-1; gate-integrated
spectra J m^-2 m^-1 sr^-1 (recorded in metadata).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. pp. 46-47
(t_delay/t_gate); Ch. 6-7 (temporally-resolved studies); Eq. 5-48
p. 119 (per-instant transfer, Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.spectrum import Spectrum
from ..core.constants import C
from ..transport.emissivity import LTESpectralModel
from ..transport.radiative import disk_integrated_radiance, emergent_radiance
from .base import PlasmaEvolution, validate_time


@dataclass(frozen=True, eq=False)
class GateIntegrator:
    """
    Renders a `PlasmaEvolution` into time-resolved and gate-integrated
    spectra using the Phase 2/3 pipeline unchanged.

    Parameters
    ----------
    spectral_model : LTESpectralModel
        Per-zone spectral physics (fixed wavelength grid).
    evolution : PlasmaEvolution
        Time-dependent plasma (e.g. `UniformPlasmaEvolution`).
    impact_parameter_m : float, optional
        If given, spatially resolved observation along that ray; if
        None (default), disk-integrated observation (the thesis'
        spatially integrated mode).
    n_impact : int, optional
        Quadrature nodes for the disk average (default 64).
    """

    spectral_model: LTESpectralModel
    evolution: PlasmaEvolution
    impact_parameter_m: Optional[float] = None
    n_impact: int = 64

    # ------------------------------------------------------------------
    def _radiance_lambda_at(self, time_s: float) -> NDArray[np.float64]:
        """Instantaneous per-wavelength radiance I_lambda(t) (W m^-2 m^-1 sr^-1)."""
        geometry = self.evolution.geometry_at(time_s)
        epsilon, kappa = self.spectral_model.geometry_properties(geometry)
        if self.impact_parameter_m is None:
            radiance_nu = disk_integrated_radiance(
                geometry, epsilon, kappa, n_impact=self.n_impact
            )
        else:
            radiance_nu = emergent_radiance(
                geometry.path_segments(float(self.impact_parameter_m)),
                epsilon,
                kappa,
            )
        return radiance_nu * C / self.spectral_model.wavelength_m**2

    def _observation_mode(self) -> str:
        if self.impact_parameter_m is None:
            return "disk-integrated"
        return f"impact_parameter={float(self.impact_parameter_m):.6g} m"

    def _zone_temperatures(self, time_s: float) -> Tuple[float, ...]:
        geometry = self.evolution.geometry_at(time_s)
        return tuple(z.temperature_K for z in geometry.zones)

    # ------------------------------------------------------------------
    def snapshot(self, time_s: float) -> Spectrum:
        """
        Instantaneous (time-resolved) spectrum at one time.

        Returns a `Spectrum` of spectral radiance (W m^-2 m^-1 sr^-1)
        with the plasma summary in the metadata.
        """
        t = validate_time(time_s)
        radiance = self._radiance_lambda_at(t)
        geometry = self.evolution.geometry_at(t)
        return Spectrum(
            wavelength_m=np.array(self.spectral_model.wavelength_m, copy=True),
            intensity=radiance,
            metadata={
                "intensity_units": "W m^-2 m^-1 sr^-1 (spectral radiance)",
                "time_s": t,
                "observation_mode": self._observation_mode(),
                "n_zones": len(geometry.zones),
                "zone_temperatures_K": self._zone_temperatures(t),
                "outer_radius_m": geometry.outer_radius_m,
            },
        )

    def time_resolved(self, times_s: Sequence[float]) -> List[Spectrum]:
        """Snapshots at each requested time (order preserved)."""
        return [self.snapshot(t) for t in times_s]

    # ------------------------------------------------------------------
    def gate_integrated(
        self,
        gate_delay_s: float,
        gate_width_s: float,
        n_time_nodes: int = 16,
        quadrature: str = "gauss",
    ) -> Spectrum:
        """
        Emission integrated over the detector gate — the recorded LIBS
        signal of pp. 46-47 (Herrera 2008).

            E_lambda = Integral_{t_d}^{t_d + t_g} I_lambda(t) dt

        Parameters
        ----------
        gate_delay_s : float
            t_delay (s, >= 0): time from plasma initiation to gate
            opening.
        gate_width_s : float
            t_gate (s, > 0): integration window length.
        n_time_nodes : int, optional
            Quadrature nodes across the gate (default 16, >= 2).
            Increase for fast decays relative to the gate width.
        quadrature : {"gauss", "trapezoid"}, optional
            Gauss-Legendre (default; interior nodes, high order for
            smooth decays) or trapezoid (includes the endpoints —
            useful when profiles are only defined piecewise).

        Returns
        -------
        Spectrum
            Spectral radiant exposure (J m^-2 m^-1 sr^-1); metadata
            records gate settings, quadrature, node times and per-node
            zone temperatures (plasma history provenance).
        """
        delay = validate_time(gate_delay_s)
        width = float(gate_width_s)
        if not (width > 0.0 and np.isfinite(width)):
            raise ValueError("gate_width_s must be finite and > 0")
        nodes = int(n_time_nodes)
        if nodes < 2:
            raise ValueError("n_time_nodes must be >= 2")

        if quadrature == "gauss":
            x, w = np.polynomial.legendre.leggauss(nodes)
            times = delay + 0.5 * width * (x + 1.0)
            weights = 0.5 * width * w
        elif quadrature == "trapezoid":
            times = np.linspace(delay, delay + width, nodes)
            weights = np.full(nodes, width / (nodes - 1))
            weights[0] *= 0.5
            weights[-1] *= 0.5
        else:
            raise ValueError(
                f"unknown quadrature {quadrature!r}; use 'gauss' or 'trapezoid'"
            )

        exposure = np.zeros(self.spectral_model.n_wavelengths)
        node_temperatures: List[Tuple[float, ...]] = []
        for t, weight in zip(times, weights):
            exposure += weight * self._radiance_lambda_at(float(t))
            node_temperatures.append(self._zone_temperatures(float(t)))

        return Spectrum(
            wavelength_m=np.array(self.spectral_model.wavelength_m, copy=True),
            intensity=exposure,
            metadata={
                "intensity_units": (
                    "J m^-2 m^-1 sr^-1 (spectral radiant exposure)"
                ),
                "gate_delay_s": delay,
                "gate_width_s": width,
                "quadrature": quadrature,
                "n_time_nodes": nodes,
                "time_nodes_s": [float(t) for t in times],
                "zone_temperatures_K_per_node": node_temperatures,
                "observation_mode": self._observation_mode(),
                "calculation_notes": (
                    "top-hat detector gate; quasi-static plasma per node "
                    "(temporal/base.py); t_delay/t_gate per Herrera 2008 "
                    "pp. 46-47"
                ),
            },
        )

"""
libssim.transport.radiative
===========================
Line-of-sight radiative transfer through a zoned plasma (Phase 3).

Physical Context (Herrera 2008)
-------------------------------
The stationary radiative transfer equation in spherical coordinates,
Eq. 5-44, p. 117,

    phi dI_nu/dr + ((1-phi^2)/r) dI_nu/dphi + kappa'_nu I_nu
        = kappa'_nu I_nu^b,

has the boundary-radiance solution Eq. 5-48, p. 119 ("detailed
approximation"): along the ray at impact parameter p,

    I_nu(R, phi) = Integral dz K(r(z))
                   * exp( - Integral_z ds kappa'_nu(r(s)) ),

with source K = kappa'_nu * I_nu^b (LTE) and the boundary condition
that no radiation enters the plasma from outside (p. 118). For a
geometry made of homogeneous zones (base.py), the exact solution over
one segment of length L with constant epsilon_nu, kappa'_nu is

    I_out = I_in * exp(-kappa*L) + S * (1 - exp(-kappa*L)),
    S = epsilon_nu / kappa'_nu  (= I_nu^b under LTE, Kirchhoff),

so walking the ordered segments evaluates Eq. 5-48 *analytically* —
no ODE stepping, no dz discretization. This is what makes the
line-of-sight integration numerically stable (Phase 3 acceptance
criterion): expm1 keeps the optically thin limit exact to first order
(I += epsilon*L) and the optically thick limit saturates cleanly at
S = I_nu^b — the blackbody ceiling of the self-absorption discussion,
Eq. 3-10, p. 55 (I = B * (1 - exp(-kappa*l)) for a uniform medium,
reproduced exactly by a one-zone geometry).

Self-absorption and self-reversal (Ch. 3, pp. 53-54) emerge from the
same update: an optically thick line core saturates toward the *local*
blackbody radiance of the outer, cooler zones, producing the central
dip of Fig. 3-3 — the Phase 3 acceptance behaviour.

Units and Conventions
---------------------
- Internal radiance is per-frequency: W m^-2 Hz^-1 sr^-1 (the thesis'
  I_nu, CGS erg s^-1 cm^-2 Hz^-1 sr^-1, in SI).
- `emergent_spectrum` converts to per-wavelength radiance
  (W m^-2 m^-1 sr^-1) with the c/lambda^2 Jacobian when packing the
  Phase 0 `Spectrum` container; units are recorded in the metadata.
- Spatially *resolved* spectra: `emergent_radiance` at one impact
  parameter. Spatially *integrated* spectra (the thesis' default
  measurement mode): `disk_integrated_radiance`, the area-weighted
  average over the projected plasma disk.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-44 p. 117;
Eqs. 5-45, 5-48 pp. 118-119; Eq. 3-10 p. 55; self-reversal pp. 53-54.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.constants import C
from ..core.spectrum import Spectrum
from .base import PathSegment, PlasmaGeometry
from .emissivity import LTESpectralModel


def _validate_zone_arrays(
    segments: Sequence[PathSegment],
    epsilon_zones: NDArray[np.float64],
    kappa_zones: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    eps = np.asarray(epsilon_zones, dtype=np.float64)
    kap = np.asarray(kappa_zones, dtype=np.float64)
    if eps.ndim != 2 or kap.ndim != 2 or eps.shape != kap.shape:
        raise ValueError(
            "epsilon_zones and kappa_zones must be 2-D arrays of equal "
            "shape (n_zones, n_wavelengths)"
        )
    if not np.all(np.isfinite(eps)) or np.any(eps < 0.0):
        raise ValueError("epsilon_zones must be finite and >= 0")
    if not np.all(np.isfinite(kap)) or np.any(kap < 0.0):
        raise ValueError("kappa_zones must be finite and >= 0")
    if len(segments) == 0:
        raise ValueError("segments must not be empty")
    max_index = max(seg.zone_index for seg in segments)
    if max_index >= eps.shape[0]:
        raise ValueError(
            f"segment zone_index {max_index} exceeds the {eps.shape[0]} "
            "zone rows provided"
        )
    return eps, kap


def optical_depth(
    segments: Sequence[PathSegment],
    kappa_zones: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Optical depth tau_nu along a line of sight.

    The exponent of Eq. 5-48, p. 119 (Herrera 2008) over the full
    chord: tau_nu = sum_segments kappa'_nu(zone) * L. Increasing tau in
    the line core is what drives the transmitted-intensity acceptance
    criterion.

    Parameters
    ----------
    segments : sequence of PathSegment
        From `PlasmaGeometry.path_segments` (order irrelevant for tau).
    kappa_zones : ndarray, shape (n_zones, n_wavelengths)
        Per-zone kappa'_nu (m^-1), e.g. from
        `LTESpectralModel.geometry_properties`.

    Returns
    -------
    ndarray, shape (n_wavelengths,)
        tau_nu (dimensionless).
    """
    kap = np.asarray(kappa_zones, dtype=np.float64)
    if kap.ndim != 2:
        raise ValueError("kappa_zones must be 2-D (n_zones, n_wavelengths)")
    tau = np.zeros(kap.shape[1], dtype=np.float64)
    for seg in segments:
        tau += kap[seg.zone_index] * seg.length_m
    return tau


def emergent_radiance(
    segments: Sequence[PathSegment],
    epsilon_zones: NDArray[np.float64],
    kappa_zones: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Spectral radiance leaving the plasma along one line of sight,
    I_nu (W m^-2 Hz^-1 sr^-1).

    Evaluates Eq. 5-48, p. 119 (Herrera 2008) exactly for piecewise
    homogeneous zones: for each segment, in far-boundary -> observer
    order,

        I <- I * exp(-tau_seg) + S * (1 - exp(-tau_seg)),
        tau_seg = kappa'_nu * L,   S = epsilon_nu / kappa'_nu,

    starting from I = 0 (no radiation enters the plasma, p. 118). The
    kappa'_nu = 0 limit is handled exactly as I <- I + epsilon_nu * L.

    Parameters
    ----------
    segments : sequence of PathSegment
        Ordered decomposition of the ray
        (`PlasmaGeometry.path_segments`).
    epsilon_zones, kappa_zones : ndarray, shape (n_zones, n_wavelengths)
        Per-zone emission coefficient (W m^-3 Hz^-1 sr^-1) and total
        absorption coefficient (m^-1).

    Returns
    -------
    ndarray, shape (n_wavelengths,)
        Emergent I_nu.

    Notes
    -----
    Numerical stability (acceptance criterion): the update is the
    analytic segment solution, monotone and bounded —
    min(I_in, S) <= I_out <= max(I_in, S) — for any tau in [0, inf);
    expm1 keeps tau -> 0 exact to first order and tau -> inf saturates
    at S (the blackbody ceiling of Eq. 3-10, p. 55).
    """
    eps, kap = _validate_zone_arrays(segments, epsilon_zones, kappa_zones)

    radiance = np.zeros(eps.shape[1], dtype=np.float64)
    for seg in segments:
        k = kap[seg.zone_index]
        e = eps[seg.zone_index]
        tau = k * seg.length_m
        absorbed = -np.expm1(-tau)  # 1 - e^-tau, accurate for tau -> 0
        with np.errstate(divide="ignore", invalid="ignore"):
            source = np.where(k > 0.0, e / np.where(k > 0.0, k, 1.0), 0.0)
        radiance = (
            radiance * np.exp(-tau)
            + source * absorbed
            # Transparent (kappa = 0) wavelengths accumulate pure
            # emission exactly: I += epsilon * L.
            + np.where(k == 0.0, e * seg.length_m, 0.0)
        )
    return radiance


def disk_integrated_radiance(
    geometry: PlasmaGeometry,
    epsilon_zones: NDArray[np.float64],
    kappa_zones: NDArray[np.float64],
    n_impact: int = 64,
) -> NDArray[np.float64]:
    """
    Area-averaged radiance over the projected plasma disk
    (W m^-2 Hz^-1 sr^-1).

        <I_nu> = (2 / R^2) * Integral_0^R I_nu(p) * p dp

    — the spatially *integrated* observation mode of the thesis
    measurements (Ch. 6), as opposed to the spatially resolved
    I_nu(p) of `emergent_radiance`. Evaluated with Gauss-Legendre
    quadrature over p in (0, R); interior nodes only, so the p = R
    tangent ray is never requested.

    Parameters
    ----------
    geometry : PlasmaGeometry
        Supplies the chord decompositions and the outer radius R.
    epsilon_zones, kappa_zones : ndarray, shape (n_zones, n_wavelengths)
        Per-zone spectral properties.
    n_impact : int, optional
        Number of quadrature nodes (default 64). The integrand behaves
        like sqrt(R - p) near the limb, so convergence is algebraic —
        raise n_impact rather than assuming spectral accuracy.

    Returns
    -------
    ndarray, shape (n_wavelengths,)
        Disk-averaged I_nu.
    """
    if int(n_impact) < 2:
        raise ValueError("n_impact must be >= 2")
    radius = geometry.outer_radius_m
    nodes, weights = np.polynomial.legendre.leggauss(int(n_impact))
    # Map (-1, 1) -> (0, R); all nodes interior (0 < p < R).
    p_values = 0.5 * radius * (nodes + 1.0)
    p_weights = 0.5 * radius * weights

    total: Optional[NDArray[np.float64]] = None
    for p, w in zip(p_values, p_weights):
        ray = emergent_radiance(
            geometry.path_segments(float(p)), epsilon_zones, kappa_zones
        )
        contribution = (2.0 / radius**2) * w * p * ray
        total = contribution if total is None else total + contribution
    assert total is not None
    return total


def emergent_spectrum(
    geometry: PlasmaGeometry,
    model: LTESpectralModel,
    impact_parameter_m: Optional[float] = None,
    n_impact: int = 64,
) -> Spectrum:
    """
    Full forward step: geometry + LTE spectral model -> `Spectrum`.

    Evaluates the per-zone properties once
    (`LTESpectralModel.geometry_properties`), solves the transfer
    either along one ray (spatially resolved) or disk-averaged
    (spatially integrated), and packs the Phase 0 `Spectrum` container
    with per-wavelength radiance

        I_lambda = I_nu * c / lambda^2      (W m^-2 m^-1 sr^-1).

    Parameters
    ----------
    geometry : PlasmaGeometry
        The zoned plasma.
    model : LTESpectralModel
        Spectral physics on its wavelength grid.
    impact_parameter_m : float, optional
        If given, the single-ray radiance at that impact parameter;
        if None (default), the disk-integrated radiance.
    n_impact : int, optional
        Quadrature nodes for the disk average (ignored for single rays).

    Returns
    -------
    Spectrum
        wavelength_m = the model grid; intensity = I_lambda; metadata
        records units, observation mode, zone count and radius.
    """
    epsilon_zones, kappa_zones = model.geometry_properties(geometry)
    if impact_parameter_m is None:
        radiance_nu = disk_integrated_radiance(
            geometry, epsilon_zones, kappa_zones, n_impact=n_impact
        )
        mode = "disk-integrated"
    else:
        radiance_nu = emergent_radiance(
            geometry.path_segments(float(impact_parameter_m)),
            epsilon_zones,
            kappa_zones,
        )
        mode = f"impact_parameter={float(impact_parameter_m):.6g} m"

    wavelength = np.array(model.wavelength_m, dtype=np.float64, copy=True)
    radiance_lambda = radiance_nu * C / wavelength**2

    return Spectrum(
        wavelength_m=wavelength,
        intensity=radiance_lambda,
        metadata={
            "intensity_units": "W m^-2 m^-1 sr^-1 (spectral radiance)",
            "observation_mode": mode,
            "n_zones": len(geometry.zones),
            "outer_radius_m": geometry.outer_radius_m,
            "calculation_notes": (
                "Eq. 5-48 (Herrera 2008, p. 119) evaluated exactly over "
                "piecewise-homogeneous zones; LTE source, no external "
                "illumination."
            ),
        },
    )

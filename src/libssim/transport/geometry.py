"""
libssim.transport.geometry
==========================
Spherical onion plasma geometry (Phase 3).

Physical Context (Herrera 2008)
-------------------------------
The MC-LIBS plasma is a sphere with radially varying conditions: the
model prescribes parabolic initial profiles for temperature and number
density (Eqs. 5-36 and 5-37, p. 116),

    T(r, 0)   = T_0   * (1 - k_1 * r^2)
    n_j(r, 0) = n_0^j * (1 - k_2 * r^2)

with per-point Saha closure for the electron/atom/ion densities
(Eqs. 5-38/5-39/5-40, p. 116 — the Appendix D system implemented by
`physics.saha.SahaSolver`), and boundary radius R between 0.1 and
0.5 cm after LTE is established (p. 116). Appendix B (Eqs. B-1..B-9,
pp. 267-269) evolves these profiles self-similarly in time — a Phase 4
(temporal) concern; this module represents one static snapshot.

`SphericalOnion` discretizes the continuous radial profiles into N
concentric shells ("zones"), each a locally uniform `PlasmaState`. The
chord decomposition needed by the transfer solution Eq. 5-48, p. 119
(rays at impact parameter p, r(z) = sqrt(z^2 + p^2)) becomes exact
per-shell path lengths from z_k = sqrt(R_k^2 - p^2).

Ownership of radial information (documented design decision)
-------------------------------------------------------------
**`SphericalOnion.boundaries_m` is the single authority on radii. The
`PlasmaState.radius_m` field of the zone states is NOT read anywhere in
the transport layer.** `PlasmaState` was designed for a single uniform
plasma, where `radius_m` is the outer boundary; in a multi-zone
geometry that per-zone field is redundant with (and subordinate to) the
geometry's boundaries. Factories in this module set each zone's
`radius_m` to its shell's outer boundary purely for human inspection.

Units: strict SI (m, K, m^-3) throughout.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eqs. 5-36, 5-37,
5-38 p. 116; Eq. 5-48 p. 119; App. B pp. 267-269.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from ..core.state import PlasmaState
from ..physics.saha import SahaSolver
from .base import PathSegment, PlasmaGeometry


@dataclass(frozen=True, eq=False)
class SphericalOnion(PlasmaGeometry):
    """
    Concentric-shell discretization of the spherical plasma.

    Parameters
    ----------
    zones : tuple of PlasmaState
        Locally uniform conditions per shell, **innermost first**.
    boundaries_m : tuple of float
        Outer radius of each shell (m), strictly increasing; the last
        entry is the plasma boundary R of Eq. 5-48, p. 119. Shell k
        occupies boundaries_m[k-1] <= r < boundaries_m[k] (inner shell
        starts at r = 0).

    Notes
    -----
    - The geometry owns the radii: zone `PlasmaState.radius_m` values
      are ignored by transport (module docstring).
    - Frozen and eq=False (identity comparison), consistent with the
      other array/state-holding containers in this project.
    - A one-zone onion is the uniform sphere used by the analytic
      validation limit I = B_nu * (1 - exp(-kappa * L)) (Eq. 3-10,
      p. 55, with l the chord length).
    """

    zones: Tuple[PlasmaState, ...]
    boundaries_m: Tuple[float, ...]

    def __post_init__(self) -> None:
        zones = tuple(self.zones)
        bounds = tuple(float(b) for b in self.boundaries_m)
        if len(zones) == 0:
            raise ValueError("at least one zone is required")
        if len(zones) != len(bounds):
            raise ValueError(
                f"zones ({len(zones)}) and boundaries_m ({len(bounds)}) "
                "must have the same length"
            )
        for zone in zones:
            if not isinstance(zone, PlasmaState):
                raise TypeError("every zone must be a PlasmaState")
        if not all(np.isfinite(b) and b > 0.0 for b in bounds):
            raise ValueError("boundaries_m must be finite and > 0")
        if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
            raise ValueError("boundaries_m must be strictly increasing")
        object.__setattr__(self, "zones", zones)
        object.__setattr__(self, "boundaries_m", bounds)

    # ------------------------------------------------------------------
    @property
    def outer_radius_m(self) -> float:
        """Plasma boundary R (m) — the r = R of Eq. 5-48, p. 119."""
        return self.boundaries_m[-1]

    @property
    def n_zones(self) -> int:
        """Number of concentric shells."""
        return len(self.zones)

    # ------------------------------------------------------------------
    def path_segments(
        self, impact_parameter_m: float
    ) -> Tuple[PathSegment, ...]:
        """
        Ordered homogeneous segments of the chord at impact parameter p.

        Geometry of Eq. 5-48, p. 119 (Herrera 2008): along the ray,
        r(z) = sqrt(z^2 + p^2), so shell boundary R_k is crossed at
        z_k = sqrt(R_k^2 - p^2) (if R_k > p). The chord enters through
        the outermost shell, descends to the deepest shell with
        R_k > p — crossed once through its full central chord — and
        exits mirror-symmetrically:

            [outermost ... deepest ... outermost]

        Segment lengths telescope to the full chord 2*sqrt(R^2 - p^2)
        (unit-tested). Grazing shells (zero-length crossings, p exactly
        on a boundary) are dropped.

        Parameters
        ----------
        impact_parameter_m : float
            0 <= p < R (m). p >= R does not intersect the plasma and
            raises (disk quadratures should use interior nodes).

        Returns
        -------
        tuple of PathSegment
            Far-boundary -> observer ordering (the integration direction
            of Eq. 5-48, no radiation entering from outside).
        """
        p = float(impact_parameter_m)
        if not (np.isfinite(p) and p >= 0.0):
            raise ValueError("impact_parameter_m must be finite and >= 0")
        radius = self.outer_radius_m
        if p >= radius:
            raise ValueError(
                f"impact parameter {p:.6g} m lies outside the plasma "
                f"(R = {radius:.6g} m); rays must satisfy 0 <= p < R"
            )

        # Half-chord coordinate of each crossed boundary: z_k > 0 only
        # for shells the ray actually penetrates (R_k > p).
        bounds = np.asarray(self.boundaries_m)
        z = np.sqrt(np.maximum(bounds**2 - p**2, 0.0))

        # Half-path length inside shell k is z_k - z_{k-1} (z_{-1} = 0);
        # the deepest crossed shell has z_{k-1} = 0 and is traversed
        # centrally with length 2*z_k, all other shells twice (in/out).
        half_lengths = np.diff(z, prepend=0.0)

        inbound: list[PathSegment] = []   # far boundary -> center
        for k in range(self.n_zones - 1, -1, -1):
            if half_lengths[k] > 0.0:
                inbound.append(PathSegment(k, float(half_lengths[k])))
        # Mirror symmetry of the chord: outbound half repeats the
        # inbound shells in reverse, except the deepest (central) shell,
        # whose two halves are merged into one segment of twice the
        # length (fewer segments, identical physics).
        deepest = inbound[-1]
        central = PathSegment(deepest.zone_index, 2.0 * deepest.length_m)
        outbound = tuple(reversed(inbound[:-1]))
        return tuple(inbound[:-1]) + (central,) + outbound

    # ------------------------------------------------------------------
    @classmethod
    def from_parabolic_profiles(
        cls,
        *,
        center_temperature_K: float,
        temperature_gradient_k1: float,
        center_densities_m3: Mapping[str, float],
        density_gradient_k2: float,
        outer_radius_m: float,
        n_zones: int,
        saha_solver: SahaSolver,
        time_s: float = 0.0,
    ) -> "SphericalOnion":
        """
        Build an onion from the thesis' parabolic initial profiles.

        Samples Eqs. 5-36 and 5-37, p. 116 (Herrera 2008) at the
        mid-radius of each of `n_zones` equal-thickness shells,

            T_i   = T_0   * (1 - k_1 * r_mid^2)
            n_j,i = n_0^j * (1 - k_2 * r_mid^2)

        and closes each zone's electron density self-consistently with
        the Saha charge equilibrium, Eq. 5-38, p. 116 (identically
        Eq. D-9, pp. 275-276), via `SahaSolver.solve_electron_density`
        — the exact per-point pipeline of the MC-LIBS model.

        Parameters
        ----------
        center_temperature_K : float
            T_0 at the plasma center (K, > 0).
        temperature_gradient_k1, density_gradient_k2 : float
            The k_1, k_2 coefficients of Eqs. 5-36/5-37 (m^-2, >= 0),
            "dependent on the gradient desired" (p. 116). Both profiles
            must stay positive inside R: k * R^2 < 1 is enforced.
        center_densities_m3 : Mapping[str, float]
            n_0^j per element at the center (m^-3, > 0) — total heavy
            (atom + ion) densities, the n_j of Eq. 5-37 / App. B.
        outer_radius_m : float
            Plasma boundary R (m, > 0); thesis range 0.1-0.5 cm
            (p. 116) is typical but not enforced.
        n_zones : int
            Number of equal-thickness shells (>= 1). Equal thickness is
            an implementation choice (documented); refine by raising
            n_zones rather than by non-uniform meshes.
        saha_solver : SahaSolver
            Phase 2 solver carrying partition data and ionization
            energies for every element in `center_densities_m3`.
        time_s : float, optional
            Timestamp stored on the zone states (default 0 — the
            "initial distribution" of Eqs. 5-36/5-37). Time evolution
            (App. B) belongs to Phase 4.

        Returns
        -------
        SphericalOnion
            Zone `radius_m` fields are set to each shell's outer
            boundary for inspection only — the geometry's boundaries
            are authoritative (module docstring).
        """
        T0 = float(center_temperature_K)
        k1 = float(temperature_gradient_k1)
        k2 = float(density_gradient_k2)
        R = float(outer_radius_m)
        n = int(n_zones)
        if not (T0 > 0.0 and np.isfinite(T0)):
            raise ValueError("center_temperature_K must be finite and > 0")
        if k1 < 0.0 or k2 < 0.0 or not np.isfinite(k1) or not np.isfinite(k2):
            raise ValueError("gradient coefficients must be finite and >= 0")
        if not (R > 0.0 and np.isfinite(R)):
            raise ValueError("outer_radius_m must be finite and > 0")
        if n < 1:
            raise ValueError("n_zones must be >= 1")
        if k1 * R**2 >= 1.0 or k2 * R**2 >= 1.0:
            raise ValueError(
                "parabolic profiles must stay positive inside the plasma: "
                "require k1*R^2 < 1 and k2*R^2 < 1 (Eqs. 5-36/5-37 are "
                "only meaningful while T, n_j > 0)"
            )
        densities0 = {
            str(el): float(v) for el, v in dict(center_densities_m3).items()
        }
        if not densities0:
            raise ValueError("center_densities_m3 must not be empty")
        for el, v in densities0.items():
            if not (v > 0.0 and np.isfinite(v)):
                raise ValueError(f"center density for {el!r} must be > 0")

        boundaries = tuple(R * (i + 1) / n for i in range(n))
        zones = []
        for i, outer in enumerate(boundaries):
            inner = boundaries[i - 1] if i > 0 else 0.0
            r_mid = 0.5 * (inner + outer)
            profile_t = 1.0 - k1 * r_mid**2   # Eq. 5-36
            profile_n = 1.0 - k2 * r_mid**2   # Eq. 5-37
            temperature = T0 * profile_t
            heavies = {el: v * profile_n for el, v in densities0.items()}
            # Eq. 5-38 closure: zone electron density from charge
            # equilibrium at the local (T, n_j).
            n_e = saha_solver.solve_electron_density(temperature, heavies)
            total_heavy = sum(heavies.values())
            zones.append(
                PlasmaState(
                    temperature_K=temperature,
                    electron_density_m3=n_e,
                    # PlasmaState totals count atoms + ions + electrons
                    # (state.py docs).
                    total_density_m3=total_heavy + n_e,
                    radius_m=outer,  # informational only; see module docs
                    time_s=time_s,
                    composition=heavies,  # normalized by PlasmaState
                )
            )
        return cls(zones=tuple(zones), boundaries_m=boundaries)

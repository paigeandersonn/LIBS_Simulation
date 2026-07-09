"""
libssim.transport.base
======================
Geometry abstractions for spatial radiative transport (Phase 3).

Physical Context (Herrera 2008)
-------------------------------
The MC-LIBS radiative model solves the stationary radiative transfer
equation in spherical coordinates (Eq. 5-44, p. 117) along straight
rays. Its boundary-radiance solution, Eq. 5-48, p. 119, parameterizes
each ray by the observation angle phi, i.e. by the chord at
perpendicular distance ("impact parameter")

    p = R * sqrt(1 - phi^2),        r(z) = sqrt(z^2 + p^2),

with z the coordinate along the ray. Radiative transfer therefore never
needs the full 3-D field — only, per ray, the ordered sequence of path
lengths through regions of locally uniform conditions.

That reduction is the contract defined here: a `PlasmaGeometry` turns an
impact parameter into an ordered tuple of `PathSegment`s (far boundary
-> observer), each referencing one homogeneous zone. The transfer solver
(radiative.py) then applies the exact homogeneous-medium solution of
Eq. 5-44 segment by segment, which is what makes the line-of-sight
integration numerically stable (Phase 3 acceptance criterion).

Design notes
------------
- Zones are full `PlasmaState` objects: the spatial model is "many local
  uniform plasmas", so every Phase 2 physics routine applies per zone
  unchanged (architecture.md layering).
- Only `path_segments` is enforced abstract at runtime; `zones` and
  `outer_radius_m` are annotated attributes of the protocol, implemented
  as fields/properties by concrete geometries (an @abstractmethod
  property cannot be overridden by a dataclass field).
- The geometry owns all radial information. `PlasmaState.radius_m` is
  **not used** by the transport layer (see geometry.py).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-44 p. 117;
Eqs. 5-45, 5-48 pp. 118-119.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..core.state import PlasmaState


@dataclass(frozen=True)
class PathSegment:
    """
    One homogeneous piece of a line of sight.

    Attributes
    ----------
    zone_index : int
        Index into the geometry's `zones` tuple (>= 0).
    length_m : float
        Geometric path length through that zone (m, > 0). Zero-length
        crossings (grazing chords) are dropped by the geometry rather
        than emitted.

    Notes
    -----
    Segments are ordered from the far plasma boundary toward the
    observer, matching the integration direction of Eq. 5-48, p. 119
    (Herrera 2008), whose boundary condition is that no radiation enters
    the plasma from outside.
    """

    zone_index: int
    length_m: float

    def __post_init__(self) -> None:
        if int(self.zone_index) != self.zone_index or self.zone_index < 0:
            raise ValueError("zone_index must be a non-negative integer")
        if not (self.length_m > 0.0 and np.isfinite(self.length_m)):
            raise ValueError("length_m must be finite and > 0")


class PlasmaGeometry(ABC):
    """
    Abstract spatial map of the plasma for radiative transport.

    A concrete geometry provides:

    - ``zones`` : tuple[PlasmaState, ...] — the locally uniform
      conditions, index-addressed by `PathSegment.zone_index`;
    - ``outer_radius_m`` : float — the outer plasma boundary R of
      Eq. 5-48, p. 119 (Herrera 2008);
    - ``path_segments(impact_parameter_m)`` — the ordered homogeneous
      decomposition of the chord at that impact parameter.

    `zones` is declared as an annotation (not an abstract property) so
    concrete geometries may implement it as a frozen-dataclass field;
    `outer_radius_m` and `path_segments` are enforced abstract.
    """

    #: Locally uniform plasma conditions, innermost zone first.
    zones: Tuple[PlasmaState, ...]

    @property
    @abstractmethod
    def outer_radius_m(self) -> float:
        """Outer plasma boundary R (m) — the r = R of Eq. 5-48, p. 119."""

    @abstractmethod
    def path_segments(
        self, impact_parameter_m: float
    ) -> Tuple[PathSegment, ...]:
        """
        Decompose the chord at the given impact parameter into ordered
        homogeneous segments (far boundary -> observer).

        Parameters
        ----------
        impact_parameter_m : float
            Perpendicular distance p of the ray from the plasma center
            (m), 0 <= p < outer_radius_m. This is the R*sqrt(1 - phi^2)
            of Eq. 5-48, p. 119 (Herrera 2008).

        Returns
        -------
        tuple of PathSegment
            Ordered segments; their lengths sum to the full chord
            length 2*sqrt(R^2 - p^2).

        Raises
        ------
        ValueError
            If the impact parameter is negative, non-finite, or lies
            outside the plasma (p >= R).
        """

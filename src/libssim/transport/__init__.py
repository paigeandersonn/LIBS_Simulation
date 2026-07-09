"""
libssim.transport
=================
Spatial transport layer (Phase 3): zoned plasma geometry and
line-of-sight radiative transfer with self-absorption.

Moves from the single uniform `PlasmaState` of Phase 2 to a spatially
varying plasma per Herrera (2008): parabolic radial profiles
(Eqs. 5-36/5-37, p. 116) discretized into a spherical onion, with the
radiative transfer solution of Eq. 5-48, p. 119 evaluated exactly over
the homogeneous shells.

Provides:
- `PathSegment`, `PlasmaGeometry` — the ray-decomposition contract
  (`base`)
- `SphericalOnion` — concentric shells + parabolic-profile factory
  (`geometry`); owns all radial information (zone
  `PlasmaState.radius_m` is not used by this layer)
- `LTESpectralModel` — per-zone (epsilon_nu, kappa'_nu) from the
  Phase 2 physics (`emissivity`)
- `optical_depth`, `emergent_radiance`, `disk_integrated_radiance`,
  `emergent_spectrum` — the transfer solver (`radiative`)

Strict SI units throughout (development_rules.md).
"""

from .base import PathSegment, PlasmaGeometry
from .geometry import SphericalOnion
from .emissivity import LTESpectralModel
from .radiative import (
    disk_integrated_radiance,
    emergent_radiance,
    emergent_spectrum,
    optical_depth,
)

__all__ = [
    "PathSegment",
    "PlasmaGeometry",
    "SphericalOnion",
    "LTESpectralModel",
    "optical_depth",
    "emergent_radiance",
    "disk_integrated_radiance",
    "emergent_spectrum",
]

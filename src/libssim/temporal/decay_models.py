r"""Closed-form time profiles and concrete plasma evolutions
(Phase 4).

Physical context and provenance
-------------------------------
The thesis evolves the plasma with a coupled ODE system: an
energy-balance equation for the temperature (Eq. 5-35, p. 115, derived
in Appendix C) and a self-similar hydrodynamic expansion for the
densities (Appendix B, Eqs. B-1..B-9, pp. 267-269). It does **not**
prescribe closed-form $T(t)$ or $n_e(t)$.

The simple profiles here are therefore *documented simplifications*,
standard in the time-resolved LIBS characterization literature:

- Power-law decays $T(t) = T_{\mathrm{ref}} (t/t_{\mathrm{ref}})^{-b}$
  and $n_e(t) = n_{\mathrm{ref}} (t/t_{\mathrm{ref}})^{-a}$ fit
  measured plasma histories well over the usual 0.5-10 µs analysis
  window; see Aguilera & Aragón, Spectrochim. Acta Part B 59 (2004)
  1861, and Cristoforetti et al., Spectrochim. Acta Part B 59 (2004)
  1907 (= thesis ref [171]).
- Exponential decay is a convenient alternative when a single time
  constant is known.

Every consumer accepts any `TimeProfile` callable, so these defaults
can be replaced by measured histories or by a numerical solution of
the thesis ODE system without touching the rest of the framework.

Evolutions
----------
- `UniformPlasmaEvolution` — **the primary, simplest option**: one
  homogeneous sphere with T(t), heavy density(t), radius(t) and fixed
  composition; n_e(t) either prescribed or closed self-consistently by
  Saha charge equilibrium (Eq. 5-38, p. 116) at each instant. This is
  the intended path for early validation on well-documented elements
  (Na, Al, Fe).
- `ExpandingOnionEvolution` — **advanced/optional**: the thesis'
  self-similar expansion (App. B) applied to the parabolic profiles of
  Eqs. 5-36/5-37, p. 116. Density evolution follows Eq. B-9, p. 269
  exactly; the temperature *profile shape* is assumed frozen in the
  self-similar coordinate r/R(t) with a user-supplied center history
  T_0(t) — a documented simplification of the full App. C energy
  equation (see class docstring).
- `CustomEvolution` — wraps any user factory t -> PlasmaGeometry
  (the escape hatch for arbitrary models).

Units: strict SI (s, K, m^-3, m).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eqs. 5-36/5-37/
5-38 p. 116; Eq. 5-35 p. 115; App. B Eqs. B-7/B-9 pp. 268-269.
Aguilera, J.A. & Aragon, C. (2004). Spectrochim. Acta B 59, 1861.
Cristoforetti, G. et al. (2004). Spectrochim. Acta B 59, 1907
[thesis ref 171].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np

from ..core.state import PlasmaState
from ..physics.saha import SahaSolver
from ..transport.base import PlasmaGeometry
from ..transport.geometry import SphericalOnion
from .base import PlasmaEvolution, TimeProfile, evaluate_profile, validate_time


# ---------------------------------------------------------------------------
# Scalar time profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Constant:
    """Time-independent profile: v(t) = value (finite)."""

    value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("value must be finite")

    def __call__(self, time_s: float) -> float:
        validate_time(time_s)
        return float(self.value)


@dataclass(frozen=True)
class ExponentialDecay:
    r"""
    Exponential decay profile:

    $$
    v(t) \;=\; v_0\,
    \exp\!\left(-\frac{t - t_{\mathrm{start}}}{\tau}\right)
    $$

    for $t \ge t_{\mathrm{start}}$; clamped to `initial_value` before
    `start_time_s` (the parameter is taken as constant until its decay
    begins — documented convention).

    Parameters
    ----------
    initial_value : float
        Value at t = start_time_s (finite).
    decay_time_s : float
        e-folding time tau (s, > 0).
    start_time_s : float, optional
        Decay onset (s, >= 0, default 0).
    """

    initial_value: float
    decay_time_s: float
    start_time_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_value):
            raise ValueError("initial_value must be finite")
        if not (self.decay_time_s > 0.0 and np.isfinite(self.decay_time_s)):
            raise ValueError("decay_time_s must be finite and > 0")
        if not (self.start_time_s >= 0.0 and np.isfinite(self.start_time_s)):
            raise ValueError("start_time_s must be finite and >= 0")

    def __call__(self, time_s: float) -> float:
        t = validate_time(time_s)
        if t <= self.start_time_s:
            return float(self.initial_value)
        return float(
            self.initial_value
            * np.exp(-(t - self.start_time_s) / self.decay_time_s)
        )


@dataclass(frozen=True)
class PowerLawDecay:
    r"""
    Power-law decay profile (the time-resolved LIBS standard —
    Aguilera & Aragón 2004; Cristoforetti et al. 2004 [thesis ref
    171]; module docstring):

    $$
    v(t) \;=\; v_{\mathrm{ref}}\,
    \left(\frac{t}{t_{\mathrm{ref}}}\right)^{-b}
    $$

    Defined for $t > 0$ only: the power law diverges at $t = 0$,
    mirroring the breakdown singularity it approximates — evaluate at
    gate times, not at zero.

    Parameters
    ----------
    reference_value : float
        Value at t = reference_time_s (finite).
    reference_time_s : float
        Anchor time t_ref (s, > 0), typically the earliest delay at
        which LTE analysis is trusted.
    exponent : float
        Decay exponent b >= 0 (measured values ~0.2-0.7 for T,
        ~1-2 for n_e in the cited literature).
    """

    reference_value: float
    reference_time_s: float
    exponent: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.reference_value):
            raise ValueError("reference_value must be finite")
        if not (
            self.reference_time_s > 0.0 and np.isfinite(self.reference_time_s)
        ):
            raise ValueError("reference_time_s must be finite and > 0")
        if not (self.exponent >= 0.0 and np.isfinite(self.exponent)):
            raise ValueError("exponent must be finite and >= 0")

    def __call__(self, time_s: float) -> float:
        t = validate_time(time_s)
        if t <= 0.0:
            raise ValueError(
                "PowerLawDecay is defined for t > 0 only (diverges at the "
                "breakdown instant); evaluate at gate times"
            )
        return float(
            self.reference_value * (t / self.reference_time_s) ** (-self.exponent)
        )


# ---------------------------------------------------------------------------
# Evolutions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class UniformPlasmaEvolution(PlasmaEvolution):
    """
    Homogeneous spherical plasma with time-varying parameters — the
    primary Phase 4 path (module docstring).

    At each requested time the profiles are evaluated and packed into a
    one-zone `SphericalOnion`, so every Phase 2/3 routine applies
    unchanged.

    Parameters
    ----------
    temperature_K : TimeProfile
        T(t) in Kelvin (> 0 at every evaluated time).
    heavy_density_m3 : TimeProfile
        Total heavy-particle (atom + ion) density n(t) in m^-3 (>= 0),
        distributed over `composition`.
    composition : Mapping[str, float]
        Fixed relative elemental fractions (stoichiometric ablation
        assumption, thesis Ch. 5; normalized by `PlasmaState`).
    radius_m : TimeProfile or float
        Plasma radius R(t) (m, > 0). A float means a static radius.
    electron_density_m3 : TimeProfile, optional
        Prescribed n_e(t) (e.g. from Stark-broadening measurements,
        Eqs. 5-17/5-19). Mutually exclusive with `saha_solver`.
    saha_solver : SahaSolver, optional
        If given (and no n_e profile), n_e(t) is closed
        self-consistently by charge equilibrium, Eq. 5-38, p. 116, at
        each instant.

    Raises
    ------
    ValueError
        If neither or both of electron_density_m3 / saha_solver are
        provided, or a profile evaluates out of physical range.
    """

    temperature_K: TimeProfile
    heavy_density_m3: TimeProfile
    composition: Mapping[str, float] = field(default_factory=dict)
    radius_m: TimeProfile | float = 1.0e-3
    electron_density_m3: Optional[TimeProfile] = None
    saha_solver: Optional[SahaSolver] = None

    def __post_init__(self) -> None:
        composition = dict(self.composition)
        if not composition:
            raise ValueError("composition must not be empty")
        object.__setattr__(self, "composition", composition)
        if (self.electron_density_m3 is None) == (self.saha_solver is None):
            raise ValueError(
                "provide exactly one of electron_density_m3 (prescribed "
                "n_e(t)) or saha_solver (Eq. 5-38 closure)"
            )
        radius = self.radius_m
        if not callable(radius):
            radius_value = float(radius)
            if not (radius_value > 0.0 and np.isfinite(radius_value)):
                raise ValueError("radius_m must be finite and > 0")
            object.__setattr__(self, "radius_m", Constant(radius_value))

    def geometry_at(self, time_s: float) -> PlasmaGeometry:
        t = validate_time(time_s)
        temperature = evaluate_profile(self.temperature_K, t, "temperature_K")
        if temperature <= 0.0:
            raise ValueError(
                f"temperature profile returned {temperature:.6g} K <= 0 "
                f"at t = {t:.6g} s"
            )
        heavy = evaluate_profile(self.heavy_density_m3, t, "heavy_density_m3")
        if heavy < 0.0:
            raise ValueError(
                f"heavy-density profile returned {heavy:.6g} m^-3 < 0 "
                f"at t = {t:.6g} s"
            )
        radius = evaluate_profile(self.radius_m, t, "radius_m")  # type: ignore[arg-type]
        if radius <= 0.0:
            raise ValueError(
                f"radius profile returned {radius:.6g} m <= 0 at t = {t:.6g} s"
            )
        heavies = {el: frac * heavy for el, frac in self.composition.items()}
        if self.electron_density_m3 is not None:
            n_e = evaluate_profile(
                self.electron_density_m3, t, "electron_density_m3"
            )
            if n_e < 0.0:
                raise ValueError(
                    f"electron-density profile returned {n_e:.6g} m^-3 < 0 "
                    f"at t = {t:.6g} s"
                )
        else:
            assert self.saha_solver is not None
            n_e = self.saha_solver.solve_electron_density(temperature, heavies)

        state = PlasmaState(
            temperature_K=temperature,
            electron_density_m3=n_e,
            total_density_m3=heavy + n_e,
            radius_m=radius,  # informational; geometry owns radii
            time_s=t,
            composition=heavies,
        )
        return SphericalOnion(zones=(state,), boundaries_m=(radius,))


@dataclass(frozen=True, eq=False)
class CustomEvolution(PlasmaEvolution):
    """
    Escape hatch: wrap any user factory t -> PlasmaGeometry.

    Use for measured multi-zone histories or a numerical solution of
    the thesis App. B/C system; the factory's return type is checked at
    every call so failures surface at the offending time.
    """

    geometry_factory: Callable[[float], PlasmaGeometry]

    def geometry_at(self, time_s: float) -> PlasmaGeometry:
        t = validate_time(time_s)
        geometry = self.geometry_factory(t)
        if not isinstance(geometry, PlasmaGeometry):
            raise TypeError(
                "geometry_factory must return a PlasmaGeometry, got "
                f"{type(geometry).__name__!r} at t = {t:.6g} s"
            )
        return geometry


@dataclass(frozen=True, eq=False)
class ExpandingOnionEvolution(PlasmaEvolution):
    r"""
    Self-similar expanding parabolic onion — **advanced/optional**
    (Herrera 2008, Appendix B applied to the Eq. 5-36/5-37 profiles).

    Density evolution is Eq. B-9, p. 269, exactly: with the parabolic
    initial profile $g_1(r) = n_0^{j} (1 - k_2 r^{2})$ (Eq. B-7,
    p. 268 = Eq. 5-37, p. 116),

    $$
    n_j(r, t) \;=\;
    \left(\frac{R_0}{R(t)}\right)^{3}
    g_1\!\left(r\, \frac{R_0}{R(t)}\right)
    $$

    which is again parabolic with center density
    $n_0^{j} (R_0/R)^{3}$ and gradient $k_2 (R_0/R)^{2}$ — so each
    snapshot is generated by the Phase 3
    `SphericalOnion.from_parabolic_profiles` factory unchanged. Note
    the invariant $k_2(t)\, R(t)^{2} = k_2 R_0^{2}$: a profile
    positive at $t = 0$ stays positive at all times.

    Documented simplification (development_rules.md): the
    *temperature* profile shape is assumed frozen in the self-similar
    coordinate $r/R(t)$,
    $T(r,t) = T_0(t)\,(1 - k_1 (r R_0/R)^{2})$, with the center
    history $T_0(t)$ supplied by the user (e.g. a `PowerLawDecay`).
    The thesis instead evolves $T$ through the energy-balance ODE
    (Eq. 5-35, p. 115 / App. C, Eq. C-21) — plug a numerical solution
    in through `CustomEvolution` when that fidelity is needed.

    Parameters
    ----------
    center_temperature_K : TimeProfile
        T_0(t) at the plasma center (K, > 0 at evaluated times).
    radius_m : TimeProfile
        Outer radius history R(t) (m, > 0), e.g. from plasma imaging
        (thesis Fig. 6-2) or the App. A expansion model.
    initial_radius_m : float
        R_0 = R(0) (m, > 0), the reference radius of Eq. B-9.
    temperature_gradient_k1, density_gradient_k2 : float
        Initial parabolic coefficients (m^-2, >= 0) of Eqs. 5-36/5-37;
        k * R_0^2 < 1 required (positivity inside the plasma).
    center_densities_m3 : Mapping[str, float]
        Initial center heavy densities n_0^j (m^-3, > 0).
    n_zones : int
        Shells per snapshot (>= 1).
    saha_solver : SahaSolver
        Per-zone n_e closure (Eq. 5-38), as in the Phase 3 factory.
    """

    center_temperature_K: TimeProfile
    radius_m: TimeProfile
    initial_radius_m: float
    temperature_gradient_k1: float
    density_gradient_k2: float
    center_densities_m3: Mapping[str, float]
    n_zones: int
    saha_solver: SahaSolver

    def __post_init__(self) -> None:
        R0 = float(self.initial_radius_m)
        if not (R0 > 0.0 and np.isfinite(R0)):
            raise ValueError("initial_radius_m must be finite and > 0")
        k1, k2 = float(self.temperature_gradient_k1), float(
            self.density_gradient_k2
        )
        if k1 < 0 or k2 < 0 or not (np.isfinite(k1) and np.isfinite(k2)):
            raise ValueError("gradient coefficients must be finite and >= 0")
        if k1 * R0**2 >= 1.0 or k2 * R0**2 >= 1.0:
            raise ValueError(
                "initial parabolic profiles must stay positive: require "
                "k1*R0^2 < 1 and k2*R0^2 < 1 (Eqs. 5-36/5-37); the "
                "self-similar invariant k*R^2 then preserves positivity "
                "for all times"
            )
        densities = {
            str(el): float(v)
            for el, v in dict(self.center_densities_m3).items()
        }
        if not densities:
            raise ValueError("center_densities_m3 must not be empty")
        object.__setattr__(self, "center_densities_m3", densities)
        if int(self.n_zones) < 1:
            raise ValueError("n_zones must be >= 1")

    def geometry_at(self, time_s: float) -> PlasmaGeometry:
        t = validate_time(time_s)
        radius = evaluate_profile(self.radius_m, t, "radius_m")
        if radius <= 0.0:
            raise ValueError(
                f"radius profile returned {radius:.6g} m <= 0 at t = {t:.6g} s"
            )
        t_center = evaluate_profile(
            self.center_temperature_K, t, "center_temperature_K"
        )
        scale = self.initial_radius_m / radius  # R_0 / R(t) of Eq. B-9
        return SphericalOnion.from_parabolic_profiles(
            center_temperature_K=t_center,
            temperature_gradient_k1=self.temperature_gradient_k1 * scale**2,
            center_densities_m3={
                el: v * scale**3  # (R_0/R)^3 dilution, Eq. B-9
                for el, v in self.center_densities_m3.items()
            },
            density_gradient_k2=self.density_gradient_k2 * scale**2,
            outer_radius_m=radius,
            n_zones=int(self.n_zones),
            saha_solver=self.saha_solver,
            time_s=t,
        )

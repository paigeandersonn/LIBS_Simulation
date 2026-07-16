r"""Spontaneous line emission, bound-bound absorption and blackbody
radiance for LTE plasmas (Phase 2).

Physical context (Herrera 2008)
-------------------------------
Line radiation in the thesis appears in two equivalent guises.

**CF-LIBS integrated line intensity** (Eq. 5-8, pp. 103-104) — the
spontaneous photon emission rate per unit volume, with $n_k^{s}$ the
LTE upper-level density from Eq. 5-1 (`boltzmann`):

$$
I_{ki} \;=\; n_k^{s}\, A_{ki}
\qquad [\mathrm{photons\; cm^{-3}\, s^{-1}}]
$$

**MC-LIBS radiative model** — the radiative transfer equation
(Eq. 5-44, p. 117):

$$
\phi\,\frac{\partial I_\nu}{\partial r}
+ \frac{1-\phi^{2}}{r}\,\frac{\partial I_\nu}{\partial \phi}
+ \kappa'_\nu I_\nu \;=\; \kappa'_\nu I_\nu^{b}
$$

whose LTE source term is $\kappa'_\nu I_\nu^{b}$ (Kirchhoff's law),
with $I_\nu^{b}$ the blackbody spectral radiance (Planck's law,
Eq. 3-11, p. 55; symbols p. 20). The bound-bound contribution to
$\kappa'_\nu$ is Eq. 5-52, p. 120:

$$
\kappa_{bb} \;=\; \frac{h\nu}{c} \sum_j \sum_{l,u}
B_{lu}^{j}\, n_s^{j}\, P_{lu}^{j}(\nu)
\left(1 - e^{-h\nu/(k_B T)}\right)
$$

where $B_{lu}$ is the Einstein absorption coefficient in the spectral
energy-density convention (thesis units cm$^3$ erg$^{-1}$ s$^{-1}$ Hz,
p. 120), $n_s$ the density of the absorbing (lower-level) species and
$P_{lu}$ the normalized line profile (Voigt, Eq. 5-53 —
`line_profiles`). The $(1 - e^{-h\nu/(k_B T)})$ factor is the LTE
stimulated-emission correction that makes $\kappa'_\nu$ a *total*
absorption coefficient (p. 117).

This module implements the per-line SI versions of these quantities;
spatial integration along the line of sight (Eqs. 5-45/5-48) is
Phase 3.

Units and Conventions (SI throughout, development_rules.md)
-----------------------------------------------------------
- Photon rate:                photons m^-3 s^-1        (Eq. 5-8)
- Emitted power density:      W m^-3
- Emission coefficient:       W m^-3 Hz^-1 sr^-1  (isotropic emission,
  factor 1/4pi; consistent with radiance in Eq. 5-44, thesis CGS
  erg s^-1 cm^-2 Hz^-1 sr^-1)
- Absorption coefficient:     m^-1                     (thesis cm^-1)
- Blackbody spectral radiance: W m^-2 Hz^-1 sr^-1 (per Hz) or
  W m^-2 m^-1 sr^-1 (per wavelength; thesis B_lambda^bb is per nm, p. 20)
- Normalized line profile P(nu): Hz^-1, with integral over nu equal 1
  (validated in line_profiles.py).

Implementation decisions (documented per development_rules.md)
--------------------------------------------------------------
- The Einstein $A \to B$ conversion,

    $$
    B_{lu} \;=\; \frac{g_u}{g_l}\,
    \frac{c^{3}}{8\pi h \nu_0^{3}}\, A_{ul}
    $$

    is not printed in the thesis (which takes $B_{lu}$ as data,
    p. 120); it is the standard detailed-balance relation in the same
    energy-density convention as the thesis units (e.g. Mihalas,
    Stellar Atmospheres). Correctness is enforced by the Kirchhoff
    identity $\epsilon_\nu / \kappa_{bb} = B_\nu(T)$ at line center
    under LTE, verified numerically in the unit tests.

- Narrow-line approximation: the photon energy $h\nu$ and the
  stimulated factor are evaluated at the line-center frequency
  $\nu_0 = c/\lambda_0$ rather than the running frequency (profile
  widths ~10 GHz against $\nu_0 \sim 10^{15}$ Hz, relative variation
  $< 10^{-4}$ across the line).
- Transition wavelengths are used as stored by the atomic layer; no
  air/vacuum correction is applied here (instrument-layer concern,
  Phase 4).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-8
pp. 103-104; Eq. 5-44 p. 117; Eq. 5-52 p. 120; Eq. 3-11 p. 55; symbols
pp. 19-23.
Mihalas, D. (1978). Stellar Atmospheres, 2nd ed. (Einstein relations).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.constants import C, H, KB
from ..atomic.transition import Transition


def _as_nonnegative(name: str, value: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and >= 0")
    return arr


def _as_positive(name: str, value: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and > 0")
    return arr


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if np.ndim(value) == 0:
        return float(value)
    return value


def transition_frequency_hz(transition: Transition) -> float:
    r"""
    Line-center frequency $\nu_0 = c / \lambda_0$ in Hz.

    Uses the transition wavelength as stored in SI by the atomic layer
    (Phase 1); see module notes on the air/vacuum caveat.
    """
    return C / transition.wavelength_m


def line_photon_emission_rate(
    transition: Transition,
    upper_level_density_m3: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Spontaneous photon emission rate of one line, $I_{ki}$
    (photons m$^{-3}$ s$^{-1}$).

    Herrera (2008), Eq. 5-8, pp. 103-104:

    $$
    I_{ki} \;=\; n_k^{s}\, A_{ki}
    $$

    (thesis units photons cm$^{-3}$ s$^{-1}$; here SI m$^{-3}$
    s$^{-1}$). The upper-level density $n_k^{s}$ comes from the
    Boltzmann relation Eq. 5-1, p. 98
    (`boltzmann.upper_level_density`), which makes $I_{ki}$ scale
    with the species density, $e^{-E_k/k_B T}/U(T)$ and $A_{ki}$ —
    the Phase 2 acceptance scaling.

    Parameters
    ----------
    transition : Transition
        Supplies the transition probability $A_{ki}$ (s$^{-1}$).
    upper_level_density_m3 : array_like of float
        $n_k^{s}$ in m$^{-3}$ (>= 0).
    """
    n_k = _as_nonnegative("upper_level_density_m3", upper_level_density_m3)
    return _scalar_or_array(n_k * transition.a_ki)


def line_power_density(
    transition: Transition,
    upper_level_density_m3: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Total radiant power density of one line (W m$^{-3}$).

    Photon rate of Eq. 5-8 (pp. 103-104) weighted by the photon
    energy $h\nu_0$:

    $$
    P \;=\; h \nu_0\, n_k^{s}\, A_{ki}
    $$

    This is the frequency-integrated, solid-angle-integrated
    emission; dividing by $4\pi$ and multiplying by the normalized
    profile $P(\nu)$ gives the emission coefficient
    (`line_emission_coefficient`).
    """
    rate = line_photon_emission_rate(transition, upper_level_density_m3)
    return _scalar_or_array(
        np.asarray(rate, dtype=np.float64) * H * transition_frequency_hz(transition)
    )


def line_emission_coefficient(
    transition: Transition,
    upper_level_density_m3: ArrayLike,
    profile_hz: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Spectral emission coefficient of one line, $\epsilon_\nu$
    (W m$^{-3}$ Hz$^{-1}$ sr$^{-1}$).

    $$
    \epsilon_\nu \;=\; \frac{h \nu_0}{4\pi}\,
    n_k^{s}\, A_{ki}\, P(\nu)
    $$

    This is the LTE source term of the radiative transfer equation,
    Eq. 5-44, p. 117 (Herrera 2008), restricted to one bound-bound
    transition: under LTE it satisfies Kirchhoff's law
    $\epsilon_\nu = \kappa_{bb}\, I_\nu^{b}$ with $\kappa_{bb}$ from
    Eq. 5-52, p. 120 and $I_\nu^{b}$ from Planck's law (Eq. 3-11,
    p. 55) — an identity used as the unit-test cross-check of this
    module.

    Parameters
    ----------
    transition : Transition
        Supplies $A_{ki}$ and the line-center frequency.
    upper_level_density_m3 : array_like of float
        LTE upper-level density $n_k^{s}$ (m$^{-3}$), from
        `boltzmann.upper_level_density`.
    profile_hz : array_like of float
        Normalized line profile $P(\nu)$ in Hz$^{-1}$ evaluated at
        the frequencies of interest (from `line_profiles`; integral
        over $\nu$ = 1). Values must be finite and >= 0.

    Returns
    -------
    float or ndarray
        $\epsilon_\nu$, broadcasting over density and profile inputs.
    """
    n_k = _as_nonnegative("upper_level_density_m3", upper_level_density_m3)
    P = _as_nonnegative("profile_hz", profile_hz)
    nu0 = transition_frequency_hz(transition)
    # photon energy x isotropic 1/4pi x volumetric decay rate (Eq. 5-8)
    # x normalized spectral distribution P(nu).
    return _scalar_or_array(
        (H * nu0 / (4.0 * np.pi)) * n_k * transition.a_ki * P
    )


def einstein_b_lu(transition: Transition) -> float:
    r"""
    Einstein absorption coefficient $B_{lu}$ in the spectral
    energy-density convention (SI: m$^3$ J$^{-1}$ s$^{-2}$).

    Standard detailed-balance relations (Mihalas 1978):

    $$
    \frac{A_{ul}}{B_{ul}} = \frac{8\pi h \nu_0^{3}}{c^{3}},
    \qquad g_l B_{lu} = g_u B_{ul}
    \;\;\Longrightarrow\;\;
    B_{lu} = \frac{g_u}{g_l}\,
    \frac{c^{3}}{8\pi h \nu_0^{3}}\, A_{ul}
    $$

    This is the B_lu^j appearing in the bound-bound absorption
    coefficient, Eq. 5-52, p. 120 of Herrera (2008), where it is treated
    as atomic data with units cm^3 erg^-1 s^-1 Hz (same convention,
    CGS). The relation itself is not printed in the thesis — see module
    Implementation Decisions.
    """
    nu0 = transition_frequency_hz(transition)
    # B_lu = (g_u/g_l) * B_ul with B_ul = A_ul * c^3/(8*pi*h*nu0^3);
    # correctness is pinned by the Kirchhoff closure test.
    return (
        (transition.g_upper / transition.g_lower)
        * C**3
        / (8.0 * np.pi * H * nu0**3)
        * transition.a_ki
    )


def line_absorption_coefficient(
    transition: Transition,
    lower_level_density_m3: ArrayLike,
    temperature_K: ArrayLike,
    profile_hz: ArrayLike,
    include_stimulated_emission: bool = True,
) -> float | NDArray[np.float64]:
    r"""
    Bound-bound absorption coefficient of one line, $\kappa_{bb}$
    (m$^{-1}$).

    Herrera (2008), Eq. 5-52, p. 120, for a single transition:

    $$
    \kappa_{bb} \;=\; \frac{h\nu_0}{c}\, B_{lu}\, n_l\, P(\nu)
    \left(1 - e^{-h\nu_0/(k_B T)}\right)
    $$

    with $B_{lu}$ from `einstein_b_lu`, $n_l$ the number density of
    the absorbing lower level of the species (thesis $n_s^{j}$) and
    $P(\nu)$ the normalized Voigt profile (Eq. 5-53, p. 120;
    `line_profiles`). The stimulated-emission factor makes this the
    *total* (net) absorption coefficient entering $\kappa'_\nu$ of
    Eqs. 5-44/5-49 (pp. 117, 119) under LTE.

    Parameters
    ----------
    transition : Transition
        Atomic data ($A_{ki}$, degeneracies, wavelength).
    lower_level_density_m3 : array_like of float
        Number density of atoms/ions in the transition's *lower*
        level (m$^{-3}$, >= 0) — from
        `boltzmann.boltzmann_population_fraction` with
        `g_lower`/`energy_lower_ev` times the species density.
    temperature_K : array_like of float
        LTE temperature (K, > 0) for the stimulated-emission factor.
    profile_hz : array_like of float
        Normalized line profile $P(\nu)$ in Hz$^{-1}$ (>= 0).
    include_stimulated_emission : bool, optional
        If False, return the raw absorption coefficient without the
        $(1 - e^{-h\nu_0/k_B T})$ correction. Default True (Eq. 5-52
        form).

    Returns
    -------
    float or ndarray
        $\kappa_{bb}$ in m$^{-1}$.

    Notes
    -----
    Narrow-line approximation: $h\nu/c$ and the stimulated factor use
    $\nu_0$ (module notes). Kirchhoff identity
    $\epsilon_\nu/\kappa_{bb} = B_\nu(\nu_0, T)$ holds exactly in
    this approximation for LTE level populations.
    """
    n_l = _as_nonnegative("lower_level_density_m3", lower_level_density_m3)
    P = _as_nonnegative("profile_hz", profile_hz)
    T = _as_positive("temperature_K", temperature_K)

    nu0 = transition_frequency_hz(transition)
    # Eq. 5-52 restricted to one line: (h*nu/c) * B_lu * n_l * P(nu).
    kappa = (H * nu0 / C) * einstein_b_lu(transition) * n_l * P
    if include_stimulated_emission:
        # (1 - e^-x) via expm1: accurate when h*nu0 << k_B*T.
        kappa = kappa * (-np.expm1(-H * nu0 / (KB * T)))
    return _scalar_or_array(kappa)


def blackbody_spectral_radiance_hz(
    frequency_hz: ArrayLike,
    temperature_K: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Planck blackbody spectral radiance per unit frequency, $B_\nu$
    (W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$).

    $$
    B_\nu(T) \;=\; \frac{2 h \nu^{3} / c^{2}}
    {e^{h\nu/(k_B T)} - 1}
    $$

    This is $I_\nu^{b}$, the LTE source radiance of the radiative
    transfer equation, Eq. 5-44, p. 117 of Herrera (2008) (thesis CGS
    units erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$); the wavelength
    form is Eq. 3-11, p. 55.

    Numerically stable at both limits: expm1 keeps the Rayleigh-Jeans
    ($h\nu \ll k_B T$) limit accurate; the Wien tail underflows
    gracefully to 0.
    """
    nu = _as_positive("frequency_hz", frequency_hz)
    T = _as_positive("temperature_K", temperature_K)
    x = H * nu / (KB * T)  # dimensionless photon energy h*nu/k_B*T
    # expm1 keeps the Rayleigh-Jeans limit (x -> 0) accurate.
    with np.errstate(over="ignore"):  # exp overflow -> inf -> B = 0 (Wien tail)
        radiance = (2.0 * H * nu**3 / C**2) / np.expm1(x)
    return _scalar_or_array(radiance)


def blackbody_spectral_radiance_wavelength(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Planck blackbody spectral radiance per unit wavelength,
    $B_\lambda$ (W m$^{-2}$ m$^{-1}$ sr$^{-1}$).

    Herrera (2008), Eq. 3-11, p. 55 ($B_\lambda^{bb}$, thesis
    practical units J s$^{-1}$ cm$^{-2}$ sr$^{-1}$ nm$^{-1}$; symbols
    p. 20):

    $$
    B_\lambda(T) \;=\; \frac{2 h c^{2} / \lambda^{5}}
    {e^{h c/(\lambda k_B T)} - 1}
    $$

    Consistency $B_\lambda = B_\nu\, c / \lambda^{2}$ is enforced in
    the unit tests. Used in Phase 3 for the self-absorption radiance
    limit (Eq. 3-10, p. 55: the line core saturates at the blackbody
    radiance).
    """
    lam = _as_positive("wavelength_m", wavelength_m)
    T = _as_positive("temperature_K", temperature_K)
    x = H * C / (lam * KB * T)
    with np.errstate(over="ignore"):
        radiance = (2.0 * H * C**2 / lam**5) / np.expm1(x)
    return _scalar_or_array(radiance)

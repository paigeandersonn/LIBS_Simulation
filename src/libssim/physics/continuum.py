r"""Bremsstrahlung (free-free) and radiative-recombination
(free-bound) continuum for LTE plasmas (Phase 2).

Physical context (Herrera 2008)
-------------------------------
The MC-LIBS radiative model decomposes the total absorption coefficient
entering the radiative transfer equation (Eq. 5-44, p. 117) as
Eq. 5-49, p. 119:

$$
\kappa'_\nu \;=\; \kappa_{f\!f} + \kappa_{f\!b} + \kappa_{bb}
$$

The continuum members implemented here (bound-bound is `emission`):

**Free-free** (inverse bremsstrahlung; Eq. 5-50, p. 119; printed CGS
coefficient $3.7\times10^{8}$):

$$
\kappa_{f\!f} \;=\;
\frac{8\pi e^{6}}{3 m_e h c \sqrt{6\pi m_e k_B}}\;
\frac{n_e}{T^{1/2}}\, z^{2}\, \frac{G}{\nu^{3}}
\left(1 - e^{-h\nu/(k_B T)}\right) \sum_j n_i^{j}
$$

**Free-bound** (radiative recombination; Eq. 5-51, p. 120):

$$
\kappa_{f\!b} \;=\; [\text{same prefactor}]\;
\frac{n_e}{T^{1/2}}\, z^{2}\, \frac{\xi_z}{\nu^{3}}\,
e^{h\nu/(k_B T)}
\left(1 - e^{-h\nu/(k_B T)}\right)^{2} \sum_j n_i^{j}
$$

**Planck mean** (ff + fb only, for the radiation-loss term $q$ of
Eq. 5-47, p. 118; Eq. 5-46, p. 118):

$$
\kappa_{\mathrm{mean}} \;=\;
\sqrt{\tfrac{128}{27} k_B}\,
\left(\frac{\pi}{m_e}\right)^{3/2}
\frac{z^{2} e^{6} G}{h \sigma c^{3}}\,
\frac{n_e}{T^{7/2}} \sum_j n_i^{j}
$$

Under LTE the continuum emission follows Kirchhoff's law (source term
of Eq. 5-44): $\epsilon_\nu = \kappa_\nu B_\nu(T)$. This reproduces
exactly the bracket structure of the line-to-continuum expression,
Eq. 5-7, p. 101: $\epsilon_c \propto
[\xi (1 - e^{-h\nu/k_B T}) + G e^{-h\nu/k_B T}]$, which is used as a
consistency check in the unit tests.

Documented ambiguity (development_rules.md)
-------------------------------------------
Eq. 5-50 as printed carries the factor "$e^{h\nu/k_B T - 1}$". This is
a typographical error for $(1 - e^{-h\nu/(k_B T)})$, established three
ways: (1) the printed prefactor is algebraically identical to the
standard thermally-averaged free-free absorption (Rybicki & Lightman,
Radiative Processes, Eq. 5.18b), whose practical CGS coefficient is
exactly the "$3.7\times10^{8}$ cgs" printed in Eq. 5-50 and whose
exponential factor is $(1 - e^{-h\nu/k T})$; (2) Kirchhoff's law then
yields the free-free emission $G e^{-h\nu/k T}$ term of Eq. 5-7,
p. 101; (3) the printed form grows exponentially with frequency, which
is unphysical for an absorption coefficient. The corrected factor is
implemented. Eq. 5-51 is implemented exactly as printed (its Kirchhoff
image reproduces the $\xi$-term of Eq. 5-7, confirming it).

Units and conventions (SI throughout)
-------------------------------------
Thesis formulas are Gaussian-CGS ("cgs" tags in Eqs. 5-50/5-51); the
SI implementation replaces $e^{6} \to e^{6}/(4\pi\varepsilon_0)^{3}$.
Inputs: frequency Hz, temperature K, densities m$^{-3}$. Outputs:
absorption coefficients m$^{-1}$; emission coefficient W m$^{-3}$
Hz$^{-1}$ sr$^{-1}$. The unit tests verify the SI prefactor against
the printed $3.7\times10^{8}$ CGS coefficient.

Numerical assumptions and limitations
-------------------------------------
- Hydrogen-like Kramers-Unsold model: $G$ (free-free Gaunt factor)
  and $\xi$ (free-bound Biberman-like correction) are ~ unity and
  slowly varying (thesis p. 118: "in most cases, the numerical value
  of G is approximately unity"); both default to 1.0 and are
  caller-adjustable.
- Two-stage plasma: ions are singly charged, $z = 1$ default
  (Eq. D-3 charge equilibrium, p. 274); $z$ is exposed for generality
  and enters as $z^{2}$ outside the ion sum, as printed.
- The free-bound expression grows $\sim e^{h\nu/k_B T}$ at photon
  energies far above thermal — the Kramers-Unsold model is meaningful
  for $h\nu$ below and around the ionization edge (LIBS UV-VIS
  window); far-UV/X-ray evaluation will overflow to inf rather than
  silently produce wrong finite numbers.
- exp terms use expm1 for accuracy in the Rayleigh-Jeans limit
  ($h\nu \ll k_B T$).

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-7 p. 101;
Eqs. 5-46, 5-47 p. 118; Eqs. 5-49, 5-50 p. 119; Eq. 5-51 p. 120.
Rybicki, G.B. & Lightman, A.P. (1979). Radiative Processes in
Astrophysics. Wiley. (Free-free absorption, Eq. 5.18b.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.constants import C, E, EPSILON0, H, KB, ME, SIGMA
from .emission import blackbody_spectral_radiance_hz

# SI Kramers prefactor of Eqs. 5-50/5-51 (p. 119-120):
#   8*pi*e^6 / (3*m_e*h*c*(6*pi*m_e*k_B)^(1/2)), Gaussian e^6 -> e^6/(4*pi*eps0)^3.
# Units: m^5 K^(1/2) Hz^3 (multiplies n_e*n_i/(T^(1/2) nu^3) to give m^-1).
# Numerically equivalent to the printed "3.7e8 cgs" (unit-test validated).
_KRAMERS_SI: float = (
    8.0
    * np.pi
    * E**6
    / (
        (4.0 * np.pi * EPSILON0) ** 3
        * 3.0
        * ME
        * H
        * C
        * np.sqrt(6.0 * np.pi * ME * KB)
    )
)

# SI prefactor of the Planck mean, Eq. 5-46 (p. 118), same e^6 substitution.
_PLANCK_MEAN_SI: float = (
    np.sqrt(128.0 * KB / 27.0)
    * (np.pi / ME) ** 1.5
    * E**6
    / ((4.0 * np.pi * EPSILON0) ** 3 * H * SIGMA * C**3)
)


def _as_positive(name: str, value: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and > 0")
    return arr


def _as_nonnegative(name: str, value: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and >= 0")
    return arr


def _validate_factors(charge_number: float, factor: float, name: str) -> None:
    if not (charge_number > 0 and np.isfinite(charge_number)):
        raise ValueError("charge_number must be finite and > 0")
    if not (factor > 0 and np.isfinite(factor)):
        raise ValueError(f"{name} must be finite and > 0")


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if np.ndim(value) == 0:
        return float(value)
    return value


def free_free_absorption_coefficient(
    frequency_hz: ArrayLike,
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
    ion_density_m3: ArrayLike,
    charge_number: float = 1.0,
    gaunt_factor: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""
    Free-free (inverse bremsstrahlung) absorption coefficient
    $\kappa_{f\!f}$ (m$^{-1}$), including stimulated emission.

    Herrera (2008), Eq. 5-50, p. 119, with the exponential factor
    corrected to $(1 - e^{-h\nu/(k_B T)})$ — see module *Documented
    ambiguity*:

    $$
    \kappa_{f\!f} \;=\; C_K\,
    \frac{n_e\, n_{\mathrm{ion}}\, z^{2}\, G}{T^{1/2}\, \nu^{3}}
    \left(1 - e^{-h\nu/(k_B T)}\right)
    $$

    $C_K$ is the Kramers prefactor
    $8\pi e^{6} / (3 m_e h c \sqrt{6\pi m_e k_B})$ in SI (printed CGS
    value $3.7\times10^{8}$).

    Parameters
    ----------
    frequency_hz : array_like of float
        Photon frequency $\nu$ (Hz, > 0).
    temperature_K : array_like of float
        Plasma temperature (K, > 0).
    electron_density_m3, ion_density_m3 : array_like of float
        $n_e$ and the summed ion density $\sum_j n_i^{j}$ (m$^{-3}$,
        >= 0) — for a multi-element plasma pass the total ion density
        from the Saha balance
        (`IonizationBalance.total_ion_density_m3`).
    charge_number : float, optional
        Effective ion charge $z$ (default 1, singly ionized two-stage
        plasma; enters as $z^{2}$ outside the ion sum, as printed).
    gaunt_factor : float, optional
        Dimensionless free-free Gaunt factor $G \approx 1$ (thesis
        p. 118).

    Returns
    -------
    float or ndarray
        $\kappa_{f\!f}$ in m$^{-1}$; broadcasts over the array inputs.

    Notes
    -----
    Low-frequency limit: $(1 - e^{-x}) \to x$ gives
    $\kappa_{f\!f} \sim \nu^{-2}$ (computed with expm1 to preserve
    this limit accurately).
    """
    nu = _as_positive("frequency_hz", frequency_hz)
    T = _as_positive("temperature_K", temperature_K)
    n_e = _as_nonnegative("electron_density_m3", electron_density_m3)
    n_i = _as_nonnegative("ion_density_m3", ion_density_m3)
    _validate_factors(charge_number, gaunt_factor, "gaunt_factor")

    x = H * nu / (KB * T)  # dimensionless photon energy h*nu/k_B*T
    kappa = (
        _KRAMERS_SI
        * n_e
        * n_i
        * charge_number**2
        * gaunt_factor
        # Stimulated-emission factor (1 - e^-x) — the typo-corrected
        # exponential of Eq. 5-50 (module docs); expm1 keeps the
        # low-frequency kappa ~ nu^-2 limit accurate.
        * (-np.expm1(-x))
        / (np.sqrt(T) * nu**3)  # Kramers T^(-1/2) nu^-3 scaling
    )
    return _scalar_or_array(kappa)


def free_bound_absorption_coefficient(
    frequency_hz: ArrayLike,
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
    ion_density_m3: ArrayLike,
    charge_number: float = 1.0,
    bound_free_correction: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""
    Free-bound (radiative recombination) absorption coefficient
    $\kappa_{f\!b}$ (m$^{-1}$), including stimulated emission.

    Herrera (2008), Eq. 5-51, p. 120, exactly as printed:

    $$
    \kappa_{f\!b} \;=\; C_K\,
    \frac{n_e\, n_{\mathrm{ion}}\, z^{2}\, \xi}{T^{1/2}\, \nu^{3}}\;
    e^{h\nu/(k_B T)} \left(1 - e^{-h\nu/(k_B T)}\right)^{2}
    $$

    (identically $(e^{x} - 1)(1 - e^{-x})$ with $x = h\nu/k_B T$, the
    form used internally via expm1 for numerical accuracy).

    Parameters
    ----------
    frequency_hz, temperature_K, electron_density_m3, ion_density_m3,
    charge_number
        As in `free_free_absorption_coefficient`.
    bound_free_correction : float, optional
        The dimensionless free-bound continuum correction factor
        $\xi_z$ of Eq. 5-51 ("takes into account the electron
        structure of the atom and usually assumes a value of unity",
        p. 120). Default 1.0.

    Returns
    -------
    float or ndarray
        $\kappa_{f\!b}$ in m$^{-1}$.

    Notes
    -----
    Kirchhoff image: $\kappa_{f\!b} B_\nu = C_K (2h/c^{2})\,
    n_e n_i z^{2} \xi / T^{1/2} \cdot (1 - e^{-h\nu/k_B T})$ — the
    $\xi$-term of Eq. 5-7, p. 101. Grows as $e^{x}$ at
    $h\nu \gg k_B T$ (model validity limit; see module notes).
    """
    nu = _as_positive("frequency_hz", frequency_hz)
    T = _as_positive("temperature_K", temperature_K)
    n_e = _as_nonnegative("electron_density_m3", electron_density_m3)
    n_i = _as_nonnegative("ion_density_m3", ion_density_m3)
    _validate_factors(charge_number, bound_free_correction, "bound_free_correction")

    x = H * nu / (KB * T)  # dimensionless photon energy h*nu/k_B*T
    with np.errstate(over="ignore"):  # far-UV overflow -> inf (documented)
        # (e^x - 1)(1 - e^-x): algebraically identical to the printed
        # e^x (1 - e^-x)^2 of Eq. 5-51; expm1 keeps the small-x limit
        # (~x^2) accurate.
        spectral_factor = np.expm1(x) * (-np.expm1(-x))
    kappa = (
        _KRAMERS_SI
        * n_e
        * n_i
        * charge_number**2
        * bound_free_correction
        * spectral_factor
        / (np.sqrt(T) * nu**3)
    )
    return _scalar_or_array(kappa)


def continuum_absorption_coefficient(
    frequency_hz: ArrayLike,
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
    ion_density_m3: ArrayLike,
    charge_number: float = 1.0,
    gaunt_factor: float = 1.0,
    bound_free_correction: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""
    Continuum part of the total absorption coefficient of Eq. 5-49,
    p. 119 (Herrera 2008): $\kappa_{f\!f} + \kappa_{f\!b}$
    (m$^{-1}$).

    The bound-bound member $\kappa_{bb}$ of Eq. 5-49 is line physics
    (`emission.line_absorption_coefficient`) and is added by the
    radiative-transfer layer (Phase 3), keeping continuum and line
    responsibilities separate.
    """
    kappa_ff = free_free_absorption_coefficient(
        frequency_hz,
        temperature_K,
        electron_density_m3,
        ion_density_m3,
        charge_number,
        gaunt_factor,
    )
    kappa_fb = free_bound_absorption_coefficient(
        frequency_hz,
        temperature_K,
        electron_density_m3,
        ion_density_m3,
        charge_number,
        bound_free_correction,
    )
    return _scalar_or_array(np.asarray(kappa_ff) + np.asarray(kappa_fb))


def continuum_emission_coefficient(
    frequency_hz: ArrayLike,
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
    ion_density_m3: ArrayLike,
    charge_number: float = 1.0,
    gaunt_factor: float = 1.0,
    bound_free_correction: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""
    LTE continuum emission coefficient $\epsilon_\nu$
    (W m$^{-3}$ Hz$^{-1}$ sr$^{-1}$).

    Kirchhoff's law applied to the continuum absorption — the LTE
    source term of the radiative transfer equation, Eq. 5-44, p. 117
    (Herrera 2008):

    $$
    \epsilon_\nu \;=\;
    (\kappa_{f\!f} + \kappa_{f\!b})\, B_\nu(T)
    $$

    Analytically this equals

    $$
    C_K\, \frac{2h}{c^{2}}\,
    \frac{n_e n_i z^{2}}{T^{1/2}}
    \left[G\, e^{-h\nu/k_B T} +
    \xi \left(1 - e^{-h\nu/k_B T}\right)\right]
    $$

    the bracket of the line-to-continuum expression Eq. 5-7, p. 101 —
    free-free emission decays as $e^{-h\nu/k_B T}$ while
    recombination emission saturates, as observed in LIBS continua
    (thesis Ch. 3, continuum = bremsstrahlung + radiative
    recombination).

    Parameters as in `continuum_absorption_coefficient`.
    """
    kappa = continuum_absorption_coefficient(
        frequency_hz,
        temperature_K,
        electron_density_m3,
        ion_density_m3,
        charge_number,
        gaunt_factor,
        bound_free_correction,
    )
    radiance = blackbody_spectral_radiance_hz(frequency_hz, temperature_K)
    # Kirchhoff's law: epsilon_nu = kappa_nu * B_nu(T), the LTE source
    # term of the RTE (Eq. 5-44).
    return _scalar_or_array(np.asarray(kappa) * np.asarray(radiance))


def planck_mean_absorption_coefficient(
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
    ion_density_m3: ArrayLike,
    charge_number: float = 1.0,
    gaunt_factor: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""
    Planck mean absorption coefficient $\kappa_{\mathrm{mean}}$
    (m$^{-1}$), free-free + free-bound only.

    Herrera (2008), Eq. 5-46, p. 118 (SI form; $\sigma$ is the
    Stefan-Boltzmann constant):

    $$
    \kappa_{\mathrm{mean}} \;=\;
    \sqrt{\tfrac{128}{27} k_B}\,
    \left(\frac{\pi}{m_e}\right)^{3/2}
    \frac{z^{2} e^{6} G}{h \sigma c^{3}}\,
    \frac{n_e}{T^{7/2}} \sum_j n_i^{j}
    $$

    Used with the rough-approximation radiance (Eq. 5-45, p. 118) to
    evaluate the radiation loss term $q$, Eq. 5-47, p. 118, of the
    plasma energy balance — a Phase 3/4 consumer. "The use of
    kappa_mean ... implies that the line structure of the spectrum is
    not included since bound-bound transitions were neglected"
    (p. 118).

    Parameters
    ----------
    temperature_K : array_like of float
        Plasma temperature (K, > 0).
    electron_density_m3, ion_density_m3 : array_like of float
        $n_e$ and summed ion density (m$^{-3}$, >= 0).
    charge_number, gaunt_factor : float, optional
        Effective charge $z$ and free-free Gaunt factor $G$
        (defaults 1).

    Returns
    -------
    float or ndarray
        $\kappa_{\mathrm{mean}}$ in m$^{-1}$.
    """
    T = _as_positive("temperature_K", temperature_K)
    n_e = _as_nonnegative("electron_density_m3", electron_density_m3)
    n_i = _as_nonnegative("ion_density_m3", ion_density_m3)
    _validate_factors(charge_number, gaunt_factor, "gaunt_factor")

    kappa = (
        _PLANCK_MEAN_SI
        * charge_number**2
        * gaunt_factor
        * n_e
        * n_i
        / T**3.5  # the n_e * sum(n_i) / T^(7/2) scaling of Eq. 5-46
    )
    return _scalar_or_array(kappa)

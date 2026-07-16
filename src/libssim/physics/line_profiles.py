r"""Doppler and Stark line widths and normalized Gaussian, Lorentzian
and Voigt line profiles (Phase 2).

Physical context (Herrera 2008)
-------------------------------
The normalized line profile $P_{lu}$ (symbols p. 22: "Normalized line
profile") shapes both the bound-bound absorption coefficient (Eq. 5-52,
p. 120) and the line emission coefficient (`emission`). In the MC-LIBS
model "the line profile is described by a Voigt function under the
assumption that the dominant line broadening mechanism is Stark
broadening which competes with Doppler broadening as the plasma evolves
in time" (p. 120).

**Stark-shifted Voigt function** (Eq. 5-53, p. 120) — the reduced Voigt
$H(a, \Lambda)/\sqrt{\pi}$, unit integral in $\Lambda$:

$$
P \;=\; \frac{a}{\pi\sqrt{\pi}} \int_{-\infty}^{+\infty}
\frac{e^{-y^{2}}\,\mathrm{d}y}{(\Lambda(\nu, t) - y)^{2} + a^{2}}
$$

**Reduced (Stark-shifted) detuning** (Eq. 5-54, p. 120):

$$
\Lambda(\nu, t) \;=\;
\frac{2\,[\nu - \nu_{0} + \Delta\nu_{\mathrm{shift}}]}{\Delta\nu_{D}}
$$

**Damping parameter** (Eq. 5-55, p. 121):

$$
a \;=\; \frac{\Delta\nu_{S}\,\sqrt{\ln 2}}{\Delta\nu_{D}}
$$

(Voigt evaluation in the thesis: modified Humlicek algorithm after
Schreier [150], p. 121; here `scipy.special.voigt_profile`, which
evaluates the same Faddeeva/Voigt function.)

Width formulas implemented (thesis Ch. 3 and Ch. 5)
---------------------------------------------------
**Doppler FWHM** (Eq. 3-1, p. 50; Gaussian, Maxwellian velocities):

$$
\Delta\lambda_{D} \;=\;
\lambda_{0}\sqrt{\frac{8\ln 2\, k_{B} T}{m c^{2}}}
$$

and its practical twin (Eq. 3-2, p. 51, $M$ in g/mol):

$$
\Delta\lambda_{D} \;=\; 7.16\times10^{-7}\,
\lambda_{0}\sqrt{T/M}
$$

**Quadratic Stark FWHM** (Lorentzian; Eq. 3-8, p. 53; SI, $n_e$ in
m$^{-3}$, $w$ the electron-impact half-width):

$$
\Delta\lambda_{S} \;=\; 2\times10^{-22}\, w\, n_{e}
\left[1 + 5.53\times10^{-6}\,\alpha\, n_{e}^{1/4}
\left(1 - 0.0068\, n_{e}^{1/6}\, T^{-1/2}\right)\right]
$$

equivalently Eq. 5-15, p. 106 in CGS ($n_e$ in cm$^{-3}$, Debye-sphere
form with $\beta = 0.75$ for neutral emitters baked into the 0.0068
above — verified numerically in the unit tests); reduced form without
ion broadening (Eq. 5-17, p. 106):
$\Delta\lambda_{S} = 2 w\, (n_{e}/10^{16}\,\mathrm{cm}^{-3})$.

**Quadratic Stark shift** (Eq. 3-9, p. 53; SI, $d/w$ the
shift-to-width ratio):

$$
\Delta\lambda_{\mathrm{shift}} \;=\; 10^{-22}\, n_{e}\, w
\left[\frac{d}{w} + 6.32\times10^{-6}\,\alpha\, n_{e}^{1/4}
\left(1 - 0.0068\, n_{e}^{1/6}\, T^{-1/2}\right)\right]
$$

**Voigt/Gaussian/Lorentzian FWHM relation** (Eq. 5-18, p. 107) — the
Whiting-type approximation CF-LIBS uses to extract the Stark width from
a fitted Voigt width:

$$
\Delta\lambda_{S} \;=\;
\Delta\lambda_{V} - \frac{\Delta\lambda_{G}^{2}}{\Delta\lambda_{V}}
$$

Out of scope here (documented boundary): van der Waals and resonance
broadening, Eqs. 3-5/3-6/3-7, pp. 52-53 — negligible against Stark +
Doppler under the thesis plasma conditions (p. 120) and deferred to
Phase 6 `physics/empirical_broadening.py`. Instrumental broadening is
Phase 4. Natural broadening is negligible (Ch. 3).

Documented ambiguity (development_rules.md)
-------------------------------------------
Eqs. 5-54 and 5-55 as printed are mutually inconsistent by
$\sqrt{\ln 2}$: with $\Delta\nu_D$ the Doppler FWHM, the damping
parameter of Eq. 5-55 pairs with the reduced detuning
$u = 2\sqrt{\ln 2}\,(\nu - \nu_0 + \Delta\nu_{\mathrm{shift}})/\Delta\nu_D$,
not with Eq. 5-54's $2(\cdots)/\Delta\nu_D$. As printed, a pure-Doppler
line would have FWHM $= \sqrt{\ln 2}\,\Delta\nu_D \approx
0.83\,\Delta\nu_D$, contradicting the definition of $\Delta\nu_D$. This
module therefore evaluates profiles with exact FWHM semantics via
`scipy.special.voigt_profile`
($\sigma = \mathrm{FWHM}_G / (2\sqrt{2\ln 2})$,
$\gamma = \mathrm{FWHM}_L / 2$), which is identical to Eq. 5-53 with
$a$ from Eq. 5-55 and the $\sqrt{\ln 2}$ restored in Eq. 5-54 — the
standard reduced-Voigt convention. The verbatim reduced form
(Eq. 5-53) is provided separately as `stark_shifted_voigt_reduced`;
the equivalence is proven in the unit tests to machine precision.

Shift sign convention: Eq. 5-54 *adds* the Stark shift to
$(\nu - \nu_0)$, so a positive $\Delta\nu_{\mathrm{shift}}$ peaks the
profile at $\nu_0 - \Delta\nu_{\mathrm{shift}}$ (lower frequency); in
wavelength space a positive $\Delta\lambda_{\mathrm{shift}}$ (Eq. 3-9)
peaks at $\lambda_0 + \Delta\lambda_{\mathrm{shift}}$. Both correspond
to the usual red shift ("the shift is usually towards the red", p. 53).

Normalization (Phase 2 acceptance criterion)
--------------------------------------------
All profile functions are analytically normalized to unit integral over
their own variable (`scipy.special.voigt_profile` is a true PDF); the
unit tests verify area = 1.0 to $\le 10^{-10}$ by adaptive quadrature.
Note for consumers: a finite sampling window misses the Lorentzian tail
mass $\sim \mathrm{FWHM}_L/(\pi W)$ for half-window $W$ — choose grids
accordingly rather than renormalizing numerically.

Units: strict SI (m, Hz, K, m^-3, kg). Profile values are per Hz (s) or
per m, matching the variable of evaluation.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eqs. 3-1, 3-2
pp. 50-51; Eqs. 3-8, 3-9 p. 53; Eqs. 5-15, 5-16, 5-17 p. 106; Eq. 5-18
p. 107; Eqs. 5-52, 5-53, 5-54 p. 120; Eq. 5-55 p. 121; symbols p. 22.

Griem, H.R. (1974). Spectral Line Broadening by Plasmas. (Quadratic
Stark formulas underlying Eqs. 3-8/3-9/5-15.)

Schreier, F. (1992). JQSRT 48, 743 (Humlicek-type Voigt evaluation,
thesis ref [150]).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import voigt_profile as _scipy_voigt

from ..core.constants import C, KB

# FWHM of a unit-variance Gaussian: 2*sqrt(2*ln 2).
_GAUSS_FWHM_PER_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _as_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not (value > 0.0 and np.isfinite(value)):
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _as_nonnegative_scalar(name: str, value: float) -> float:
    value = float(value)
    if not (value >= 0.0 and np.isfinite(value)):
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if np.ndim(value) == 0:
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Line widths
# ---------------------------------------------------------------------------
def doppler_fwhm_m(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
    emitter_mass_kg: float,
) -> float | NDArray[np.float64]:
    r"""
    Doppler (Gaussian) FWHM in wavelength, $\Delta\lambda_D$ (m).

    Herrera (2008), Eq. 3-1, p. 50:

    $$
    \Delta\lambda_D \;=\;
    \lambda_0 \sqrt{\frac{8 \ln 2\, k_B T}{m c^2}}
    $$

    valid for a Maxwellian velocity distribution (thermal equilibrium,
    p. 50).

    Parameters
    ----------
    wavelength_m : array_like of float
        Transition wavelength $\lambda_0$ (m, > 0).
    temperature_K : array_like of float
        Absolute temperature $T$ (K, > 0).
    emitter_mass_kg : float
        Mass $m$ of the radiating atom/ion (kg, > 0); for molar-mass
        input use `doppler_fwhm_practical_m` (Eq. 3-2).

    Returns
    -------
    float or ndarray
        FWHM in meters; broadcasts over wavelength and temperature.
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    T = np.asarray(temperature_K, dtype=np.float64)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("wavelength_m must be finite and > 0")
    if np.any(T <= 0) or not np.all(np.isfinite(T)):
        raise ValueError("temperature_K must be finite and > 0")
    m = _as_positive_scalar("emitter_mass_kg", emitter_mass_kg)

    # Eq. 3-1: thermal-to-light speed ratio sqrt(k_B*T/(m*c^2)) times the
    # Maxwellian FWHM factor sqrt(8*ln 2), scaled by lambda0.
    width = lam * np.sqrt(8.0 * np.log(2.0) * KB * T / (m * C**2))
    return _scalar_or_array(width)


def doppler_fwhm_practical_m(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
    molar_mass_g_mol: float,
) -> float | NDArray[np.float64]:
    r"""
    Doppler FWHM via the thesis practical formula, Eq. 3-2, p. 51:

    $$
    \Delta\lambda_D \;=\; 7.16\times10^{-7}\,\lambda_0\,\sqrt{T/M}
    $$

    with $M$ the atomic/molecular weight in g mol$^{-1}$. Numerically
    the rounded twin of Eq. 3-1 (agreement $\sim 3\times10^{-4}$,
    unit-test checked); prefer `doppler_fwhm_m` for exactness — this
    form exists for direct traceability to the thesis and to
    tabulated $M$.
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    T = np.asarray(temperature_K, dtype=np.float64)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("wavelength_m must be finite and > 0")
    if np.any(T <= 0) or not np.all(np.isfinite(T)):
        raise ValueError("temperature_K must be finite and > 0")
    M = _as_positive_scalar("molar_mass_g_mol", molar_mass_g_mol)
    return _scalar_or_array(7.16e-7 * lam * np.sqrt(T / M))


def _stark_ion_bracket(
    electron_density_m3: NDArray[np.float64],
    ion_broadening_alpha: float,
    temperature_K: ArrayLike | None,
    coefficient: float,
) -> NDArray[np.float64]:
    """Common ion-broadening bracket of Eqs. 3-8/3-9, p. 53 (SI units).

    The `coefficient` selects the form: 5.53e-6 is the *width* variant
    (Eq. 3-8, full bracket [1 + ion term], electron term normalized to 1)
    while 6.32e-6 is the *shift* variant (Eq. 3-9, bare ion term to be
    added to the tabulated d/w ratio). Both coefficients are the CGS
    Griem factors (1.75 and 2.0) times 10^-5.5 from the cm^-3 -> m^-3
    density rescaling (module docs).
    """
    if ion_broadening_alpha == 0.0:
        # Electron-impact only: width bracket collapses to 1, shift ion
        # term to 0 (the CF-LIBS working forms, Eq. 5-17 / d/w alone).
        return np.asarray(1.0 if coefficient == 5.53e-6 else 0.0)
    if temperature_K is None:
        raise ValueError(
            "temperature_K is required when ion_broadening_alpha > 0 "
            "(Debye correction term of Eqs. 3-8/3-9)"
        )
    T = np.asarray(temperature_K, dtype=np.float64)
    if np.any(T <= 0) or not np.all(np.isfinite(T)):
        raise ValueError("temperature_K must be finite and > 0")
    n_e = electron_density_m3
    # 0.0068*n_e^(1/6)/sqrt(T) is Eq. 5-15's Debye-sphere correction
    # (1 - 0.75*n_D^(-1/3)) rewritten in SI with beta = 0.75 folded in.
    debye_term = 1.0 - 0.0068 * n_e ** (1.0 / 6.0) / np.sqrt(T)
    ion_term = coefficient * ion_broadening_alpha * n_e**0.25 * debye_term
    return np.asarray(1.0 + ion_term if coefficient == 5.53e-6 else ion_term)


def stark_fwhm_m(
    electron_impact_halfwidth_m: float,
    electron_density_m3: ArrayLike,
    ion_broadening_alpha: float = 0.0,
    temperature_K: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    r"""
    Quadratic-Stark (Lorentzian) FWHM in wavelength,
    $\Delta\lambda_{\mathrm{Stark}}$ (m).

    Herrera (2008), Eq. 3-8, p. 53 (SI form; $n_e$ in m$^{-3}$, $w$
    in m):

    $$
    \Delta\lambda_S \;=\; 2\times10^{-22}\, w\, n_e
    \left[1 + 5.53\times10^{-6}\,\alpha\, n_e^{1/4}
    \left(1 - 0.0068\, n_e^{1/6}\, T^{-1/2}\right)\right]
    $$

    With $\alpha = 0$ (electron impact only) this reduces to the
    CF-LIBS working formula (Eq. 5-17, p. 106),
    $\Delta\lambda_S = 2 w\,(n_e / 10^{16}\,\mathrm{cm}^{-3})$; the
    full bracket is the CGS Eq. 5-15, p. 106 with the neutral-emitter
    Debye coefficient $\beta = 0.75$ folded into 0.0068 (both
    correspondences verified numerically in the unit tests). Stark
    profiles are Lorentzian (Ch. 3, p. 53; "Lorentzian in nature",
    p. 106).

    Parameters
    ----------
    electron_impact_halfwidth_m : float
        Electron-impact half-width $w$ (m, >= 0) at the reference
        density $10^{22}$ m$^{-3}$ ($= 10^{16}$ cm$^{-3}$), tabulated
        per line (thesis refs [131, 139]; `Transition.stark_width`
        when populated).
    electron_density_m3 : array_like of float
        Electron density $n_e$ (m$^{-3}$, >= 0).
    ion_broadening_alpha : float, optional
        Dimensionless ion-broadening parameter $\alpha$ (>= 0,
        default 0 — "normally neglected due to the negligible
        contribution of ion-broadening under typical LIBS
        conditions", p. 106).
    temperature_K : array_like of float, optional
        Required only when `ion_broadening_alpha` > 0 (Debye term).

    Returns
    -------
    float or ndarray
        Lorentzian FWHM in meters.
    """
    w = _as_nonnegative_scalar(
        "electron_impact_halfwidth_m", electron_impact_halfwidth_m
    )
    alpha = _as_nonnegative_scalar("ion_broadening_alpha", ion_broadening_alpha)
    n_e = np.asarray(electron_density_m3, dtype=np.float64)
    if np.any(n_e < 0) or not np.all(np.isfinite(n_e)):
        raise ValueError("electron_density_m3 must be finite and >= 0")

    bracket = _stark_ion_bracket(n_e, alpha, temperature_K, 5.53e-6)
    # FWHM = 2w at the tabulation reference density 1e22 m^-3 (1e16 cm^-3),
    # scaled linearly in n_e (Eq. 3-8 / Eq. 5-17).
    return _scalar_or_array(2.0e-22 * w * n_e * bracket)


def stark_shift_m(
    electron_impact_halfwidth_m: float,
    shift_to_width_ratio: float,
    electron_density_m3: ArrayLike,
    ion_broadening_alpha: float = 0.0,
    temperature_K: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    r"""
    Quadratic-Stark line shift in wavelength,
    $\Delta\lambda_{\mathrm{shift}}$ (m).

    Herrera (2008), Eq. 3-9, p. 53 (SI form):

    $$
    \Delta\lambda_{\mathrm{shift}} \;=\; 10^{-22}\, n_e\, w
    \left[\frac{d}{w} + 6.32\times10^{-6}\,\alpha\, n_e^{1/4}
    \left(1 - 0.0068\, n_e^{1/6}\, T^{-1/2}\right)\right]
    $$

    where $d/w$ is the tabulated shift-to-width ratio (dimensionless,
    signed). Positive values are red shifts ("the shift is usually
    towards the red and is the same for ions or electrons", p. 53).

    Sign caveat (documented ambiguity): the sign joining the ion term
    to $d/w$ is typeset ambiguously in the thesis; the Griem
    convention with the ion contribution *added* to $d/w$ (as
    extracted) is implemented. With $\alpha = 0$ the result is exactly
    $(d/w)\, w\, n_e / 10^{22}$.

    Returns the shift in meters (sign follows $d/w$); feed it to the
    wavelength-domain profile, or convert with
    `fwhm_wavelength_to_frequency` (magnitude) for Eq. 5-54's
    $\Delta\nu_{\mathrm{shift}}$.
    """
    w = _as_nonnegative_scalar(
        "electron_impact_halfwidth_m", electron_impact_halfwidth_m
    )
    ratio = float(shift_to_width_ratio)
    if not np.isfinite(ratio):
        raise ValueError("shift_to_width_ratio must be finite")
    alpha = _as_nonnegative_scalar("ion_broadening_alpha", ion_broadening_alpha)
    n_e = np.asarray(electron_density_m3, dtype=np.float64)
    if np.any(n_e < 0) or not np.all(np.isfinite(n_e)):
        raise ValueError("electron_density_m3 must be finite and >= 0")

    ion_term = _stark_ion_bracket(n_e, alpha, temperature_K, 6.32e-6)
    # shift = w * (n_e / 1e22 m^-3) * [d/w + ion term]; sign follows the
    # tabulated d/w (Eq. 3-9).
    return _scalar_or_array(1.0e-22 * w * n_e * (ratio + ion_term))


def fwhm_wavelength_to_frequency(
    fwhm_m: ArrayLike,
    wavelength_m: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Convert a narrow-line width from wavelength (m) to frequency (Hz):
    $\Delta\nu = c\,\Delta\lambda / \lambda_0^2$ (first-order
    Jacobian; relative error $\sim \Delta\lambda/\lambda_0 < 10^{-4}$
    for LIBS line widths). Bridges the Ch. 3 wavelength-domain widths
    (Eqs. 3-1, 3-8) to the frequency-domain Voigt of Eqs. 5-53..5-55.
    """
    dlam = np.asarray(fwhm_m, dtype=np.float64)
    lam = np.asarray(wavelength_m, dtype=np.float64)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("wavelength_m must be finite and > 0")
    if np.any(dlam < 0) or not np.all(np.isfinite(dlam)):
        raise ValueError("fwhm_m must be finite and >= 0")
    return _scalar_or_array(C * dlam / lam**2)


def fwhm_frequency_to_wavelength(
    fwhm_hz: ArrayLike,
    wavelength_m: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""Inverse of `fwhm_wavelength_to_frequency`:
    $\Delta\lambda = \lambda_0^2\, \Delta\nu / c$."""
    dnu = np.asarray(fwhm_hz, dtype=np.float64)
    lam = np.asarray(wavelength_m, dtype=np.float64)
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        raise ValueError("wavelength_m must be finite and > 0")
    if np.any(dnu < 0) or not np.all(np.isfinite(dnu)):
        raise ValueError("fwhm_hz must be finite and >= 0")
    return _scalar_or_array(lam**2 * dnu / C)


# ---------------------------------------------------------------------------
# Normalized profiles (unit integral over their own variable)
# ---------------------------------------------------------------------------
def _normalized_voigt(
    detuning: NDArray[np.float64],
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
) -> NDArray[np.float64]:
    """scipy Voigt PDF with FWHM parameterization; needs one width > 0."""
    if gaussian_fwhm == 0.0 and lorentzian_fwhm == 0.0:
        raise ValueError(
            "at least one of gaussian_fwhm and lorentzian_fwhm must be > 0"
        )
    sigma = gaussian_fwhm / _GAUSS_FWHM_PER_SIGMA  # Gaussian std dev
    gamma = lorentzian_fwhm / 2.0                  # Lorentzian HWHM
    # scipy handles the pure limits itself: sigma=0 -> Cauchy PDF,
    # gamma=0 -> normal PDF; both stay exactly unit-area.
    return _scipy_voigt(detuning, sigma, gamma)


def voigt_profile_hz(
    frequency_hz: ArrayLike,
    center_frequency_hz: float,
    gaussian_fwhm_hz: float,
    lorentzian_fwhm_hz: float,
    stark_shift_hz: float = 0.0,
) -> float | NDArray[np.float64]:
    r"""
    Normalized (unit-area) Stark-shifted Voigt profile $P(\nu)$ in
    Hz$^{-1}$.

    Physical form of Herrera (2008) Eqs. 5-53/5-54/5-55 (pp. 120-121):
    the reduced Voigt $H(a, u)/\sqrt{\pi}$ with damping parameter
    $a = \sqrt{\ln 2}\,\Delta\nu_L/\Delta\nu_G$ (Eq. 5-55) and reduced
    detuning
    $u = 2\sqrt{\ln 2}\,(\nu - \nu_0 + \Delta\nu_{\mathrm{shift}})/\Delta\nu_G$
    (Eq. 5-54 with the $\sqrt{\ln 2}$ restored — see the module's
    *Documented ambiguity*), mapped back to frequency. Evaluated with
    `scipy.special.voigt_profile` (Faddeeva function; thesis used a
    Humlicek/Schreier algorithm, p. 121 — same function, different
    numerics).

    This is the $P_{lu}$ of the bound-bound absorption coefficient
    (Eq. 5-52, p. 120) and of the line emission coefficient
    (`emission`). Analytically normalized:
    $\int P(\nu)\,\mathrm{d}\nu = 1$ (Phase 2 acceptance criterion,
    unit-tested to $\le 10^{-10}$).

    Parameters
    ----------
    frequency_hz : array_like of float
        Evaluation frequencies $\nu$ (Hz).
    center_frequency_hz : float
        Unperturbed line-center frequency $\nu_0$ (Hz, > 0), e.g.
        `emission.transition_frequency_hz`.
    gaussian_fwhm_hz : float
        Doppler (Gaussian) FWHM $\Delta\nu_D$ (Hz, >= 0) — Eq. 3-1 via
        `doppler_fwhm_m` + `fwhm_wavelength_to_frequency`.
        Instrumental Gaussian broadening may be folded in quadrature
        (Phase 4).
    lorentzian_fwhm_hz : float
        Stark (Lorentzian) FWHM $\Delta\nu_S$ (Hz, >= 0) — Eq. 3-8 via
        `stark_fwhm_m` + conversion. At least one width must be > 0.
    stark_shift_hz : float, optional
        $\Delta\nu_{\mathrm{shift}}$ of Eq. 5-54 (Hz, signed;
        default 0). Positive values peak the profile at
        $\nu_0 - \Delta\nu_{\mathrm{shift}}$ (red), matching
        Eq. 5-54's "+" inside the bracket.

    Returns
    -------
    float or ndarray
        $P(\nu)$ in Hz$^{-1}$, same shape as `frequency_hz`.
    """
    nu = np.asarray(frequency_hz, dtype=np.float64)
    nu0 = _as_positive_scalar("center_frequency_hz", center_frequency_hz)
    dnu_G = _as_nonnegative_scalar("gaussian_fwhm_hz", gaussian_fwhm_hz)
    dnu_L = _as_nonnegative_scalar("lorentzian_fwhm_hz", lorentzian_fwhm_hz)
    shift = float(stark_shift_hz)
    if not np.isfinite(shift):
        raise ValueError("stark_shift_hz must be finite")

    detuning = nu - nu0 + shift  # Eq. 5-54 sign convention
    return _scalar_or_array(_normalized_voigt(detuning, dnu_G, dnu_L))


def voigt_profile_wavelength_m(
    wavelength_m: ArrayLike,
    center_wavelength_m: float,
    gaussian_fwhm_m: float,
    lorentzian_fwhm_m: float,
    stark_shift_m: float = 0.0,
) -> float | NDArray[np.float64]:
    r"""
    Normalized Stark-shifted Voigt profile $P(\lambda)$ in m$^{-1}$.

    Wavelength-domain counterpart of `voigt_profile_hz` (narrow-line
    treatment: the profile is normalized in $\lambda$ directly; widths
    from Eqs. 3-1/3-8). A positive `stark_shift_m` (Eq. 3-9 red shift)
    peaks the profile at $\lambda_0 + \Delta\lambda_{\mathrm{shift}}$
    — note the sign is opposite to the frequency-domain convention
    because $\lambda$ and $\nu$ run oppositely.
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    lam0 = _as_positive_scalar("center_wavelength_m", center_wavelength_m)
    dlam_G = _as_nonnegative_scalar("gaussian_fwhm_m", gaussian_fwhm_m)
    dlam_L = _as_nonnegative_scalar("lorentzian_fwhm_m", lorentzian_fwhm_m)
    shift = float(stark_shift_m)
    if not np.isfinite(shift):
        raise ValueError("stark_shift_m must be finite")

    detuning = lam - lam0 - shift  # red shift -> larger wavelength
    return _scalar_or_array(_normalized_voigt(detuning, dlam_G, dlam_L))


def gaussian_profile_hz(
    frequency_hz: ArrayLike,
    center_frequency_hz: float,
    fwhm_hz: float,
) -> float | NDArray[np.float64]:
    r"""
    Normalized Gaussian profile (Hz$^{-1}$) — the pure-Doppler limit
    of the Voigt (Eq. 3-1 broadening alone, p. 50: "the intensity
    distribution has a Gaussian profile").
    """
    fwhm = _as_positive_scalar("fwhm_hz", fwhm_hz)
    return voigt_profile_hz(frequency_hz, center_frequency_hz, fwhm, 0.0)


def lorentzian_profile_hz(
    frequency_hz: ArrayLike,
    center_frequency_hz: float,
    fwhm_hz: float,
) -> float | NDArray[np.float64]:
    r"""
    Normalized Lorentzian profile (Hz$^{-1}$) — the pure-Stark limit
    of the Voigt ("the Stark width ... is Lorentzian in nature",
    p. 106).
    """
    fwhm = _as_positive_scalar("fwhm_hz", fwhm_hz)
    return voigt_profile_hz(frequency_hz, center_frequency_hz, 0.0, fwhm)


def stark_shifted_voigt_reduced(
    damping_a: ArrayLike,
    reduced_detuning: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Verbatim reduced Voigt of Herrera (2008), Eq. 5-53, p. 120:

    $$
    P(a, \Lambda) \;=\; \frac{a}{\pi\sqrt{\pi}}
    \int_{-\infty}^{+\infty}
    \frac{e^{-y^{2}}\,\mathrm{d}y}{(\Lambda - y)^{2} + a^{2}}
    \;=\; \frac{H(a, \Lambda)}{\sqrt{\pi}}
    $$

    with unit integral over the reduced detuning $\Lambda$. Provided
    for direct traceability to the thesis; equals
    `scipy.special.voigt_profile(Lambda, 1/sqrt(2), a)` exactly (unit
    tested). Use `damping_parameter` (Eq. 5-55) for $a$, and note the
    $\Lambda$ convention discussion in the module docstring.
    """
    a = np.asarray(damping_a, dtype=np.float64)
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError("damping_a must be finite and >= 0")
    u = np.asarray(reduced_detuning, dtype=np.float64)
    # With sigma = 1/sqrt(2), scipy's Voigt PDF equals H(a, u)/sqrt(pi)
    # identically — Eq. 5-53 without any rescaling (unit-tested against
    # direct numerical evaluation of the defining integral).
    return _scalar_or_array(_scipy_voigt(u, 1.0 / np.sqrt(2.0), a))


def damping_parameter(
    lorentzian_fwhm: ArrayLike,
    gaussian_fwhm: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Voigt damping parameter of Herrera (2008), Eq. 5-55, p. 121:

    $$
    a \;=\; \frac{\Delta\nu_S\,\sqrt{\ln 2}}{\Delta\nu_D}
    $$

    with both FWHM in the same units (Hz or m). Equal to
    $\gamma/(\sigma\sqrt{2})$ of the scipy parameterization — the
    identity behind the profile-equivalence unit test.
    """
    dL = np.asarray(lorentzian_fwhm, dtype=np.float64)
    dG = np.asarray(gaussian_fwhm, dtype=np.float64)
    if np.any(dL < 0) or not np.all(np.isfinite(dL)):
        raise ValueError("lorentzian_fwhm must be finite and >= 0")
    if np.any(dG <= 0) or not np.all(np.isfinite(dG)):
        raise ValueError("gaussian_fwhm must be finite and > 0")
    return _scalar_or_array(dL * np.sqrt(np.log(2.0)) / dG)


# ---------------------------------------------------------------------------
# FWHM algebra (CF-LIBS width extraction)
# ---------------------------------------------------------------------------
def lorentzian_fwhm_from_voigt(
    voigt_fwhm: ArrayLike,
    gaussian_fwhm: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Stark (Lorentzian) FWHM from a fitted Voigt FWHM, Herrera (2008),
    Eq. 5-18, p. 107:

    $$
    \Delta\lambda_{\mathrm{Stark}} \;=\;
    \Delta\lambda_V - \frac{\Delta\lambda_G^2}{\Delta\lambda_V}
    $$

    used by CF-LIBS to deduct the Gaussian (Doppler + instrumental)
    contribution before applying Eq. 5-17 for $n_e$. Whiting-type
    approximation, accurate to ~1-2% (unit-tested against the exact
    numerical Voigt FWHM); requires
    $\Delta\lambda_V \ge \Delta\lambda_G$.
    """
    dV = np.asarray(voigt_fwhm, dtype=np.float64)
    dG = np.asarray(gaussian_fwhm, dtype=np.float64)
    if np.any(dV <= 0) or not np.all(np.isfinite(dV)):
        raise ValueError("voigt_fwhm must be finite and > 0")
    if np.any(dG < 0) or not np.all(np.isfinite(dG)):
        raise ValueError("gaussian_fwhm must be finite and >= 0")
    if np.any(dG > dV):
        raise ValueError(
            "gaussian_fwhm cannot exceed voigt_fwhm (Eq. 5-18 domain)"
        )
    # Eq. 5-18: deduct the Gaussian contribution from the fitted Voigt
    # width; exact in both pure limits (dG=0 -> dV; dG=dV -> 0).
    return _scalar_or_array(dV - dG**2 / dV)


def voigt_fwhm_estimate(
    gaussian_fwhm: ArrayLike,
    lorentzian_fwhm: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Approximate Voigt FWHM — the inverse of Eq. 5-18, p. 107
    (Herrera 2008):

    $$
    \Delta\lambda_V \;=\; \frac{\Delta\lambda_L}{2} +
    \sqrt{\frac{\Delta\lambda_L^2}{4} + \Delta\lambda_G^2}
    $$

    Same Whiting-type accuracy (~1-2%) as
    `lorentzian_fwhm_from_voigt`; exact in both pure limits.
    """
    dG = np.asarray(gaussian_fwhm, dtype=np.float64)
    dL = np.asarray(lorentzian_fwhm, dtype=np.float64)
    if np.any(dG < 0) or not np.all(np.isfinite(dG)):
        raise ValueError("gaussian_fwhm must be finite and >= 0")
    if np.any(dL < 0) or not np.all(np.isfinite(dL)):
        raise ValueError("lorentzian_fwhm must be finite and >= 0")
    # Solving Eq. 5-18 for the Voigt width (quadratic in dV, positive root).
    return _scalar_or_array(dL / 2.0 + np.sqrt(dL**2 / 4.0 + dG**2))

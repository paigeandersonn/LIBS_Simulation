# Methods

This page explains the physical model implemented in `libssim`: what is
computed at each stage of the forward pipeline and the governing
equations, so the simulation can be understood — and reproduced —
without reading the source. Equation tags "(H x-y, p. z)" cite the
source dissertation:

> Herrera, K.K. (2008). *From Sample to Signal in Laser-Induced
> Breakdown Spectroscopy: An Experimental Assessment of Existing
> Algorithms and Theoretical Modeling Approaches.* PhD Dissertation,
> University of Florida.

Boxes marked **Deviation** are documented departures from, or
corrections to, the dissertation — they affect reproducibility and are
stated explicitly. Full per-function documentation lives in the
[API Reference](reference.md).

---

## Overview of the forward model

`libssim` simulates the light a gated spectrometer records from a
laser-induced plasma. It implements the *forward* (simulation)
direction of the CF-LIBS / MC-LIBS formalism of Herrera (2008) as a
fixed pipeline:

**plasma state → LTE population physics → per-zone
emissivity/absorption → line-of-sight radiative transfer → temporal
gate integration → instrument response → comparison metrics.**

The model assumes local thermodynamic equilibrium (LTE), two ionization
stages per element (I ↔ II), and a spherically symmetric plasma.

Two immutable containers carry data through the pipeline
([core](reference/core.md)):

- `PlasmaState` — one locally uniform plasma at one instant:
  temperature $T$, electron density $n_e$, total density, radius, time,
  and elemental composition fractions (normalized to sum to 1).
- `Spectrum` — wavelength/intensity arrays plus a metadata dictionary
  that accumulates full provenance (plasma history, gate settings,
  instrument parameters, preprocessing steps) as the spectrum moves
  through the chain.

All quantities are strict SI internally; every physics routine cites
the dissertation equation it implements, and each is unit-tested
against analytic identities (see
[Numerical implementation and verification](#numerical-implementation-and-verification)).

## Atomic and spectroscopic data

Transition data (wavelength, transition probability $A_{ki}$, level
energies, statistical weights) and first ionization energies are
transcribed from the NIST Atomic Spectra Database (Kramida et al.); a
defensive CSV parser for ASD exports
([atomic.parsers](reference/atomic.md)) validates every critical field.
Air wavelengths are used as stored — no air/vacuum correction is
applied in the physics layer (a narrow-line, instrument-layer concern).

Stark broadening inputs are the electron-impact half-width $w$,
tabulated at the reference density
$n_e = 10^{22}\ \mathrm{m^{-3}} = 10^{16}\ \mathrm{cm^{-3}}$, and the
shift-to-width ratio $d/w$ (Griem; thesis refs [131, 139]). Entries
transcribed with reduced confidence are flagged in the data layer
([validation.atomic_data](reference/validation.md)) so they can be
re-verified before publication-grade use.

## Equilibrium plasma state (LTE)

A single temperature $T = T_{\mathrm{exc}} = T_{\mathrm{ion}}$ governs
all populations ([physics.boltzmann](reference/physics.md),
[physics.saha](reference/physics.md)).

**Boltzmann level populations** (H 5-1, p. 98): the fraction of a
species in bound level $k$ is

$$
n_k^{s} \;=\; n_{\mathrm{tot}}^{s}\,
\frac{g_k\, e^{-E_k/(k_B T)}}{U^{s}(T)}
$$

**Saha ionization balance** (H 5-2, p. 98 / H D-1, p. 274): each
element $j$ splits between neutral and singly-ionized stages according
to the Saha function

$$
s^{j}(T) \;=\; \frac{n_i^{j}\, n_e}{n_a^{j}} \;=\;
2\,\frac{U^{II}(T)}{U^{I}(T)}
\left(\frac{2\pi m_e k_B T}{h^{2}}\right)^{3/2}
\exp\!\left(-\frac{\chi_j - \Delta\chi}{k_B T}\right)
$$

The system is closed by mass conservation (H D-2) and charge
equilibrium (H D-3), giving the explicit stage densities (H D-10)
$n_i^{j} = n^{j} s^{j}/(s^{j} + n_e)$ and a self-consistent electron
density from the implicit equation (H D-9)

$$
n_e \;=\; \sum_j \frac{n^{j}\, s^{j}(T)}{s^{j}(T) + n_e}
$$

solved by bracketed root finding (Brent's method, rtol $10^{-14}$; the
root is unique and bracketed because the residual is strictly
increasing). Computing $n_a^{j} = n^{j} - n_i^{j}$ rather than
evaluating both stages independently makes mass conservation exact in
floating point.

!!! note "Deviation — ionization-potential lowering"
    The dissertation defines $\Delta\chi$ (pp. 105, 274) but gives no
    formula. The standard Debye-shielding estimate
    $\Delta\chi_z = z e^{2}/(4\pi\varepsilon_0 \lambda_D)$ (Griem 1964;
    Drawin & Felenbok 1965) is provided, with default
    $\Delta\chi = 0$ — its magnitude (0.05–0.2 eV under LIBS
    conditions) is small against ionization energies of several eV.

**LTE validity diagnostics** are computed alongside every balance: the
McWhirter criterion (H 5-3, p. 99), with $\Delta E$ the largest
transition gap in eV,

$$
n_e \;\gg\; 1.6\times10^{12}\; T^{1/2} (\Delta E)^{3}
\quad [\mathrm{cm^{-3}}]
$$

and the Debye-sphere particle count (H 5-16, p. 106),
$n_D = 1.72\times10^{9}\, T_{\mathrm{eV}}^{3/2}/n_e^{1/2}$
($n_e$ in cm$^{-3}$). Both are implemented in SI from first principles
and validated against the printed CGS coefficients.

## Partition functions

The internal partition function normalizes every population expression
([physics.partition](reference/physics.md)). Its definition (H p. 260)
is the direct Boltzmann sum over bound levels:

$$
U(T) \;=\; \sum_i g_i\, e^{-E_i/(k_B T)}
$$

Three composable evaluation strategies are provided, and the one used
per element is recorded:

1. **Direct summation** over transcribed NIST level lists (primary;
   Na I complete through $n = 6$, Al I through 6s).
2. **Linear interpolation** on tabulated $(T, U)$ grids (the Drawin &
   Felenbok / Halenka style of the dissertation's own sources).
3. **Irwin (1981) polynomial**,
   $\ln U = \sum_k a_k (\ln T)^{k}$, as an out-of-grid fallback.

Evaluation outside all validity ranges raises an error rather than
extrapolating — partition functions grow steeply at high $T$ and
silent extrapolation is physically unsafe. The low-temperature limit
$U(T) \to g_0$ is a verification identity.

Near the ionization limit the level sum is cutoff-dependent
(Inglis–Teller microfield dissolution of high-$n$ states). An optional
hydrogenic tail augmentation ($2n^{2}$ weights at
$\chi - R_y/n^{2}$) exists but is deliberately not applied by default,
because the physically meaningful $n_{\max}$ depends on the plasma
density being simulated; the truncation deficit of the shipped lists
is a few percent at 15 kK.

## Line emission and absorption

Per transition ([physics.emission](reference/physics.md)), the
spontaneous photon rate (H 5-8, pp. 103–104) is
$I_{ki} = n_k A_{ki}$, and the spectral emission coefficient for
isotropic emission is

$$
\epsilon_\nu \;=\; \frac{h\nu_0}{4\pi}\, n_k\, A_{ki}\, P(\nu)
$$

with $P(\nu)$ the unit-area line profile. The bound-bound absorption
coefficient with its LTE stimulated-emission correction (H 5-52,
p. 120) is

$$
\kappa_{bb} \;=\; \frac{h\nu_0}{c}\, B_{lu}\, n_l\, P(\nu)
\left(1 - e^{-h\nu_0/(k_B T)}\right)
$$

!!! note "Deviation — Einstein coefficient conversion"
    The dissertation takes $B_{lu}$ as data; `libssim` derives it from
    the tabulated $A_{ul}$ via the standard detailed-balance relation
    (Mihalas 1978),

    $$
    B_{lu} \;=\; \frac{g_u}{g_l}\,
    \frac{c^{3}}{8\pi h \nu_0^{3}}\, A_{ul}
    $$

    The convention is pinned by enforcing the Kirchhoff identity
    $\epsilon_\nu/\kappa_{bb} = B_\nu(T)$ numerically in the test
    suite.

A narrow-line approximation evaluates $h\nu$ and the stimulated factor
at the line center $\nu_0$ (relative error $< 10^{-4}$ across a LIBS
line). Planck functions $B_\nu$ and $B_\lambda$ (H 3-11, p. 55) are
implemented expm1-stably in both the Wien and Rayleigh–Jeans limits.

## Line profiles and broadening

Lines are Voigt profiles — Doppler (Gaussian) $\otimes$ Stark
(Lorentzian) — evaluated with the Faddeeva function
(`scipy.special.voigt_profile`; the dissertation used a
Humlíček/Schreier algorithm for the same function)
([physics.line_profiles](reference/physics.md)).

**Doppler FWHM** (H 3-1, p. 50), for a Maxwellian velocity
distribution:

$$
\Delta\lambda_D \;=\;
\lambda_0 \sqrt{\frac{8 \ln 2\, k_B T}{m c^{2}}}
$$

**Quadratic Stark FWHM** (H 3-8, p. 53; SI form, $n_e$ in m$^{-3}$):

$$
\Delta\lambda_S \;=\; 2\times10^{-22}\, w\, n_e
\left[1 + 5.53\times10^{-6}\,\alpha\, n_e^{1/4}
\left(1 - 0.0068\, n_e^{1/6}\, T^{-1/2}\right)\right]
$$

linear in $n_e$ with the tabulated half-width $w$; the optional
ion-broadening bracket (parameter $\alpha$, with its Debye correction)
defaults to $\alpha = 0$, negligible under LIBS conditions (H p. 106).
The quadratic Stark **shift** (H 3-9, p. 53) is implemented with the
red-shift sign convention.

The dissertation's reduced Voigt (H 5-53, p. 120) with damping
parameter $a = \sqrt{\ln 2}\,\Delta\nu_S/\Delta\nu_D$ (H 5-55, p. 121)
is

$$
P(a, \Lambda) \;=\; \frac{a}{\pi\sqrt{\pi}}
\int_{-\infty}^{+\infty}
\frac{e^{-y^{2}}\,\mathrm{d}y}{(\Lambda - y)^{2} + a^{2}}
$$

!!! note "Deviations — Stark shift sign and reduced-Voigt consistency"
    The $\pm$ joining the ion term to $d/w$ in H 3-9 is typeset
    ambiguously; the Griem convention (ion term added to $d/w$) is
    implemented. Separately, the printed pair H 5-54/5-55 is mutually
    inconsistent by $\sqrt{\ln 2}$: as printed, a pure-Doppler line
    would not have FWHM $\Delta\nu_D$. Profiles are therefore
    evaluated with exact FWHM semantics (equivalent to H 5-53 with the
    $\sqrt{\ln 2}$ restored in H 5-54); the equivalence to the
    verbatim reduced form is proven to machine precision in the tests.

All profiles are analytically normalized to unit area
($\int P\,\mathrm{d}\nu = 1$, tested to $\le 10^{-10}$). The
Whiting-type FWHM algebra (H 5-18, p. 107),
$\Delta\lambda_S = \Delta\lambda_V - \Delta\lambda_G^{2}/\Delta\lambda_V$,
is provided for Stark-width extraction in the validation workflow. Van
der Waals, resonance, and natural broadening are explicitly out of
scope (negligible against Stark + Doppler under these conditions).

## Continuum radiation

The continuum is the hydrogen-like Kramers–Unsöld model
([physics.continuum](reference/physics.md)). The total absorption
coefficient entering the transfer equation decomposes as (H 5-49,
p. 119)

$$
\kappa'_\nu \;=\; \kappa_{f\!f} + \kappa_{f\!b} + \kappa_{bb}
$$

**Free-free** (inverse bremsstrahlung; H 5-50, p. 119):

$$
\kappa_{f\!f} \;=\;
\frac{8\pi e^{6}}{3 m_e h c \sqrt{6\pi m_e k_B}}\;
\frac{n_e}{T^{1/2}}\, z^{2}\, \frac{G}{\nu^{3}}
\left(1 - e^{-h\nu/(k_B T)}\right) \sum_j n_i^{j}
$$

**Free-bound** (radiative recombination; H 5-51, p. 120): the same
prefactor with
$(\xi_z/\nu^{3})\, e^{h\nu/(k_B T)}
\left(1 - e^{-h\nu/(k_B T)}\right)^{2}$
in place of the free-free frequency factor. The Gaunt factor $G$ and
Biberman-like factor $\xi$ default to 1 (H p. 118) and are
caller-adjustable. Continuum emission follows Kirchhoff's law,
$\epsilon_\nu = \kappa_\nu B_\nu(T)$, which reproduces the bracket
structure of the line-to-continuum expression (H 5-7, p. 101) — used
as a consistency check in the tests. The Planck-mean ff+fb absorption
coefficient (H 5-46, p. 118) is available for the radiative
energy-loss term.

!!! note "Deviation — typographical error in H 5-50"
    Eq. 5-50 as printed carries the factor $e^{h\nu/k_B T - 1}$, which
    grows exponentially with frequency — unphysical for an absorption
    coefficient. It is corrected to
    $\left(1 - e^{-h\nu/(k_B T)}\right)$, established three ways: the
    printed prefactor matches the standard thermally-averaged
    free-free absorption of Rybicki & Lightman (whose CGS coefficient
    is exactly the printed $3.7\times10^{8}$), Kirchhoff's law then
    reproduces the free-free term of H 5-7, and the corrected form is
    bounded.

The thesis formulas are Gaussian-CGS; the SI implementation replaces
$e^{6} \to e^{6}/(4\pi\varepsilon_0)^{3}$ and is validated against the
printed $3.7\times10^{8}$ CGS coefficient.

## Plasma geometry and radiative transfer

The plasma is a sphere with radially varying conditions, discretized
as a spherical "onion": $N$ concentric homogeneous shells, each a
locally uniform `PlasmaState`
([transport.geometry](reference/transport.md)). The dissertation's
parabolic initial profiles (H 5-36/5-37, p. 116),

$$
T(r, 0) = T_0 \left(1 - k_1 r^{2}\right),
\qquad
n_j(r, 0) = n_0^{j} \left(1 - k_2 r^{2}\right)
$$

are sampled per shell with per-zone Saha closure; a single zone
recovers the uniform sphere. Per zone, an adapter layer
([transport.emissivity](reference/transport.md)) evaluates the Phase 2
physics once — Saha stage densities, Boltzmann populations, Voigt
profiles, line and continuum coefficients — and returns
$(\epsilon_\nu, \kappa'_\nu)$ on a fixed wavelength grid. Lines
without tabulated Stark data receive Doppler-only (Gaussian) profiles;
transitions of elements absent from a zone contribute nothing there.

Radiative transfer solves the stationary RTE in spherical coordinates
(H 5-44, p. 117),

$$
\phi\,\frac{\partial I_\nu}{\partial r}
+ \frac{1-\phi^{2}}{r}\,\frac{\partial I_\nu}{\partial \phi}
+ \kappa'_\nu I_\nu \;=\; \kappa'_\nu I_\nu^{b}
$$

along straight chords at impact parameter $p$ (H 5-48, p. 119), with
the boundary condition that no radiation enters from outside. Because
every zone is homogeneous, the solution is evaluated **analytically
per segment** ([transport.radiative](reference/transport.md)):

$$
I_{\mathrm{out}} \;=\; I_{\mathrm{in}}\, e^{-\kappa L}
+ S \left(1 - e^{-\kappa L}\right),
\qquad
S \;=\; \frac{\epsilon_\nu}{\kappa'_\nu} \;=\; B_\nu
\ \text{(Kirchhoff)}
$$

No ODE stepping is needed; expm1 keeps the optically thin limit exact
and the optically thick limit saturates at the blackbody ceiling
(H 3-10, p. 55). Observables are the spatially resolved radiance at
one impact parameter or the disk-integrated radiance (area-weighted
average over the projected disk — the standard spatially integrated
LIBS measurement). Self-absorption and self-reversal **emerge**
naturally from cool outer zones saturating the line core toward their
local blackbody radiance — no ad hoc self-absorption parameter. The
optical depth $\tau_\nu = \sum \kappa' L$ is reported per line of
sight.

## Temporal evolution and gate integration

The plasma is treated quasi-statically: at each quadrature instant it
is rendered as a static snapshot with the full transport solution
([temporal](reference/temporal.md)). Parameter histories are
closed-form profiles — power-law decays

$$
T(t) = T_{\mathrm{ref}} \left(\frac{t}{t_{\mathrm{ref}}}\right)^{-b},
\qquad
n_e(t) = n_{\mathrm{ref}} \left(\frac{t}{t_{\mathrm{ref}}}\right)^{-a}
$$

(Aguilera & Aragón 2004; Cristoforetti et al. 2004) or exponential
decay — and any user-supplied $t \mapsto$ value callable can be
substituted.

!!! note "Deviation — closed-form histories instead of the ODE system"
    The dissertation evolves the plasma with an energy-balance ODE for
    $T$ (H 5-35, p. 115 / App. C) and a self-similar hydrodynamic
    expansion for the densities (App. B). These are not solved;
    the closed-form profiles above are a documented simplification,
    standard in the time-resolved LIBS characterization literature. A
    numerical solution of the thesis system can be plugged in through
    the same interface.

Two evolution models are provided: a **uniform sphere** with
per-instant Saha closure of $n_e$ (primary; used for validation), and
a **self-similar expanding onion** (App. B Eq. B-9 densities with a
frozen profile shape) as the spatially resolved option.

The recorded signal is the gate integral over the detector window
(top-hat gate; $t_{\mathrm{delay}}$/$t_{\mathrm{gate}}$ definitions
H pp. 46–47):

$$
E_\lambda \;=\; \int_{t_d}^{t_d + t_g} I_\lambda(t)\, \mathrm{d}t
$$

evaluated by Gauss–Legendre quadrature (default) or the trapezoid
rule, with the node count recorded in the spectrum metadata.

## Instrument model

Instrument effects are chained in fixed physical order, each stage
optional ([instrument](reference/instrument.md)): LSF convolution →
spectral collection efficiency → pixel binning → detector noise.

**Line-spread function.** The working model is Gaussian (H p. 59),
with triangular (equal entrance/exit slits, H Fig. 3-5, p. 64) and
Voigt (Gaussian core plus a calibrated Lorentzian component, as
measured for echelle-type instrument functions; used in the Hansen
validation case below) options. The FWHM comes from the slit bandpass
(H 3-19, p. 57) with a diffraction-limited validity floor (H 3-20,
p. 59), and Gaussian widths combine in quadrature (H 3-22, p. 60):

$$
\Delta\lambda_s = R_d\, W_{\mathrm{slit}},
\qquad
\Delta\lambda_G = \sqrt{\Delta\lambda_D^{2} + \Delta\lambda_I^{2}}
$$

The discrete kernel is normalized to unit sum (flux-conserving on the
grid), and $\ge 3$ samples per FWHM are enforced.

**Collection efficiency** (H 5-9..5-11, p. 104): pure per-wavelength
scaling
$F_{\mathrm{det}}(\lambda) = F_{\mathrm{rel}}(\lambda)\,
F_{\mathrm{abs}}$, with $F_{\mathrm{rel}}$ from tabulated calibration
curves and $F_{\mathrm{abs}}$ a single radiometric scale factor.

**Detector noise** (standard detector statistics — a documented
extension; the dissertation treats noise operationally): per
wavelength sample, mutually independent,

$$
\mathrm{output} \;=\;
\mathrm{Poisson}(\mathrm{signal} + \mathrm{dark} +
\mathrm{background})
+ \mathcal{N}\!\left(0,\, \sigma_{\mathrm{read}}^{2}\right)
$$

with a mandatory integer seed and a dedicated RNG, so identical seeds
give bit-identical noisy spectra.

## Numerical implementation and verification

Python 3 / NumPy / SciPy; strict SI internally; frozen (immutable)
dataclasses; vectorized pure functions; no global state.
Numerical-stability choices: expm1 for all $(1 - e^{-x})$ factors,
Brent bracketing for the Saha root, refusal to extrapolate partition
tables, Voigt via the Faddeeva function.

Every physics module is verified against analytic identities as unit
tests:

- profile normalization $\int P\,\mathrm{d}\nu = 1$ to $\le 10^{-10}$;
- Kirchhoff closure $\epsilon/\kappa = B_\nu$ and
  $B_\lambda = B_\nu\, c/\lambda^{2}$;
- Saha mass conservation exact in floating point; population fractions
  sum to 1;
- SI forms reproduce the printed CGS coefficients (McWhirter, Debye
  sphere, Stark bracket, Kramers $3.7\times10^{8}$);
- Whiting FWHM algebra accurate to 1–2 %;
- gate integral $\propto t_{\mathrm{gate}}$ for a constant plasma;
- one-zone transfer reproduces
  $I = B\left(1 - e^{-\kappa \ell}\right)$ exactly;
- limit behavior: $T \to 0$ neutral limit, $U(T) \to g_0$, optically
  thin/thick limits, Wien/Rayleigh–Jeans limits.

## Validation methodology

Experimental spectra are loaded from CSV
([analysis](reference/analysis.md)) or digitized from embedded figure
images in published PDFs (with the digitization procedure and
estimated uncertainties stated per case). Preprocessing mirrors the
CF-LIBS preparation (H p. 103)
([validation.preprocessing](reference/validation.md)): crop to the
modeled window, baseline subtraction from line-free windows (constant
or linear fit), normalization, and resampling onto the common model
grid — every step recorded in the spectrum metadata.

The primary comparison metric
([validation.metrics](reference/validation.md)) is the linear
correlation coefficient (H 5-56, p. 122 — the MC-LIBS cost function):

$$
R \;=\; \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_i (x_i - \bar{x})^{2}\, \sum_i (y_i - \bar{y})^{2}}}
$$

supported by peak-position matching (wavelength calibration and line
identification), peak intensity ratios (population and $A$-value
physics), and the normalized rms residual (overall shape). Parameter
sweeps (e.g. gate delay at fixed width — the temporally resolved study
pattern of thesis Ch. 6–7) and comparison plotting are provided by the
[analysis](reference/analysis.md) helpers; the reusable
simulate-and-compare pipeline is
[validation.workflow](reference/validation.md).

Validation cases drive the uniform-plasma forward model with
literature-sourced plasma conditions ($T$, $n_e$ from Stark
measurements). First real-data case: the Na I D doublet against the
Hansen thesis spectrum at 500 ns delay, $R = 0.9894$.

## Suggested tables/figures for the journal manuscript

- Fig. M1: pipeline block diagram (Overview).
- Fig. M2: onion geometry + chord/impact-parameter sketch (Transfer).
- Table M1: symbols and units.
- Table M2: atomic data per line with sources/grades (Atomic data).
- Table M3: verification identities and tolerances (Verification).
- Table M4: validation cases with plasma conditions and sources
  (Validation).

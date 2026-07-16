# Methods Section — Detailed Outline

Draft outline for the journal-article methods section, built from what is
actually implemented in `libssim`. Equation tags "(H x-y, p. z)" refer to
Herrera (2008), the source dissertation; the paper should renumber them as
its own equations. Items marked **[deviation]** are documented departures
from / corrections to the dissertation that belong in the paper because
they affect reproducibility.

---

## 2. Methods

### 2.1 Overview of the forward model
- One-paragraph pipeline statement with a figure (block diagram):
  **plasma state → LTE population physics → per-zone emissivity/absorption
  → line-of-sight radiative transfer → temporal gate integration →
  instrument response → comparison metrics.**
- Scope statement: forward (simulation) direction of the CF-LIBS/MC-LIBS
  formalism of Herrera (2008); LTE, two ionization stages (I ↔ II),
  spherically symmetric plasma.
- Software statement: open-source Python package (`libssim`), strict SI
  units internally, pure/immutable data structures, every physics module
  unit-tested against analytic identities (details in §2.11).

### 2.2 Atomic and spectroscopic data
- Transition data (wavelength, $A_{ki}$, level energies, statistical
  weights) and first ionization energies transcribed from the NIST Atomic
  Spectra Database (Kramida et al.); robust CSV parser for ASD exports
  (`atomic/parsers.py`) with validation of critical fields.
- Air wavelengths as stored; no air/vacuum correction in the physics layer
  (narrow-line; instrument-layer concern).
- Stark electron-impact half-widths $w$ (at reference density
  $n_e = 10^{22}\ \mathrm{m^{-3}} = 10^{16}\ \mathrm{cm^{-3}}$) and
  shift-to-width ratios $d/w$ from tabulated sources (Griem; thesis
  refs [131,139]).
- Table (paper): lines used per element with $\lambda$, $A_{ki}$,
  $E_{\mathrm{upper}}$, $g$, $w$, and ASD accuracy grades.

### 2.3 Plasma state and equilibrium populations (LTE)
- LTE assumption: single temperature
  $T = T_{\mathrm{exc}} = T_{\mathrm{ion}}$ (justify with gate-delay
  regime studied).
- **Boltzmann level populations** (H 5-1, p. 98) → `physics/boltzmann.py`:

    $$
    n_k \;=\; n_{\mathrm{tot}}\,
    \frac{g_k\, e^{-E_k/(k_B T)}}{U(T)}
    $$

- **Saha ionization balance**, two stages, per element (H 5-2 p. 98 /
  D-1 p. 274): Saha function $s^{j}(T)$; closure by mass conservation
  (D-2) and charge equilibrium (D-3); explicit stage densities
  (D-8, D-10) → `physics/saha.py`.
- Self-consistent electron density: implicit equation (D-9)

    $$
    n_e \;=\; \sum_j \frac{n^{j}\, s^{j}(T)}{s^{j}(T) + n_e}
    $$

    solved by bracketed root finding (Brent's method, rtol $10^{-14}$;
    unique positive root proven by monotonicity). Mass conservation exact
    in floating point by construction ($n_a^{j} = n^{j} - n_i^{j}$).

- **[deviation]** Ionization-potential lowering $\Delta\chi$: dissertation
  defines the quantity but gives no formula; implemented as the standard
  Debye estimate $\Delta\chi_z = z e^{2}/(4\pi\varepsilon_0 \lambda_D)$
  (Griem 1964; Drawin & Felenbok 1965), default 0 (magnitude
  0.05–0.2 eV under LIBS conditions).
- **LTE validity diagnostics** reported with every simulation:
  McWhirter criterion (H 5-3, p. 99) and Debye-sphere particle count
  (H 5-16, p. 106), both re-derived in SI and validated against the
  printed CGS coefficients.

### 2.4 Partition functions
- Definition: direct Boltzmann sum over bound levels,
  $U(T) = \sum_i g_i\, e^{-E_i/(k_B T)}$ (H p. 260)
  → `physics/partition.py`.
- Three composable evaluation strategies (state which was used per element):
  1. direct summation over transcribed NIST level lists (primary; Na I
     complete through $n = 6$, Al I through 6s);
  2. linear interpolation on tabulated $(T, U)$ grids (Drawin & Felenbok /
     Halenka style);
  3. Irwin (1981) polynomial $\ln U = \sum_k a_k (\ln T)^{k}$ as an
     out-of-grid fallback.
- No extrapolation outside validity ranges (raises instead — safety
  decision worth one sentence).
- High-$n$ Rydberg truncation caveat (Inglis–Teller microfield
  dissolution); optional hydrogenic tail augmentation ($2n^{2}$ weights at
  $\chi - R_y/n^{2}$), deliberately not applied by default; quantify
  truncation error (few % at 15 kK for Na/Al lists).

### 2.5 Line emission and absorption coefficients
- Spontaneous photon rate $I_{ki} = n_k A_{ki}$ (H 5-8, pp. 103–104);
  spectral emission coefficient
  $\epsilon_\nu = (h\nu_0/4\pi)\, n_k A_{ki} P(\nu)$
  → `physics/emission.py`.
- **[deviation]** Einstein $B_{lu}$ from $A_{ul}$ via the standard
  detailed-balance relations (Mihalas 1978) — not printed in the
  dissertation, which takes $B$ as data.
- Bound-bound absorption coefficient with LTE stimulated-emission factor
  (H 5-52, p. 120):

    $$
    \kappa_{bb} \;=\; \frac{h\nu_0}{c}\, B_{lu}\, n_l\, P(\nu)
    \left(1 - e^{-h\nu_0/(k_B T)}\right)
    $$

- Narrow-line approximation ($h\nu$ and stimulated factor evaluated at
  $\nu_0$; relative error $< 10^{-4}$).
- Internal consistency: Kirchhoff identity
  $\epsilon_\nu/\kappa_{bb} = B_\nu(T)$ enforced numerically in the test
  suite (state this — it pins the $A \leftrightarrow B$ convention).
- Planck functions $B_\nu$, $B_\lambda$ (H 3-11, p. 55), expm1-stable at
  both limits.

### 2.6 Line profiles and broadening
- Voigt profile: Doppler (Gaussian) $\otimes$ Stark (Lorentzian)
  → `physics/line_profiles.py`, evaluated with the Faddeeva function
  (scipy `voigt_profile`; dissertation used a Humlíček/Schreier algorithm —
  same function).
- Doppler FWHM (H 3-1, p. 50):
  $\Delta\lambda_D = \lambda_0 \sqrt{8 \ln 2\, k_B T/(m c^{2})}$.
- Quadratic Stark FWHM (H 3-8, p. 53, SI form): linear in $n_e$, tabulated
  $w$ at $10^{22}\ \mathrm{m^{-3}}$, optional ion-broadening bracket with
  Debye correction; $\alpha = 0$ default (negligible under LIBS
  conditions, H p. 106).
- Quadratic Stark shift (H 3-9, p. 53); red-shift sign convention;
  **[deviation]** ambiguous $\pm$ joining the ion term to $d/w$ resolved
  to the Griem convention.
- **[deviation]** Printed reduced-Voigt pair (H 5-54/5-55) is mutually
  inconsistent by $\sqrt{\ln 2}$; profiles are evaluated with exact FWHM
  semantics (equivalent to 5-53 with the $\sqrt{\ln 2}$ restored);
  equivalence to the verbatim reduced form proven to machine precision in
  tests.
- All profiles analytically unit-normalized
  ($\int P\,\mathrm{d}\nu = 1$; tested $\le 10^{-10}$); Lorentzian
  tail-mass note for finite grids.
- Whiting-type FWHM algebra (H 5-18, p. 107) for Stark-width extraction
  (used in the validation workflow, §2.12).
- Explicitly out of scope: van der Waals, resonance, natural broadening
  (negligible vs. Stark + Doppler under these conditions).

### 2.7 Continuum radiation
- Kramers–Unsöld hydrogen-like model → `physics/continuum.py`:
  free-free (inverse bremsstrahlung, H 5-50, p. 119) and free-bound
  (radiative recombination, H 5-51, p. 120) absorption coefficients;
  Gaunt factor $G$ and Biberman factor $\xi \approx 1$ defaults.
- **[deviation]** Typographical error in printed Eq. 5-50 exponential
  factor corrected to $\left(1 - e^{-h\nu/(k_B T)}\right)$; justified
  three ways (Rybicki & Lightman coefficient match, Kirchhoff image of
  Eq. 5-7, physical boundedness).
- Continuum emission via Kirchhoff's law
  $\epsilon_\nu = \kappa_\nu B_\nu(T)$; Planck-mean absorption
  coefficient (H 5-46, p. 118) available for the energy-loss term.
- SI conversion of the Gaussian-CGS forms
  ($e^{6} \to e^{6}/(4\pi\varepsilon_0)^{3}$), validated against the
  printed $3.7\times10^{8}$ CGS coefficient.

### 2.8 Plasma geometry and radiative transfer
- Spherical "onion" discretization: $N$ concentric homogeneous shells,
  each a locally uniform plasma state → `transport/geometry.py`; parabolic
  initial $T(r)$ and $n(r)$ profiles (H 5-36/5-37, p. 116) with per-zone
  Saha closure; single-zone case = uniform sphere.
- Per-zone spectral properties: $\epsilon_\nu(\lambda)$,
  $\kappa'_\nu(\lambda)$ with
  $\kappa' = \kappa_{ff} + \kappa_{fb} + \kappa_{bb}$ (H 5-49, p. 119)
  → `transport/emissivity.py`.
- Formal solution of the RTE (H 5-44, p. 117) along chords at impact
  parameter $p$ (H 5-48, p. 119) evaluated **analytically per segment**
  → `transport/radiative.py`:

    $$
    I_{\mathrm{out}} \;=\; I_{\mathrm{in}}\, e^{-\kappa L}
    + S \left(1 - e^{-\kappa L}\right),
    \qquad
    S \;=\; \epsilon/\kappa \;=\; B_\nu \ \text{(Kirchhoff)}.
    $$

    No ODE stepping; expm1 keeps the optically thin limit exact and the
    thick limit saturates at the blackbody ceiling (H 3-10, p. 55).

- Observables: spatially resolved radiance at one impact parameter, or
  disk-integrated radiance (area-weighted average over the projected
  disk) — the standard spatially integrated LIBS measurement.
- Emergent behavior to state: self-absorption and self-reversal arise
  naturally from cool outer zones (no ad hoc self-absorption parameter).
- Optical depth $\tau_\nu = \sum \kappa' L$ reported per line of sight.

### 2.9 Temporal evolution and gate integration
- Quasi-static assumption: plasma rendered as a static snapshot at each
  quadrature instant → `temporal/`.
- Parameter histories: power-law decays
  $T(t) = T_{\mathrm{ref}}\, (t/t_{\mathrm{ref}})^{-b}$,
  $n_e(t) = n_{\mathrm{ref}}\, (t/t_{\mathrm{ref}})^{-a}$
  (Aguilera & Aragón 2004; Cristoforetti et al. 2004) or exponential;
  **[deviation]** documented simplification — the dissertation's ODE
  energy balance (H 5-35 / App. C) and self-similar expansion (App. B)
  are not solved; any user profile can be substituted.
- Evolution models: uniform sphere with per-instant Saha closure of $n_e$
  (primary, used for validation); self-similar expanding onion
  (App. B Eq. B-9 densities, frozen profile shape) as the
  spatially resolved option.
- Detector gate (top-hat;
  $t_{\mathrm{delay}}$/$t_{\mathrm{gate}}$ definitions H pp. 46–47):

    $$
    E_\lambda \;=\; \int_{t_d}^{t_d + t_g} I_\lambda(t)\, \mathrm{d}t
    $$

    Gauss–Legendre quadrature (default; node count stated), trapezoid
    alternative.

### 2.10 Instrument model
- Chain (each stage optional, order fixed) → `instrument/response.py`:
  LSF convolution → spectral collection efficiency → pixel binning →
  detector noise.
- Line-spread function: Gaussian working model (H p. 59), with triangular
  (equal entrance/exit slits, H Fig. 3-5, p. 64) and Voigt (Gaussian core
  plus calibrated Lorentzian component, as measured for echelle-type
  instrument functions; used in the Hansen validation case, §2.12)
  options; FWHM from slit bandpass
  $\Delta\lambda_s = R_d\, W_{\mathrm{slit}}$ (H 3-19, p. 57),
  diffraction-limited floor (H 3-20, p. 59); Gaussian widths combine in
  quadrature, $\Delta\lambda_G = \sqrt{\Delta\lambda_D^{2} +
  \Delta\lambda_I^{2}}$ (H 3-22, p. 60); discrete flux-conserving kernel
  on a uniform grid, $\ge 3$ samples per FWHM enforced.
- Collection efficiency
  $F_{\mathrm{det}}(\lambda) = F_{\mathrm{rel}}(\lambda)\,
  F_{\mathrm{abs}}$ (H 5-9..5-11, p. 104); $F_{\mathrm{rel}}$ from
  tabulated calibration curves, $F_{\mathrm{abs}}$ a single radiometric
  scale factor.
- Detector noise (Janesick-standard, documented extension): Poisson shot
  noise on signal + dark + background, additive Gaussian read noise;
  seeded RNG → bit-reproducible noisy spectra.

### 2.11 Numerical implementation and verification
- Python 3 / NumPy / SciPy; strict SI internally; frozen dataclasses;
  vectorized pure functions; no global state.
- Numerical-stability choices: expm1 for all $(1 - e^{-x})$ factors,
  Brent bracketing for the Saha root, refusal to extrapolate partition
  tables, Voigt via Faddeeva.
- Verification strategy (worth a short subsection or table):
  - analytic identities as unit tests: profile normalization
    $\le 10^{-10}$, Kirchhoff closure $\epsilon/\kappa = B_\nu$,
    $B_\lambda = B_\nu\, c/\lambda^{2}$, Saha mass conservation exact,
    population fractions sum to 1, SI forms reproduce the printed CGS
    coefficients (McWhirter, Debye sphere, Stark bracket, Kramers
    $3.7\times10^{8}$), Whiting FWHM accuracy 1–2 %, gate integral
    $\propto t_{\mathrm{gate}}$ for a constant plasma, one-zone transfer
    reproduces $I = B\left(1 - e^{-\kappa \ell}\right)$ exactly;
  - limit behavior: $T \to 0$ neutral limit, $U(T) \to g_0$, optically
    thin/thick limits, Wien/Rayleigh–Jeans limits.
- Code availability statement (repository, version/tag, license).

### 2.12 Validation methodology
- Experimental data handling: literature spectra digitized from embedded
  figure images in published PDFs (state digitization procedure and
  estimated wavelength/intensity uncertainty) or loaded from CSV
  → `analysis.load_spectrum_csv`.
- Preprocessing mirroring the CF-LIBS preparation (H p. 103)
  → `validation/preprocessing.py`: crop to modeled window; baseline
  subtraction from line-free windows (constant or linear fit);
  normalization; resampling onto the common model grid; all steps recorded
  in spectrum metadata (provenance).
- Comparison metrics → `validation/metrics.py`:
  - primary: linear correlation coefficient $R$ (H 5-56, p. 122 — the
    MC-LIBS cost function);
  - supporting: peak-position matching (wavelength calibration + line
    identification), peak intensity ratios (population/$A$-value physics),
    normalized rms residual (overall shape).
- Validation cases (results section will expand; methods states the
  design): literature-sourced plasma conditions ($T$, $n_e$ from Stark
  measurements) drive the uniform-plasma forward model; e.g. Na I D
  doublet vs. Hansen thesis spectrum at 500 ns delay ($R = 0.9894$).
- Atomic-data provenance discipline: NIST ASD transcription with flagged
  low-confidence entries; representative Stark widths flagged for
  replacement with tabulated coefficients in quantitative width work.

### Suggested tables/figures for the section
- Fig. M1: pipeline block diagram (§2.1).
- Fig. M2: onion geometry + chord/impact-parameter sketch (§2.8).
- Table M1: symbols and units.
- Table M2: atomic data per line with sources/grades (§2.2).
- Table M3: verification identities and tolerances (§2.11).
- Table M4: validation cases with plasma conditions and sources (§2.12).

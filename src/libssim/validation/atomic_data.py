r"""Real atomic data for validation elements (Phase 4 validation
workflow).

Provenance (read before publication-grade use)
----------------------------------------------
All numeric values below are transcribed from the NIST Atomic Spectra
Database (ASD; Kramida et al., https://physics.nist.gov/asd) as of this
module's writing: air wavelengths, transition probabilities, level
energies, statistical weights and ionization energies.
**Verify against the current ASD release before publishing results.**
Entries with reduced confidence (J-merged blocks, 3-4 decimal
energies) are marked ``VERIFY`` inline and collected in each element's
``ElementAtomicData.verify_notes`` so nothing needs re-discovery.

Partition-function strategy (three tiers, lowest effort first)
--------------------------------------------------------------
1. **Transcribed level lists (default here).** Direct Boltzmann
   summation via `PartitionFunctionTable.from_levels` (Herrera p. 260
   definition). Na I is complete through the n = 6 shell (~4.60 eV),
   Al I through 6s (~5.24 eV): the truncation deficit is pushed above
   ~12-13 kK and is a few percent at 15 kK.
2. **Irwin-form polynomial fallback.** Each setup also registers a
   $\ln U = \mathrm{poly}(\ln T)$ fit through
   `PartitionFunctionProvider.with_polynomial`-style composition
   (table primary, polynomial where the table does not cover T).
   The shipped coefficients are FITTED TO THE SAME LEVEL LISTS —
   they add smoothness and a drop-in slot, not independent accuracy.
   Replace them with published Irwin (1981, ApJS 45, 621) or
   Halenka & Madej (2002) coefficients for literature-grade high-T
   values: build a `PartitionFunctionPolynomial` with those
   coefficients and put it in `ElementAtomicData.neutral_polynomial`.
3. **Full NIST ASD level download (best).** Export the ASD levels page
   to CSV (columns: energy, g) and load with `load_levels_csv`; feed
   the result to an `ElementAtomicData` spec. Accuracy is then limited
   only by the high-n cutoff question below.

High-n Rydberg caveat (thesis p. 100)
-------------------------------------
Near the ionization limit the level sum is cutoff-dependent: high-$n$
states are dissolved by plasma microfields (Inglis-Teller; at
$n_e \sim 10^{23}$ m$^{-3}$ only $n \lesssim 7$ survive for alkalis),
which is one of the "challenges associated with partition function
calculation" the thesis cites (p. 100). `hydrogenic_tail_levels`
provides an *optional, explicitly-chosen* augmentation ($2n^{2}$
weights at $\chi - R_y/n^{2}$); it is deliberately NOT applied by
default because the right $n_{\max}$ depends on the plasma density
being simulated.

Stark widths
------------
`Transition.stark_width` values (electron-impact half-width $w$ in
meters at the $10^{22}$ m$^{-3}$ reference, Eq. 3-8 / Eq. 5-17
convention)
remain *representative order-of-magnitude values* — replace with
tabulated coefficients (thesis refs [131, 139]) for quantitative
line-width work.

References
----------
Kramida, A. et al., NIST ASD, https://physics.nist.gov/asd.
Irwin, A.W. (1981). Astrophys. J. Suppl. 45, 621.
Halenka, J. & Madej, J. (2002). Acta Astronomica 52, 195.
Herrera, K.K. (2008), p. 100 (partition-function challenges), p. 260
(level summation); Eq. 3-8 p. 53 (Stark convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..atomic.transition import Transition
from ..core.constants import C, EV, H
from ..physics.partition import (
    PartitionFunctionPolynomial,
    PartitionFunctionProvider,
    PartitionFunctionTable,
    partition_function_from_levels,
)
from ..physics.saha import SahaSolver

#: One bound level: (energy above ground in eV, statistical weight g).
LevelData = Tuple[float, float]

#: Atomic mass unit (kg) — CODATA 2018.
_U_KG = 1.66053906660e-27

#: Default partition-table temperature grid (K).
_PARTITION_GRID = np.linspace(500.0, 30000.0, 600)

#: eV per cm^-1 (h*c*100/e), for ASD level exports in wavenumbers.
_EV_PER_CM1 = H * C * 100.0 / EV


# ---------------------------------------------------------------------------
# Sodium (NIST ASD transcription)
# ---------------------------------------------------------------------------
#: Minimal Na I list (original Phase 4 transcription, kept for
#: comparison studies): complete to ~4.35 eV.
NA_I_LEVELS_MINIMAL: Tuple[LevelData, ...] = (
    (0.000000, 2),   # 3s 2S1/2
    (2.102297, 2),   # 3p 2P1/2
    (2.104429, 4),   # 3p 2P3/2
    (3.191351, 2),   # 4s 2S1/2
    (3.616872, 4),   # 3d 2D3/2
    (3.616986, 6),   # 3d 2D5/2
    (3.752571, 2),   # 4p 2P1/2
    (3.753302, 4),   # 4p 2P3/2
    (4.116400, 2),   # 5s 2S1/2
    (4.283450, 10),  # 4d 2D (J merged)
    (4.288500, 14),  # 4f 2F (J merged)
    (4.344590, 2),   # 5p 2P1/2
    (4.344920, 4),   # 5p 2P3/2
)

#: Improved Na I list: minimal list plus the n = 6 shell block —
#: truncation now at ~4.60 eV (vs 5.139 eV ionization limit).
NA_I_LEVELS: Tuple[LevelData, ...] = NA_I_LEVELS_MINIMAL + (
    (4.511660, 2),   # 6s 2S1/2                       VERIFY (4 d.p.)
    (4.575400, 10),  # 5d 2D (J merged)               VERIFY (merged)
    (4.577700, 14),  # 5f 2F (J merged)               VERIFY (merged)
    (4.577900, 18),  # 5g 2G (J merged)               VERIFY (merged)
    (4.601850, 6),   # 6p 2P (J merged)               VERIFY (merged)
)

#: Na II (Ne-like closed shell): first excited level ~33 eV, so U = 1
#: exactly at LIBS temperatures.
NA_II_LEVELS: Tuple[LevelData, ...] = ((0.0, 1),)

NA_IONIZATION_EV = 5.139077
NA_MASS_KG = 22.98977 * _U_KG

#: Na I D resonance doublet (air wavelengths; A_ki from NIST ASD).
#: stark_width (electron-impact HWHM w at 1e22 m^-3 = 1e16 cm^-3):
#: from the measured FWHM compilation of Konjevic, Lesage, Fuhr &
#: Wiese, J. Phys. Chem. Ref. Data 31, 819 (2002), "Numerical results
#: for Na I", 3s-3p: D2 0.38/0.15/0.22 A FWHM at 3.50/1.23/2.55e17
#: cm^-3, D1 0.41/0.13/0.18 A at the same densities (T = 38-49 kK;
#: measured-to-theory ratios 0.80-1.15) -> ~0.10 A FWHM per 1e17
#: cm^-3 -> w = 5.0e-13 m HWHM at the 1e22 m^-3 reference. Measured
#: at 38-49 kK; LIBS-range (~10-15 kK) extrapolation uncertainty is
#: a few tens of percent (semiclassical T-dependence is weak). The
#: previous placeholder (3.0e-12 m) was ~6x broader than every
#: tabulated measurement.
NA_I_LINES: Tuple[Transition, ...] = (
    Transition(  # D2
        element="Na", ion_stage=1, wavelength_m=588.99509e-9,
        energy_lower_ev=0.0, energy_upper_ev=2.104429,
        a_ki=6.16e7, g_lower=2, g_upper=4, stark_width=5.0e-13,
    ),
    Transition(  # D1
        element="Na", ion_stage=1, wavelength_m=589.59237e-9,
        energy_lower_ev=0.0, energy_upper_ev=2.102297,
        a_ki=6.14e7, g_lower=2, g_upper=2, stark_width=5.0e-13,
    ),
)

# ---------------------------------------------------------------------------
# Aluminum (NIST ASD transcription)
# ---------------------------------------------------------------------------
#: Minimal Al I list (original transcription): complete to ~4.68 eV.
AL_I_LEVELS_MINIMAL: Tuple[LevelData, ...] = (
    (0.000000, 2),   # 3p 2P1/2 (ground)
    (0.013893, 4),   # 3p 2P3/2
    (3.142721, 2),   # 4s 2S1/2
    (4.021483, 4),   # 3d 2D3/2
    (4.021589, 6),   # 3d 2D5/2
    (4.084527, 2),   # 4p 2P1/2
    (4.087096, 4),   # 4p 2P3/2
    (4.672970, 2),   # 5s 2S1/2
)

#: Improved Al I list: adds the next configurations to ~5.24 eV.
AL_I_LEVELS: Tuple[LevelData, ...] = AL_I_LEVELS_MINIMAL + (
    (4.826800, 6),   # 5p 2P (J merged)               VERIFY (merged)
    (5.058000, 10),  # 4d 2D (J merged)               VERIFY (merged, 3 d.p.)
    (5.236500, 2),   # 6s 2S1/2                       VERIFY (3-4 d.p.)
)

#: Al II (Mg-like): ground 1S0, the 3s3p 3P block near 4.64 eV, and
#: the 3s3p 1P level (matters only above ~20 kK).
AL_II_LEVELS: Tuple[LevelData, ...] = (
    (0.000000, 1),
    (4.634950, 1),   # 3P0
    (4.637650, 3),   # 3P1
    (4.643160, 5),   # 3P2
    (7.420700, 3),   # 1P1                            VERIFY (4 d.p.)
)

AL_IONIZATION_EV = 5.985769
AL_MASS_KG = 26.981538 * _U_KG

#: Al I resonance doublet (shared upper level 4s 2S1/2 -> classic
#: ~1:2 intensity ratio 394.4 : 396.15).
AL_I_LINES: Tuple[Transition, ...] = (
    Transition(
        element="Al", ion_stage=1, wavelength_m=394.40058e-9,
        energy_lower_ev=0.0, energy_upper_ev=3.142721,
        a_ki=4.99e7, g_lower=2, g_upper=2, stark_width=1.5e-12,
    ),
    Transition(
        element="Al", ion_stage=1, wavelength_m=396.15200e-9,
        energy_lower_ev=0.013893, energy_upper_ev=3.142721,
        a_ki=9.85e7, g_lower=4, g_upper=2, stark_width=1.5e-12,
    ),
)


# ---------------------------------------------------------------------------
# Helpers: better data, one step at a time
# ---------------------------------------------------------------------------
def load_levels_csv(
    path: Union[str, Path],
    energy_unit: str = "eV",
    energy_column: int = 0,
    g_column: int = 1,
    delimiter: str = ",",
    skip_header: int = 0,
) -> Tuple[LevelData, ...]:
    """
    Load a NIST ASD levels export as (energy_eV, g) tuples.

    The one-step upgrade path to full-accuracy partition functions:
    export the ASD "Levels" page for a species to CSV (energy + g
    columns; '#' comments allowed), load it here, and put the result in
    an `ElementAtomicData` spec.

    Parameters
    ----------
    path : str or Path
        CSV file with numeric energy and g columns.
    energy_unit : {"eV", "cm-1"}, optional
        ASD exports offer both; cm^-1 values are converted via
        h*c/e (1 cm^-1 = 1.2398e-4 eV).
    energy_column, g_column : int, optional
        Zero-based column indices (defaults 0 and 1).
    delimiter, skip_header : optional
        Passed to numpy.genfromtxt.

    Returns
    -------
    tuple of (energy_ev, g)
        Sorted by energy; validated finite, energies >= 0, g > 0.
    """
    unit = energy_unit.strip().lower()
    if unit not in ("ev", "cm-1"):
        raise ValueError("energy_unit must be 'eV' or 'cm-1'")
    data = np.genfromtxt(
        Path(path), delimiter=delimiter, skip_header=skip_header, comments="#"
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if max(energy_column, g_column) >= data.shape[1]:
        raise ValueError(
            f"file has {data.shape[1]} columns; requested "
            f"{energy_column} and {g_column}"
        )
    energy = data[:, energy_column]
    g = data[:, g_column]
    valid = np.isfinite(energy) & np.isfinite(g)
    if not np.any(valid):
        raise ValueError(f"no valid (energy, g) rows in {path}")
    energy, g = energy[valid], g[valid]
    if unit == "cm-1":
        energy = energy * _EV_PER_CM1
    if np.any(energy < 0) or np.any(g <= 0):
        raise ValueError("levels require energy >= 0 and g > 0")
    order = np.argsort(energy)
    return tuple(
        (float(e), float(w)) for e, w in zip(energy[order], g[order])
    )


def hydrogenic_tail_levels(
    ionization_ev: float,
    n_min: int,
    n_max: int,
    rydberg_ev: float = 13.605693,
) -> Tuple[LevelData, ...]:
    r"""
    OPTIONAL high-$n$ augmentation for one-valence-electron
    (alkali-like) species: pseudo-levels at
    $E_n = \chi - R_y/n^{2}$ with the full shell weight
    $g_n = 2n^{2}$.

    Deliberately not applied by default: the physically meaningful
    $n_{\max}$ is density-dependent (Inglis-Teller microfield
    dissolution; $n_{\max} \sim 7$ at $n_e \sim 10^{23}$ m$^{-3}$),
    which is exactly the thesis' "challenges associated with partition
    function calculation" (p. 100). Choose `n_max` consciously for the
    plasma you simulate, e.g.:

        levels = NA_I_LEVELS + hydrogenic_tail_levels(5.139, 7, 7)
    """
    if not (ionization_ev > 0 and np.isfinite(ionization_ev)):
        raise ValueError("ionization_ev must be finite and > 0")
    if n_min < 2 or n_max < n_min:
        raise ValueError("require 2 <= n_min <= n_max")
    levels = []
    for n in range(int(n_min), int(n_max) + 1):
        energy = ionization_ev - rydberg_ev / n**2
        if energy <= 0:
            raise ValueError(
                f"hydrogenic level n={n} falls below the ground state; "
                "n_min too small for this ionization energy"
            )
        levels.append((float(energy), float(2 * n**2)))
    return tuple(levels)


def fit_irwin_polynomial(
    levels: Sequence[LevelData],
    temperature_range_K: Tuple[float, float] = (500.0, 30000.0),
    degree: int = 11,
    n_fit_points: int = 400,
) -> PartitionFunctionPolynomial:
    r"""
    Irwin-form polynomial,
    $\ln U = \sum_k a_k\, (\ln T)^{k}$, fitted to the direct level
    summation.

    This provides a smooth fallback and the exact container where
    *published* Irwin (1981) / Halenka & Madej (2002) coefficients
    belong — a fit to our own level list adds no independent accuracy
    (documented limitation).

    Numerics: the fit runs in numpy's scaled domain (well-conditioned
    at degree ~11) and is converted exactly to the plain power basis
    that `PartitionFunctionPolynomial` / Irwin's published tables use.
    Residuals for the shipped Na/Al lists are <~ 0.3% relative
    (unit-tested); the limiting feature is the low-temperature
    freeze-out knee — one reason published Irwin fits start at 1000 K.
    """
    lo, hi = (float(t) for t in temperature_range_K)
    grid = np.linspace(lo, hi, int(n_fit_points))
    g = [w for _, w in levels]
    e_ev = [e for e, _ in levels]
    u_values = np.asarray(partition_function_from_levels(g, e_ev, grid))
    # Scaled-domain fit (conditioning), then exact conversion to the
    # unscaled ln T power basis of Irwin's convention.
    fitted = np.polynomial.Polynomial.fit(
        np.log(grid), np.log(u_values), int(degree)
    )
    coefficients = fitted.convert().coef
    return PartitionFunctionPolynomial(
        coefficients=tuple(float(c) for c in coefficients),
        temperature_range_K=(lo, hi),
    )


# ---------------------------------------------------------------------------
# Declarative element specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ElementAtomicData:
    """
    Everything needed to build an `ElementSetup`, as plain declarative
    data — adding a new element is one instance of this class.

    Parameters
    ----------
    element : str
        Symbol ("Na", "Al", ...).
    lines : tuple of Transition
        Transcribed line data (air wavelengths, A_ki, ...).
    neutral_levels, ion_levels : tuple of (energy_eV, g)
        Bound levels for stages I and II (from the constants here,
        `load_levels_csv`, and/or `hydrogenic_tail_levels`).
    ionization_ev, mass_kg : float
        First ionization energy and atomic mass.
    neutral_polynomial, ion_polynomial : PartitionFunctionPolynomial, optional
        High-T fallbacks composed with the level tables by the
        provider. If None, an Irwin-form fit to the level list is
        generated automatically (slot for published coefficients).
    provenance : str
        Data-source statement, surfaced on the built ElementSetup.
    verify_notes : tuple of str
        Entries flagged for re-verification against current NIST ASD.
    """

    element: str
    lines: Tuple[Transition, ...]
    neutral_levels: Tuple[LevelData, ...]
    ion_levels: Tuple[LevelData, ...]
    ionization_ev: float
    mass_kg: float
    neutral_polynomial: Optional[PartitionFunctionPolynomial] = None
    ion_polynomial: Optional[PartitionFunctionPolynomial] = None
    provenance: str = "NIST ASD transcription (verify before publication)"
    verify_notes: Tuple[str, ...] = ()


#: Sodium specification (module-level so studies can inspect/replace it).
SODIUM_DATA = ElementAtomicData(
    element="Na",
    lines=NA_I_LINES,
    neutral_levels=NA_I_LEVELS,
    ion_levels=NA_II_LEVELS,
    ionization_ev=NA_IONIZATION_EV,
    mass_kg=NA_MASS_KG,
    provenance=(
        "Na: NIST ASD transcription; Na I levels complete through the "
        "n=6 shell (~4.60 eV of 5.139 eV limit); Na II closed-shell. "
        "D-line Stark widths from the measured compilation of "
        "Konjevic et al., J. Phys. Chem. Ref. Data 31, 819 (2002) "
        "(~0.10 A FWHM per 1e17 cm^-3 at 38-49 kK)."
    ),
    verify_notes=(
        "Na I 6s/5d/5f/5g/6p block: J-merged energies at 3-4 decimals",
        "Rydberg tail above n=6 omitted (density-dependent cutoff; "
        "see hydrogenic_tail_levels)",
        "Na D Stark widths measured at 38-49 kK (Konjevic 2002); "
        "10-15 kK LIBS use extrapolates the weak semiclassical "
        "T-dependence (tens-of-percent uncertainty)",
    ),
)

#: Aluminum specification.
ALUMINUM_DATA = ElementAtomicData(
    element="Al",
    lines=AL_I_LINES,
    neutral_levels=AL_I_LEVELS,
    ion_levels=AL_II_LEVELS,
    ionization_ev=AL_IONIZATION_EV,
    mass_kg=AL_MASS_KG,
    provenance=(
        "Al: NIST ASD transcription; Al I levels to ~5.24 eV of the "
        "5.986 eV limit; Al II through 3s3p 1P. Stark widths are "
        "representative Griem-order values."
    ),
    verify_notes=(
        "Al I 5p/4d/6s entries: J-merged energies at 3-4 decimals",
        "Al II 1P1 energy at 4 decimals",
    ),
)


# ---------------------------------------------------------------------------
# Element setup container + builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ElementSetup:
    """
    Everything the validation workflow needs for one element: lines,
    a Saha solver carrying its partition data, masses and provenance.
    """

    element: str
    transitions: Tuple[Transition, ...]
    saha_solver: SahaSolver
    atomic_masses_kg: Mapping[str, float]
    provenance: str
    verify_notes: Tuple[str, ...] = ()

    @property
    def line_wavelengths_m(self) -> Tuple[float, ...]:
        """Expected line positions (m), for peak matching."""
        return tuple(t.wavelength_m for t in self.transitions)


def build_setup(
    data: ElementAtomicData,
    temperature_grid_K: Optional[np.ndarray] = None,
    include_polynomial_fallback: bool = True,
) -> ElementSetup:
    """
    Build an `ElementSetup` from a declarative spec.

    Registers level-summation tables (primary) and Irwin-form
    polynomials (fallback outside the table grid) with one
    `PartitionFunctionProvider`; the Phase 2 provider composes them
    per temperature point automatically.
    """
    grid = (
        _PARTITION_GRID
        if temperature_grid_K is None
        else np.asarray(temperature_grid_K, dtype=np.float64)
    )

    def table(levels: Tuple[LevelData, ...]) -> PartitionFunctionTable:
        return PartitionFunctionTable.from_levels(
            [g for _, g in levels], [e for e, _ in levels], grid
        )

    tables = {
        (data.element, 1): table(data.neutral_levels),
        (data.element, 2): table(data.ion_levels),
    }
    polynomials = {}
    if include_polynomial_fallback:
        polynomials[(data.element, 1)] = (
            data.neutral_polynomial
            if data.neutral_polynomial is not None
            else fit_irwin_polynomial(data.neutral_levels)
        )
        polynomials[(data.element, 2)] = (
            data.ion_polynomial
            if data.ion_polynomial is not None
            else fit_irwin_polynomial(data.ion_levels)
        )

    provider = PartitionFunctionProvider(
        tables=tables, polynomials=polynomials
    )
    solver = SahaSolver(
        partition_provider=provider,
        ionization_energies_ev={data.element: data.ionization_ev},
    )
    return ElementSetup(
        element=data.element,
        transitions=data.lines,
        saha_solver=solver,
        atomic_masses_kg={data.element: data.mass_kg},
        provenance=data.provenance,
        verify_notes=data.verify_notes,
    )


def sodium_setup() -> ElementSetup:
    """
    Na I D-doublet validation setup (NIST data; module provenance).

    Improved level list (n <= 6): U_I(10 kK) ~ 3.3, truncation deficit
    pushed above ~12 kK; Na II closed-shell U = 1. Irwin-form
    polynomial fallback registered (fitted to the same levels — see
    module docstring for the literature-coefficient upgrade).
    """
    return build_setup(SODIUM_DATA)


def aluminum_setup() -> ElementSetup:
    """
    Al I 394.4/396.15 nm doublet validation setup (NIST data).

    Improved level list (to ~5.24 eV): U_I(10 kK) ~ 6.1. Irwin-form
    polynomial fallback registered (see module docstring).
    """
    return build_setup(ALUMINUM_DATA)

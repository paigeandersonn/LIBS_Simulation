r"""Atomic and ionic partition functions $U(T)$ for LTE plasma
modeling (Phase 2).

Physical context (Herrera 2008)
-------------------------------
The internal partition function $U(T)$ is the temperature-dependent
statistical weight of a species — the normalization that converts a
total species number density into individual level populations. In the
thesis it is defined in the symbols list as "U(T): Partition function,
dimensionless" (p. 23) and enters every population and intensity
expression of the CF-LIBS / MC-LIBS formalism:

- Boltzmann level population (Eq. 5-1, p. 98):
  $n_k^{s} = n_{\mathrm{tot}}^{s}\, g_k
  \exp(-E_k / k_B T_{\mathrm{exc}}) / U^{s}(T)$
- Saha ionization balance (Eq. 5-2, p. 98; ratio $U^{II}/U^{I}$) and
  its MC-LIBS form (Eq. D-1, p. 274; ratio $U_i^{j}/U_a^{j}$).
- Integrated line intensity (Eq. 5-8, pp. 103-104; divides by
  $U^{s}(T)$).

Herrera notes (p. 260) that accurate partition functions must "[take]
into account all the bound quantum states of an atom or ion", i.e. the
direct Boltzmann sum over bound levels:

$$
U(T) \;=\; \sum_i g_i\, \exp\!\left(-\frac{E_i}{k_B T}\right)
\qquad \text{(definition)}
$$

and sources numerical values from published tabulations (p. 133):
Drawin & Felenbok [130], Halenka [169], Halenka & Madej [170].

Implementation decisions (documented per development_rules.md)
--------------------------------------------------------------
The dissertation cites tabulated $U(T)$ sources but does not prescribe
an evaluation algorithm. This module therefore provides three standard,
literature-backed strategies:

1. Direct summation over bound levels
   (`partition_function_from_levels`) — the definition referenced on
   p. 260. Exact for the levels supplied; truncation of the level list
   is the user's responsibility.
2. Linear interpolation on a tabulated $(T, U)$ grid
   (`PartitionFunctionTable`) — the primary strategy, mirroring the
   thesis use of Drawin & Felenbok / Halenka tables. Linear
   interpolation is monotonicity-preserving and cannot overshoot
   between grid points.
3. Polynomial fallback (`PartitionFunctionPolynomial`) in the Irwin
   (1981, Astrophys. J. Suppl. 45, 621) form
   $\ln U = \sum_k a_k (\ln T)^k$, used when the requested temperature
   falls outside the tabulated grid. This functional form is the
   de-facto standard for compact partition-function fits and extends
   beyond the dissertation, which is silent on the functional form of
   its sources.

`PartitionFunctionProvider` composes strategies 2 and 3 per species
(element, ion stage I/II/III) and is the single entry point used by the
Saha (saha.py) and Boltzmann (boltzmann.py) modules.

Units and Conventions
---------------------
- Temperature: Kelvin (SI). Level energies: eV (as stored in atomic data,
  e.g. `Transition`), converted internally to Joules via
  `libssim.core.constants.EV`.
- U(T) is dimensionless.
- Ion stages follow the atomic-layer convention (see atomic/parsers.py):
  1 = neutral (I), 2 = singly ionized (II), 3 = doubly ionized (III).
  Herrera's CF-LIBS detects only I and II lines (p. 103); stage III is
  supported because the Saha chain I <-> II <-> III needs U up to III.

Numerical assumptions and limitations
-------------------------------------
- No ionization-potential lowering correction is applied to $U(T)$
  itself; $\Delta\chi$ enters the Saha exponent (Eq. 5-2 / D-1), not
  the level sum.
- Evaluation outside both the table grid and the polynomial validity
  range raises ValueError rather than extrapolating: partition
  functions grow steeply at high $T$ and silent extrapolation is
  physically unsafe.
- Low-temperature limit: $U(T) \to g_0$ (ground-level degeneracy) as
  $T \to 0$, a validation identity used in the unit tests.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Symbols p. 23;
Eqs. 5-1, 5-2 p. 98; Eq. 5-8 pp. 103-104; Eq. D-1 p. 274; partition
function sources p. 133 (refs [130], [169], [170]); bound-state discussion
p. 260.
Irwin, A.W. (1981). Astrophys. J. Suppl. 45, 621 (polynomial form).
Drawin, H.W. & Felenbok, P. (1965). Data for Plasmas in Local
Thermodynamic Equilibrium. Gauthier-Villars, Paris.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.constants import KB, EV

# Ion-stage bounds supported in Phase 2 (I = neutral ... III = doubly ionized).
_MIN_ION_STAGE = 1
_MAX_ION_STAGE = 3

#: Registry key: (element symbol upper-cased, ion stage). Element matching is
#: case-insensitive, mirroring atomic/parsers.py behaviour.
SpeciesKey = Tuple[str, int]


def _species_key(element: str, ion_stage: int) -> SpeciesKey:
    """Normalize (element, ion_stage) into a case-insensitive registry key."""
    elem = str(element).strip().upper()
    if not elem:
        raise ValueError("element symbol must be a non-empty string")
    stage = int(ion_stage)
    if not (_MIN_ION_STAGE <= stage <= _MAX_ION_STAGE):
        raise ValueError(
            f"ion_stage must be in [{_MIN_ION_STAGE}, {_MAX_ION_STAGE}] "
            f"(1=I neutral, 2=II, 3=III; Herrera p. 103 uses I-II lines, "
            f"III is needed to close the Saha chain), got {ion_stage}"
        )
    return elem, stage


def _validate_temperature(temperature_K: ArrayLike) -> NDArray[np.float64]:
    """Return temperatures as a 1-D float array, enforcing T > 0 and finite."""
    T = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
    if T.size == 0:
        raise ValueError("temperature_K must contain at least one value")
    if not np.all(np.isfinite(T)):
        raise ValueError("temperature_K must be finite")
    if np.any(T <= 0.0):
        raise ValueError("temperature_K must be > 0 K")
    return T


def partition_function_from_levels(
    g_values: ArrayLike,
    energies_ev: ArrayLike,
    temperature_K: ArrayLike,
) -> float | NDArray[np.float64]:
    r"""
    Direct Boltzmann summation of the partition function over bound
    levels.

    Implements the definition referenced by Herrera (2008), p. 260 —
    partition functions that "[take] into account all the bound
    quantum states of an atom or ion":

    $$
    U(T) \;=\; \sum_i g_i\, \exp\!\left(-\frac{E_i}{k_B T}\right)
    $$

    This is the same statistical sum that normalizes the Boltzmann
    level populations of Eq. 5-1, p. 98.

    Parameters
    ----------
    g_values : array_like of float, shape (n_levels,)
        Statistical weights $g_i = 2J_i + 1$ of each bound level
        (dimensionless, > 0).
    energies_ev : array_like of float, shape (n_levels,)
        Level energies $E_i$ above the ground state in eV (>= 0).
        Converted to Joules internally via `core.constants.EV`.
    temperature_K : array_like of float
        Temperature(s) in Kelvin (> 0).

    Returns
    -------
    float or ndarray
        $U(T)$, dimensionless. Scalar input returns a float; array
        input returns an array of the same shape.

    Notes
    -----
    - Exact for the level list supplied; the physical accuracy is
      limited by truncation of the level list (high-lying states
      matter at high $T$).
    - $E_i \gg k_B T$ terms underflow harmlessly to 0.0 in IEEE-754.
    - Low-$T$ limit: $U \to g_0$ of the $E = 0$ level(s), used as a
      validation identity.
    """
    g = np.asarray(g_values, dtype=np.float64)
    E_ev = np.asarray(energies_ev, dtype=np.float64)
    if g.ndim != 1 or E_ev.ndim != 1 or g.shape != E_ev.shape or g.size == 0:
        raise ValueError(
            "g_values and energies_ev must be non-empty 1-D arrays of equal length"
        )
    if np.any(g <= 0.0):
        raise ValueError("statistical weights g_i must be > 0")
    if np.any(E_ev < 0.0) or not np.all(np.isfinite(E_ev)):
        raise ValueError("level energies must be finite and >= 0 eV")

    # Flatten T so the (n_T, n_levels) outer broadcast below works for
    # scalar, 1-D, or N-D temperature input alike; reshape at return.
    T = _validate_temperature(temperature_K).ravel()
    scalar_input = np.ndim(temperature_K) == 0

    energies_J = E_ev * EV  # eV -> J (SI) via core constant
    # Broadcast: (n_T, 1) temperatures against (n_levels,) energies.
    exponent = -energies_J[np.newaxis, :] / (KB * T[:, np.newaxis])
    # Boltzmann-weighted degeneracy sum; E_i >> k_B*T terms underflow
    # harmlessly to 0 (they contribute nothing physically).
    U = np.sum(g[np.newaxis, :] * np.exp(exponent), axis=1)

    return float(U[0]) if scalar_input else U.reshape(np.shape(temperature_K))


@dataclass(frozen=True, eq=False)
class PartitionFunctionTable:
    """
    Tabulated partition function U(T) on a temperature grid, evaluated by
    linear interpolation.

    This is the primary evaluation strategy, mirroring Herrera's use of
    published tabulations (p. 133: Drawin & Felenbok [130], Halenka [169],
    Halenka & Madej [170]) for the U(T) factors of Eqs. 5-1/5-2 (p. 98),
    5-8 (pp. 103-104) and D-1 (p. 274).

    Parameters
    ----------
    temperatures_K : array_like of float, shape (n,)
        Strictly increasing temperature grid in Kelvin (> 0), n >= 2.
    values : array_like of float, shape (n,)
        Tabulated U(T) at each grid point (dimensionless, > 0).

    Notes
    -----
    - Linear interpolation preserves monotonicity and cannot overshoot;
      the interpolation error is bounded by the grid spacing of the source
      table (documented limitation — refine the grid, not the method).
    - Evaluation strictly outside [T_min, T_max] raises ValueError; use a
      `PartitionFunctionPolynomial` fallback via the provider instead of
      extrapolating.
    - Arrays are copied and marked read-only to honour immutability
      (development_rules.md). `eq=False`: identity comparison only.
    """

    temperatures_K: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        T = np.array(self.temperatures_K, dtype=np.float64, copy=True)
        U = np.array(self.values, dtype=np.float64, copy=True)
        if T.ndim != 1 or U.ndim != 1 or T.shape != U.shape:
            raise ValueError(
                "temperatures_K and values must be 1-D arrays of equal length"
            )
        if T.size < 2:
            raise ValueError("a partition function table needs >= 2 grid points")
        if not np.all(np.isfinite(T)) or not np.all(np.isfinite(U)):
            raise ValueError("table entries must be finite")
        if np.any(T <= 0.0):
            raise ValueError("grid temperatures must be > 0 K")
        if np.any(np.diff(T) <= 0.0):
            raise ValueError("temperatures_K must be strictly increasing")
        if np.any(U <= 0.0):
            raise ValueError("partition function values must be > 0")
        # Arrays were defensively copied above; locking the buffers makes
        # the frozen dataclass deeply immutable (development_rules.md).
        T.setflags(write=False)
        U.setflags(write=False)
        object.__setattr__(self, "temperatures_K", T)
        object.__setattr__(self, "values", U)

    @classmethod
    def from_levels(
        cls,
        g_values: ArrayLike,
        energies_ev: ArrayLike,
        temperature_grid_K: ArrayLike,
    ) -> "PartitionFunctionTable":
        """
        Build a table by direct Boltzmann summation over bound levels
        (Herrera p. 260 definition) on the given temperature grid.

        Convenient bridge from atomic level data (e.g. NIST levels parsed
        by the atomic layer) to the interpolation strategy.
        """
        grid = _validate_temperature(temperature_grid_K)
        U = partition_function_from_levels(g_values, energies_ev, grid)
        return cls(temperatures_K=grid, values=np.asarray(U))

    @property
    def temperature_range_K(self) -> Tuple[float, float]:
        """(T_min, T_max) of the tabulated grid in Kelvin."""
        return float(self.temperatures_K[0]), float(self.temperatures_K[-1])

    def covers(self, temperature_K: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask: which temperatures fall inside the grid range."""
        T = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
        lo, hi = self.temperature_range_K
        return (T >= lo) & (T <= hi)

    def __call__(self, temperature_K: ArrayLike) -> float | NDArray[np.float64]:
        """
        Interpolate U(T) at the requested temperature(s) in Kelvin.

        Raises
        ------
        ValueError
            If any temperature lies outside the tabulated range (no
            extrapolation — see module notes).
        """
        T = _validate_temperature(temperature_K)
        if not np.all(self.covers(T)):
            lo, hi = self.temperature_range_K
            raise ValueError(
                f"temperature outside tabulated range [{lo:.6g}, {hi:.6g}] K; "
                "register a polynomial fallback with the provider instead of "
                "extrapolating"
            )
        # Piecewise-linear interpolation: monotone between grid points,
        # cannot overshoot the tabulated values (module docs).
        U = np.interp(T, self.temperatures_K, self.values)
        scalar_input = np.ndim(temperature_K) == 0
        return float(U[0]) if scalar_input else U.reshape(np.shape(temperature_K))


@dataclass(frozen=True, eq=False)
class PartitionFunctionPolynomial:
    r"""
    Polynomial partition function fit in the Irwin (1981) form:

    $$
    \ln U(T) \;=\; \sum_{k=0}^{n} a_k\, (\ln T)^{k},
    \qquad T \text{ in Kelvin.}
    $$

    Fallback strategy for temperatures outside a tabulated grid. The
    dissertation sources $U(T)$ from tabulations (p. 133) without
    giving a functional form; the Irwin ln-ln polynomial is the
    standard compact representation in the astrophysics/plasma
    literature and is adopted here as a documented extension (see
    ai_instructions.md).

    Parameters
    ----------
    coefficients : tuple of float
        $(a_0, a_1, \ldots, a_n)$ in ascending power of $\ln T$.
    temperature_range_K : tuple of float
        $(T_{\min}, T_{\max})$ validity window of the fit in Kelvin.
        Irwin's original fits are valid for 1000-16000 K; supply the
        range quoted by whatever source the coefficients come from.

    Notes
    -----
    - Evaluation outside the validity window raises ValueError
      (partition functions diverge if a polynomial fit is
      extrapolated).
    - exp/ln formulation guarantees $U > 0$ for any real
      coefficients.
    """

    coefficients: Tuple[float, ...]
    temperature_range_K: Tuple[float, float]

    def __post_init__(self) -> None:
        coeffs = tuple(float(c) for c in self.coefficients)
        if len(coeffs) == 0:
            raise ValueError("coefficients must be non-empty")
        if not all(np.isfinite(coeffs)):
            raise ValueError("coefficients must be finite")
        lo, hi = (float(x) for x in self.temperature_range_K)
        if not (0.0 < lo < hi):
            raise ValueError(
                "temperature_range_K must satisfy 0 < T_min < T_max"
            )
        object.__setattr__(self, "coefficients", coeffs)
        object.__setattr__(self, "temperature_range_K", (lo, hi))

    def covers(self, temperature_K: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask: which temperatures fall inside the validity window."""
        T = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
        lo, hi = self.temperature_range_K
        return (T >= lo) & (T <= hi)

    def __call__(self, temperature_K: ArrayLike) -> float | NDArray[np.float64]:
        """
        Evaluate U(T) at the requested temperature(s) in Kelvin.

        Raises
        ------
        ValueError
            If any temperature lies outside the fit validity window.
        """
        T = _validate_temperature(temperature_K)
        if not np.all(self.covers(T)):
            lo, hi = self.temperature_range_K
            raise ValueError(
                f"temperature outside polynomial validity range "
                f"[{lo:.6g}, {hi:.6g}] K"
            )
        # Irwin form: ln U = sum_k a_k (ln T)^k; exponentiating guarantees
        # U > 0 for any real coefficients.
        ln_U = np.polynomial.polynomial.polyval(np.log(T), self.coefficients)
        U = np.exp(ln_U)
        scalar_input = np.ndim(temperature_K) == 0
        return float(U[0]) if scalar_input else U.reshape(np.shape(temperature_K))


@dataclass(frozen=True, eq=False)
class PartitionFunctionProvider:
    """
    Registry of partition functions per species (element, ion stage I-III),
    with table interpolation as the primary strategy and a polynomial
    fallback outside the tabulated range.

    This is the single U(T) entry point consumed by the Saha solver
    (Eq. 5-2 p. 98 / Eq. D-1 p. 274 need the U^II/U^I ratio), the Boltzmann
    populations (Eq. 5-1, p. 98) and the emission intensities (Eq. 5-8,
    pp. 103-104). Keeping the data source behind one interface preserves
    the element-agnostic physics layer required by the architecture: no
    element-specific numbers live in this module.

    Parameters
    ----------
    tables : mapping of (element, ion_stage) -> PartitionFunctionTable
        Primary tabulated data (e.g. digitized Drawin & Felenbok / Halenka
        tables, or tables built with `PartitionFunctionTable.from_levels`).
    polynomials : mapping of (element, ion_stage) -> PartitionFunctionPolynomial
        Optional fallbacks used where the table does not cover T.

    Notes
    -----
    - Element matching is case-insensitive ("ce" == "Ce"); ion stages
      follow atomic/parsers.py: 1=I, 2=II, 3=III.
    - The provider is immutable: `with_table` / `with_polynomial` return
      new providers (composition over mutation, development_rules.md).
    - Resolution order per temperature point: table (if in range), then
      polynomial (if in range), else ValueError. Deterministic and
      side-effect free.
    """

    tables: Mapping[SpeciesKey, PartitionFunctionTable] = field(
        default_factory=dict
    )
    polynomials: Mapping[SpeciesKey, PartitionFunctionPolynomial] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        norm_tables: dict[SpeciesKey, PartitionFunctionTable] = {}
        for (elem, stage), table in dict(self.tables).items():
            if not isinstance(table, PartitionFunctionTable):
                raise TypeError(
                    f"tables[{(elem, stage)!r}] must be a PartitionFunctionTable"
                )
            norm_tables[_species_key(elem, stage)] = table
        norm_polys: dict[SpeciesKey, PartitionFunctionPolynomial] = {}
        for (elem, stage), poly in dict(self.polynomials).items():
            if not isinstance(poly, PartitionFunctionPolynomial):
                raise TypeError(
                    f"polynomials[{(elem, stage)!r}] must be a "
                    "PartitionFunctionPolynomial"
                )
            norm_polys[_species_key(elem, stage)] = poly
        object.__setattr__(self, "tables", norm_tables)
        object.__setattr__(self, "polynomials", norm_polys)

    # --- Introspection -------------------------------------------------
    @property
    def species(self) -> list[SpeciesKey]:
        """Sorted list of all (ELEMENT, ion_stage) keys with any data."""
        return sorted(set(self.tables) | set(self.polynomials))

    def has_species(self, element: str, ion_stage: int) -> bool:
        """True if any partition function data exists for the species."""
        key = _species_key(element, ion_stage)
        return key in self.tables or key in self.polynomials

    # --- Immutable builders --------------------------------------------
    def with_table(
        self, element: str, ion_stage: int, table: PartitionFunctionTable
    ) -> "PartitionFunctionProvider":
        """Return a new provider with `table` registered for the species."""
        tables = dict(self.tables)
        tables[_species_key(element, ion_stage)] = table
        return PartitionFunctionProvider(tables=tables, polynomials=self.polynomials)

    def with_polynomial(
        self, element: str, ion_stage: int, polynomial: PartitionFunctionPolynomial
    ) -> "PartitionFunctionProvider":
        """Return a new provider with `polynomial` registered for the species."""
        polys = dict(self.polynomials)
        polys[_species_key(element, ion_stage)] = polynomial
        return PartitionFunctionProvider(tables=self.tables, polynomials=polys)

    # --- Evaluation ------------------------------------------------------
    def partition_function(
        self,
        element: str,
        ion_stage: int,
        temperature_K: ArrayLike,
    ) -> float | NDArray[np.float64]:
        """
        Evaluate U(T) for one species, dimensionless.

        Supplies the U^s(T) of Boltzmann Eq. 5-1 (p. 98) and intensity
        Eq. 5-8 (pp. 103-104), and the U^I / U^II / U^III factors of the
        Saha balance, Eq. 5-2 (p. 98) / Eq. D-1 (p. 274), of Herrera (2008).

        Parameters
        ----------
        element : str
            Element symbol (case-insensitive), e.g. "Ce", "Al".
        ion_stage : int
            1 = neutral (I), 2 = singly ionized (II), 3 = doubly ionized
            (III) — same convention as `Transition.ion_stage`.
        temperature_K : array_like of float
            Temperature(s) in Kelvin (> 0, finite).

        Returns
        -------
        float or ndarray
            U(T); scalar input returns a float, array input an array of
            the same shape.

        Raises
        ------
        KeyError
            If no data at all is registered for the species.
        ValueError
            If some temperature is covered by neither the table grid nor
            the polynomial validity window (no silent extrapolation).
        """
        key = _species_key(element, ion_stage)
        table = self.tables.get(key)
        poly = self.polynomials.get(key)
        if table is None and poly is None:
            raise KeyError(
                f"no partition function data registered for {key[0]} "
                f"(ion stage {key[1]}); available: {self.species}"
            )

        T = _validate_temperature(temperature_K)
        # NaN marks "not yet resolved"; each strategy below fills only
        # the slots it covers, so resolution order is table -> polynomial.
        U = np.full(T.shape, np.nan, dtype=np.float64)

        if table is not None:
            in_table = table.covers(T)
            if np.any(in_table):
                U[in_table] = np.interp(
                    T[in_table], table.temperatures_K, table.values
                )
        if poly is not None:
            # Polynomial fallback only for points the table left open.
            remaining = np.isnan(U) & poly.covers(T)
            if np.any(remaining):
                ln_U = np.polynomial.polynomial.polyval(
                    np.log(T[remaining]), poly.coefficients
                )
                U[remaining] = np.exp(ln_U)

        uncovered = np.isnan(U)
        if np.any(uncovered):
            parts = []
            if table is not None:
                lo, hi = table.temperature_range_K
                parts.append(f"table covers [{lo:.6g}, {hi:.6g}] K")
            if poly is not None:
                lo, hi = poly.temperature_range_K
                parts.append(f"polynomial covers [{lo:.6g}, {hi:.6g}] K")
            bad = T[uncovered]
            raise ValueError(
                f"U(T) for {key[0]} (ion stage {key[1]}) undefined at "
                f"T in [{bad.min():.6g}, {bad.max():.6g}] K; "
                + "; ".join(parts)
                + ". Extrapolation is disallowed (see module notes)."
            )

        scalar_input = np.ndim(temperature_K) == 0
        return float(U[0]) if scalar_input else U.reshape(np.shape(temperature_K))

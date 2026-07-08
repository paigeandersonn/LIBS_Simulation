"""
libssim.physics.boltzmann
=========================
Boltzmann level populations for LTE plasmas (Phase 2).

Physical Context (Herrera 2008)
-------------------------------
Under local thermodynamic equilibrium a single excitation temperature
T_exc governs how the atoms/ions of a species s distribute themselves
over their bound energy levels. The thesis states this as the Boltzmann
relationship, Eq. 5-1, p. 98:

    n_k^s = n_tot^s * g_k * exp(-E_k / (k_B * T_exc)) / U^s(T)

where n_k^s is the number density in upper level k, n_tot^s the total
number density of species s (one element in one ionization stage),
g_k the level degeneracy, E_k the level energy and U^s(T) the partition
function (symbols pp. 19-23; U(T) on p. 23).

The upper-level density n_k^s computed here is exactly the population
that feeds the integrated line intensity of Eq. 5-8, pp. 103-104
(I_ki = n_k^s * A_ki), implemented in emission.py. The linearized form of
the same physics is the Boltzmann-plot equation, Eq. 5-20, p. 109, used
by CF-LIBS to invert for T — a Phase 5 concern, cited here for
traceability only.

Units and Conventions
---------------------
- Level energies arrive in eV (as stored on `Transition`) and are
  converted to Joules via `core.constants.EV` before dividing by k_B*T
  (SI throughout, development_rules.md).
- Number densities are m^-3; temperatures are Kelvin.
- "Species" means one element in one ionization stage (CF-LIBS usage,
  p. 103): pass the neutral-stage density with a stage-I transition, the
  singly-ionized density with a stage-II transition, etc. (densities come
  from the Saha balance in saha.py).

Numerical Assumptions and Limitations
-------------------------------------
- Functions are pure and vectorized; no global state.
- exp underflow for E_k >> k_B*T flushes to 0.0 (correct physical limit).
- `boltzmann_population_fraction` uses an externally supplied U(T)
  (partition.py); it is exact relative to the true total density only in
  so far as U(T) accounts for all bound levels (Herrera p. 260).
- `level_population_fractions` self-normalizes over the supplied level
  list, so the fractions sum to 1.0 by construction (Phase 2 validation
  criterion) — but they refer to that truncated level set, not to a
  hypothetical complete one.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-1 p. 98;
Eq. 5-8 pp. 103-104; Eq. 5-20 p. 109; symbols pp. 19-23.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.constants import KB, EV
from ..atomic.transition import Transition
from .partition import partition_function_from_levels


def _validate_positive_temperature(temperature_K: ArrayLike) -> NDArray[np.float64]:
    """Temperatures as float array; enforce finite and > 0 K."""
    T = np.asarray(temperature_K, dtype=np.float64)
    if T.size == 0:
        raise ValueError("temperature_K must contain at least one value")
    if not np.all(np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError("temperature_K must be finite and > 0 K")
    return T


def boltzmann_population_fraction(
    g_k: ArrayLike,
    energy_ev: ArrayLike,
    partition_function: ArrayLike,
    temperature_K: ArrayLike,
) -> float | NDArray[np.float64]:
    """
    Fraction of a species residing in one bound level under LTE.

    Implements the Boltzmann relationship of Herrera (2008), Eq. 5-1,
    p. 98, divided through by the total species density:

        n_k^s / n_tot^s = g_k * exp(-E_k / (k_B * T)) / U^s(T)

    Parameters
    ----------
    g_k : array_like of float
        Statistical weight(s) g_k = 2J_k + 1 of the level(s) (> 0).
    energy_ev : array_like of float
        Level energy E_k above the species ground state in eV (>= 0).
        Converted to Joules internally via `core.constants.EV`.
    partition_function : array_like of float
        U^s(T), dimensionless (> 0), from
        `partition.PartitionFunctionProvider` evaluated at the same
        temperature(s).
    temperature_K : array_like of float
        Excitation temperature T_exc in Kelvin (> 0). Under LTE this is
        the single plasma temperature of `PlasmaState` (T_e = T_exc =
        T_ion; state.py docs).

    Returns
    -------
    float or ndarray
        Dimensionless population fraction(s); lies in [0, 1] whenever
        U(T) is consistent with (i.e. at least as large as the g_k
        Boltzmann term of) the supplied level. Inputs broadcast under
        normal NumPy rules (e.g. arrays of levels against a scalar
        temperature); all-scalar input returns a float.

    Notes
    -----
    The fraction is exact only if U^s(T) sums over all bound states
    (Herrera p. 260). Fractions over a *truncated* level list that must
    sum to exactly 1 are provided by `level_population_fractions`.
    """
    g = np.asarray(g_k, dtype=np.float64)
    E_ev = np.asarray(energy_ev, dtype=np.float64)
    U = np.asarray(partition_function, dtype=np.float64)
    T = _validate_positive_temperature(temperature_K)

    if np.any(g <= 0.0) or not np.all(np.isfinite(g)):
        raise ValueError("g_k must be finite and > 0")
    if np.any(E_ev < 0.0) or not np.all(np.isfinite(E_ev)):
        raise ValueError("energy_ev must be finite and >= 0")
    if np.any(U <= 0.0) or not np.all(np.isfinite(U)):
        raise ValueError("partition_function must be finite and > 0")

    energy_J = E_ev * EV  # eV -> J (SI) via core constant
    fraction = g * np.exp(-energy_J / (KB * T)) / U

    if fraction.ndim == 0:
        return float(fraction)
    return fraction


def upper_level_density(
    transition: Transition,
    species_density_m3: ArrayLike,
    partition_function: ArrayLike,
    temperature_K: ArrayLike,
) -> float | NDArray[np.float64]:
    """
    LTE number density of the upper level of a transition, n_k^s (m^-3).

    Direct application of Herrera (2008), Eq. 5-1, p. 98, using the
    transition's upper-level data (g_k = `g_upper`, E_k =
    `energy_upper_ev`):

        n_k^s = n_tot^s * g_k * exp(-E_k / (k_B * T)) / U^s(T)

    This n_k^s is the population that multiplies A_ki in the integrated
    line intensity, Eq. 5-8, pp. 103-104 (emission.py).

    Parameters
    ----------
    transition : Transition
        Atomic transition; supplies g_upper and energy_upper_ev (eV,
        converted to J internally).
    species_density_m3 : array_like of float
        n_tot^s in m^-3 (>= 0): density of the element *in the
        transition's ionization stage* (neutral density for stage-I
        lines, ion density for stage-II), as returned by the Saha
        balance (saha.py, Eqs. D-8/D-10, pp. 275-276).
    partition_function : array_like of float
        U^s(T) for the same species and temperature(s), dimensionless.
    temperature_K : array_like of float
        Temperature in Kelvin (> 0).

    Returns
    -------
    float or ndarray
        n_k^s in m^-3; broadcasts over the array inputs.
    """
    n_s = np.asarray(species_density_m3, dtype=np.float64)
    if np.any(n_s < 0.0) or not np.all(np.isfinite(n_s)):
        raise ValueError("species_density_m3 must be finite and >= 0")

    fraction = boltzmann_population_fraction(
        transition.g_upper,
        transition.energy_upper_ev,
        partition_function,
        temperature_K,
    )
    density = n_s * fraction
    if np.ndim(density) == 0:
        return float(density)
    return density


def level_population_fractions(
    g_values: ArrayLike,
    energies_ev: ArrayLike,
    temperature_K: float,
) -> NDArray[np.float64]:
    """
    Normalized Boltzmann fractions over a supplied level list.

    Same statistics as Eq. 5-1, p. 98 (Herrera 2008), but with U(T)
    computed by direct summation over exactly the levels supplied
    (`partition.partition_function_from_levels`, the p. 260 definition),
    so that the returned fractions sum to 1.0 by construction — the
    Phase 2 validation criterion "population fractions must sum to 1.0"
    (validation_strategy.md).

    Parameters
    ----------
    g_values : array_like of float, shape (n_levels,)
        Statistical weights of the levels (> 0).
    energies_ev : array_like of float, shape (n_levels,)
        Level energies in eV (>= 0); converted to J internally.
    temperature_K : float
        Temperature in Kelvin (> 0); scalar, one distribution at a time.

    Returns
    -------
    ndarray, shape (n_levels,)
        Population fractions n_k / n_tot, summing to 1.0 (within one
        floating-point rounding of the final normalization).

    Notes
    -----
    Truncation caveat: the fractions describe the supplied (finite) level
    set. Adding higher levels redistributes population, most visibly at
    high temperature.
    """
    if np.ndim(temperature_K) != 0:
        raise ValueError("temperature_K must be a scalar here (one distribution)")
    g = np.asarray(g_values, dtype=np.float64)
    E_ev = np.asarray(energies_ev, dtype=np.float64)

    # Validation (shapes, positivity) is delegated to the partition sum.
    U = partition_function_from_levels(g, E_ev, float(temperature_K))

    energy_J = E_ev * EV
    weights = g * np.exp(-energy_J / (KB * float(temperature_K)))
    return weights / U

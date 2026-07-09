"""
libssim.physics.saha
====================
Saha ionization balance with strict mass conservation (Phase 2).

Physical Context (Herrera 2008)
-------------------------------
Under LTE a single ionization temperature controls how each element j
splits between its neutral (I) and singly-ionized (II) stages. The thesis
gives the Saha equation twice, in identical physics:

- Eq. 5-2, p. 98 (CF-LIBS form):
      n_e * n^II / n^I
        = 2 * (2*pi*m_e*k_B*T_ion)^(3/2) / h^3
          * (U^II(T)/U^I(T)) * exp(-(chi - Delta_chi)/(k_B*T_ion))
- Eq. D-1, p. 274 (MC-LIBS form, Appendix D), defining the Saha function
      s^j(T) = n_i^j * n_e / n_a^j
             = 2 * (U_i^j/U_a^j) * (2*pi*m_e*k_B*T/h^2)^(3/2)
               * exp(-(chi_j - Delta_chi)/(k_B*T))
  where chi_j is the first ionization potential of constituent j and
  Delta_chi the lowering of the ionization potential due to the electric
  field of surrounding electrons (defined verbally on pp. 105 and 274).

Appendix D closes the system with the conservation constraints
(pp. 274-276):

- Eq. D-2, p. 274 : mass conservation per element,  n_i^j + n_a^j = n^j
- Eq. D-3, p. 274 : charge equilibrium,             sum_j n_i^j = n_e
- Eq. D-8, p. 275 : n_a^j = n^j * n_e / (s^j(T) + n_e)
- Eq. D-9, pp. 275-276 : n_e = sum_j n^j * s^j(T) / (s^j(T) + n_e),
  an implicit equation with "a unique positive solution"
- Eq. D-10, p. 276: n_i^j = n^j * s^j(T) / (s^j(T) + n_e)

LTE validity diagnostics from Chapter 5 are provided alongside:

- Eq. 5-3, p. 99  : McWhirter criterion
      n_e >> 1.6e12 * T_K^(1/2) * (Delta_E)^3   [cm^-3, Delta_E in eV]
- Eq. 5-16, p. 106: particles in the Debye sphere
      n_D = 1.72e9 * T_eV^(3/2) / n_e^(1/2)     [n_e in cm^-3]

Implementation Decisions (documented per development_rules.md)
--------------------------------------------------------------
- Two ionization stages (I <-> II), exactly as in Appendix D and the
  thesis plasma model ("atoms + singly-charged ions + electrons",
  state.py / App. B). Extension to stage III is deliberately out of
  scope here; partition.py already serves U up to III for when a
  three-stage chain is added.
- Strict mass conservation: the balance computes n_i^j from Eq. D-10 and
  then n_a^j = n^j - n_i^j from Eq. D-2, so n_a^j + n_i^j == n^j is exact
  in floating point (Phase 2 acceptance: relative error <= 1e-10; here
  it is identically 0).
- Delta_chi: the thesis defines the quantity (pp. 105, 274) but gives no
  formula. It therefore defaults to 0 and can be supplied explicitly.
  `ionization_potential_lowering_ev` offers the standard Debye-shielding
  estimate Delta_chi_z = z * e^2 / (4*pi*eps0*lambda_D) (Griem 1964;
  Drawin & Felenbok 1965 = thesis ref [130]) as a documented extension
  beyond the dissertation.
- Eq. D-9 is solved with `scipy.optimize.brentq` on
  g(n_e) = n_e - sum_j n^j s^j/(s^j + n_e); g is strictly increasing with
  g(0) < 0 < g(N_heavy), so the root is unique and bracketed — matching
  the thesis statement of a unique positive solution.

Units: strict SI internally (m^-3, K, J); ionization energies enter in
eV (atomic-data convention) and are converted via `core.constants.EV`.
The thesis' practical CGS forms (Eqs. 5-3, 5-16) are implemented in SI
from first principles and validated against the printed CGS coefficients
in the unit tests.

Numerical Assumptions and Limitations
-------------------------------------
- exp underflow of s^j(T) at very low T flushes to 0 -> fully neutral
  limit (correct physics; acceptance: neutral fraction -> 1 as T -> 0).
- n_e = 0 with s^j > 0 yields full ionization from Eqs. D-8/D-10; the
  0/0 case (s^j = 0 and n_e = 0) resolves to the neutral limit.
- Ions are assumed singly charged (z = 1) in charge equilibrium, as in
  Eq. D-3.

References
----------
Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
Spectroscopy. PhD Dissertation, University of Florida. Eq. 5-2 p. 98;
Eq. 5-3 p. 99; Eq. 5-16 p. 106; Eqs. D-1..D-10 pp. 274-276.
Griem, H.R. (1964). Plasma Spectroscopy. McGraw-Hill.
Drawin, H.W. & Felenbok, P. (1965). Data for Plasmas in Local
Thermodynamic Equilibrium. Gauthier-Villars, Paris. [thesis ref 130]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from ..core.constants import E as _ELEM_CHARGE
from ..core.constants import EPSILON0, EV, H, KB, ME
from ..core.state import PlasmaState
from .partition import PartitionFunctionProvider

# Relative tolerance for the Eq. D-9 root (well below the 1e-10
# conservation acceptance threshold; brentq converges superlinearly).
_NE_RTOL = 1e-14


def _validate_temperature(temperature_K: ArrayLike) -> NDArray[np.float64]:
    """Temperatures as float array; enforce finite and > 0 K."""
    T = np.asarray(temperature_K, dtype=np.float64)
    if not np.all(np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError("temperature_K must be finite and > 0 K")
    return T


def saha_factor(
    temperature_K: ArrayLike,
    u_neutral: ArrayLike,
    u_ion: ArrayLike,
    ionization_energy_ev: float,
    lowering_ev: float = 0.0,
) -> float | NDArray[np.float64]:
    """
    Saha function s(T) = n_i * n_e / n_a in m^-3.

    Implements Herrera (2008) Eq. D-1, p. 274 (identically Eq. 5-2,
    p. 98):

        s(T) = 2 * (U_II/U_I) * (2*pi*m_e*k_B*T / h^2)^(3/2)
                 * exp(-(chi - Delta_chi) / (k_B*T))

    Parameters
    ----------
    temperature_K : array_like of float
        Ionization temperature in Kelvin (> 0); under LTE equal to the
        single plasma temperature (p. 98).
    u_neutral, u_ion : array_like of float
        Partition functions U^I(T) and U^II(T) of the neutral and
        singly-ionized stages (dimensionless, > 0), e.g. from
        `partition.PartitionFunctionProvider`.
    ionization_energy_ev : float
        First ionization potential chi in eV (> 0), converted to Joules
        internally via `core.constants.EV`.
    lowering_ev : float, optional
        Lowering of the ionization potential Delta_chi in eV
        (0 <= Delta_chi < chi). The thesis defines the quantity
        (pp. 105, 274) without a formula; see
        `ionization_potential_lowering_ev` for the standard estimate.
        Default 0.

    Returns
    -------
    float or ndarray
        s(T) in m^-3. Scalar inputs return a float.

    Notes
    -----
    - At low T the exponential underflows to 0 (fully neutral limit).
    - s(T) is strictly increasing in T for fixed U-ratio, which drives
      the monotonic-ionization acceptance criterion.
    """
    T = _validate_temperature(temperature_K)
    U_a = np.asarray(u_neutral, dtype=np.float64)
    U_i = np.asarray(u_ion, dtype=np.float64)
    if np.any(U_a <= 0.0) or np.any(U_i <= 0.0):
        raise ValueError("partition functions must be > 0")
    chi = float(ionization_energy_ev)
    dchi = float(lowering_ev)
    if not (chi > 0.0 and np.isfinite(chi)):
        raise ValueError("ionization_energy_ev must be finite and > 0")
    if not (0.0 <= dchi < chi):
        raise ValueError(
            "lowering_ev must satisfy 0 <= Delta_chi < chi "
            "(effective ionization energy must stay positive)"
        )

    # Effective ionization energy chi - Delta_chi (Debye lowering), eV -> J.
    chi_eff_J = (chi - dchi) * EV
    # Electron translational phase-space factor (2*pi*m_e*k_B*T/h^2)^(3/2),
    # units m^-3 — the inverse cubed thermal de Broglie wavelength.
    thermal = (2.0 * np.pi * ME * KB * T / H**2) ** 1.5
    # Eq. D-1: leading 2 = electron spin degeneracy; s(T) carries m^-3.
    s = 2.0 * (U_i / U_a) * thermal * np.exp(-chi_eff_J / (KB * T))

    if np.ndim(s) == 0:
        return float(s)
    return s


def mcwhirter_minimum_electron_density_m3(
    temperature_K: ArrayLike,
    largest_gap_ev: float,
) -> float | NDArray[np.float64]:
    """
    Minimum electron density for LTE (McWhirter criterion), in m^-3.

    Herrera (2008), Eq. 5-3, p. 99 (after Thorne [25]):

        n_e >> 1.6e12 * T_K^(1/2) * (Delta_E)^3    [cm^-3]

    converted to SI (x 1e6 -> 1.6e18 m^-3), with Delta_E the largest
    transition energy gap considered, in eV.

    Returns the right-hand side; LTE requires the actual n_e to exceed
    it by a comfortable margin (">>", i.e. a necessary, not sufficient,
    condition — see implementation_plan.md Phase 2 gap check).
    """
    T = _validate_temperature(temperature_K)
    dE = float(largest_gap_ev)
    if not (dE > 0.0 and np.isfinite(dE)):
        raise ValueError("largest_gap_ev must be finite and > 0")
    # Thesis prints 1.6e12 cm^-3 (Eq. 5-3); x 1e6 converts to m^-3.
    n_min = 1.6e18 * np.sqrt(T) * dE**3
    if np.ndim(n_min) == 0:
        return float(n_min)
    return n_min


def debye_length_m(
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
) -> float | NDArray[np.float64]:
    """
    Electron Debye shielding length lambda_D in meters:

        lambda_D = sqrt(eps0 * k_B * T / (n_e * e^2))

    The thesis uses Debye shielding through the Debye-sphere count of
    Eq. 5-16, p. 106 (and symbols p. 21) without printing the length
    formula; this is the standard SI definition (Griem 1964) from which
    Eq. 5-16 follows — the unit tests verify that correspondence.
    """
    T = _validate_temperature(temperature_K)
    n_e = np.asarray(electron_density_m3, dtype=np.float64)
    if np.any(n_e <= 0.0) or not np.all(np.isfinite(n_e)):
        raise ValueError("electron_density_m3 must be finite and > 0")
    # Electron-only shielding: ion mobility neglected on the shielding
    # timescale (standard for LIBS diagnostics).
    lam = np.sqrt(EPSILON0 * KB * T / (n_e * _ELEM_CHARGE**2))
    if np.ndim(lam) == 0:
        return float(lam)
    return lam


def debye_sphere_particle_count(
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
) -> float | NDArray[np.float64]:
    """
    Number of electrons inside the Debye sphere, n_D (dimensionless).

    Herrera (2008), Eq. 5-16, p. 106:

        n_D = 1.72e9 * T_eV^(3/2) / n_e^(1/2)     [n_e in cm^-3]

    computed here in SI as n_D = (4/3)*pi*lambda_D^3*n_e, which
    reproduces the printed CGS coefficient. n_D >> 1 is required for
    collective (Debye-shielded) plasma behaviour and enters the quadratic
    Stark width, Eq. 5-15, p. 106.

    Note: the thesis symbols list labels n_D in cm^-3 (p. 21); it is a
    particle *count* and dimensionless — the documented ambiguity is
    resolved in favour of the count (development_rules.md).
    """
    T = _validate_temperature(temperature_K)
    n_e = np.asarray(electron_density_m3, dtype=np.float64)
    lam = debye_length_m(T, n_e)
    # Electron count in a sphere of radius lambda_D; evaluating this SI
    # form reproduces Eq. 5-16's CGS coefficient 1.72e9 (unit-tested).
    n_D = (4.0 / 3.0) * np.pi * np.asarray(lam) ** 3 * n_e
    if np.ndim(n_D) == 0:
        return float(n_D)
    return n_D


def ionization_potential_lowering_ev(
    temperature_K: ArrayLike,
    electron_density_m3: ArrayLike,
    charge_after: int = 1,
) -> float | NDArray[np.float64]:
    """
    Debye-shielding estimate of the ionization-potential lowering
    Delta_chi, in eV.

    The thesis uses Delta_chi in Eqs. 5-2 (p. 98), 5-13/5-14 (p. 105)
    and D-1 (p. 274) and defines it as the "lowering of the ionization
    potential of atoms due to the electric field of surrounding
    electrons" (p. 274) — but provides no formula. Following
    ai_instructions.md, the standard estimate of Griem (1964) /
    Drawin & Felenbok (1965, thesis ref [130]) is implemented:

        Delta_chi_z = z * e^2 / (4*pi*eps0*lambda_D)

    with z = `charge_after`, the charge of the ion *produced* by the
    ionization (1 for neutral -> singly ionized).

    Typical LIBS magnitude: ~0.05-0.2 eV at n_e ~ 1e22-1e24 m^-3 and
    T ~ 1e4 K — small against chi of several eV, which is why
    Delta_chi = 0 is an acceptable default in `saha_factor`.
    """
    z = int(charge_after)
    if z < 1:
        raise ValueError("charge_after must be >= 1")
    lam = np.asarray(
        debye_length_m(temperature_K, electron_density_m3), dtype=np.float64
    )
    # Coulomb energy of the freed charge z*e at the Debye radius — the
    # shielding energy no longer needed to escape (Griem 1964).
    dchi_J = z * _ELEM_CHARGE**2 / (4.0 * np.pi * EPSILON0 * lam)
    dchi_ev = dchi_J / EV
    if dchi_ev.ndim == 0:
        return float(dchi_ev)
    return dchi_ev


@dataclass(frozen=True)
class IonizationBalance:
    """
    Frozen result of a two-stage Saha balance at one plasma condition.

    Attributes
    ----------
    temperature_K : float
        Temperature at which the balance was evaluated (K).
    electron_density_m3 : float
        Electron density used in Eqs. D-8/D-10 (m^-3).
    neutral_density_m3 : Mapping[str, float]
        n_a^j per element (m^-3) — Eq. D-2 rearranged (n^j - n_i^j).
    ion_density_m3 : Mapping[str, float]
        n_i^j per element (m^-3) — Eq. D-10, p. 276.

    Notes
    -----
    Mass conservation n_a^j + n_i^j = n^j holds exactly by construction
    (see `SahaSolver.balance`). Charge equilibrium (Eq. D-3) holds only
    if n_e was obtained from `SahaSolver.solve_electron_density`.
    """

    temperature_K: float
    electron_density_m3: float
    neutral_density_m3: Mapping[str, float]
    ion_density_m3: Mapping[str, float]

    def __post_init__(self) -> None:
        if not (self.temperature_K > 0.0 and np.isfinite(self.temperature_K)):
            raise ValueError("temperature_K must be finite and > 0 K")
        if not (
            self.electron_density_m3 >= 0.0
            and np.isfinite(self.electron_density_m3)
        ):
            raise ValueError("electron_density_m3 must be finite and >= 0")
        neutral = dict(self.neutral_density_m3)
        ion = dict(self.ion_density_m3)
        if set(neutral) != set(ion):
            raise ValueError("neutral and ion mappings must share the same keys")
        for mapping, name in ((neutral, "neutral"), (ion, "ion")):
            for element, value in mapping.items():
                if not (value >= 0.0 and np.isfinite(value)):
                    raise ValueError(
                        f"{name} density for {element!r} must be finite and >= 0"
                    )
        object.__setattr__(self, "neutral_density_m3", neutral)
        object.__setattr__(self, "ion_density_m3", ion)

    @property
    def species(self) -> list[str]:
        """Element labels present in the balance."""
        return list(self.neutral_density_m3.keys())

    def elemental_density_m3(self, element: str) -> float:
        """Total elemental density n^j = n_a^j + n_i^j (Eq. D-2, p. 274)."""
        return (
            self.neutral_density_m3[element] + self.ion_density_m3[element]
        )

    def ionization_fraction(self, element: str) -> float:
        """n_i^j / n^j in [0, 1]; 0 for a vanishing element density."""
        total = self.elemental_density_m3(element)
        if total == 0.0:
            return 0.0
        return self.ion_density_m3[element] / total

    def neutral_fraction(self, element: str) -> float:
        """n_a^j / n^j in [0, 1]; 1 for a vanishing element density."""
        return 1.0 - self.ionization_fraction(element)

    @property
    def total_ion_density_m3(self) -> float:
        """sum_j n_i^j — left side of charge equilibrium Eq. D-3, p. 274."""
        return float(sum(self.ion_density_m3.values()))


@dataclass(frozen=True, eq=False)
class SahaSolver:
    """
    Two-stage (I <-> II) Saha ionization solver per Herrera (2008),
    Appendix D, pp. 274-276.

    Composition over inheritance: the solver owns a
    `PartitionFunctionProvider` (partition.py) and a table of first
    ionization potentials, and exposes pure evaluation methods.

    Parameters
    ----------
    partition_provider : PartitionFunctionProvider
        Source of U^I(T) and U^II(T) per element (Eq. D-1 numerator/
        denominator).
    ionization_energies_ev : Mapping[str, float]
        First ionization potential chi_j in eV per element label
        (case-insensitive match), e.g. {"Al": 5.9858, "Fe": 7.9024}
        (NIST values; data injected, never hardcoded here —
        architecture.md).
    lowering_ev : float, optional
        Constant Delta_chi in eV applied to every element (default 0;
        see `ionization_potential_lowering_ev` for an estimate). The
        thesis gives no prescription, so the simplest documented choice
        is a caller-controlled constant.
    """

    partition_provider: PartitionFunctionProvider
    ionization_energies_ev: Mapping[str, float] = field(default_factory=dict)
    lowering_ev: float = 0.0

    def __post_init__(self) -> None:
        energies = {
            str(k).strip().upper(): float(v)
            for k, v in dict(self.ionization_energies_ev).items()
        }
        for label, chi in energies.items():
            if not (chi > 0.0 and np.isfinite(chi)):
                raise ValueError(
                    f"ionization energy for {label!r} must be finite and > 0 eV"
                )
        dchi = float(self.lowering_ev)
        if dchi < 0.0 or not np.isfinite(dchi):
            raise ValueError("lowering_ev must be finite and >= 0")
        object.__setattr__(self, "ionization_energies_ev", energies)
        object.__setattr__(self, "lowering_ev", dchi)

    # ------------------------------------------------------------------
    def _chi_ev(self, element: str) -> float:
        key = str(element).strip().upper()
        try:
            return self.ionization_energies_ev[key]
        except KeyError:
            raise KeyError(
                f"no first ionization energy registered for {element!r}; "
                f"available: {sorted(self.ionization_energies_ev)}"
            ) from None

    def saha_factor(
        self, element: str, temperature_K: ArrayLike
    ) -> float | NDArray[np.float64]:
        """
        s^j(T) of Eq. D-1, p. 274, for one element, in m^-3.

        U^I and U^II are drawn from the partition provider (ion stages
        1 and 2), chi_j from the registered ionization energies.
        """
        U_a = self.partition_provider.partition_function(
            element, 1, temperature_K
        )
        U_i = self.partition_provider.partition_function(
            element, 2, temperature_K
        )
        return saha_factor(
            temperature_K,
            U_a,
            U_i,
            self._chi_ev(element),
            lowering_ev=self.lowering_ev,
        )

    # ------------------------------------------------------------------
    def balance(
        self,
        temperature_K: float,
        electron_density_m3: float,
        elemental_densities_m3: Mapping[str, float],
    ) -> IonizationBalance:
        """
        Split each element between neutral and ion stages at given
        (T, n_e), conserving elemental mass exactly.

        Implements Eqs. D-10 (p. 276) and D-2 (p. 274) of Herrera (2008):

            n_i^j = n^j * s^j(T) / (s^j(T) + n_e)      (D-10)
            n_a^j = n^j - n_i^j                        (D-2, rearranged)

        Using D-2 for the neutral density (instead of evaluating D-8
        independently) makes n_a^j + n_i^j == n^j exact in floating
        point — the Phase 2 acceptance criterion (<= 1e-10 relative
        error) is satisfied identically.

        Parameters
        ----------
        temperature_K : float
            Plasma temperature (K), > 0.
        electron_density_m3 : float
            Electron density (m^-3), >= 0 — e.g. `PlasmaState.n_e`
            (Stark-broadening measurement, Eqs. 5-17/5-19) or the
            self-consistent value from `solve_electron_density`.
        elemental_densities_m3 : Mapping[str, float]
            n^j per element (m^-3, >= 0): total density of atoms + ions
            of each element (thesis n_j, App. B/D).

        Returns
        -------
        IonizationBalance

        Notes
        -----
        Limits: s -> 0 (low T) gives the fully neutral plasma; n_e = 0
        with s > 0 gives full ionization; s = 0 and n_e = 0 resolves to
        neutral (the T -> 0 limit dominates).
        """
        T = float(temperature_K)
        _validate_temperature(T)
        n_e = float(electron_density_m3)
        if not (n_e >= 0.0 and np.isfinite(n_e)):
            raise ValueError("electron_density_m3 must be finite and >= 0")

        neutral: dict[str, float] = {}
        ion: dict[str, float] = {}
        for element, n_j in elemental_densities_m3.items():
            n_j = float(n_j)
            if not (n_j >= 0.0 and np.isfinite(n_j)):
                raise ValueError(
                    f"elemental density for {element!r} must be finite and >= 0"
                )
            s = float(self.saha_factor(element, T))
            if s == 0.0:  # exp underflow: fully neutral limit (also 0/0 guard)
                frac_ion = 0.0
            else:
                # s/(s+n_e) lies in (0, 1], so 0 <= n_i <= n_j always.
                frac_ion = s / (s + n_e)
            n_i = n_j * frac_ion          # Eq. D-10, p. 276
            # Subtracting instead of evaluating Eq. D-8 independently makes
            # n_a + n_i == n_j exact in floating point (acceptance <= 1e-10).
            n_a = n_j - n_i               # Eq. D-2, p. 274
            ion[element] = n_i
            neutral[element] = n_a

        return IonizationBalance(
            temperature_K=T,
            electron_density_m3=n_e,
            neutral_density_m3=neutral,
            ion_density_m3=ion,
        )

    # ------------------------------------------------------------------
    def solve_electron_density(
        self,
        temperature_K: float,
        elemental_densities_m3: Mapping[str, float],
    ) -> float:
        """
        Self-consistent electron density from charge equilibrium.

        Solves Eq. D-9, pp. 275-276 of Herrera (2008):

            n_e = sum_j n^j * s^j(T) / (s^j(T) + n_e)

        equivalent to the root of the strictly increasing function

            g(n_e) = n_e - sum_j n^j s^j / (s^j + n_e),

        which has g(0) <= 0 <= g(N_heavy) — the "unique positive
        solution" stated in the thesis. Solved with
        `scipy.optimize.brentq` (bracketing, guaranteed convergence,
        rtol = 1e-14).

        Returns
        -------
        float
            n_e in m^-3 (0 for a plasma cold enough that every s^j
            underflows to 0, i.e. fully neutral).

        Notes
        -----
        Two-stage model: at most one electron per heavy particle, so
        n_e <= sum_j n^j, which provides the upper bracket.
        """
        T = float(temperature_K)
        _validate_temperature(T)

        elements = list(elemental_densities_m3.keys())
        n_heavy = np.array(
            [float(elemental_densities_m3[el]) for el in elements], dtype=float
        )
        if np.any(n_heavy < 0.0) or not np.all(np.isfinite(n_heavy)):
            raise ValueError("elemental densities must be finite and >= 0")
        s = np.array([float(self.saha_factor(el, T)) for el in elements])

        total_heavy = float(np.sum(n_heavy))
        # If every s^j*n^j vanishes (empty plasma or full exp underflow),
        # Eq. D-9 has no positive root: the plasma is exactly neutral.
        if total_heavy == 0.0 or float(np.sum(s * n_heavy)) == 0.0:
            return 0.0

        def charge_imbalance(n_e: float) -> float:
            # sum written to be exact at n_e = 0 for s = 0 terms
            with np.errstate(invalid="ignore", divide="ignore"):
                terms = np.where(s > 0.0, n_heavy * s / (s + n_e), 0.0)
            return n_e - float(np.sum(terms))

        # g(0) = -sum_{s>0} n^j < 0; g(total_heavy) > 0 (s/(s+N) < 1).
        root: float = brentq(charge_imbalance, 0.0, total_heavy, rtol=_NE_RTOL, maxiter=200)  # type: ignore[assignment]  # scipy stubs mistype rtol/return
        return float(root)

    # ------------------------------------------------------------------
    def balance_from_state(self, state: PlasmaState) -> IonizationBalance:
        """
        Saha balance for a `PlasmaState`, taking T and n_e from the state.

        The per-element totals are derived from the state fields as

            n^j = C_j * (n_tot - n_e)

        i.e. composition fractions distribute the *heavy-particle*
        density, since `PlasmaState.total_density_m3` counts atoms +
        singly-charged ions + electrons (state.py, after App. B/D of the
        thesis) while the n^j of Eq. D-2 count atoms + ions only. This
        reading is a documented implementation decision.

        Returns
        -------
        IonizationBalance
            Mass conservation per element is exact; charge equilibrium
            holds only to the extent the state's n_e is Saha-consistent.
        """
        # PlasmaState's total counts atoms + ions + electrons, while the
        # n^j of Eq. D-2 count heavies only — remove the electrons before
        # splitting by composition fractions (documented decision).
        n_heavy = state.total_density_m3 - state.electron_density_m3
        elemental = {
            element: fraction * n_heavy
            for element, fraction in state.composition.items()
        }
        return self.balance(
            state.temperature_K, state.electron_density_m3, elemental
        )

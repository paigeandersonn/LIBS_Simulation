"""Frozen dataclass representing a single atomic spectral transition
(line).

Designed to be the core data structure returned by any AtomicDatabase
implementation. Follows the project's philosophy of immutability and
clear separation between data and physics.

Inspired by Herrera (2008) PhD thesis modeling approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Transition:
    """
    Immutable representation of an atomic transition.
    """

    element: str
    ion_stage: int
    wavelength_m: float
    energy_lower_ev: float
    energy_upper_ev: float
    a_ki: float
    g_lower: int
    g_upper: int

    # Optional fields for future use
    stark_width: Optional[float] = None
    empirical_intensity: Optional[float] = None
    lower_config: Optional[str] = None
    upper_config: Optional[str] = None

    def __post_init__(self):
        if self.wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        if self.a_ki < 0:
            raise ValueError("a_ki must be non-negative")
        if self.g_lower <= 0 or self.g_upper <= 0:
            raise ValueError("Statistical weights must be positive integers")
        if self.energy_upper_ev < self.energy_lower_ev:
            raise ValueError("energy_upper_ev must be >= energy_lower_ev")

    @property
    def wavelength_nm(self) -> float:
        from ..core.units import m_to_nm
        return m_to_nm(self.wavelength_m)

    @property
    def energy_diff_ev(self) -> float:
        return self.energy_upper_ev - self.energy_lower_ev

    def __repr__(self) -> str:
        return (
            f"Transition({self.element} {self.ion_stage}, "
            f"λ={self.wavelength_nm:.4f} nm, "
            f"A={self.a_ki:.2e} s⁻¹, "
            f"g={self.g_lower}→{self.g_upper})"
        )
"""
libssim.core.constants
======================
Fundamental physical constants in SI units.

These values are used throughout the LIBS forward model for:
- Energy conversions (eV <-> J, temperature in K)
- Planck's law for continuum radiation
- Saha ionization equilibrium
- Doppler broadening calculations
- etc.

Values are taken from CODATA 2018 / NIST recommended values (rounded
to match precision commonly used in spectroscopy literature, including
Herrera 2008 thesis pp. 20-25).

References
----------
- Herrera, K.K. (2008). ... LIST OF CONSTANTS (pp. 20)
  c, e, ε₀, h, k_B, m_e, N_A, σ
- NIST CODATA: https://physics.nist.gov/cuu/Constants/

All constants are module-level floats (immutable by convention).
"""

from __future__ import annotations

# Speed of light in vacuum
C: float = 2.99792458e8          # m s^{-1}

# Planck's constant
H: float = 6.62607015e-34        # J s

# Boltzmann constant
KB: float = 1.380649e-23         # J K^{-1}

# Electron mass
ME: float = 9.1093837015e-31     # kg

# Elementary charge
E: float = 1.602176634e-19       # C

# Vacuum permittivity
EPSILON0: float = 8.8541878128e-12  # F m^{-1} = C² N^{-1} m^{-2}

# Avogadro's number
NA: float = 6.02214076e23        # mol^{-1}

# Stefan-Boltzmann constant
SIGMA: float = 5.670374419e-8    # W m^{-2} K^{-4}

# Additional useful constants for LIBS spectroscopy
HBAR: float = H / (2 * 3.141592653589793)   # J s
EV: float = E                    # 1 eV = 1.602176634e-19 J
KB_EV: float = KB / E            # ≈ 8.617333262145e-5 eV K^{-1}
C_NM_PS: float = C * 1e9 / 1e12  # 0.299792458 nm/ps
ALPHA: float = 7.2973525693e-3

def _get_constants_dict() -> dict[str, float]:
    """Return a dictionary of all defined constants (for introspection / logging)."""
    return {
        "C": C,
        "H": H,
        "KB": KB,
        "ME": ME,
        "E": E,
        "EPSILON0": EPSILON0,
        "NA": NA,
        "SIGMA": SIGMA,
        "HBAR": HBAR,
        "EV": EV,
        "KB_EV": KB_EV,
        "C_NM_PS": C_NM_PS,
        "ALPHA": ALPHA,
    }
"""
libssim.core.constants
======================
Fundamental physical constants and derived conversion factors used
throughout the LIBS forward model.

These constants are centralized in one module to ensure consistency
across all calculations involving plasma thermodynamics, atomic
populations, radiative processes, and spectroscopic conversions.

Constants Glossary
------------------
The following table provides the symbol, value, and scientific
context for each constant used in the model.

| Constant     | Symbol   | Value                  | Description |
|--------------|----------|------------------------|-----------|
| `C`          | c        | `2.99792458e8`         | Speed of light in vacuum (m s⁻¹). Fundamental constant that relates wavelength and frequency of electromagnetic radiation. Used for Doppler broadening calculations and photon energy conversions. |
| `H`          | h        | `6.62607015e-34`       | Planck's constant (J s). Defines the relationship between the energy and frequency of a photon (E = h * frequency). Essential for calculating photon energies from observed wavelengths. |
| `KB`         | k_B      | `1.380649e-23`         | Boltzmann constant (J K⁻¹). Relates macroscopic temperature to microscopic thermal energy. Central to Saha-Boltzmann equilibrium calculations for level populations and ionization balance. |
| `ME`         | m_e      | `9.1093837015e-31`     | Rest mass of the electron (kg). Important in plasma physics for calculating electron thermal velocities and certain broadening mechanisms. |
| `E`          | e        | `1.602176634e-19`      | Elementary charge (C). Smallest unit of electric charge. Primarily used here as the conversion factor between Joules and electronvolts. |
| `EPSILON0`   | ε₀       | `8.8541878128e-12`     | Vacuum permittivity (F m⁻¹). Measures the electric polarizability of free space. Appears in expressions for plasma frequency and Debye shielding length. |
| `NA`         | N_A      | `6.02214076e23`        | Avogadro's number (mol⁻¹). Number of constituent particles in one mole. Used when converting between atomic number densities and molar composition. |
| `SIGMA`      | σ        | `5.670374419e-8`       | Stefan-Boltzmann constant (W m⁻² K⁻⁴). Governs the total power radiated by a blackbody. Relevant for modeling continuum emission from the plasma. |
| `HBAR`       | ℏ        | `H / (2π)`             | Reduced Planck's constant (J s). Commonly used in quantum mechanical expressions involving angular momentum and angular frequency. |
| `EV`         | —        | `1.602176634e-19`      | Conversion factor: energy of 1 electronvolt in Joules. Provides a convenient bridge between SI energy units and the electronvolt scale standard in atomic physics. |
| `KB_EV`      | k_B      | `KB / E`               | Boltzmann constant expressed in eV K⁻¹ (≈ 8.617333 × 10⁻⁵). Widely used in plasma spectroscopy because temperatures are frequently expressed in electronvolts. |
| `C_NM_PS`    | —        | `C × 1e9 / 1e12`       | Speed of light expressed in nm ps⁻¹ (≈ 0.299792458). Practical unit for time-resolved spectroscopy and estimating light transit times across the plasma volume. |
| `ALPHA`      | α        | `7.2973525693e-3`      | Fine-structure constant (dimensionless). Characterizes the strength of the electromagnetic interaction. Appears in higher-order corrections in atomic structure calculations. |
Notes
-----
- All values are taken from CODATA 2018 / NIST recommendations.
- The module also provides the helper function `_get_constants_dict()` for
  programmatic access to all constants.
- These constants are referenced in Herrera (2008), particularly in the
  discussion of plasma thermodynamics and atomic modeling (pp. 20–25).

References
----------
- Herrera, K.K. (2008). From Sample to Signal in Laser-Induced Breakdown
  Spectroscopy. PhD Dissertation, University of Florida.
- NIST CODATA: https://physics.nist.gov/cuu/Constants/
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
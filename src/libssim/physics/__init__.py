"""
libssim.physics
===============
Local Thermodynamic Equilibrium (LTE) physics modules (Phase 2).

Calculates populations, line shapes, and emission/absorption for a
single, uniform point of plasma, per Herrera (2008) Ch. 5 and
Appendix D. All equations are traceable to the dissertation via the
equation/page citations in each module.

Provides:
- Partition functions U(T): tabulated interpolation with polynomial
  fallback (`partition`)
- Saha ionization balance with exact mass conservation and LTE validity
  diagnostics (`saha`)
- Boltzmann level populations (`boltzmann`)
- Line emission / bound-bound absorption and blackbody radiance
  (`emission`)
- Doppler/Stark widths and normalized Gaussian/Lorentzian/Voigt
  profiles (`line_profiles`)
- Free-free and free-bound continuum (`continuum`)

Strict SI units throughout (development_rules.md); atomic data enter in
their native units (eV, tabulated Stark widths) and are converted via
`libssim.core.constants` / `libssim.core.units`.
"""

# Partition functions
from .partition import (
    PartitionFunctionProvider,
    PartitionFunctionTable,
    PartitionFunctionPolynomial,
    partition_function_from_levels,
)

# Saha ionization + LTE validity diagnostics
from .saha import (
    SahaSolver,
    IonizationBalance,
    saha_factor,
    mcwhirter_minimum_electron_density_m3,
    debye_length_m,
    debye_sphere_particle_count,
    ionization_potential_lowering_ev,
)

# Boltzmann populations (functions, not a class)
from .boltzmann import (
    boltzmann_population_fraction,
    upper_level_density,
    level_population_fractions,
)

# Emission & absorption (functions, not a class)
from .emission import (
    transition_frequency_hz,
    line_photon_emission_rate,
    line_power_density,
    line_emission_coefficient,
    line_absorption_coefficient,
    einstein_b_lu,
    blackbody_spectral_radiance_hz,
    blackbody_spectral_radiance_wavelength,
)

# Line widths & normalized profiles (functions, not a class)
from .line_profiles import (
    doppler_fwhm_m,
    doppler_fwhm_practical_m,
    stark_fwhm_m,
    stark_shift_m,
    fwhm_wavelength_to_frequency,
    fwhm_frequency_to_wavelength,
    voigt_profile_hz,
    voigt_profile_wavelength_m,
    gaussian_profile_hz,
    lorentzian_profile_hz,
    stark_shifted_voigt_reduced,
    damping_parameter,
    lorentzian_fwhm_from_voigt,
    voigt_fwhm_estimate,
)

# Continuum (functions, not a class)
from .continuum import (
    free_free_absorption_coefficient,
    free_bound_absorption_coefficient,
    continuum_absorption_coefficient,
    continuum_emission_coefficient,
    planck_mean_absorption_coefficient,
)

__all__ = [
    # partition
    "PartitionFunctionProvider",
    "PartitionFunctionTable",
    "PartitionFunctionPolynomial",
    "partition_function_from_levels",
    # saha
    "SahaSolver",
    "IonizationBalance",
    "saha_factor",
    "mcwhirter_minimum_electron_density_m3",
    "debye_length_m",
    "debye_sphere_particle_count",
    "ionization_potential_lowering_ev",
    # boltzmann
    "boltzmann_population_fraction",
    "upper_level_density",
    "level_population_fractions",
    # emission
    "transition_frequency_hz",
    "line_photon_emission_rate",
    "line_power_density",
    "line_emission_coefficient",
    "line_absorption_coefficient",
    "einstein_b_lu",
    "blackbody_spectral_radiance_hz",
    "blackbody_spectral_radiance_wavelength",
    # line_profiles
    "doppler_fwhm_m",
    "doppler_fwhm_practical_m",
    "stark_fwhm_m",
    "stark_shift_m",
    "fwhm_wavelength_to_frequency",
    "fwhm_frequency_to_wavelength",
    "voigt_profile_hz",
    "voigt_profile_wavelength_m",
    "gaussian_profile_hz",
    "lorentzian_profile_hz",
    "stark_shifted_voigt_reduced",
    "damping_parameter",
    "lorentzian_fwhm_from_voigt",
    "voigt_fwhm_estimate",
    # continuum
    "free_free_absorption_coefficient",
    "free_bound_absorption_coefficient",
    "continuum_absorption_coefficient",
    "continuum_emission_coefficient",
    "planck_mean_absorption_coefficient",
]

"""
libssim.physics
===============

Local Thermodynamic Equilibrium (LTE) physics modules.
"""

# Partition functions
from .partition import (
    PartitionFunctionProvider,
    PartitionFunctionTable,
    PartitionFunctionPolynomial,
    partition_function_from_levels,
)

# Saha ionization
from .saha import (
    SahaSolver,
    IonizationBalance,
)

# Boltzmann populations (functions, not a class)
from .boltzmann import (
    boltzmann_population_fraction,
    upper_level_density,
    level_population_fractions,
)

# Emission & absorption
# from .emission import EmissionCoefficients     # ← commented out for now

# Line profiles
# from .line_profiles import LineProfile         # ← commented out for now

# Continuum
# from .continuum import ContinuumEmission       # ← commented out for now

__all__ = [
    "PartitionFunctionProvider",
    "PartitionFunctionTable",
    "PartitionFunctionPolynomial",
    "partition_function_from_levels",
    "SahaSolver",
    "IonizationBalance",
    "boltzmann_population_fraction",
    "upper_level_density",
    "level_population_fractions",
]
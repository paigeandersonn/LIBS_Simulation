"""
libssim.validation
==================
Validation workflow (Phase 4): compare synthetic spectra against
experimental LIBS measurements for well-documented elements.

Provides:
- `sodium_setup` / `aluminum_setup` — NIST-transcribed atomic data
  bundled as `ElementSetup` (`atomic_data`)
- preprocessing: `crop`, `subtract_background`, `normalize`
- metrics: `compute_metrics`, `ValidationMetrics`, `PeakMatch`,
  `find_peak_positions`, `match_peaks`, `intensity_ratio`,
  `rms_residual` — primary metric R of Eq. 5-56, p. 122
- workflow: `PlasmaConditions`, `InstrumentSettings`, `ValidationCase`,
  `ValidationResult`, `surrogate_experiment`

Worked examples: examples/validate_sodium.py and
examples/validate_aluminum.py; real-data guidance in
data/experimental/README.md.
"""

from .atomic_data import (
    ALUMINUM_DATA,
    SODIUM_DATA,
    ElementAtomicData,
    ElementSetup,
    aluminum_setup,
    build_setup,
    fit_irwin_polynomial,
    hydrogenic_tail_levels,
    load_levels_csv,
    sodium_setup,
)
from .metrics import (
    PeakMatch,
    ValidationMetrics,
    compute_metrics,
    find_peak_positions,
    intensity_ratio,
    match_peaks,
    rms_residual,
)
from .preprocessing import crop, normalize, subtract_background
from .workflow import (
    InstrumentSettings,
    NoiseStudy,
    PlasmaConditions,
    ValidationCase,
    ValidationResult,
    noise_study,
    surrogate_experiment,
)

__all__ = [
    "ElementSetup",
    "ElementAtomicData",
    "SODIUM_DATA",
    "ALUMINUM_DATA",
    "sodium_setup",
    "aluminum_setup",
    "build_setup",
    "load_levels_csv",
    "fit_irwin_polynomial",
    "hydrogenic_tail_levels",
    "crop",
    "subtract_background",
    "normalize",
    "PeakMatch",
    "ValidationMetrics",
    "compute_metrics",
    "find_peak_positions",
    "match_peaks",
    "intensity_ratio",
    "rms_residual",
    "PlasmaConditions",
    "InstrumentSettings",
    "ValidationCase",
    "ValidationResult",
    "surrogate_experiment",
    "NoiseStudy",
    "noise_study",
]

"""Reusable simulate-and-compare pipeline (Phase 4 validation
workflow).

Composes the existing layers without modification:

    ElementSetup (validation.atomic_data)
      + PlasmaConditions  ->  UniformPlasmaEvolution + GateIntegrator
      + InstrumentSettings -> InstrumentResponse
      = ValidationCase.synthetic_*()

    experimental Spectrum -> preprocessing -> resample -> normalize
      -> metrics (R of Eq. 5-56 + peaks + rms) = ValidationResult

The uniform single-zone evolution is the deliberate starting point for
Na/Al validation (Phase 4 plan); swap `ValidationCase.build_evolution`
output for an `ExpandingOnionEvolution`/`CustomEvolution` through
subclassing or the `evolution_override` hook when spatial structure is
needed.

When no measured spectrum is available, `surrogate_experiment`
generates a clearly-labeled synthetic stand-in (full pipeline + noise)
so the workflow can be exercised end-to-end; its metadata carries
``surrogate=True`` and every report states it. A surrogate validates
the *workflow*, never the *physics* — drop real lab CSVs into
data/experimental/ for that (see data/experimental/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..analysis import resample
from ..core.spectrum import Spectrum
from ..instrument.noise import NoiseModel
from ..instrument.optics import CollectionOptics
from ..instrument.response import InstrumentResponse
from ..instrument.spectrometer import InstrumentalProfile
from ..temporal.base import PlasmaEvolution, TimeProfile
from ..temporal.decay_models import Constant, UniformPlasmaEvolution
from ..temporal.integrator import GateIntegrator
from ..transport.emissivity import LTESpectralModel
from .atomic_data import ElementSetup
from .metrics import ValidationMetrics, compute_metrics
from .preprocessing import crop, normalize, subtract_background

#: Scalars are promoted to Constant profiles for convenience.
ProfileLike = Union[TimeProfile, float]


def _as_profile(value: ProfileLike) -> TimeProfile:
    if callable(value):
        return value
    return Constant(float(value))


@dataclass(frozen=True, eq=False)
class PlasmaConditions:
    """
    Physical conditions of one validation run (literature-sourced).

    Parameters
    ----------
    temperature_K : TimeProfile or float
        T(t); floats mean a constant plasma.
    heavy_density_m3 : TimeProfile or float
        Total heavy (atom + ion) density of the emitting element(s).
    electron_density_m3 : TimeProfile or float, optional
        Prescribed n_e(t) — the usual case when literature reports
        Stark-based measurements. If None, n_e is Saha-closed from the
        element itself (only sensible for single-element plasmas).
    composition : Mapping[str, float], optional
        Elemental fractions; defaults to the setup's pure element.
    radius_m : float
        Plasma radius (m); thesis-typical 1-5 mm (p. 116).
    gate_delay_s, gate_width_s : float
        Detector gate (pp. 46-47).
    n_time_nodes : int
        Gate quadrature nodes (default 8).
    """

    temperature_K: ProfileLike
    heavy_density_m3: ProfileLike
    electron_density_m3: Optional[ProfileLike] = None
    composition: Optional[Mapping[str, float]] = None
    radius_m: float = 1.5e-3
    gate_delay_s: float = 1.0e-6
    gate_width_s: float = 1.0e-6
    n_time_nodes: int = 8


@dataclass(frozen=True, eq=False)
class InstrumentSettings:
    """
    Instrument description of one validation run.

    Defaults model a typical 0.5 m Czerny-Turner + ICCD LIBS setup;
    override from the experiment's reported parameters.
    """

    reciprocal_dispersion_nm_per_mm: float = 1.6
    slit_width_um: float = 50.0
    lsf_shape: str = "gaussian"
    aberration_fwhm_m: float = 0.0
    #: Lorentzian FWHM (nm) of a "voigt" LSF — from an
    #: instrument-function calibration (spectrometer module notes).
    lsf_lorentzian_fwhm_nm: float = 0.0
    absolute_factor: float = 1.0
    read_noise_rms_counts: float = 0.0
    dark_mean_counts: float = 0.0
    n_pixels: Optional[int] = None

    def instrumental_profile(self) -> InstrumentalProfile:
        return InstrumentalProfile(
            reciprocal_dispersion_nm_per_mm=(
                self.reciprocal_dispersion_nm_per_mm
            ),
            slit_width_um=self.slit_width_um,
            aberration_fwhm_m=self.aberration_fwhm_m,
            shape=self.lsf_shape,
            lorentzian_fwhm_m=self.lsf_lorentzian_fwhm_nm * 1.0e-9,
        )

    def response(self, include_noise: bool = True) -> InstrumentResponse:
        noise: Optional[NoiseModel] = None
        if include_noise and (
            self.read_noise_rms_counts > 0 or self.dark_mean_counts > 0
        ):
            noise = NoiseModel(
                read_noise_rms_counts=self.read_noise_rms_counts,
                dark_mean_counts=self.dark_mean_counts,
            )
        elif include_noise:
            noise = NoiseModel()  # shot noise only
        return InstrumentResponse(
            instrumental_profile=self.instrumental_profile(),
            collection_optics=CollectionOptics(
                absolute_factor=self.absolute_factor
            ),
            n_pixels=self.n_pixels,
            noise_model=noise,
        )


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of one synthetic-vs-experimental comparison."""

    case_name: str
    synthetic: Spectrum
    experimental: Spectrum
    metrics: ValidationMetrics
    is_surrogate: bool

    def report(self) -> str:
        """Formatted validation report (print or write to file)."""
        header = f"Validation report: {self.case_name}"
        lines = [header, "=" * len(header)]
        if self.is_surrogate:
            lines.append(
                "NOTE: experimental data is a SYNTHETIC SURROGATE — this "
                "run validates the workflow, not the physics."
            )
        lines.append(self.metrics.summary())
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class ValidationCase:
    """
    One element + conditions + instrument on a fixed wavelength grid.

    Parameters
    ----------
    name : str
        Report label, e.g. "Na I D doublet".
    setup : ElementSetup
        From `validation.atomic_data`.
    conditions : PlasmaConditions
    instrument : InstrumentSettings
    wavelength_m : ndarray
        Uniform, strictly increasing model grid (m) — uniformity is
        required by the LSF convolution.
    include_continuum : bool
        Free-free + free-bound continuum in the model (default True).
    n_impact : int
        Disk-integration nodes (observation is spatially integrated,
        matching typical LIBS collection).
    evolution_override : PlasmaEvolution, optional
        Advanced hook: replace the default uniform evolution (e.g. an
        `ExpandingOnionEvolution`) while keeping the rest of the
        pipeline.
    """

    name: str
    setup: ElementSetup
    conditions: PlasmaConditions
    instrument: InstrumentSettings
    wavelength_m: NDArray[np.float64]
    include_continuum: bool = True
    n_impact: int = 32
    evolution_override: Optional[PlasmaEvolution] = None

    def __post_init__(self) -> None:
        grid = np.array(self.wavelength_m, dtype=np.float64, copy=True)
        if grid.ndim != 1 or grid.size < 8:
            raise ValueError("wavelength_m must be a 1-D grid with >= 8 points")
        steps = np.diff(grid)
        if np.any(steps <= 0):
            raise ValueError("wavelength_m must be strictly increasing")
        if np.max(np.abs(steps - steps.mean())) > 1e-6 * steps.mean():
            raise ValueError(
                "wavelength_m must be uniform (required by the LSF "
                "convolution)"
            )
        grid.setflags(write=False)
        object.__setattr__(self, "wavelength_m", grid)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """
        Formatted table of every input condition — printed by the
        examples so each comparison documents its parameters
        (validation plan requirement).
        """
        conditions = self.conditions
        instrument = self.instrument
        profile = instrument.instrumental_profile()
        grid = self.wavelength_m

        def show(value: object) -> str:
            if callable(value) and not isinstance(value, (int, float)):
                return repr(value)
            return f"{float(value):.6g}"  # type: ignore[arg-type]

        n_e = conditions.electron_density_m3
        rows = [
            ("case", self.name),
            ("element", self.setup.element),
            (
                "lines (nm)",
                ", ".join(
                    f"{w * 1e9:.3f}" for w in self.setup.line_wavelengths_m
                ),
            ),
            ("T(t) [K]", show(conditions.temperature_K)),
            ("heavy density(t) [m^-3]", show(conditions.heavy_density_m3)),
            (
                "n_e(t) [m^-3]",
                "Saha closure (Eq. 5-38)" if n_e is None else show(n_e),
            ),
            ("plasma radius [m]", f"{conditions.radius_m:.3g}"),
            ("gate delay [s]", f"{conditions.gate_delay_s:.3g}"),
            ("gate width [s]", f"{conditions.gate_width_s:.3g}"),
            ("gate time nodes", str(conditions.n_time_nodes)),
            ("slit width [um]", f"{instrument.slit_width_um:.3g}"),
            (
                "dispersion [nm/mm]",
                f"{instrument.reciprocal_dispersion_nm_per_mm:.3g}",
            ),
            (
                "instrumental FWHM [nm]",
                f"{profile.fwhm_m * 1e9:.4f} ({profile.shape})",
            ),
            (
                "wavelength grid",
                f"{grid[0] * 1e9:.2f}-{grid[-1] * 1e9:.2f} nm, "
                f"{grid.size} pts",
            ),
            ("continuum included", str(self.include_continuum)),
        ]
        width = max(len(key) for key, _ in rows)
        return "\n".join(f"{key.ljust(width)} : {val}" for key, val in rows)

    # ------------------------------------------------------------------
    def build_evolution(self) -> PlasmaEvolution:
        if self.evolution_override is not None:
            return self.evolution_override
        conditions = self.conditions
        composition = conditions.composition or {self.setup.element: 1.0}
        if conditions.electron_density_m3 is not None:
            return UniformPlasmaEvolution(
                temperature_K=_as_profile(conditions.temperature_K),
                heavy_density_m3=_as_profile(conditions.heavy_density_m3),
                composition=composition,
                radius_m=conditions.radius_m,
                electron_density_m3=_as_profile(
                    conditions.electron_density_m3
                ),
            )
        return UniformPlasmaEvolution(
            temperature_K=_as_profile(conditions.temperature_K),
            heavy_density_m3=_as_profile(conditions.heavy_density_m3),
            composition=composition,
            radius_m=conditions.radius_m,
            saha_solver=self.setup.saha_solver,
        )

    def build_integrator(self) -> GateIntegrator:
        model = LTESpectralModel(
            saha_solver=self.setup.saha_solver,
            wavelength_m=self.wavelength_m,
            transitions=self.setup.transitions,
            atomic_masses_kg=self.setup.atomic_masses_kg,
            include_continuum=self.include_continuum,
        )
        return GateIntegrator(
            spectral_model=model,
            evolution=self.build_evolution(),
            impact_parameter_m=None,  # disk-integrated observation
            n_impact=self.n_impact,
        )

    # ------------------------------------------------------------------
    def _gate_spectrum(self) -> Spectrum:
        conditions = self.conditions
        return self.build_integrator().gate_integrated(
            conditions.gate_delay_s,
            conditions.gate_width_s,
            n_time_nodes=conditions.n_time_nodes,
        )

    def synthetic_noise_free(self) -> Spectrum:
        """Physics + resolution + efficiency, deterministic (no noise)."""
        return self.instrument.response(include_noise=False).noise_free(
            self._gate_spectrum()
        )

    def synthetic_noisy(self, seed: int) -> Spectrum:
        """Full pipeline including seeded detector noise."""
        return self.instrument.response(include_noise=True).apply(
            self._gate_spectrum(), seed=seed
        )

    # ------------------------------------------------------------------
    def validate(
        self,
        experimental: Spectrum,
        background_windows: Optional[Sequence[Tuple[float, float]]] = None,
        background_fit: str = "constant",
        normalize_mode: str = "peak",
        peak_tolerance_m: float = 5.0e-11,
    ) -> ValidationResult:
        """
        Compare the noise-free synthetic spectrum against an
        experimental one.

        Pipeline: crop experimental to the model window -> optional
        background subtraction -> resample onto the model grid ->
        normalize both -> metrics (R of Eq. 5-56 + rms + peak matching
        against the setup's line list).
        """
        processed = crop(
            experimental,
            float(self.wavelength_m[0]),
            float(self.wavelength_m[-1]),
        )
        if background_windows:
            processed = subtract_background(
                processed, background_windows, fit=background_fit
            )
        processed = resample(processed, self.wavelength_m)
        processed = normalize(processed, normalize_mode)

        synthetic = normalize(self.synthetic_noise_free(), normalize_mode)

        metrics = compute_metrics(
            synthetic,
            processed,
            expected_lines_m=self.setup.line_wavelengths_m,
            peak_tolerance_m=peak_tolerance_m,
        )
        return ValidationResult(
            case_name=self.name,
            synthetic=synthetic,
            experimental=processed,
            metrics=metrics,
            is_surrogate=bool(experimental.metadata.get("surrogate", False)),
        )


def surrogate_experiment(
    case: ValidationCase,
    seed: int,
    peak_counts: float = 5.0e4,
    read_noise_rms_counts: Optional[float] = None,
    dark_mean_counts: Optional[float] = None,
    background_mean_counts: float = 0.0,
) -> Spectrum:
    """
    Synthetic stand-in for a measured spectrum (module docstring).

    Runs the case's own full pipeline, auto-scales the expected signal
    to `peak_counts` at the maximum (a stand-in absolute calibration),
    and applies seeded detector noise. The result is labeled
    ``metadata["surrogate"] = True`` and reports will say so.

    Realism knobs (proof-of-concept comparisons should *not* look
    noise-free):

    - `peak_counts` sets the signal-to-noise scale: shot noise at the
      peak is sqrt(peak_counts)/peak_counts (~1.6% at 4000 counts,
      ~0.4% at 50000).
    - `read_noise_rms_counts` / `dark_mean_counts` default to the
      case's instrument settings (floored at 5 / 50 counts so a
      surrogate is never noiseless).
    - `background_mean_counts` adds a flat continuum/stray-light
      pedestal (with shot noise) that the validation pipeline must
      remove via its `background_windows` — exercising the same
      preprocessing a real spectrum needs.
    """
    if peak_counts <= 0 or not np.isfinite(peak_counts):
        raise ValueError("peak_counts must be finite and > 0")
    clean = case.synthetic_noise_free()
    peak = float(np.max(clean.intensity))
    if peak <= 0:
        raise ValueError("case produces no signal; cannot build surrogate")
    scale = peak_counts / peak
    scaled = Spectrum(
        wavelength_m=clean.wavelength_m,
        intensity=clean.intensity * scale,
        metadata=dict(clean.metadata),
    )
    read_noise = (
        max(case.instrument.read_noise_rms_counts, 5.0)
        if read_noise_rms_counts is None
        else float(read_noise_rms_counts)
    )
    dark = (
        max(case.instrument.dark_mean_counts, 50.0)
        if dark_mean_counts is None
        else float(dark_mean_counts)
    )
    noisy = NoiseModel(
        read_noise_rms_counts=read_noise,
        dark_mean_counts=dark,
        background_mean_counts=float(background_mean_counts),
    ).apply(scaled, seed=seed)
    metadata = dict(noisy.metadata)
    metadata["surrogate"] = True
    metadata["surrogate_note"] = (
        "synthetic stand-in generated by libssim itself (auto-scaled to "
        f"peak ~ {peak_counts:.3g} counts); replace with a measured "
        "spectrum for physics validation"
    )
    return Spectrum(
        wavelength_m=noisy.wavelength_m,
        intensity=noisy.intensity,
        metadata=metadata,
    )


@dataclass(frozen=True)
class NoiseStudy:
    """
    Noise-robustness statistics over repeated surrogate realizations.

    Quantifies how detector noise degrades the two working metrics:
    the correlation R (Eq. 5-56) and a doublet peak-intensity ratio.
    """

    r_values: Tuple[float, ...]
    ratio_values: Tuple[float, ...]
    clean_ratio: float

    @property
    def r_mean(self) -> float:
        return float(np.mean(self.r_values))

    @property
    def r_std(self) -> float:
        return float(np.std(self.r_values))

    @property
    def ratio_mean(self) -> float:
        return float(np.mean(self.ratio_values))

    @property
    def ratio_std(self) -> float:
        return float(np.std(self.ratio_values))

    def summary(self) -> str:
        return "\n".join(
            [
                f"noise realizations   : {len(self.r_values)}",
                f"R (Eq. 5-56)         : {self.r_mean:.4f} +/- {self.r_std:.4f}",
                (
                    f"peak ratio           : {self.ratio_mean:.3f} "
                    f"+/- {self.ratio_std:.3f} "
                    f"(clean synthetic: {self.clean_ratio:.3f})"
                ),
            ]
        )


def noise_study(
    case: ValidationCase,
    seeds: Sequence[int],
    line_a_m: float,
    line_b_m: float,
    half_window_m: float,
    peak_counts: float = 4.0e3,
    read_noise_rms_counts: Optional[float] = None,
    dark_mean_counts: Optional[float] = None,
    background_mean_counts: float = 0.0,
    background_windows: Optional[Sequence[Tuple[float, float]]] = None,
    normalize_mode: str = "peak",
) -> NoiseStudy:
    """
    Repeat the surrogate -> validate loop over several noise seeds and
    collect R plus the line_a/line_b peak ratio per realization.

    The scatter of the ratio is the noise-induced uncertainty of the
    doublet diagnostic (self-absorption indicator); its offset from the
    clean-synthetic ratio exposes the peak-picking bias of taking a
    window *maximum* on noisy data (extreme-value bias — area-based
    ratios would reduce it; documented observation, not hidden).
    """
    from .metrics import intensity_ratio  # local import: avoid cycle noise

    if len(seeds) < 2:
        raise ValueError("provide at least 2 seeds for meaningful statistics")
    clean = case.synthetic_noise_free()
    clean_ratio = intensity_ratio(clean, line_a_m, line_b_m, half_window_m)
    r_values = []
    ratio_values = []
    for seed in seeds:
        surrogate = surrogate_experiment(
            case,
            seed=int(seed),
            peak_counts=peak_counts,
            read_noise_rms_counts=read_noise_rms_counts,
            dark_mean_counts=dark_mean_counts,
            background_mean_counts=background_mean_counts,
        )
        result = case.validate(
            surrogate,
            background_windows=background_windows,
            normalize_mode=normalize_mode,
        )
        r_values.append(result.metrics.r_correlation)
        ratio_values.append(
            intensity_ratio(
                result.experimental, line_a_m, line_b_m, half_window_m
            )
        )
    return NoiseStudy(
        r_values=tuple(r_values),
        ratio_values=tuple(ratio_values),
        clean_ratio=float(clean_ratio),
    )

"""
Worked validation example: the Al I 394.4 / 396.15 nm resonance doublet.

Proof-of-concept comparison (Phase 4): a CLEAN synthetic spectrum vs a
realistic NOISY "experimental-like" version (Poisson shot + Gaussian
read noise + flat background pedestal, ~4000 peak counts). See
validate_sodium.py for the rationale; run that one first.

Physics focus: both Al lines share the upper level (4s 2S1/2,
3.1427 eV), so their optically thin intensity ratio is fixed by
transition probabilities alone: I(394.4)/I(396.15) = (nu*A)_394 /
(nu*A)_396 ~ 0.51 â€” a clean atomic-data check with no temperature
sensitivity. Self-absorption pushes the ratio toward 1 (both are
resonance lines; 396.15 saturates first because its lower level,
2P3/2 with g=4, holds twice the population).

Plasma conditions (literature-representative, refine per experiment):
    T(t)   = 11500 K * (t / 1 us)^-0.4
    n_e(t) = 8.0e22 m^-3 * (t / 1 us)^-1.1
    Al heavy density 1e21 m^-3, R = 1.5 mm
    gate: t_delay = 1.5 us, t_gate = 2.0 us
(Aguilera & Aragon, SAB 59 (2004) 1861; Cristoforetti et al., SAB 59
(2004) 1907.)

USING REAL DATA: pass --experimental <csv> or drop a measured file at
data/experimental/aluminum_394_396.csv â€” the surrogate is skipped and
the identical pipeline runs on the measurement (REAL-DATA SWAP below).

Run:  python examples/validate_aluminum.py [--output-dir DIR] [--seed N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from libssim.analysis import load_spectrum_csv, plot_comparison
from libssim.temporal.decay_models import PowerLawDecay
from libssim.validation import (
    InstrumentSettings,
    PlasmaConditions,
    ValidationCase,
    aluminum_setup,
    intensity_ratio,
    noise_study,
    normalize,
    surrogate_experiment,
)

DEFAULT_EXPERIMENTAL = Path("data/experimental/aluminum_394_396.csv")

AL_394_M, AL_396_M = 394.40058e-9, 396.15200e-9

#: Surrogate realism (see validate_sodium.py).
PEAK_COUNTS = 4.0e3
READ_NOISE = 15.0
BACKGROUND = 250.0

#: Line-free window edges for background fitting.
BACKGROUND_WINDOWS = ((393.0e-9, 393.8e-9), (396.8e-9, 397.5e-9))

#: Zoom window for the line-detail figure (nm).
ZOOM_NM = (394.0, 396.7)

COMMENTS = """
What this comparison shows
--------------------------
- R ~ 0.99 with a flat, zero-centered residual: the model reproduces
  the noisy spectrum to within its own noise floor. Watch the residual
  panel for *structure* at the line positions â€” that, not the noise
  amplitude, would signal physics discrepancies against real data.
- The 394.4/396.15 ratio is temperature-insensitive (shared upper
  level): its clean value tracks A_ki transcription + self-absorption
  only, and the noise study gives the precision with which a
  measurement of this SNR could test it.
- The continuum pedestal (free-free + free-bound is ON in this case,
  plus the surrogate's flat background) is removed by the line-free
  window fit â€” check the fitted windows stay line-free if you change
  plasma conditions.

REAL-DATA SWAP: place a measured CSV at data/experimental/
aluminum_394_396.csv (or use --experimental); everything downstream of
the load is unchanged.
"""


def build_case() -> ValidationCase:
    setup = aluminum_setup()
    grid = np.linspace(393.0e-9, 397.5e-9, 901)  # 5 pm step
    conditions = PlasmaConditions(
        temperature_K=PowerLawDecay(1.15e4, 1.0e-6, 0.4),
        heavy_density_m3=1.0e21,
        electron_density_m3=PowerLawDecay(8.0e22, 1.0e-6, 1.1),
        radius_m=1.5e-3,
        gate_delay_s=1.5e-6,
        gate_width_s=2.0e-6,
        n_time_nodes=8,
    )
    instrument = InstrumentSettings(
        reciprocal_dispersion_nm_per_mm=1.6,
        slit_width_um=50.0,
        read_noise_rms_counts=READ_NOISE,
        dark_mean_counts=100.0,
    )
    return ValidationCase(
        name="Al I 394.4/396.15 nm doublet",
        setup=setup,
        conditions=conditions,
        instrument=instrument,
        wavelength_m=grid,
    )


def main(
    experimental_csv: Path | None = None,
    output_dir: Path = Path("examples/output/surrogate"),
    seed: int = 20260712,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case = build_case()

    print("Input conditions")
    print("----------------")
    print(case.describe())
    print()

    # ---- atomic data provenance + partition-function improvement -------
    from libssim.physics.partition import partition_function_from_levels
    from libssim.validation.atomic_data import AL_I_LEVELS_MINIMAL

    setup = case.setup
    print("Atomic data")
    print("-----------")
    print(setup.provenance)
    for note in setup.verify_notes:
        print(f"  VERIFY: {note}")
    provider = setup.saha_solver.partition_provider
    print("U_I(T), improved (~5.24 eV) level list vs minimal (~4.68 eV):")
    for temperature in (8000.0, 10000.0, 12000.0, 15000.0):
        u_improved = provider.partition_function("Al", 1, temperature)
        u_minimal = partition_function_from_levels(
            [g for _, g in AL_I_LEVELS_MINIMAL],
            [e for e, _ in AL_I_LEVELS_MINIMAL],
            temperature,
        )
        delta = u_improved / u_minimal - 1.0
        print(
            f"  {temperature:7.0f} K : {u_improved:6.3f} vs {u_minimal:6.3f}"
            f"  ({delta:+.1%})"
        )
    print(
        "  (Irwin-form polynomial fallback registered beyond the table "
        "grid; swap in published coefficients or a full ASD level CSV "
        "via load_levels_csv for literature-grade U.)"
    )
    print()

    # REAL-DATA SWAP: a measured CSV takes priority over the surrogate.
    csv_path = experimental_csv or (
        DEFAULT_EXPERIMENTAL if DEFAULT_EXPERIMENTAL.exists() else None
    )
    if csv_path is not None:
        experimental = load_spectrum_csv(csv_path)
        print(f"Loaded experimental spectrum: {csv_path}")
    else:
        experimental = surrogate_experiment(
            case,
            seed=seed,
            peak_counts=PEAK_COUNTS,
            read_noise_rms_counts=READ_NOISE,
            background_mean_counts=BACKGROUND,
        )
        shot_pct = 100.0 / np.sqrt(PEAK_COUNTS)
        print(
            "No measured spectrum found - using a realistic NOISY "
            f"SURROGATE (seed={seed}): peak ~ {PEAK_COUNTS:.0f} counts "
            f"(~{shot_pct:.1f}% shot noise at peak), read "
            f"{READ_NOISE:.0f} counts rms, background {BACKGROUND:.0f} "
            f"counts. Drop a real file at {DEFAULT_EXPERIMENTAL} for "
            "physics validation."
        )

    result = case.validate(experimental, background_windows=BACKGROUND_WINDOWS)
    print()
    print(result.report())

    half_window = 1.5e-10
    clean = normalize(case.synthetic_noise_free(), "peak")
    ratio_clean = intensity_ratio(clean, AL_394_M, AL_396_M, half_window)
    ratio_noisy = intensity_ratio(
        result.experimental, AL_394_M, AL_396_M, half_window
    )
    print()
    print(f"394.4/396.15 ratio  clean synthetic        : {ratio_clean:.3f}")
    print(f"394.4/396.15 ratio  noisy experimental-like: {ratio_noisy:.3f}")
    print("  (optically thin limit ~0.51; self-absorption pushes toward 1)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for suffix, window in (("", None), ("_zoom", ZOOM_NM)):
            path = output_dir / f"aluminum_comparison{suffix}.png"
            fig = plot_comparison(
                result.synthetic,
                result.experimental,
                labels=("clean synthetic", "noisy experimental-like"),
                wavelength_range=window,
                title=(
                    f"{case.name} - R = {result.metrics.r_correlation:.4f}"
                ),
                save_path=path,
            )
            plt.close(fig)
            print(f"Saved figure: {path}")
    except ImportError:
        print("(matplotlib not installed - skipping figures)")

    study = noise_study(
        case,
        seeds=range(1, 13),
        line_a_m=AL_394_M,
        line_b_m=AL_396_M,
        half_window_m=half_window,
        peak_counts=PEAK_COUNTS,
        read_noise_rms_counts=READ_NOISE,
        background_mean_counts=BACKGROUND,
        background_windows=BACKGROUND_WINDOWS,
    )
    print()
    print("Noise study (12 realizations)")
    print("-----------------------------")
    print(study.summary())
    print(COMMENTS)
    return result, study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experimental", type=Path, default=None,
        help="CSV with a measured Al spectrum (wavelength_nm,intensity)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/surrogate"))
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    main(args.experimental, args.output_dir, args.seed)

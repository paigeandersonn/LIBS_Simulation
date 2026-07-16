"""
Worked validation example: the Na I D resonance doublet.

Proof-of-concept comparison (Phase 4): a CLEAN synthetic spectrum is
generated first, then a realistic NOISY "experimental-like" version of
it (Poisson shot noise + Gaussian read noise + a flat background
pedestal at ~4000 peak counts, i.e. ~1.6% shot noise at the peak).
Comparing the two exercises the full validation loop â€” background
subtraction, resampling, normalization, correlation, peak matching,
doublet-ratio diagnostics â€” with honest, noise-limited metrics instead
of the trivial R = 1.0000 of a nearly noiseless surrogate.

Element choice: sodium first (Phase 4 validation plan) â€” two strong,
well-isolated lines (D2 588.995 nm, D1 589.592 nm) with textbook
atomic data and a clean qualitative diagnostic: the optically thin
D2/D1 peak ratio equals the upper-level degeneracy ratio g(D2)/g(D1)
= 2, and self-absorption compresses it toward 1 (Herrera Ch. 3,
Fig. 3-2).

Plasma conditions (literature-representative, refine per experiment):
    T(t)   = 10500 K * (t / 1 us)^-0.4     [power-law decay]
    n_e(t) = 1.0e23 m^-3 * (t / 1 us)^-1.0 [= 1e17 cm^-3 at 1 us]
    Na heavy density 5e20 m^-3 (trace analyte), R = 1.5 mm
    gate: t_delay = 1.0 us, t_gate = 1.0 us
Power-law histories follow the time-resolved characterization
literature (Aguilera & Aragon, SAB 59 (2004) 1861; Cristoforetti et
al., SAB 59 (2004) 1907); magnitudes are typical published LIBS values.

Instrument: 0.5 m Czerny-Turner-like (R_d = 1.6 nm/mm), 50 um slit
-> 0.08 nm Gaussian LSF (Eq. 3-19); ICCD-like noise.

USING REAL DATA: pass --experimental <csv> or drop a measured file at
data/experimental/sodium_d_lines.csv (format: data/experimental/
README.md). The script then skips the noisy surrogate entirely â€” the
comparison pipeline is identical either way (that is the swap point
marked REAL-DATA SWAP below).

Run:  python examples/validate_sodium.py [--output-dir DIR] [--seed N]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from libssim.analysis import load_spectrum_csv, plot_comparison
from libssim.temporal.decay_models import PowerLawDecay
from libssim.validation import (
    InstrumentSettings,
    PlasmaConditions,
    ValidationCase,
    intensity_ratio,
    noise_study,
    normalize,
    sodium_setup,
    surrogate_experiment,
)

#: Default drop-in location for a measured spectrum.
DEFAULT_EXPERIMENTAL = Path("data/experimental/sodium_d_lines.csv")

D2_M, D1_M = 588.99509e-9, 589.59237e-9

#: Realism of the experimental-like surrogate: ~1.6% shot noise at the
#: peak, visible read noise, and a background pedestal (~6% of peak)
#: that the pipeline must remove.
PEAK_COUNTS = 4.0e3
READ_NOISE = 15.0
BACKGROUND = 250.0

#: Line-free edges of the window, used to fit the background
#: (preprocessing.subtract_background) â€” the same step a real ICCD
#: spectrum needs.
BACKGROUND_WINDOWS = ((587.5e-9, 588.2e-9), (590.4e-9, 591.0e-9))

#: Zoom window for the line-detail figure (nm).
ZOOM_NM = (588.5, 590.1)

COMMENTS = """
What this comparison shows
--------------------------
- R ~ 0.99 (not 1.0000): the residual panel is dominated by shot +
  read noise scattered around zero, which is exactly what a
  well-matched model against a real spectrum of this SNR should look
  like. Structure in the residual (systematic lobes at line centers or
  wings) â€” not amplitude â€” is what would indicate model error.
- Background handling is real: the surrogate carries a flat ~250-count
  pedestal that `subtract_background` removes from the line-free
  window edges before comparison, the same preprocessing a measured
  spectrum needs.
- D2/D1 ratio under noise: the noise study reports the ratio's
  mean +/- std across realizations. The scatter is the noise floor of
  the self-absorption diagnostic. The small bias vs the clean ratio
  comes from taking window *maxima* on noisy data: extreme-value bias
  inflates both peaks, relatively more the weaker one, nudging the
  ratio low â€” fitted or area-based ratios would reduce it.
- The R(T) sweep still recovers the input temperature, now with
  noise-limited contrast: iteration works at realistic SNR.

REAL-DATA SWAP: place a measured CSV at data/experimental/
sodium_d_lines.csv (or use --experimental). Everything downstream of
the load â€” preprocessing, metrics, figures, noise study excepted â€” is
unchanged.
"""


def build_case() -> ValidationCase:
    setup = sodium_setup()
    grid = np.linspace(587.5e-9, 591.0e-9, 701)  # 5 pm step, uniform
    conditions = PlasmaConditions(
        temperature_K=PowerLawDecay(1.05e4, 1.0e-6, 0.4),
        heavy_density_m3=5.0e20,
        electron_density_m3=PowerLawDecay(1.0e23, 1.0e-6, 1.0),
        radius_m=1.5e-3,
        gate_delay_s=1.0e-6,
        gate_width_s=1.0e-6,
        n_time_nodes=8,
    )
    instrument = InstrumentSettings(
        reciprocal_dispersion_nm_per_mm=1.6,
        slit_width_um=50.0,
        read_noise_rms_counts=READ_NOISE,
        dark_mean_counts=100.0,
    )
    return ValidationCase(
        name="Na I D doublet",
        setup=setup,
        conditions=conditions,
        instrument=instrument,
        wavelength_m=grid,
    )


def temperature_sweep(case: ValidationCase, experimental, output_path: Path):
    """R(T0): the iteration loop â€” where does correlation peak?"""
    temperatures = np.linspace(8500.0, 12500.0, 9)
    r_values = []
    for t0 in temperatures:
        candidate = replace(
            case,
            conditions=replace(
                case.conditions,
                temperature_K=PowerLawDecay(float(t0), 1.0e-6, 0.4),
            ),
        )
        r_values.append(
            candidate.validate(
                experimental, background_windows=BACKGROUND_WINDOWS
            ).metrics.r_correlation
        )
    best = temperatures[int(np.argmax(r_values))]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(temperatures, r_values, "o-")
        ax.axvline(best, color="0.6", ls="--", label=f"best: {best:.0f} K")
        ax.set_xlabel("assumed center temperature at 1 us (K)")
        ax.set_ylabel("R (Eq. 5-56)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Na I D: correlation vs assumed temperature")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("(matplotlib not installed - skipping the R(T) figure)")
    return temperatures, r_values, best


def main(
    experimental_csv: Path | None = None,
    output_dir: Path = Path("examples/output/surrogate"),
    seed: int = 20260711,
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
    from libssim.validation.atomic_data import NA_I_LEVELS_MINIMAL

    setup = case.setup
    print("Atomic data")
    print("-----------")
    print(setup.provenance)
    for note in setup.verify_notes:
        print(f"  VERIFY: {note}")
    provider = setup.saha_solver.partition_provider
    print("U_I(T), improved (n<=6) level list vs minimal (~4.35 eV) list:")
    for temperature in (8000.0, 10000.0, 12000.0, 15000.0):
        u_improved = provider.partition_function("Na", 1, temperature)
        u_minimal = partition_function_from_levels(
            [g for _, g in NA_I_LEVELS_MINIMAL],
            [e for e, _ in NA_I_LEVELS_MINIMAL],
            temperature,
        )
        delta = u_improved / u_minimal - 1.0
        print(
            f"  {temperature:7.0f} K : {u_improved:6.3f} vs {u_minimal:6.3f}"
            f"  ({delta:+.1%})"
        )
    print(
        "  (Irwin-form polynomial fallback registered beyond the table "
        "grid; swap in published Irwin/Halenka coefficients or a full "
        "ASD level CSV via load_levels_csv for literature-grade U.)"
    )
    print()

    # ---- experimental-like data ----------------------------------------
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
            "counts. Drop a real file at "
            f"{DEFAULT_EXPERIMENTAL} for physics validation."
        )

    # ---- validate (clean synthetic vs noisy experimental-like) ---------
    result = case.validate(
        experimental, background_windows=BACKGROUND_WINDOWS
    )
    print()
    print(result.report())

    # ---- doublet ratio diagnostics --------------------------------------
    half_window = 1.5e-10
    clean = normalize(case.synthetic_noise_free(), "peak")
    ratio_clean = intensity_ratio(clean, D2_M, D1_M, half_window)
    ratio_noisy = intensity_ratio(result.experimental, D2_M, D1_M, half_window)
    print()
    print(f"D2/D1 peak ratio  clean synthetic        : {ratio_clean:.3f}")
    print(f"D2/D1 peak ratio  noisy experimental-like: {ratio_noisy:.3f}")
    print("  (optically thin limit: 2.00; self-absorption pulls it lower)")

    # ---- figures: full window + line zoom, overlay + residual -----------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for suffix, window in (("", None), ("_zoom", ZOOM_NM)):
            path = output_dir / f"sodium_comparison{suffix}.png"
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
        print("(matplotlib not installed - skipping the comparison figures)")

    # ---- noise robustness of the metrics --------------------------------
    study = noise_study(
        case,
        seeds=range(1, 13),
        line_a_m=D2_M,
        line_b_m=D1_M,
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

    # ---- iteration demo --------------------------------------------------
    sweep_path = output_dir / "sodium_r_vs_temperature.png"
    _, _, best = temperature_sweep(case, experimental, sweep_path)
    print(f"\nSaved R(T) sweep figure: {sweep_path}")
    print(f"R peaks at T0 ~ {best:.0f} K (assumed truth: 10500 K)")
    print(COMMENTS)
    return result, study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experimental", type=Path, default=None,
        help="CSV with a measured Na spectrum (wavelength_nm,intensity)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/surrogate"))
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    main(args.experimental, args.output_dir, args.seed)

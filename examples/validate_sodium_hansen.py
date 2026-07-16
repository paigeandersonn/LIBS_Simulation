"""
REAL-DATA validation: Na I D doublet vs a measured LIBS spectrum.

Experimental reference
----------------------
P. B. Hansen, PhD thesis, "Laser-Induced Breakdown Spectroscopy Under
Martian Conditions" (docs/Literature/hansen_peder-LIBS_martian_conditions.pdf),
Figure 4.4.5 (thesis p. 78), bottom-left panel — digitized by
examples/digitize_hansen_naD.py into
data/experimental/sodium_d_lines_hansen_500ns.csv.

Measurement: carbonate pellet (CaCO3+MgCO3+MnCO3+Na2CO3; Na 12.68 at%)
in a simulated Martian atmosphere (0.7 kPa, mainly CO2), Nd:YAG 35 mJ,
delay 500 ns, gate 50 ns, 30 shots accumulated, LTB Aryelle Butterfly
echelle + Andor iStar ICCD (50x50 um2 slit), intensity-calibrated.

Model conditions — all pinned by the same thesis (no tuning):
    T   = 12 822 K   (+/- 292; multi-element Saha-Boltzmann, Fig. 4.4.6a)
    n_e = 7e22 m^-3  (H-alpha Stark width, Fig. 4.4.4)
    gate: t_d = 500 ns, t_g = 50 ns (T, n_e effectively constant across
          a 50 ns gate -> Constant profiles)
    R   = 2.0 mm     (plasma extent ~4 mm total, Sec. 4.4.3)
    instrument: Gaussian LSF, FWHM 0.0623 nm at 589.3 nm (thesis
          Hg-lamp instrument function, Fig. 3.3.5 — the description
          the thesis's own simulations use; a measured Voigt
          decomposition is available via hansen_instrument(voigt=True))

The ONE free parameter is the sodium heavy-particle density n_Na
(atoms + ions, m^-3): the absolute ablated-mass column is not reported
in the thesis (its own two-zone fits treat column densities as fit
parameters as well). n_Na is chosen by maximizing R (Herrera Eq. 5-56)
over a log grid — reported openly below. Everything else is fixed a
priori. The D2/D1 peak ratio (2.0 optically thin; measured 1.34 here)
is then an independent check of the self-absorption physics, since the
ratio is not part of the R-based selection in any direct way.

Run:  python examples/validate_sodium_hansen.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from libssim.analysis import load_spectrum_csv
from libssim.validation import (
    InstrumentSettings,
    PlasmaConditions,
    ValidationCase,
    intensity_ratio,
    sodium_setup,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTAL_CSV = (
    REPO / "data" / "experimental" / "hansen_thesis"
    / "sodium_d_lines_hansen_500ns.csv"
)
OUTPUT_DIR = REPO / "examples" / "output" / "hansen_na_500ns"

D2_M, D1_M = 588.99509e-9, 589.59237e-9

# --- thesis-pinned conditions (see module docstring for provenance) ---
T_K = 12822.0
NE_M3 = 7.0e22
GATE_DELAY_S = 500e-9
GATE_WIDTH_S = 50e-9
RADIUS_M = 2.0e-3
#: echelle instrument function at 589.3 nm, from the thesis's own
#: Hg-lamp characterization (Ch. 3):
#: - pure-Gaussian description (Fig. 3.3.5 linear fit, what the
#:   thesis's own simulations use — and the headline config here):
#:     FWHM = 9.149e-5 * lambda_nm + 8.417e-3 nm -> 0.0623 nm
#: - Voigt decomposition (Fig. 3.3.4 linear fits; available via
#:   hansen_instrument(voigt=True) for sensitivity studies):
#:     Gaussian  FWHM_sigma = 6.617e-5 * lambda_nm + 5.037e-3 nm
#:     Lorentzian FWHM_gamma = 2.802e-5 * lambda_nm + 4.522e-3 nm
#:     -> 0.0440 / 0.0210 nm at 589.3 nm (Hg 577/579 nm points
#:     scatter up to ~0.04 nm above the gamma fit).
#: Sensitivity result (documented in the output README): with the
#: Voigt LSF, R is flat (~0.9885) for gamma 0.021-0.050 nm and the
#: model D2/D1 rises only 1.12 -> 1.23 — the instrument Lorentzian
#: alone does not close the width/ratio budget, and the pure-Gaussian
#: description scores best (R = 0.9904).
INSTR_FWHM_NM = 9.149e-5 * 589.3 + 8.417e-3
INSTR_GAUSS_FWHM_NM = 6.617e-5 * 589.3 + 5.037e-3
INSTR_LORENTZ_FWHM_NM = 2.802e-5 * 589.3 + 4.522e-3
SLIT_UM = 50.0

#: line-free window edges for background fitting (nm -> m)
BACKGROUND_WINDOWS = ((587.0e-9, 588.3e-9), (590.6e-9, 592.5e-9))

ZOOM_NM = (588.3, 590.3)

# dataviz reference palette (light mode)
COL_EXP = "#2a78d6"      # categorical slot 1 (blue)  - measured
COL_MODEL = "#1baf7a"    # categorical slot 2 (aqua)  - model
COL_MODEL_RED = "#e34948"  # categorical red - model, simple slide variant
INK = "#0b0b0b"          # primary ink
INK_2 = "#52514e"        # secondary ink
MUTED = "#898781"        # axis/tick labels
GRID = "#e1e0d9"         # hairline grid
BASELINE = "#c3c2b7"     # axis lines / zero line
SURFACE = "#fcfcfb"      # chart surface


def hansen_instrument(voigt: bool = False) -> InstrumentSettings:
    """
    Thesis-measured echelle instrument function at 589.3 nm.

    Default: the pure-Gaussian description (Fig. 3.3.5) used by the
    thesis's own simulations. voigt=True: the Fig. 3.3.4 Voigt
    decomposition (Gaussian core + Lorentzian component) — kept for
    sensitivity studies (constants block above).
    """
    if voigt:
        return InstrumentSettings(
            # dispersion x slit = the Gaussian part of the Voigt LSF
            reciprocal_dispersion_nm_per_mm=(
                INSTR_GAUSS_FWHM_NM / (SLIT_UM * 1e-3)
            ),
            slit_width_um=SLIT_UM,
            lsf_shape="voigt",
            lsf_lorentzian_fwhm_nm=INSTR_LORENTZ_FWHM_NM,
        )
    return InstrumentSettings(
        reciprocal_dispersion_nm_per_mm=INSTR_FWHM_NM / (SLIT_UM * 1e-3),
        slit_width_um=SLIT_UM,
    )


def load_hansen_spectrum(register_wavelength: bool = True):
    """
    Load the digitized measurement; optionally apply a wavelength
    registration shift.

    The registration shift is estimated model-independently: parabolic
    apex fits to the two D lines are compared with the NIST air
    wavelengths and the mean offset is removed. It absorbs the echelle
    wavelength calibration (Hg-lamp auto-alignment, ~half a detector
    pixel here), figure digitization, and any unmodeled Stark shift in
    one number, which is reported openly.
    """
    from libssim.core.spectrum import Spectrum

    spectrum = load_spectrum_csv(EXPERIMENTAL_CSV)
    if not register_wavelength:
        return spectrum, 0.0

    offsets = []
    wl = spectrum.wavelength_m
    for line_m in (D2_M, D1_M):
        window = np.abs(wl - line_m) < 0.10e-9
        idx = np.where(window)[0]
        apex = idx[np.argmax(spectrum.intensity[idx])]
        sl = slice(max(apex - 6, 0), apex + 7)
        coeffs = np.polyfit(wl[sl], spectrum.intensity[sl], 2)
        offsets.append(-coeffs[1] / (2.0 * coeffs[0]) - line_m)
    shift = -float(np.mean(offsets))
    metadata = dict(spectrum.metadata)
    metadata["wavelength_registration_shift_m"] = shift
    registered = Spectrum(
        wavelength_m=wl + shift,
        intensity=spectrum.intensity,
        metadata=metadata,
    )
    return registered, shift


def build_case(na_heavy_density_m3: float) -> ValidationCase:
    grid = np.linspace(587.0e-9, 592.5e-9, 1101)  # 5 pm step
    conditions = PlasmaConditions(
        temperature_K=T_K,
        heavy_density_m3=float(na_heavy_density_m3),
        electron_density_m3=NE_M3,
        radius_m=RADIUS_M,
        gate_delay_s=GATE_DELAY_S,
        gate_width_s=GATE_WIDTH_S,
        n_time_nodes=3,  # T, n_e constant across the 50 ns gate
    )
    instrument = hansen_instrument()
    return ValidationCase(
        name="Na I D doublet vs Hansen 500 ns (Martian-chamber LIBS)",
        setup=sodium_setup(),
        conditions=conditions,
        instrument=instrument,
        wavelength_m=grid,
    )


def scan_density(experimental, densities_m3) -> tuple[float, list]:
    """R(n_Na) over a density grid; returns (best density, rows)."""
    rows = []
    for density in densities_m3:
        result = build_case(density).validate(
            experimental, background_windows=BACKGROUND_WINDOWS
        )
        ratio = intensity_ratio(
            result.synthetic, D2_M, D1_M, half_window_m=1.5e-10
        )
        rows.append((float(density), result.metrics.r_correlation, ratio))
    best = max(rows, key=lambda row: row[1])[0]
    return best, rows


def slide_axes_style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def slide_figure(result, ratio_exp, ratio_model, na_density_m3,
                 window_nm, save_path, legend_loc="upper right"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl_nm = result.synthetic.wavelength_m * 1e9
    experimental = result.experimental.intensity
    synthetic = result.synthetic.intensity
    residual = experimental - synthetic

    fig = plt.figure(figsize=(10.5, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(
        2, 1, height_ratios=(3.0, 1.15), hspace=0.07,
        left=0.085, right=0.975, top=0.835, bottom=0.105,
    )
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(wl_nm, experimental, color=COL_EXP, lw=1.9,
            label="measured spectrum", zorder=3)
    ax.plot(wl_nm, synthetic, color=COL_MODEL, lw=1.9,
            label="libssim model", zorder=4)
    slide_axes_style(ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel("normalized intensity", fontsize=13, color=INK)

    # direct labels (identity not by color alone)
    ax.annotate("measured", xy=(588.72, 0.62), color=COL_EXP,
                fontsize=11.5, fontweight="bold", ha="right")
    ax.annotate("model", xy=(589.78, 0.55), color="#0d8a5e",
                fontsize=11.5, fontweight="bold", ha="left")

    # line markers
    for nm, name in ((588.995, "D2"), (589.592, "D1")):
        ax.axvline(nm, color=GRID, lw=0.7, zorder=1)
        ax.text(nm, 1.055, f"{name}\n{nm:.3f}", ha="center", va="bottom",
                fontsize=9, color=MUTED, linespacing=1.1)

    metrics = result.metrics
    ax.text(
        0.015, 0.97,
        (
            f"R = {metrics.r_correlation:.4f}\n"
            f"rms residual = {metrics.rms_residual:.3f}\n"
            f"D2/D1 peak ratio:\n"
            f"  measured {ratio_exp:.2f}, model {ratio_model:.2f}\n"
            "  (optically thin limit 2.0\n"
            "   → both self-absorbed)\n"
            f"fitted $n_{{Na}}$ = "
            f"{na_density_m3 / 1e22:.1f}×10²² m⁻³"
        ),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color=INK_2, linespacing=1.4,
    )
    if isinstance(legend_loc, tuple):
        loc, anchor = legend_loc
        legend = ax.legend(loc=loc, bbox_to_anchor=anchor, frameon=False,
                           fontsize=10.5, borderaxespad=0.2)
    else:
        legend = ax.legend(loc=legend_loc, frameon=False, fontsize=11,
                           borderaxespad=0.6)
    for text in legend.get_texts():
        text.set_color(INK_2)

    ax_res.plot(wl_nm, residual, color=INK_2, lw=1.1)
    ax_res.axhline(0.0, color=BASELINE, lw=1.0)
    slide_axes_style(ax_res)
    limit = 1.15 * float(np.max(np.abs(residual)))
    ax_res.set_ylim(-limit, limit)
    ax_res.set_ylabel("residual\n(meas − model)", fontsize=10.5, color=INK)
    ax_res.set_xlabel("wavelength (nm)", fontsize=13, color=INK)

    if window_nm is not None:
        ax.set_xlim(*window_nm)
    else:
        ax.set_xlim(wl_nm[0], wl_nm[-1])
    ax.set_ylim(-0.05, 1.12)

    fig.text(0.085, 0.955, "Na I D doublet: forward model vs measured "
             "LIBS spectrum", fontsize=15.5, color=INK, fontweight="bold")
    fig.text(0.085, 0.916,
             "Measured: Hansen (DTU thesis), carbonate pellet, simulated "
             "Martian atmosphere (0.7 kPa CO$_2$), delay 500 ns, "
             "gate 50 ns",
             fontsize=10.5, color=INK_2)
    fig.text(0.085, 0.883,
             "Model pinned to thesis diagnostics (T = 12 822 K, "
             "$n_e$ = 7×10²² m⁻³, R = 2 mm) — one fitted parameter: "
             "$n_{Na}$",
             fontsize=10.5, color=INK_2)

    fig.savefig(save_path, dpi=220, facecolor=SURFACE)
    import matplotlib.pyplot as plt  # noqa: F811 (already imported)
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def simple_slide_figure(result, ratio_exp, ratio_model, save_path,
                        window_nm=(588.5, 590.0)):
    """
    Minimal single-panel overlay for slides: no residual panel, one
    title, one metrics box, measured in blue / model in red.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    wl_nm = result.synthetic.wavelength_m * 1e9

    fig, ax = plt.subplots(figsize=(10.0, 6.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.095, right=0.965, top=0.895, bottom=0.125)

    ax.plot(wl_nm, result.experimental.intensity, color=COL_EXP, lw=2.2,
            label="Measured spectrum", zorder=3)
    ax.plot(wl_nm, result.synthetic.intensity, color=COL_MODEL_RED, lw=2.2,
            label="libssim model", zorder=4)

    slide_axes_style(ax)
    ax.set_xlim(*window_nm)
    ax.set_ylim(-0.04, 1.12)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(labelsize=12.5)
    ax.set_xlabel("Wavelength (nm)", fontsize=14.5, color=INK)
    ax.set_ylabel("Normalized intensity", fontsize=14.5, color=INK)
    ax.set_title("Na I D Doublet: Simulated vs Measured LIBS Spectrum",
                 fontsize=17, color=INK, fontweight="bold", pad=16)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=13,
                       borderaxespad=0.4)
    for text in legend.get_texts():
        text.set_color(INK_2)

    metrics = result.metrics
    ax.text(
        0.975, 0.965,
        "\n".join([
            f"R = {metrics.r_correlation:.4f}",
            f"rms residual = {metrics.rms_residual:.3f}",
            f"D2/D1 peak ratio: measured {ratio_exp:.2f}, "
            f"model {ratio_model:.2f}",
        ]),
        transform=ax.transAxes, va="top", ha="right",
        multialignment="left",
        fontsize=12, color=INK_2, linespacing=1.75,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=SURFACE,
                  edgecolor=GRID, linewidth=0.9),
    )

    fig.text(0.965, 0.022,
             "Data: P. B. Hansen, DTU PhD thesis — delay 500 ns, "
             "gate 50 ns",
             ha="right", fontsize=8.5, color=MUTED)

    fig.savefig(save_path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def scan_figure(rows_coarse, rows_fine, best, save_path):
    """R(n_Na): the single fitted parameter has one clear optimum."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(set(rows_coarse) | set(rows_fine))
    densities = np.array([row[0] for row in rows])
    r_values = np.array([row[1] for row in rows])

    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.82, bottom=0.14)
    ax.plot(densities, r_values, "-", color=COL_EXP, lw=1.9, zorder=3)
    ax.plot(densities, r_values, "o", color=COL_EXP, ms=4.5, zorder=4)
    ax.axvline(best, color=BASELINE, lw=1.0, ls="--", zorder=2)
    ax.text(best * 1.12, 0.845, f"best: {best / 1e22:.1f}×10²² m⁻³",
            fontsize=10.5, color=INK_2)
    ax.set_xscale("log")
    slide_axes_style(ax)
    ax.set_xlabel("assumed Na heavy density  $n_{Na}$  (m$^{-3}$)",
                  fontsize=12.5, color=INK)
    ax.set_ylabel("correlation R (Eq. 5-56)", fontsize=12.5, color=INK)
    fig.text(0.115, 0.945, "One-parameter fit: R($n_{Na}$) has a single "
             "optimum", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.115, 0.885,
             "All other inputs fixed by thesis diagnostics; "
             "self-absorption sets the optimum",
             fontsize=10.5, color=INK_2)
    fig.savefig(save_path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTAL_CSV.exists():
        raise SystemExit(
            f"{EXPERIMENTAL_CSV} not found - run "
            "examples/digitize_hansen_naD.py first"
        )
    experimental, shift = load_hansen_spectrum()
    print(f"Loaded measured spectrum: {EXPERIMENTAL_CSV.name} "
          f"({experimental.wavelength_m.size} points)")
    print(f"Wavelength registration: {shift * 1e12:+.1f} pm shift applied "
          "(line apexes vs NIST air wavelengths; absorbs echelle "
          "calibration + digitization + unmodeled Stark shift)")
    print()

    # ---- the one free parameter: n_Na, by R over a log grid ------------
    coarse = np.logspace(20.5, 23.5, 13)
    best_coarse, rows = scan_density(experimental, coarse)
    refined = best_coarse * np.logspace(-0.25, 0.25, 9)
    best, rows_fine = scan_density(experimental, refined)

    print("n_Na scan (coarse):   density [m^-3]   R        model D2/D1")
    for density, r_value, ratio in rows:
        marker = "  <-- best" if density == best_coarse else ""
        print(f"   {density:11.3e}   {r_value:8.5f}   {ratio:5.3f}{marker}")
    r_best = max(r_value for _, r_value, _ in rows_fine)
    print(f"refined best n_Na = {best:.3e} m^-3 (R = {r_best:.5f})")
    print()

    case = build_case(best)
    print("Input conditions")
    print("----------------")
    print(case.describe())
    print()
    print("Atomic data provenance")
    print("----------------------")
    print(case.setup.provenance)
    print()

    result = case.validate(experimental, background_windows=BACKGROUND_WINDOWS)
    print(result.report())
    print()

    half_window = 1.5e-10
    ratio_exp = intensity_ratio(result.experimental, D2_M, D1_M, half_window)
    ratio_model = intensity_ratio(result.synthetic, D2_M, D1_M, half_window)
    print(f"D2/D1 peak ratio  measured : {ratio_exp:.3f}")
    print(f"D2/D1 peak ratio  model    : {ratio_model:.3f}")
    print("  (optically thin limit 2.00; self-absorption compresses "
          "toward 1 - thesis reports strong absorption effects at "
          "these column densities)")

    slide_figure(result, ratio_exp, ratio_model, best, None,
                 OUTPUT_DIR / "overlay_detailed.png")
    slide_figure(result, ratio_exp, ratio_model, best, ZOOM_NM,
                 OUTPUT_DIR / "overlay_detailed_zoom.png",
                 legend_loc=("upper left", (0.005, 0.50)))
    scan_figure(rows, rows_fine, best,
                OUTPUT_DIR / "density_scan.png")
    simple_slide_figure(result, ratio_exp, ratio_model,
                        OUTPUT_DIR / "slide_overlay.png")


if __name__ == "__main__":
    main()

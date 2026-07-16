"""
Multi-zone (SphericalOnion) variant of the Hansen Na I D validation.

Tests whether radial plasma structure improves the fit over the
single-zone result of examples/validate_sodium_hansen.py (same measured
spectrum, same instrument model, same one fitted parameter).

Radial structure — pinned to the SAME thesis, no new free parameters
--------------------------------------------------------------------
Hansen's own two-zone fits to this very spectrum (Fig. 4.4.17, thesis
p. 88 / PDF p. 102) give, at 500 ns delay:

    T_inner  ~ 14 700 K      T_outer  ~ 7 700 K
    n_e,inner ~ 7e22 m^-3    n_e,outer ~ 3.5e22 m^-3

(the multi-element Saha-Boltzmann average, 12 822 K, lies between the
two, closer to the inner zone — thesis Sec. 4.4.3; the H-alpha n_e
coincides with the inner zone.)

Here those anchor a frozen parabolic onion (Herrera Eqs. 5-36/5-37
form), discretized into N equal-thickness shells and static across the
50 ns gate:

    T(r)    = T0   (1 - k1 (r/R)^2),  T0  = 14 700 K, k1 = 1 - 7700/14700
    n_e(r)  = ne0  (1 - ke (r/R)^2),  ne0 = 7e22,     ke = 1 - 3.5/7.0
    n_Na(r) = n0   (1 - ke (r/R)^2)   (same ablated-matter falloff)

n_e is PRESCRIBED per shell (as in the single-zone run) rather than
Saha-closed, because most electrons come from the non-Na matrix.
The single fitted parameter is again the center density n0, chosen by
maximizing R (Eq. 5-56).

Run:  python examples/validate_sodium_hansen_onion.py
"""
from __future__ import annotations

import numpy as np

import validate_sodium_hansen as base
from libssim.core.state import PlasmaState
from libssim.temporal.decay_models import CustomEvolution
from libssim.transport.geometry import SphericalOnion
from libssim.validation import (
    PlasmaConditions,
    ValidationCase,
    intensity_ratio,
    sodium_setup,
)

# --- thesis-pinned radial anchors (Fig. 4.4.17 at 500 ns) --------------
T_CORE_K = 14700.0
T_EDGE_K = 7700.0
NE_CORE_M3 = 7.0e22
NE_EDGE_M3 = 3.5e22
K1 = 1.0 - T_EDGE_K / T_CORE_K       # temperature gradient (frozen)
KE = 1.0 - NE_EDGE_M3 / NE_CORE_M3   # n_e and heavy-density gradient
N_ZONES = 12

RADIUS_M = base.RADIUS_M


def onion_geometry(n0_na_m3: float, time_s: float) -> SphericalOnion:
    """Frozen parabolic onion evaluated at shell mid-radii."""
    edges = np.linspace(0.0, RADIUS_M, N_ZONES + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    zones = []
    for r in mids:
        x2 = (r / RADIUS_M) ** 2
        temperature = T_CORE_K * (1.0 - K1 * x2)
        n_e = NE_CORE_M3 * (1.0 - KE * x2)
        n_na = n0_na_m3 * (1.0 - KE * x2)
        zones.append(PlasmaState(
            temperature_K=temperature,
            electron_density_m3=n_e,
            total_density_m3=n_na + n_e,
            radius_m=RADIUS_M,
            time_s=time_s,
            composition={"Na": n_na},
        ))
    return SphericalOnion(zones=tuple(zones),
                          boundaries_m=tuple(edges[1:]))


def build_case(n0_na_m3: float) -> ValidationCase:
    grid = np.linspace(587.0e-9, 592.5e-9, 1101)
    conditions = PlasmaConditions(
        temperature_K=T_CORE_K,           # informational (core values);
        heavy_density_m3=float(n0_na_m3),  # geometry comes from override
        electron_density_m3=NE_CORE_M3,
        radius_m=RADIUS_M,
        gate_delay_s=base.GATE_DELAY_S,
        gate_width_s=base.GATE_WIDTH_S,
        n_time_nodes=2,  # static geometry across the 50 ns gate
    )
    instrument = base.hansen_instrument()
    return ValidationCase(
        name="Na I D doublet vs Hansen 500 ns - multi-zone onion",
        setup=sodium_setup(),
        conditions=conditions,
        instrument=instrument,
        wavelength_m=grid,
        evolution_override=CustomEvolution(
            geometry_factory=lambda t: onion_geometry(n0_na_m3, t)
        ),
    )


def scan_density(experimental, densities_m3):
    rows = []
    for density in densities_m3:
        result = build_case(density).validate(
            experimental, background_windows=base.BACKGROUND_WINDOWS
        )
        ratio = intensity_ratio(result.synthetic, base.D2_M, base.D1_M,
                                half_window_m=1.5e-10)
        rows.append((float(density), result.metrics.r_correlation, ratio))
    best = max(rows, key=lambda row: row[1])[0]
    return best, rows


def core_residual_rms(result, half_nm=0.10):
    """rms of (measured - model) within +/-half_nm of each line core."""
    wl_nm = result.synthetic.wavelength_m * 1e9
    residual = result.experimental.intensity - result.synthetic.intensity
    mask = (np.abs(wl_nm - base.D2_M * 1e9) < half_nm) | (
        np.abs(wl_nm - base.D1_M * 1e9) < half_nm
    )
    return float(np.sqrt(np.mean(residual[mask] ** 2)))


def comparison_figure(result_onion, result_single, save_path, window_nm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl_nm = result_onion.synthetic.wavelength_m * 1e9
    measured = result_onion.experimental.intensity
    onion = result_onion.synthetic.intensity
    single = result_single.synthetic.intensity

    fig = plt.figure(figsize=(10.5, 6.6), facecolor=base.SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.35), hspace=0.07,
                          left=0.085, right=0.975, top=0.835, bottom=0.105)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(wl_nm, measured, color=base.COL_EXP, lw=1.9,
            label="measured spectrum", zorder=3)
    ax.plot(wl_nm, single, color=base.MUTED, lw=1.6, ls=(0, (4, 2)),
            label="single-zone model", zorder=4)
    ax.plot(wl_nm, onion, color=base.COL_MODEL, lw=1.9,
            label="multi-zone onion model", zorder=5)
    base.slide_axes_style(ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel("normalized intensity", fontsize=13, color=base.INK)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(*window_nm)

    r_on = result_onion.metrics.r_correlation
    r_1z = result_single.metrics.r_correlation
    ax.text(
        0.015, 0.97,
        (
            f"R: single-zone {r_1z:.4f} → onion {r_on:.4f}\n"
            f"rms: {result_single.metrics.rms_residual:.3f} → "
            f"{result_onion.metrics.rms_residual:.3f}\n"
            "radial T, $n_e$ pinned to Hansen's own\n"
            "two-zone fit (Fig. 4.4.17): 14.7→7.7 kK,\n"
            "$n_e$ 7→3.5×10²² m⁻³ (core→edge)"
        ),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, color=base.INK_2, linespacing=1.45,
    )
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10.5,
                       borderaxespad=0.3)
    for text in legend.get_texts():
        text.set_color(base.INK_2)

    ax_res.plot(wl_nm, measured - single, color=base.MUTED, lw=1.1,
                ls=(0, (4, 2)), label="single-zone")
    ax_res.plot(wl_nm, measured - onion, color=base.INK_2, lw=1.2,
                label="onion")
    ax_res.axhline(0.0, color=base.BASELINE, lw=1.0)
    base.slide_axes_style(ax_res)
    limit = 1.15 * float(np.max(np.abs(
        np.concatenate([measured - single, measured - onion])
    )))
    ax_res.set_ylim(-limit, limit)
    ax_res.set_ylabel("residual\n(meas − model)", fontsize=10.5,
                      color=base.INK)
    ax_res.set_xlabel("wavelength (nm)", fontsize=13, color=base.INK)
    legend_res = ax_res.legend(loc="lower right", frameon=False,
                               fontsize=9.5, ncol=2, borderaxespad=0.2)
    for text in legend_res.get_texts():
        text.set_color(base.INK_2)

    fig.text(0.085, 0.955,
             "Na I D doublet: single-zone vs multi-zone onion model",
             fontsize=15.5, color=base.INK, fontweight="bold")
    fig.text(0.085, 0.916,
             "Measured: Hansen (DTU thesis), carbonate pellet, simulated "
             "Martian atmosphere (0.7 kPa CO$_2$), delay 500 ns, "
             "gate 50 ns",
             fontsize=10.5, color=base.INK_2)
    fig.text(0.085, 0.883,
             "One fitted parameter each ($n_{Na}$); onion T(r), $n_e$(r) "
             "anchored to the thesis two-zone fit — no new free "
             "parameters",
             fontsize=10.5, color=base.INK_2)

    fig.savefig(save_path, dpi=220, facecolor=base.SURFACE)
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def main() -> None:
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experimental, shift = base.load_hansen_spectrum()
    print(f"Loaded measured spectrum: {base.EXPERIMENTAL_CSV.name} "
          f"(registration {shift * 1e12:+.1f} pm)")
    print(f"Onion: {N_ZONES} shells, T {T_CORE_K:.0f}->{T_EDGE_K:.0f} K, "
          f"n_e {NE_CORE_M3:.1e}->{NE_EDGE_M3:.1e} m^-3 (parabolic, "
          "frozen over the gate)")
    print()

    # ---- fit n0 (same single free parameter as the single-zone run) ----
    coarse = np.logspace(21.5, 23.5, 9)
    best_coarse, rows = scan_density(experimental, coarse)
    refined = best_coarse * np.logspace(-0.25, 0.25, 7)
    best, rows_fine = scan_density(experimental, refined)

    print("n0 scan (coarse):   center density [m^-3]   R        model D2/D1")
    for density, r_value, ratio in rows:
        marker = "  <-- best" if density == best_coarse else ""
        print(f"   {density:11.3e}          {r_value:8.5f}   "
              f"{ratio:5.3f}{marker}")
    r_best = max(r for _, r, _ in rows_fine)
    print(f"refined best n0 = {best:.3e} m^-3 (R = {r_best:.5f})")
    print()

    result_onion = build_case(best).validate(
        experimental, background_windows=base.BACKGROUND_WINDOWS
    )

    # refit the single-zone baseline here (never reuse a stale optimum)
    best_single_coarse, _ = base.scan_density(experimental,
                                              np.logspace(21.5, 23.5, 9))
    single_refined = best_single_coarse * np.logspace(-0.25, 0.25, 7)
    best_single, _ = base.scan_density(experimental, single_refined)
    print(f"single-zone baseline refit: n_Na = {best_single:.3e} m^-3")
    result_single = base.build_case(best_single).validate(
        experimental, background_windows=base.BACKGROUND_WINDOWS
    )

    print(result_onion.report())
    print()

    half = 1.5e-10
    rows_out = []
    for tag, result in (("single-zone", result_single),
                        ("onion", result_onion)):
        ratio_model = intensity_ratio(result.synthetic, base.D2_M,
                                      base.D1_M, half)
        rows_out.append((
            tag,
            result.metrics.r_correlation,
            result.metrics.rms_residual,
            ratio_model,
            core_residual_rms(result),
        ))
    ratio_meas = intensity_ratio(result_onion.experimental, base.D2_M,
                                 base.D1_M, half)

    print("Comparison (measured D2/D1 = "
          f"{ratio_meas:.3f}; optically thin 2.00)")
    print("model         R         rms      D2/D1   core-rms(+/-0.1 nm)")
    for tag, r_value, rms, ratio, core in rows_out:
        print(f"{tag:12s}  {r_value:.4f}   {rms:.4f}   {ratio:5.3f}   "
              f"{core:.4f}")

    for suffix, window in (("", (587.0, 592.5)), ("_zoom", base.ZOOM_NM)):
        comparison_figure(
            result_onion, result_single,
            base.OUTPUT_DIR / f"onion_comparison{suffix}.png",
            window,
        )


if __name__ == "__main__":
    main()

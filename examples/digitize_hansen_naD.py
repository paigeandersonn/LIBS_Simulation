"""
Extract the measured Na I D-doublet spectrum from Hansen's thesis figure.

Source (in-repo PDF):
    P. B. Hansen, PhD thesis, "Laser-Induced Breakdown Spectroscopy
    Under Martian Conditions", docs/Literature/
    hansen_peder-LIBS_martian_conditions.pdf — Figure 4.4.5 (thesis
    p. 78 / PDF p. 92), bottom-left panel: Na I D2/D1 with Voigt fit.
    Carbonate pellet (CaCO3+MgCO3+MnCO3+Na2CO3, Na 12.68 at%),
    simulated Martian atmosphere (0.7 kPa, mainly CO2; thesis p. 20),
    delay 500 ns, gate (integration) 50 ns, 35 mJ, 30 accumulated
    shots; LTB Aryelle Butterfly echelle + Andor iStar ICCD, 50x50 um2
    entrance slit, FWHM_instr = 0.065 nm at 656.3 nm (Sec. 4.4.2).

Method (programmatic WebPlotDigitizer equivalent):
    The panel is embedded in the PDF as a 1973x1166 PNG (xref 1806).
    Axes spines are located as the longest black pixel runs; tick marks
    just outside the spines are calibrated against the printed labels
    (x: 587..592 nm, y: 0..800). Per pixel column the curve's vertical
    stroke span is the union of data-curve pixels (matplotlib C0 blue)
    and the overlaid dashed Voigt-fit pixels (black, in-axes); the
    dashed fit coincides with the data to within the line width where
    it occludes it. The path estimate is the span midpoint, corrected
    to stroke-top/bottom minus half a line width around the two apexes
    and the inter-peak valley, where flank stroke bleed-in biases a
    midpoint. Axis-calibration linearity residual < 0.002 nm.

Outputs:
    data/experimental/hansen_thesis/sodium_d_lines_hansen_500ns.csv
    data/experimental/hansen_thesis/fig445_NaD_panel.png  (source panel)
    examples/output/hansen_na_500ns/digitization_qc.png   (QC overlay)

Run:  python examples/digitize_hansen_naD.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "docs" / "Literature" / "hansen_peder-LIBS_martian_conditions.pdf"
PANEL_XREF = 1806  # bottom-left panel image of Fig. 4.4.5 (PDF p. 92)
DATA_DIR = REPO / "data" / "experimental" / "hansen_thesis"
PANEL_PNG = DATA_DIR / "fig445_NaD_panel.png"
OUT_CSV = DATA_DIR / "sodium_d_lines_hansen_500ns.csv"
QC_PNG = (
    REPO / "examples" / "output" / "hansen_na_500ns" / "digitization_qc.png"
)

X_TICK_NM = np.array([587.0, 588.0, 589.0, 590.0, 591.0, 592.0])
Y_TICK_SIG = np.array([800.0, 600.0, 400.0, 200.0, 0.0])  # top -> bottom
C0_BLUE = np.array([0x1F, 0x77, 0xB4]) / 255.0  # matplotlib default C0
MARGIN = 6  # px kept clear of the spines (spine stroke + antialiasing)


def extract_panel() -> None:
    import fitz  # PyMuPDF

    doc = fitz.open(PDF)
    info = doc.extract_image(PANEL_XREF)
    PANEL_PNG.parent.mkdir(parents=True, exist_ok=True)
    PANEL_PNG.write_bytes(info["image"])
    print(f"extracted panel: {PANEL_PNG} ({info['width']}x{info['height']})")


def longest_run(mask_1d: np.ndarray) -> int:
    best = cur = 0
    for v in mask_1d:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def cluster_centers(indices: np.ndarray) -> list[float]:
    out: list[float] = []
    start = prev = None
    for i in indices:
        if prev is None or i - prev > 3:
            if start is not None:
                out.append((start + prev) / 2.0)
            start = i
        prev = i
    if start is not None:
        out.append((start + prev) / 2.0)
    return out


def digitize() -> None:
    img = mpimg.imread(PANEL_PNG)
    if img.shape[2] == 4:  # composite any alpha over white
        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha + (1 - alpha)
    height, width, _ = img.shape

    red, green, blue_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    black = (red < 0.35) & (green < 0.35) & (blue_ch < 0.35)

    col_runs = np.array([longest_run(black[:, x]) for x in range(width)])
    row_runs = np.array([longest_run(black[y, :]) for y in range(height)])
    left, right = np.where(col_runs > 0.55 * height)[0][[0, -1]]
    top, bottom = np.where(row_runs > 0.55 * width)[0][[0, -1]]

    xt = cluster_centers(np.where(black[bottom + 8, left:right + 1])[0] + left)
    yt = cluster_centers(np.where(black[top:bottom + 1, left - 8])[0] + top)
    if len(xt) != len(X_TICK_NM) or len(yt) != len(Y_TICK_SIG):
        raise RuntimeError(
            f"tick detection failed: {len(xt)} x ticks, {len(yt)} y ticks"
        )
    cal_x = np.polyfit(xt, X_TICK_NM, 1)
    cal_y = np.polyfit(yt, Y_TICK_SIG, 1)
    res_x = float(np.max(np.abs(np.polyval(cal_x, xt) - X_TICK_NM)))
    print(f"axis calibration residual: {res_x * 1e3:.1f} pm (x)")

    is_blue = (
        np.sqrt(((img - C0_BLUE.reshape(1, 1, 3)) ** 2).sum(axis=2)) < 0.30
    )
    inner = np.zeros_like(is_blue)
    inner[top + MARGIN:bottom - MARGIN + 1, left + MARGIN:right - MARGIN + 1] = True
    is_blue &= inner
    black_in = black & inner
    union = is_blue | black_in

    cols = np.arange(left + MARGIN, right - MARGIN + 1)
    y_mid = np.full(cols.shape, np.nan)
    y_top = np.full(cols.shape, np.nan)
    y_bot = np.full(cols.shape, np.nan)
    overlay_used = np.zeros(cols.shape, dtype=bool)
    for k, x in enumerate(cols):
        rows = np.where(union[:, x])[0]
        if rows.size:
            y_top[k], y_bot[k] = rows.min(), rows.max()
            y_mid[k] = 0.5 * (rows[0] + rows[-1])
            overlay_used[k] = bool(black_in[:, x].any())

    ok = ~np.isnan(y_mid)
    wl_all = np.polyval(cal_x, cols)
    base_cols = ok & (wl_all > 587.0) & (wl_all < 588.0)
    stroke_w = float(np.median((y_bot - y_top)[base_cols]))
    half_w = stroke_w / 2.0
    print(f"stroke width on baseline: {stroke_w:.1f} px")

    # midpoint is biased at extrema by flank stroke bleed-in; use the
    # stroke envelope there instead (path = top + w/2 at maxima, etc.)
    def col_index(nm: float) -> int:
        return int(np.argmin(np.abs(wl_all - nm)))

    half_cols = 15  # ~60 pm neighborhood
    for apex_nm in (588.995, 589.592):
        i0 = col_index(apex_nm)
        seg = slice(i0 - half_cols, i0 + half_cols + 1)
        fix = y_top[seg] + half_w
        y_mid[seg] = np.where(np.isnan(fix), y_mid[seg],
                              np.minimum(y_mid[seg], fix))
    valley = slice(col_index(589.10), col_index(589.45) + 1)
    fix = y_bot[valley] - half_w
    y_mid[valley] = np.where(np.isnan(fix), y_mid[valley],
                             np.maximum(y_mid[valley], fix))

    y_filled = np.interp(cols.astype(float), cols[ok].astype(float), y_mid[ok])
    first, last = np.where(ok)[0][[0, -1]]
    keep = slice(first, last + 1)
    wl_nm = wl_all[keep]
    sig = np.polyval(cal_y, y_filled[keep])
    flag = overlay_used[keep]

    i2 = np.argmax(np.where(np.abs(wl_nm - 588.995) < 0.25, sig, -np.inf))
    i1 = np.argmax(np.where(np.abs(wl_nm - 589.592) < 0.25, sig, -np.inf))
    print(f"D2 peak {sig[i2]:.1f} at {wl_nm[i2]:.3f} nm; "
          f"D1 peak {sig[i1]:.1f} at {wl_nm[i1]:.3f} nm; "
          f"ratio {sig[i2] / sig[i1]:.3f}")

    header = f"""\
# Experimental LIBS spectrum: Na I D doublet (D2 588.995 nm, D1 589.592 nm)
# Source: P. B. Hansen, PhD thesis "Laser-Induced Breakdown Spectroscopy
#   Under Martian Conditions" (docs/Literature/hansen_peder-LIBS_martian_conditions.pdf),
#   Figure 4.4.5 (thesis p. 78 / PDF p. 92), bottom-left panel.
# Sample: pressed pellet of mixed carbonates CaCO3+MgCO3+MnCO3+Na2CO3
#   (Na 12.68 at%), in experimentally simulated Martian atmosphere
#   (0.7 kPa, mainly CO2; thesis p. 20).
# Acquisition: delay time 500 ns, integration (gate) time 50 ns, 35 mJ
#   on sample, 10 Hz, 30 accumulated shots; LTB Aryelle Butterfly
#   echelle + Andor iStar ICCD; entrance slit 50x50 um2; intensity-
#   calibrated (signal prop. photons/nm); FWHM_instr = 0.065 nm at
#   656.3 nm (thesis Sec. 4.4.2).
# Plasma parameters measured in the same work at 500 ns delay:
#   n_e = 7e22 m^-3 (H-alpha Stark broadening, thesis Fig. 4.4.4)
#   T = (12822 +/- 292) K (multi-element Saha-Boltzmann, Fig. 4.4.6a)
#   plasma extent ~4 mm total (thesis Sec. 4.4.3).
# Extraction: see examples/digitize_hansen_naD.py (method in its
#   docstring). {int(np.sum(flag))} of {wl_nm.size} columns had the dashed
#   Voigt-fit overlay contributing to the stroke span (flag column).
#   {wl_nm.size} points, {wl_nm[0]:.3f}-{wl_nm[-1]:.3f} nm, ~{np.mean(np.diff(wl_nm)) * 1e3:.1f} pm/point.
# columns: wavelength_nm,intensity,fit_overlay_contributed
"""
    with open(OUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for w, s, o in zip(wl_nm, sig, flag):
            f.write(f"{w:.5f},{s:.3f},{int(o)}\n")
    print(f"wrote {OUT_CSV} ({wl_nm.size} points)")

    QC_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[0].plot(cols[keep], y_filled[keep], "r-", lw=0.7)
    axes[0].set_title("digitized trace (red) over source panel")
    axes[0].set_axis_off()
    axes[1].imshow(img)
    axes[1].plot(cols[keep], y_filled[keep], "r-", lw=0.9)
    axes[1].set_xlim(880, 1160)
    axes[1].set_ylim(500, 60)
    axes[1].set_title("peak region")
    axes[1].set_axis_off()
    fig.suptitle("QC: Hansen Fig. 4.4.5 Na D panel digitization")
    fig.savefig(QC_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {QC_PNG}")


if __name__ == "__main__":
    extract_panel()
    digitize()

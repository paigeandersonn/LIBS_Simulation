# Na I D doublet vs Hansen (500 ns) — real-data validation outputs

Measured spectrum: P. B. Hansen, PhD thesis, *Laser-Induced Breakdown
Spectroscopy Under Martian Conditions* (DTU), Fig. 4.4.5 — carbonate
pellet (Na 12.68 at%), simulated Martian atmosphere (0.7 kPa CO₂),
delay 500 ns, gate 50 ns. Data + full provenance:
`data/experimental/hansen_thesis/`.

Model inputs pinned to the thesis's own diagnostics (T = 12 822 K
Saha–Boltzmann; nₑ = 7×10²² m⁻³ Hα Stark; R = 2 mm; echelle LSF
0.0623 nm at 589.3 nm from the measured instrument function, Fig.
3.3.5). One fitted parameter: the Na heavy density. A +8.2 pm
wavelength-registration shift (line apexes vs NIST) is applied —
consistent with the thesis's own calibration-deviation function
(Eq. 3.3.1, deviations up to ~0.015 nm).

## Headline results (after the Stark-width + registration correction)

Na D Stark widths now use the measured compilation of Konjević et
al., JPCRD 31, 819 (2002) — the earlier placeholder was ~6× too
broad and was silently compensating via a low fitted n_Na.

| model | R | rms | model D2/D1 | core rms |
|---|---|---|---|---|
| single-zone (n_Na = 1.8×10²³) | 0.9904 | 0.024 | 1.12 | 0.087 |
| 12-shell onion (n₀ = 6.8×10²²) | **0.9918** | **0.022** | 1.11 | **0.078** |

Measured D2/D1 = 1.34 · peaks matched to −0.1 / +2.6 pm.

**Open item (width budget):** with literature Stark widths, both
models need deep saturation to reproduce the ~0.15 nm measured line
widths, under-predicting the D2/D1 ratio (1.11–1.12 vs 1.34). The
two leading suspects were tested and ruled out as full explanations:

- **Instrument Lorentzian (tested — insufficient).** A "voigt" LSF
  shape was added to `libssim.instrument.spectrometer` (unit-tested;
  Gaussian core per Eq. 3-19 + calibrated Lorentzian FWHM) and run
  with the thesis's measured Voigt decomposition of the instrument
  function (Fig. 3.3.4: 0.0440 nm Gaussian + 0.0210 nm Lorentzian at
  589.3 nm). Result: R flat at ~0.9885 for γ_instr = 0.021–0.050 nm
  while the model D2/D1 rises only 1.12 → 1.23 — even γ at the top of
  the measured 577/579 nm scatter does not close the budget, and the
  thesis's pure-Gaussian description (0.0623 nm) scores best. The
  headline config therefore stays pure-Gaussian
  (`hansen_instrument(voigt=True)` reproduces the study).
- **Resonance (self-)broadening (computed — negligible).** Herrera
  Eq. 3-7 with the fitted n_Na and the Saha neutral fraction (Na is
  >99% ionized at 12.8 kK, nₑ = 7×10²² m⁻³) gives Δλ_res ≈ 2×10⁻⁵ nm.
- **Still open:** (a) ensemble smearing of the median spectrum — the
  measurement medians 8 pellet positions × repeats, and the thesis's
  own wavelength-deviation function (Eq. 3.3.1) shows
  position-dependent offsets that would Gaussian-smear an ensemble;
  (b) the 38–49 kK → 12.8 kK Stark-width extrapolation. Neither can
  be pinned from this single spectrum; the time-resolved series
  (750/1000/1250 ns) is the discriminating follow-up.

## Files

| File | What it shows |
|---|---|
| `slide_overlay.png` | Minimal single-panel overlay for slides (measured blue / model red, metrics box) |
| `overlay_detailed.png` | Full-window overlay + residual panel with conditions and metrics |
| `overlay_detailed_zoom.png` | Same, zoomed to 588.3–590.3 nm line detail |
| `density_scan.png` | R(n_Na): the one fitted parameter has a single clear optimum |
| `onion_comparison.png` / `_zoom.png` | Single-zone vs 12-shell onion (T, nₑ radial profiles pinned to Hansen's two-zone fit, Fig. 4.4.17). With corrected Stark widths the onion wins on all fit metrics (earlier "no gain" verdict was an artifact of the inflated placeholder width) |
| `digitization_qc.png` | QC overlay proving the figure-digitization traced the published curve |

## Regenerate

```bash
python examples/digitize_hansen_naD.py          # PDF figure -> CSV (once)
python examples/validate_sodium_hansen.py       # single-zone + 4 figures
python examples/validate_sodium_hansen_onion.py # onion comparison figures
```

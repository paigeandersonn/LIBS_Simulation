# Digitized spectra from the Hansen thesis (real measured LIBS data)

Source: P. B. Hansen, PhD thesis, *Laser-Induced Breakdown
Spectroscopy Under Martian Conditions* (DTU) —
`docs/Literature/hansen_peder-LIBS_martian_conditions.pdf`.

| File | Contents |
|---|---|
| `sodium_d_lines_hansen_500ns.csv` | Na I D doublet, 586.8–592.9 nm, 1474 pts (~4 pm/pt), digitized from Fig. 4.4.5 (thesis p. 78 / PDF p. 92), bottom-left panel. Full provenance + extraction notes in the CSV header |
| `fig445_NaD_panel.png` | The source figure panel as embedded in the PDF (1973×1166 PNG, xref 1806) — kept for provenance |

## Measurement conditions (thesis Secs. 3, 4.4)

- Sample: pressed pellet of CaCO₃+MgCO₃+MnCO₃+Na₂CO₃ (Na 12.68 at%)
- Atmosphere: simulated Martian, 0.7 kPa, mainly CO₂ (thesis p. 20)
- Delay 500 ns, gate (integration) 50 ns, 35 mJ on sample, 10 Hz,
  30 accumulated shots; LTB Aryelle Butterfly echelle + Andor iStar
  ICCD, 50×50 µm² slit; intensity-calibrated (∝ photons/nm);
  FWHM_instr = 0.065 nm at 656.3 nm
- Thesis-reported plasma diagnostics at 500 ns:
  T = (12 822 ± 292) K (multi-element Saha–Boltzmann, Fig. 4.4.6a);
  nₑ = 7×10²² m⁻³ (Hα Stark, Fig. 4.4.4); plasma extent ~4 mm
  (Sec. 4.4.3). Two-zone fit (Fig. 4.4.17): T 14.7/7.7 kK,
  nₑ 7/3.5×10²² m⁻³ (inner/outer)

## Extraction method

Programmatic digitization (WebPlotDigitizer-equivalent) by
`examples/digitize_hansen_naD.py`: axes auto-calibrated from tick
marks (residual < 2 pm); curve traced per pixel column with occlusion
handling for the overlaid Voigt fit and extrema correction. QC overlay:
`examples/output/hansen_na_500ns/digitization_qc.png`.

Caveat: digitized from the published figure, not raw detector data —
features below ~4 pm and the exact noise floor inherit plot resolution.

Regenerate any time: `python examples/digitize_hansen_naD.py`

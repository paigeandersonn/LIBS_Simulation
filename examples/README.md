# libssim validation examples

Worked end-to-end validation cases for Phase 4 (temporal evolution +
instrument response) on well-documented elements.

## Running

```bash
python examples/validate_sodium.py            # Na I D doublet (start here)
python examples/validate_aluminum.py          # Al I 394.4/396.15 nm doublet
python examples/validate_sodium.py --experimental path/to/your.csv

# REAL-DATA case (measured spectrum, digitized from Hansen's thesis):
python examples/digitize_hansen_naD.py            # PDF figure -> CSV (once)
python examples/validate_sodium_hansen.py         # R = 0.989 vs measurement
python examples/validate_sodium_hansen_onion.py   # multi-zone comparison
```

`validate_sodium_hansen.py` is the worked real-data example: Na I D
doublet at 500 ns delay / 50 ns gate from P. B. Hansen's PhD thesis
(carbonate pellet, simulated Martian atmosphere, 0.7 kPa CO2;
`docs/Literature/hansen_peder-LIBS_martian_conditions.pdf`, Fig.
4.4.5). All plasma/instrument inputs are pinned to the thesis's own
diagnostics (T = 12 822 K from its multi-element Saha-Boltzmann plot,
n_e = 7e22 m^-3 from H-alpha Stark broadening); the single fitted
parameter is the Na heavy density. `validate_sodium_hansen_onion.py`
repeats the comparison with a 12-shell SphericalOnion whose radial T,
n_e profiles are anchored to the thesis's own two-zone fit (verdict:
no R gain — see the output folder README).

Outputs are organized per case, each with a README describing every
figure and how to regenerate it:

- `examples/output/hansen_na_500ns/` — real-data figures (slide
  overlay, detailed overlay + residual, R(n_Na) scan, onion
  comparison, digitization QC)
- `examples/output/surrogate/` — synthetic-surrogate workflow demos
- measured data + provenance: `data/experimental/hansen_thesis/`

Each script prints a validation report (correlation coefficient R of
Herrera Eq. 5-56, rms residual, peak matching, doublet intensity
ratios) and saves figures to `examples/output/`.

**Without a measured spectrum, the scripts run against a clearly-labeled
synthetic surrogate.** That exercises every stage of the workflow
(loading, preprocessing, simulation, comparison, plotting) but validates
the *workflow*, not the *physics* — drop real data into
`data/experimental/` (see the README there) for physics validation.

## Where to get real Na / Al LIBS spectra

- **Your own lab / collaborators** — the ideal source: ICCD exports as
  `wavelength_nm,intensity` CSV, with the gate delay/width, slit width,
  grating dispersion and (if available) Stark-based n_e recorded. The
  `PlasmaConditions` / `InstrumentSettings` fields map one-to-one to
  those log entries.
- **NIST LIBS Database** (https://physics.nist.gov/PhysRefData/ASD/LIBS/)
  — ASD-derived simulated LIBS spectra per element; useful as an
  independent reference for line positions and relative intensities
  (note it is itself a model, not a measurement).
- **Literature with reported conditions** — supplementary data or
  digitized figures from papers that tabulate T(t) and n_e(t), e.g.:
  - Aguilera & Aragón, *Spectrochim. Acta B* **59** (2004) 1861 —
    laser-induced plasma characterization with time-resolved T and n_e
    (the source of the power-law default profiles);
  - Cristoforetti et al., *Spectrochim. Acta B* **59** (2004) 1907
    (Herrera's ref [171]);
  - Cremers & Radziemski, *Handbook of Laser-Induced Breakdown
    Spectroscopy* (Wiley) — representative spectra and conditions.

## Improving accuracy (documented upgrade paths)

- **Partition functions**: the built-in tables now sum extended NIST
  level lists (Na I complete through the n=6 shell at ~4.60 eV; Al I to
  ~5.24 eV), pushing the truncation deficit above ~12 kK, and every
  setup registers an Irwin-*form* polynomial fallback beyond the table
  grid. Two one-step upgrades remain: (a) export the full ASD levels
  page to CSV and load it with `libssim.validation.load_levels_csv`
  into an `ElementAtomicData` spec; (b) replace the fitted fallback
  coefficients with published Irwin (1981) / Halenka & Madej (2002)
  values (`ElementAtomicData.neutral_polynomial`). For alkali high-n
  Rydberg tails, `hydrogenic_tail_levels` exists but requires a
  consciously chosen density-dependent cutoff (see its docstring —
  the thesis p. 100 caveat).
- **Stark widths**: `libssim.validation.atomic_data` ships
  representative (Griem-order) widths; substitute tabulated
  coefficients (Herrera refs [131, 139]) in the `Transition`
  definitions for quantitative line-width work.
- **Spatial structure**: pass an `ExpandingOnionEvolution` (or any
  `CustomEvolution`) via `ValidationCase(evolution_override=...)` to
  model cool-periphery self-reversal.

## Extending to other elements

Copy the pattern in `libssim/validation/atomic_data.py`: transcribe the
NIST lines/levels/ionization energy, build an `ElementSetup`, and reuse
the whole workflow unchanged. Nothing element-specific lives outside
the atomic-data module.

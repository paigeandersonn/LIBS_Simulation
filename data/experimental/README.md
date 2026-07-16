# Experimental spectra drop-in directory

Place measured LIBS spectra here for the validation examples. When a
drop-in file below is absent, those examples fall back to a
clearly-labeled synthetic surrogate.

## Files in the repository

| Location | Contents |
|---|---|
| `hansen_thesis/` | REAL measured Na I D spectrum digitized from P. B. Hansen's PhD thesis (Fig. 4.4.5, 500 ns delay) + source panel + provenance README. Used by `examples/validate_sodium_hansen.py` |

Each digitized-literature dataset lives in its own subfolder with a
README recording conditions, thesis-reported diagnostics, and the
extraction method/regeneration command.

## Drop-in files (not shipped)

| File | Used by | Window |
|---|---|---|
| `sodium_d_lines.csv` | `examples/validate_sodium.py` | 587.5–591.0 nm |
| `aluminum_394_396.csv` | `examples/validate_aluminum.py` | 393.0–397.5 nm |

## Format

Plain CSV, `#` comments allowed, wavelength in nm (other units via
`load_spectrum_csv(..., wavelength_unit=...)`):

```csv
# source: <instrument / paper>, delay 1.0 us, gate 1.0 us, slit 50 um
wavelength_nm,intensity
587.500,1023.4
587.505,1030.1
...
```

## Record the provenance

A spectrum is only as useful as its metadata. Alongside each file, note
(in the header comments or a sidecar file):

- gate delay and gate width (the `PlasmaConditions` gate fields),
- slit width and grating dispersion (`InstrumentSettings`),
- any reported T / n_e and how they were measured (Boltzmann plot,
  Stark broadening — Herrera Eqs. 5-17/5-19),
- background/dark handling already applied by the acquisition software
  (so it is not subtracted twice via `subtract_background`).

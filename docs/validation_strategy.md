# Validation Strategy

A module is not considered complete until it passes both its software tests **and** its physics validation criteria.

## General Principles

Validation should follow this order:

1. Mathematical correctness (round-trips, identities)
2. Limiting cases and analytical solutions
3. Numerical stability and convergence
4. Comparison with literature values
5. Comparison with experimental data

## Phase 0 – Infrastructure

**PlasmaState**
- Must be immutable
- Must store values in SI units only

**Unit Conversions**
- All conversions must have relative error < 1e-12
- Round-trip conversions must recover the original value

## Phase 1 – Atomic Database

- Parser must return `Transition` objects with no `None` values in critical fields
- Units must be converted to SI
- Missing optional fields must be handled gracefully

## Phase 2 – Local LTE Physics

**Saha Solver**
- Total elemental abundance conserved to ≤ 1e-10 relative error
- Neutral fraction approaches 1 at low temperature
- Ionization increases monotonically with temperature

**Boltzmann Populations**
- Population fractions must sum to 1.0

**Line Profiles**
- Area under Voigt profile must integrate to 1.0 (±1e-10)

## Phase 3 – Spatial Transport

- Optically thick resonance lines must show self-reversal
- Increasing optical depth must decrease transmitted intensity in the line core

## Phase 4 – Temporal & Instrument

- Integrated intensity must scale correctly with gate width
- Measured FWHM after convolution must match theoretical slit function

## Phase 5 – Optimization

- Optimizer must recover known input parameters from synthetic spectra

## Phase 6 – Actinides

- Isotope wavelength shifts must be reproduced
- Relative intensities must match specified isotopic abundances

## Definition of Done

A phase/module is complete only when:
- All unit tests pass
- Physics validation criteria are satisfied with documented evidence
- Documentation is complete
- Numerical assumptions and limitations are documented
- Acceptance criteria are met
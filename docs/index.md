# libssim

**libssim** is an open-source Python framework for forward modeling of Laser-Induced Breakdown Spectroscopy (LIBS) plasmas.

It draws inspiration from the work in:

> Herrera, K.K. (2008). *From Sample to Signal in Laser-Induced Breakdown Spectroscopy: An Experimental Assessment of Existing Algorithms and Theoretical Modeling Approaches*. PhD Dissertation, University of Florida.

## Features

- Immutable and validated `PlasmaState`
- Rich `Spectrum` container with metadata
- Element-agnostic atomic data layer (`Transition`, NIST-style CSV parser)
- Local LTE physics: partition functions, Saha ionization balance with
  exact mass conservation, Boltzmann populations, line emission and
  bound-bound absorption, Doppler/Stark/Voigt line profiles, and
  free-free / free-bound continuum — every equation cited back to the
  dissertation by equation and page number
- Strict SI units throughout
- Designed to support Monte Carlo LIBS-style optimization

## Quick Start

```python
from libssim.core import PlasmaState

state = PlasmaState(
    temperature_K=10000.0,
    electron_density_m3=1e23,
    total_density_m3=2e23,
    radius_m=1e-3,
    time_s=1e-6,
    composition={"Al": 0.95, "Mg": 0.05}
)

print(state)
```

## Documentation

See the [API Reference](reference.md) for full documentation.
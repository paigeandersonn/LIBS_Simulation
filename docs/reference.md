# API Reference

The API mirrors the physical pipeline: a plasma **state** is populated
by LTE **physics**, emission is carried through the plasma by
**radiative transport**, evolved and gated in **time**, degraded by the
**instrument**, and finally compared against measurements by the
**analysis** and **validation** layers.

Every physics routine cites the governing equation and page of the
reference dissertation (Herrera 2008) it implements, and all quantities
are strict SI unless a unit suffix says otherwise (`_nm`, `_um`, …).

| Layer | Modules | What it provides |
|---|---|---|
| [Core](reference/core.md) | `state`, `spectrum`, `constants`, `units` | Immutable plasma state, spectrum container, constants, unit helpers |
| [Atomic Data](reference/atomic.md) | `transition`, `base`, `parsers` | Transition/level records and NIST-style parsers |
| [Physics](reference/physics.md) | `partition`, `saha`, `boltzmann`, `emission`, `line_profiles`, `continuum` | LTE populations, line emission, Voigt profiles, continuum |
| [Transport](reference/transport.md) | `base`, `geometry`, `emissivity`, `radiative` | Spherical-onion geometry and the radiative transfer equation |
| [Temporal](reference/temporal.md) | `base`, `decay_models`, `integrator` | Plasma evolution models and detector gate integration |
| [Instrument](reference/instrument.md) | `spectrometer`, `optics`, `noise`, `response` | Line-spread function, spectral efficiency, detector noise |
| [Analysis](reference/analysis.md) | `analysis` | Spectrum I/O, resampling, comparison and plotting helpers |
| [Validation](reference/validation.md) | `atomic_data`, `preprocessing`, `metrics`, `workflow` | Element setups and the simulate-and-compare workflow |

> Herrera, K.K. (2008). *From Sample to Signal in Laser-Induced
> Breakdown Spectroscopy: An Experimental Assessment of Existing
> Algorithms and Theoretical Modeling Approaches.* PhD Dissertation,
> University of Florida.

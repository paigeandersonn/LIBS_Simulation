# libssim Architecture Overview

libssim follows a layered architecture designed for modularity and testability.

## Layered Structure
Final Spectrum
↑
Instrument Response + Noise
↑
Radiative Transfer + Spatial Integration
↑
Emission + Line Profiles + Continuum
↑
Level Populations + Saha Ionization
↑
Atomic Database (Transition objects)
↑
PlasmaState (immutable physical conditions)
text


## Core Design Decisions

- `PlasmaState` is the central immutable data object.
- Physics calculations are separated from data containers.
- The atomic layer is fully decoupled from physics.
- All internal calculations use SI units.
- Every layer should be independently testable.
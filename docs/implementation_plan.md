
**Key Design Decisions:**
- `PlasmaState` is the central immutable data object.
- Physics calculations are separated from data containers.
- The atomic layer is fully decoupled from physics.
- All internal calculations use SI units.
- Every layer should be independently testable.

---

## Phase Overview

| Phase | Focus                        | Primary Goal                                      | Estimated Effort |
|-------|------------------------------|---------------------------------------------------|------------------|
| 0     | Immutable Core               | Foundational data structures and SI units         | 1–2 days        |
| 1     | Atomic Database              | Element-agnostic atomic data layer                | 2 days          |
| 2     | Local LTE Physics            | Populations, emission, and line profiles          | 3 days          |
| 3     | Spatial Transport            | Self-absorption and radiative transfer            | 3 days          |
| 4     | Time & Instrument            | Temporal integration and detector response        | 3 days          |
| 5     | API & Optimization           | Top-level simulator and parameter fitting         | 3 days          |
| 6     | Plutonium Actinide Overrides | Isotope handling and sparse data support          | 4 days          |

---

## Phase 0: Core Infrastructure (Completed)

**Objective:** Establish universal physical truths, SI unit enforcement, and state management.

**Key Deliverables:**
- `core/constants.py`
- `core/units.py`
- `core/state.py` (`PlasmaState` dataclass – immutable)
- `core/spectrum.py` (`Spectrum` dataclass)

**Acceptance Criteria:**
- `PlasmaState` can be instantiated with valid SI values.
- Unit conversion functions return mathematically correct results (relative error < 1e-12).
- Round-trip conversions recover the original value.

---

## Phase 1: Atomic Database Abstraction (Completed)

**Objective:** Parse quantum data so the physics engine remains agnostic to the element being simulated.

**Key Deliverables:**
- `atomic/transition.py` (`Transition` dataclass)
- `atomic/base.py` (Abstract `AtomicDatabase`)
- `atomic/parsers.py` (`CSVAtomicDatabase` + NIST-style parser)

**Acceptance Criteria:**
- Parser successfully loads `nist_cerium.csv` (or equivalent).
- All `Transition` objects have non-`None` values in critical fields.
- Units are converted to SI.
- Missing optional fields are handled gracefully.

---

## Phase 2: Local LTE Physics (In Progress)

**Objective:** Calculate populations, line shapes, and continuum for a single, uniform point of plasma.

**Key Deliverables:**
- `physics/partition.py`
- `physics/saha.py`
- `physics/boltzmann.py`
- `physics/emission.py`
- `physics/line_profiles.py`
- `physics/continuum.py`

**Acceptance Criteria:**
- Total elemental mass is conserved to ≤ 1e-10 relative error.
- Area under the Voigt profile integrates to 1.0 (±1e-10).
- Neutral fraction approaches 1.0 at low temperature.
- Ionization fraction increases monotonically with temperature.
- Emission coefficient scales correctly with upper-state population, A_ki, and photon energy.

**Gap Check:**
Do I understand the McWhirter criterion and Debye sphere concept well enough to validate LTE conditions?

---

## Phase 3: Spatial Transport

**Objective:** Build the plasma geometry, map temperature/density gradients, and simulate self-absorption.

**Key Deliverables:**
- `transport/base.py`
- `transport/geometry.py` (e.g. `SphericalOnion`)
- `transport/radiative.py` (Radiative Transfer Equation solver)

**Acceptance Criteria:**
- Optically thick resonance lines produce visible self-reversal.
- Increasing optical depth decreases transmitted intensity in the line core.
- Line-of-sight integration is numerically stable.

**Gap Check:**
Can I efficiently broadcast NumPy arrays so the line-of-sight integration does not become a performance bottleneck?

---

## Phase 4: Time Integration & Instrumentation

**Objective:** Convert pristine physics into a realistic laboratory signal.

**Key Deliverables:**
- `temporal/decay_models.py`
- `temporal/integrator.py`
- `instrument/optics.py`
- `instrument/spectrometer.py`
- `instrument/noise.py`

**Acceptance Criteria:**
- Integrated intensity scales correctly with gate width.
- Gaussian convolution produces the expected FWHM for a given slit width.
- Noise statistics behave correctly (Poisson variance ≈ mean, Gaussian read noise, dark current offset).

---

## Phase 5: API & Optimization Framework

**Objective:** Create a clean top-level interface and enable automated validation.

**Key Deliverables:**
- `simulator.py` (`LIBSSimulator`)
- `optimization/base.py`
- `optimization/scipy_fitter.py`

**Acceptance Criteria:**
- `simulator.simulate()` produces a `Spectrum` and writes a log file.
- The optimizer can recover known plasma parameters from a synthetic spectrum.

---

## Phase 6: Plutonium Actinide Overrides

**Objective:** Extend the validated engine to handle sparse actinide data and isotopic effects.

**Key Deliverables:**
- Updates to `atomic/parsers.py` (Blaise plutonium support)
- `atomic/isotopes.py`
- `physics/empirical_broadening.py`

**Acceptance Criteria:**
- Inputting a 50/50 isotopic ratio produces two distinct, correctly scaled peaks with the expected wavelength shift.
- Empirical broadening fallbacks are only applied when specific atomic data are missing.

---

## General Acceptance Criteria

Before any phase is considered complete, it must satisfy **all** of the following:

- All unit tests pass.
- Physics validation passes against analytical limits, conservation laws, or literature values.
- Full documentation is complete (docstrings + relevant updates to `docs/`).
- Code follows the rules defined in `development_rules.md`.
- No hardcoded element-specific logic exists outside the atomic database layer (except for justified overrides in Phase 6).
- Numerical assumptions and limitations are documented.

---

## Cross-Cutting Requirements

These apply across all phases:

- **Reproducibility**: Same configuration + same random seed must produce identical results.
- **Numerical Stability**: No NaNs, Infs, or unstable behavior under normal operating conditions.
- **Testing**: Physics modules must include validation against limiting cases and conservation laws.
- **Documentation**: Every public function and class must document physical meaning, units, assumptions, and references.

---

## Definition of Done

A phase is considered complete only when:
- Code passes all unit tests
- Physics validation criteria are satisfied with documented evidence
- Documentation is complete
- Acceptance criteria are met
- The implementation follows `development_rules.md`

Only then should work proceed to the next phase.
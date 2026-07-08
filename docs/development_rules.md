# Development Rules

These rules apply to all code and documentation in the libssim repository.

## Scientific Principles

- Use **SI units** internally.
- Clearly document every physical assumption.
- Prefer physically correct implementations over shortcuts.
- Keep all equations traceable to the dissertation or published literature.
- If implementation details are ambiguous, explicitly state the ambiguity and justify the chosen approach.

## Software Design Principles

- Favor **composition** over inheritance.
- Use **immutable dataclasses** (`frozen=True`) whenever practical.
- Every module must have a single, clearly defined responsibility.
- Avoid hidden global state.
- Functions should be deterministic when possible.

## Numerical Methods

- Prefer NumPy vectorization.
- Use SciPy routines for standard numerical tasks.
- Prioritize numerical stability.
- Document convergence behavior and limiting cases.

## Documentation Requirements

Every public class and function must include:
- Purpose and physical interpretation
- Parameters and return values (with units)
- References to equations or dissertation sections
- Numerical assumptions and limitations

## Testing Requirements

- Every module must have unit tests.
- Physics modules must validate against conservation laws, analytical limits, and (where available) literature values.
- Acceptance criteria must be satisfied before considering a phase complete.
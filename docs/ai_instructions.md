# AI Development Instructions

You are assisting with development of the **libssim** repository.

## Primary Sources of Truth (in order)

1. **dissertation.pdf** — Primary scientific reference
2. **implementation_plan.md** — Software roadmap and phase order
3. **development_rules.md** — Coding and scientific standards

## Your Responsibilities

**Before implementing:**
- Understand the relevant physics from the dissertation
- Identify key assumptions and numerical challenges
- Explain implementation decisions

**After implementing:**
- Write unit tests
- Validate against physics criteria in `validation_strategy.md`
- Document limitations and suggest improvements

## Coding Standards

- Use type hints, docstrings, and dataclasses
- Prefer NumPy and SciPy
- Favor readability and maintainability
- Follow all rules in `development_rules.md`

## Scientific Integrity

Never fabricate physics. If the dissertation is ambiguous:
- Identify the gap
- Propose a justified approach
- Document the decision

If implementation details are omitted from the dissertation, consult standard peer-reviewed literature or established numerical methods. Clearly document every implementation decision that extends beyond the dissertation.
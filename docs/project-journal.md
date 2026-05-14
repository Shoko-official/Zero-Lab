# Project Journal

## 2026-05-13

Context:

- The repository contained generated Python artifacts but no tracked source files.
- The first durable work establishes the planning foundation for a professional AlphaZero and MuZero platform.

Decisions:

- Use Zero Lab as the project-facing name.
- Keep public documentation in English.
- Treat chess as the default showcase domain.
- Keep toy games as correctness and CI domains.
- Make evaluation, replay, and hardware profiling first-class systems.
- Delay native search and distributed execution until profiling proves the need.

Artifacts added:

- README.
- Implementation plan.
- Evaluation plan.
- Merge request roadmap.
- Engineering standards.
- Reference map.
- Git ignore rules for generated Python artifacts.

Open questions:

- Which exact hardware target should define the first performance profile.
- Whether the first native backend should be Rust or C++.
- Whether the initial chess adapter should depend on an existing rules library or a small internal representation.
- Which experiment tracker, if any, should be adopted after the first local run format is stable.

## 2026-05-14

Context:

- Chess should be treated as a base environment now, not only as a later showcase domain.

Decision:

- Use `python-chess` for legal move generation and game-rule correctness.
- Keep Chess behind the same game-state contract as toy games.
- Use a fixed `8 x 8 x 73` AlphaZero-style action space for Chess.

Implementation:

- Added a Chess adapter with legal action masks, action decoding, FEN serialization, current-player perspective observations, outcomes, and underpromotion handling.
- Added `zero-lab list-games` so built-in adapters are visible from the CLI.
- Added tests for standard opening legal moves, move application, illegal actions, checkmate outcomes, perspective observations, serialization, and underpromotions.

Verification:

- `python -m pytest` passed.
- `python -m ruff check .` passed.
- `python -m mypy src tests` passed.
- `python -m pip install -e .` passed.
- `zero-lab smoke-test --run-dir "$env:TEMP\zero-lab-smoke" --seed 1` passed.
- `zero-lab list-games` passed.

Next entry template:

```text
Date:

Context:

Decision:

Implementation:

Verification:

Follow-up:
```

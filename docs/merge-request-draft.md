# Merge Request Draft

## Title

Add repository foundation and base game adapters

## Summary

This merge request establishes the executable foundation for Zero Lab and adds the first base game adapters, including Chess.

It adds package metadata, the `zero-lab` CLI, runtime configuration, logging, deterministic Python seeding, game-state contracts, Tic Tac Toe, Connect Four, Chess, tests, and repository hygiene.

## Motivation

The project needs a serious foundation before algorithmic implementation starts. AlphaZero and MuZero systems are easy to prototype poorly and hard to evaluate honestly. This MR sets the project direction around correctness, reproducibility, staged implementation, and reviewable engineering work.

Chess is included as a base domain now, because future search, replay, and model-target code should be designed against realistic legal-action behavior from the start.

## Changes

- Added package metadata and tool configuration.
- Added the `zero-lab` CLI with `smoke-test`, `show-config`, and `list-games`.
- Added runtime configuration, logging, and deterministic Python seeding.
- Added game-state and game-rules protocols.
- Added Tic Tac Toe and Connect Four adapters.
- Added a Chess adapter backed by `python-chess` legal move generation.
- Added a fixed Chess action space with `8 x 8 x 73` AlphaZero-style action IDs.
- Added Chess legal action masks, move application, FEN serialization, perspective observations, outcomes, and underpromotion handling.
- Added tests for CLI behavior, runtime config, seeding, toy-game invariants, and Chess invariants.
- Updated README and MR draft.

## Scope Boundaries

Included:

- Planning documentation.
- Evaluation structure.
- MR roadmap.
- Project standards.
- Repository hygiene for generated files.
- Base game adapters.
- Chess as a first-class base environment.

Not included:

- MCTS implementation.
- Training code.
- Search code.
- Runtime dependency changes.
- Benchmarks or performance claims.
- CI configuration.

## Verification

Performed local checks:

- `git diff --check` passed.
- `python -m pytest` passed.
- `python -m ruff check .` passed.
- `python -m mypy src tests` passed.
- `zero-lab list-games` passed.

## Review Focus

Please focus review on:

- Whether the implementation phases are realistic and sequenced correctly.
- Whether the evaluation gates are strong enough to prevent unsupported claims.
- Whether the environment contract is sufficient for MCTS and replay.
- Whether the Chess action encoding is acceptable as the base contract before search implementation.

## Risks

- The roadmap is intentionally ambitious. Timelines depend heavily on available hardware and evaluation budget.
- The current plan defaults to chess as the showcase domain. This can change, but the first public domain should remain narrow and measurable.
- Native search and distributed execution are deferred until profiling proves they are needed. This avoids early complexity, but means peak performance work comes later.

## Follow-Up Work

- Create package metadata and CI in the repository foundation MR.
- Add the environment contract and toy game adapters.
- Implement correct PUCT search before optimizing.
- Establish the first reproducible toy-domain learning run.
- Add chess only after the core training and evaluation loop is trustworthy.

# Merge Request Draft

## Title

Establish Zero Lab planning foundation

## Summary

This merge request establishes the public planning foundation for Zero Lab, a hardware-aware AlphaZero and MuZero research engineering platform.

It adds the project README, implementation plan, evaluation strategy, merge request roadmap, engineering standards, project journal, reference map, and Git ignore rules for generated Python artifacts.

## Motivation

The project needs a serious foundation before algorithmic implementation starts. AlphaZero and MuZero systems are easy to prototype poorly and hard to evaluate honestly. This MR sets the project direction around correctness, reproducibility, hardware measurement, staged implementation, and reviewable engineering work.

The goal is to make the repository credible from the first visible step: clear scope, explicit gates, realistic timelines, and no unsupported performance claims.

## Changes

- Added a top-level README that defines the Zero Lab project, current status, documentation map, and operating principle.
- Added a full implementation plan covering architecture, modules, phases, time estimates, hardware strategy, risks, and success criteria.
- Added an evaluation plan covering correctness, search, learning, strength, hardware metrics, promotion policy, and reporting artifacts.
- Added a merge request roadmap that breaks the project into reviewable implementation units.
- Added engineering standards for code quality, documentation, experiments, reviews, and dependencies.
- Added a project journal to track decisions and implementation history.
- Added a reference map linking the project direction to AlphaZero, MuZero, Gumbel MuZero, Sampled MuZero, EfficientZero, KataGo, LightZero, PyTorch compile, TensorRT, and Ray RLlib.
- Added `.gitignore` entries for Python caches, build outputs, virtual environments, coverage outputs, and local environment files.

## Scope Boundaries

Included:

- Planning documentation.
- Evaluation structure.
- MR roadmap.
- Project standards.
- Repository hygiene for generated files.

Not included:

- Algorithm implementation.
- Training code.
- Search code.
- Runtime dependency changes.
- Benchmarks or performance claims.
- CI configuration.

## Verification

Performed local checks:

- `git diff --check` passed.
- Whitespace scan passed on the new tracked documentation files.
- ASCII scan passed on the new tracked documentation files.
- Style scan passed for forbidden separators and generated-looking wording.

No unit tests were run because this MR only adds documentation and ignore rules.

## Review Focus

Please focus review on:

- Whether the implementation phases are realistic and sequenced correctly.
- Whether the evaluation gates are strong enough to prevent unsupported claims.
- Whether the MR roadmap is granular enough for high-quality review.
- Whether the documentation reads as a professional engineering foundation.
- Whether any scope should be moved earlier or later before implementation starts.

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

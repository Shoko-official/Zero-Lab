# Zero Lab

Zero Lab is a planned hardware-aware research and engineering platform for AlphaZero, MuZero, and modern tree-search reinforcement learning systems.

## Current Status

The repository is at the early implementation stage. It has project metadata, a minimal CLI, runtime configuration, logging, deterministic Python seeding, toy-game adapters, and a Chess adapter backed by legal move generation.

## Project Shape

- A game-agnostic AlphaZero engine for perfect-information domains.
- Chess as a base environment, not only a later showcase.
- A MuZero engine for learned-dynamics planning.
- A shared Monte Carlo Tree Search core with batched neural inference.
- A reproducible training and replay pipeline.
- A benchmark suite that measures playing strength, sample efficiency, throughput, latency, memory use, and hardware utilization.
- A documentation structure written for maintainers, reviewers, and technically strong visitors.

## Documentation Map

- [Implementation Plan](docs/implementation-plan.md)
- [Evaluation Plan](docs/evaluation-plan.md)
- [Merge Request Roadmap](docs/mr-roadmap.md)
- [Engineering Standards](docs/engineering-standards.md)
- [Project Journal](docs/project-journal.md)
- [Reference Map](docs/reference-map.md)

## Quick Start

```bash
python -m pip install -e .[dev]
zero-lab smoke-test
zero-lab list-games
python -m pytest
```


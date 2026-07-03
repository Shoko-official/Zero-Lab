# Zero Lab

<p align="center">
  <img src="assets/logo/zero-lab.png" width="100%">
</p>

**Zero Lab** is a hardware-aware research and engineering platform for AlphaZero, MuZero, and modern tree-search reinforcement learning systems.

The repository is currently focused on a correctness-first AlphaZero foundation: deterministic game adapters, PUCT search, self-play replay, trainer-facing batches, PyTorch tensor boundaries, losses, evaluation reports, checkpoint promotion metadata, batched inference, and public Chess showcase evaluation.

## Current Capabilities

- Installable Python package with typed public modules.
- Runtime config, logging, and deterministic Python seeding.
- Game adapters for Tic Tac Toe, Connect Four, and Chess.
- AlphaZero and MuZero model contracts.
- AlphaZero PUCT search with visit-count policy targets.
- Optional batched inference path for independent search roots.
- Self-play episode generation and JSONL replay records.
- Replay dataset streaming and AlphaZero training batch collation.
- PyTorch-ready AlphaZero batches, loss helpers, replay-backed training, checkpoints, and checkpoint-backed smoke evaluation.
- Fixed-seed evaluation reports for baseline agents.
- Checkpoint promotion reports with candidate Elo confidence intervals.
- Chess showcase reports with legal UCI move records and final FENs.
- CLI commands for smoke tests, search demos, replay summaries, evaluation, and Chess showcase runs.

## Not In Scope Yet

- Heavy Chess training.
- Production model architectures.
- Distributed self-play workers.
- External engine adjudication.
- SPRT and full rating ladders.
- Native MCTS storage or accelerator-specific inference kernels.

## Quick Start

Zero Lab targets Python 3.12 for local development and CI. On Windows with multiple installed
interpreters, replace `python` with `py -3.12`.

```bash
python -m pip install -e .[dev]
zero-lab smoke-test
zero-lab list-games
zero-lab search-demo --simulations 16
zero-lab self-play-demo --simulations 4 --output runs/self-play/tic-tac-toe.jsonl
zero-lab replay-summary runs/self-play/tic-tac-toe.jsonl
zero-lab replay-batch-summary runs/self-play/tic-tac-toe.jsonl --batch-size 32
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl --run-dir runs/train/tic-tac-toe --batch-size 1 --steps 1
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl --run-dir runs/train/tic-tac-toe-mlp --batch-size 1 --steps 1 --model mlp --hidden-size 128
zero-lab evaluate-linear-checkpoint runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt --simulations 1
zero-lab promote-linear-checkpoint --champion runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt --candidate runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt --champion-commit HEAD --candidate-commit HEAD --simulations 1
zero-lab evaluate --games tic_tac_toe --simulations 4
zero-lab chess-evaluate --max-plies 4 --simulations 1
python -m pytest
```

## Main Commands

| Command | Purpose |
| --- | --- |
| `zero-lab smoke-test` | Validate runtime config, logging, and seeding. |
| `zero-lab list-games` | List built-in game adapters. |
| `zero-lab search-demo --simulations 16` | Run a deterministic AlphaZero search smoke scenario. |
| `zero-lab self-play-demo --simulations 4 --output runs/self-play/tic-tac-toe.jsonl` | Generate a Tic Tac Toe self-play replay file. |
| `zero-lab replay-summary runs/self-play/tic-tac-toe.jsonl` | Summarize replay episodes and outcomes. |
| `zero-lab replay-batch-summary runs/self-play/tic-tac-toe.jsonl --batch-size 32` | Summarize trainer-facing AlphaZero batches. |
| `zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl --steps 1` | Train a bounded replay-backed AlphaZero smoke model and write a checkpoint. |
| `zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl --steps 1 --model mlp --hidden-size 128` | Train the non-linear MLP AlphaZero smoke model. |
| `zero-lab evaluate-linear-checkpoint runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt --simulations 1` | Evaluate a linear AlphaZero checkpoint against a random baseline. |
| `zero-lab promote-linear-checkpoint --champion checkpoints/champion.pt --candidate checkpoints/candidate.pt --champion-commit HEAD --candidate-commit HEAD` | Build a promotion report for two linear AlphaZero checkpoints. |
| `zero-lab evaluate --games tic_tac_toe --simulations 4` | Run fixed-seed baseline evaluation. |
| `zero-lab chess-evaluate --max-plies 4 --simulations 1` | Run the lightweight Chess showcase evaluation. |

## Repository Layout

```text
src/zero_lab/core/        Runtime config, logging, and seeding
src/zero_lab/games/       Game contracts plus Tic Tac Toe, Connect Four, and Chess
src/zero_lab/search/      AlphaZero search, evaluators, targets, and batched inference
src/zero_lab/models/      AlphaZero and MuZero model contracts
src/zero_lab/self_play/   AlphaZero self-play episode generation
src/zero_lab/replay/      Replay records, JSONL storage, summaries, and datasets
src/zero_lab/training/    AlphaZero batches, PyTorch tensors, losses, loops, and checkpoints
src/zero_lab/evaluation/  Baselines, match reports, promotion reports, and Chess showcase
tests/                    Unit and behavior tests
docs/                     Maintainer-facing documentation
```

## Documentation

- [AlphaZero Search](docs/alpha-zero-search.md)
- [Batched Inference](docs/alpha-zero-batched-inference.md)
- [Self-Play and Replay](docs/self-play-replay.md)
- [AlphaZero Training Batches](docs/alpha-zero-training-batches.md)
- [PyTorch AlphaZero Batches](docs/pytorch-alpha-zero-batches.md)
- [AlphaZero Losses](docs/alpha-zero-losses.md)
- [AlphaZero Replay Training](docs/alpha-zero-training-loop.md)
- [AlphaZero Evaluation Harness](docs/alpha-zero-evaluation.md)
- [AlphaZero Promotion Reports](docs/alpha-zero-promotion.md)
- [Chess Showcase Evaluation](docs/chess-showcase-evaluation.md)
- [Development Setup](docs/development.md)

## Verification

Run the same gates used for review:

```bash
python -m pytest
python -m ruff check .
python -m mypy src tests
```

## Engineering Direction

Zero Lab keeps correctness and reproducibility ahead of scale. The Python implementation is the reference path for API design, game contracts, search behavior, replay schemas, and evaluation reports. Native or accelerator-specific components should be added only after profiling shows a real bottleneck and the reference behavior is already covered by tests.

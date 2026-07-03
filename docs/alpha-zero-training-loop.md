# AlphaZero Replay Training

Zero Lab includes a first replay-backed AlphaZero training loop. It is intentionally small:
the loop proves the replay-to-loss-to-checkpoint path before larger model and worker layers are
introduced.

## Scope

The current implementation provides:

- Lazy JSONL replay streaming into PyTorch training batches.
- Bounded training runs with caller-provided model and optimizer.
- Policy and value optimization through the shared AlphaZero loss.
- Checkpoint save and load for model state, optimizer state, and training metadata.
- Resume support through `AlphaZeroTrainerConfig.resume_from`.
- A `train-alpha-zero` CLI smoke path for Tic Tac Toe, Connect Four, and Chess replay files.
- Linear and MLP policy-value model choices for bounded training smoke runs.
- An `evaluate-linear-checkpoint` CLI smoke path for fixed-seed evaluation against a random baseline.
- A `promote-linear-checkpoint` CLI smoke path for champion versus candidate promotion reports.

It does not yet include:

- Production model architectures.
- Model-backed evaluation agents for non-linear architectures.
- Distributed replay workers.
- Mixed precision or accelerator-specific tuning.
- Checkpoint promotion into a rating ladder.

## CLI Flow

Generate a small replay file:

```bash
zero-lab self-play-demo --simulations 4 --seed 3 --output runs/self-play/tic-tac-toe.jsonl
```

Inspect trainer-facing batches:

```bash
zero-lab replay-batch-summary runs/self-play/tic-tac-toe.jsonl --batch-size 1
```

Run one bounded training step with the default linear model:

```bash
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl \
  --run-dir runs/train/tic-tac-toe \
  --batch-size 1 \
  --steps 1 \
  --seed 3
```

Run the same smoke flow with the MLP model:

```bash
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl \
  --run-dir runs/train/tic-tac-toe-mlp \
  --batch-size 1 \
  --steps 1 \
  --model mlp \
  --hidden-size 128 \
  --seed 3
```

By default, the command writes:

```text
runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt
```

The default CLI model is a small linear policy-value model. The `--model mlp` option selects a
single-hidden-layer `MLPAlphaZeroModel` for non-linear smoke runs. Both paths exist as recovery
checks for the training loop, not as production AlphaZero architectures.

Evaluate the saved linear checkpoint:

```bash
zero-lab evaluate-linear-checkpoint \
  runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt \
  --game tic_tac_toe \
  --simulations 1 \
  --games-per-side 1 \
  --seed 3
```

The command loads the checkpoint into `LinearAlphaZeroModel`, wraps it in `TorchAlphaZeroEvaluator`,
then runs fixed-seed matches against `random_legal`.

Compare two retained linear checkpoints:

```bash
zero-lab promote-linear-checkpoint \
  --champion runs/train/tic-tac-toe/checkpoints/champion.pt \
  --candidate runs/train/tic-tac-toe/checkpoints/candidate.pt \
  --champion-commit aaa1111 \
  --candidate-commit bbb2222 \
  --game tic_tac_toe \
  --simulations 1 \
  --games-per-side 1 \
  --seed 3
```

The command reuses the promotion-report schema and adds `checkpoint_metadata` for both loaded
linear checkpoints.

## JSON Output

`train-alpha-zero` prints a JSON summary containing:

- `game`: selected game adapter.
- `input`: replay file path.
- `batch_size`: requested batch size.
- `model`: selected model architecture.
- `hidden_size`: requested MLP hidden size when `model` is `mlp`.
- `steps`: total optimizer steps represented by the checkpoint.
- `samples`: total replay samples consumed by the checkpoint.
- `last_total_loss`, `last_policy_loss`, and `last_value_loss`.
- `checkpoint_path`: saved checkpoint path.
- `run_dir`: runtime artifact directory.

`evaluate-linear-checkpoint` prints a JSON evaluation report containing:

- `checkpoint.path`: checkpoint path.
- `checkpoint.metadata`: saved training metadata.
- `checkpoint.observation_size` and `checkpoint.action_size`.
- `config`: match config and evaluated agents.
- `games`, `matches`, `scores`, `seeds`, and `limitations`.

`promote-linear-checkpoint` prints an AlphaZero promotion report containing:

- `champion` and `candidate`: checkpoint name, URI, and commit hash.
- `checkpoint_metadata`: loaded training metadata for both checkpoints.
- `candidate_elo_confidence_interval`.
- `promotion.decision`.
- `results`: fixed-seed match report.

## Resume

Resume from an existing checkpoint with:

```bash
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl \
  --run-dir runs/train/tic-tac-toe \
  --batch-size 1 \
  --steps 2 \
  --resume-from runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt
```

The loop loads model and optimizer state, restores saved `steps` and `samples`, skips replay
batches already represented by the checkpoint, then continues until the requested additional
bounded run completes.

## Python API

Use `train_alpha_zero_from_replay` when a caller owns the model and optimizer:

```python
from pathlib import Path

import torch

from zero_lab.games.toy import TicTacToeGame
from zero_lab.training.alpha_zero import AlphaZeroTrainerConfig, train_alpha_zero_from_replay

games = {"tic_tac_toe": TicTacToeGame()}
model = MyAlphaZeroModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

result = train_alpha_zero_from_replay(
    "runs/self-play/tic-tac-toe.jsonl",
    games=games,
    model=model,
    optimizer=optimizer,
    config=AlphaZeroTrainerConfig(
        batch_size=32,
        max_steps=100,
        checkpoint_path=Path("runs/train/tic-tac-toe/checkpoints/tic_tac_toe-alpha-zero.pt"),
    ),
)
```

The model must return `(policy_logits, predicted_values)` for an observation tensor. Policy logits
must have shape `(batch_size, action_size)`. Predicted values may have shape `(batch_size,)` or
`(batch_size, 1)`.

## Checkpoint Contract

Checkpoints are schema-versioned PyTorch payloads containing:

- `schema_version`.
- `model_state_dict`.
- `optimizer_state_dict`.
- `steps`.
- `samples`.
- `last_total_loss`.
- `last_policy_loss`.
- `last_value_loss`.

Unsupported schema versions fail fast during load.

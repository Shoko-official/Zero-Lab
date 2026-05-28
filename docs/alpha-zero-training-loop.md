# AlphaZero Training Loop

Zero Lab now includes a replay-backed AlphaZero training loop with checkpoint save and resume
support.

## Scope

The current implementation provides:

- Streaming PyTorch training batches from replay JSONL files.
- Repeated optimizer-backed AlphaZero training steps.
- Fixed-step run limits for smoke runs and local experiments.
- Checkpoint save and load for model and optimizer state.
- Resume support that skips already completed replay batches in the same replay stream.
- A `zero-lab train-alpha-zero` CLI smoke command.

It does not yet include:

- Learning-rate schedules.
- Mixed precision.
- Validation loops.
- Promotion evaluation.
- Replay sampling windows.

## Python API

`train_alpha_zero_from_replay` accepts replay paths, a game registry, a PyTorch model, an optimizer,
and an `AlphaZeroTrainerConfig`.

The trainer is model-agnostic. The model must follow the training-step output contract:

```text
(policy_logits, predicted_values)
```

Checkpoints contain:

- Schema version.
- Model state dict.
- Optimizer state dict.
- Completed training steps.
- Consumed samples.
- Last policy, value, and total loss.

## CLI

Train a small linear policy-value model from a Tic Tac Toe replay file:

```bash
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl --game tic_tac_toe --steps 1 --batch-size 1
```

The command writes a checkpoint under the runtime `run_dir` by default:

```text
runs/default/checkpoints/tic_tac_toe-alpha-zero.pt
```

Resume from a prior checkpoint:

```bash
zero-lab train-alpha-zero runs/self-play/tic-tac-toe.jsonl --resume-from runs/default/checkpoints/tic_tac_toe-alpha-zero.pt
```

## Next Step

The next production step is an evaluation harness that can compare checkpoints against baselines
with reproducible reports.

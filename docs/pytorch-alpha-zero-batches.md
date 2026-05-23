# PyTorch AlphaZero Batches

Zero Lab now includes a PyTorch tensor boundary for AlphaZero training batches.

## Scope

The current implementation provides:

- Conversion from `AlphaZeroTrainingBatch` into PyTorch tensors.
- Device selection through `torch.device` or device strings.
- Floating dtype selection for observations, policy targets, and value targets.
- Boolean legal-action masks.
- Integer selected actions.
- Integer current-player tensors.
- Lazy replay-to-PyTorch batch streaming.

It does not yet include:

- Model modules.
- Optimizer setup.
- Mixed precision.
- Pinned-memory transfer.
- Checkpointing.

## Tensor Contract

`TorchAlphaZeroTrainingBatch` stores:

- `observations`: float tensor with shape `(batch_size, observation_size)`.
- `legal_action_masks`: bool tensor with shape `(batch_size, action_size)`.
- `target_policies`: float tensor with shape `(batch_size, action_size)`.
- `target_values`: float tensor with shape `(batch_size,)`.
- `selected_actions`: long tensor with shape `(batch_size,)`.
- `current_players`: long tensor with shape `(batch_size,)`.

All tensors must live on the same device. Policy and value targets must use the same floating dtype
as observations.

## Replay Streaming

Use `iter_torch_alpha_zero_training_batches` to stream replay shards directly into PyTorch-ready
batches:

```python
from zero_lab.games.toy import TicTacToeGame
from zero_lab.training.alpha_zero import iter_torch_alpha_zero_training_batches

games = {"tic_tac_toe": TicTacToeGame()}

for batch in iter_torch_alpha_zero_training_batches(
    "runs/self-play/tic-tac-toe.jsonl",
    games=games,
    batch_size=32,
    device="cpu",
):
    ...
```

## Next Step

The next production step is an optimizer-backed training step over these tensors.

See `docs/alpha-zero-losses.md` for the policy and value loss boundary.

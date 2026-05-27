# AlphaZero Training Batches

Zero Lab now includes a framework-light batch collation path from replay records to AlphaZero
training batches.

## Scope

The current implementation provides:

- Validated `AlphaZeroTrainingBatch` records.
- Observation and legal-action-mask reconstruction from serialized replay states.
- Policy target validation.
- Scalar value target validation.
- Selected-action and current-player validation.
- Lazy replay-to-batch iteration.
- Batch summary reporting for replay inspection.

It does not yet include:

- Pinned-memory transfer.
- Optimizer setup.
- Loss functions.
- Checkpointing.
- Training loops.

## Batch Contract

Each `AlphaZeroTrainingBatch` stores:

- Canonical observations.
- Legal-action masks.
- Visit-count policy targets.
- Scalar value targets.
- Selected actions.
- Current players.
- Shared batch, observation, and action dimensions.

The batch can also produce an `AlphaZeroBatch` for model inference, which keeps the training target
contract separate from the model input contract.

## Replay Collation

`iter_alpha_zero_training_batches` streams replay samples from one or more JSONL replay files. For
each sample, it:

- Resolves the replay game name through the provided game registry.
- Deserializes the stored state.
- Confirms the stored current player matches the state.
- Rebuilds the canonical observation.
- Rebuilds the legal-action mask.
- Preserves policy, value, and selected-action targets.

The implementation is intentionally lazy so larger replay shards can be inspected or consumed
without loading the full corpus into memory.

## CLI

Summarize the training batches that a replay file would produce:

```bash
zero-lab replay-batch-summary runs/self-play/tic-tac-toe.jsonl --batch-size 32
```

Drop incomplete final batches from the summary:

```bash
zero-lab replay-batch-summary runs/self-play/tic-tac-toe.jsonl --batch-size 32 --drop-remainder
```

## Next Step

The next production step is policy and value losses over PyTorch tensors.

See `docs/pytorch-alpha-zero-batches.md` for the PyTorch tensor boundary.

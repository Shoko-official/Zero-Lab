# AlphaZero Training Step

Zero Lab now includes an optimizer-backed PyTorch training step for AlphaZero batches.

## Scope

The current implementation provides:

- Model forward execution over `TorchAlphaZeroTrainingBatch.observations`.
- Policy, value, and total loss computation through the AlphaZero loss boundary.
- Backpropagation.
- Optimizer stepping.
- Optional gradient clipping.
- Scalar training metrics for loss components and gradient norm.
- CPU tests for parameter updates, weighted losses, gradient clipping, and output validation.

It does not yet include:

- Learning-rate schedules.
- Mixed precision.
- Checkpointing.
- Replay training loops.
- Validation loops.

## Model Output Contract

`run_alpha_zero_training_step` expects a PyTorch module whose forward pass returns:

```text
(policy_logits, predicted_values)
```

`policy_logits` must have shape `(batch_size, action_size)`.

`predicted_values` may have shape `(batch_size,)` or `(batch_size, 1)`. Column values are squeezed
before loss computation.

## Step Behavior

Each call:

- Sets the model to training mode.
- Clears optimizer gradients with `set_to_none=True`.
- Runs model inference.
- Computes the configured AlphaZero loss.
- Runs `backward`.
- Clips gradients when `max_grad_norm` is configured.
- Steps the optimizer.
- Returns scalar metrics.

## Next Step

The next production step is a replay-backed training loop that repeatedly consumes PyTorch batches,
records metrics, and saves resumable checkpoints.

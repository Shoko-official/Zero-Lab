# AlphaZero Losses

Zero Lab now includes PyTorch policy and value losses for AlphaZero training.

## Scope

The current implementation provides:

- Policy cross-entropy against visit-count target distributions.
- Scalar value loss with MSE.
- Scalar value loss with Huber.
- Weighted total loss composition.
- Validation for policy logits, policy targets, predicted values, and value targets.
- Numerical CPU tests for policy, value, and total losses.

It does not yet include:

- Learning-rate schedules.
- Mixed precision.
- Checkpointing.
- A full training loop.

## Policy Loss

`alpha_zero_policy_loss` expects unnormalized policy logits with shape
`(batch_size, action_size)` and target policies with the same shape.

Target policies are visit-count distributions produced by search. The loss validates that targets:

- Are floating point tensors.
- Match the logits shape.
- Live on the same device as the logits.
- Contain only finite values.
- Contain no negative probabilities.
- Sum to one per sample.

The implementation uses PyTorch cross entropy with probability targets.

## Value Loss

`alpha_zero_value_loss` expects predicted scalar values and target scalar values with shape
`(batch_size,)`.

The default value loss is MSE. Huber loss is available through `loss_kind="huber"` and
`huber_delta`.

## Total Loss

`alpha_zero_loss` combines policy and value losses:

```text
total = policy_weight * policy_loss + value_weight * value_loss
```

The default configuration gives both components equal weight and uses MSE for value targets.

## Next Step

The loss functions are now consumed by `run_alpha_zero_training_step`.

The next production step is a replay-backed training loop with checkpoint save and resume support.

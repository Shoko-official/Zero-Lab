# AlphaZero Batched Inference

Zero Lab now includes a batched AlphaZero inference path for evaluating multiple search leaves with
one model call when the states share compatible model dimensions.

## Scope

The current implementation provides:

- `AlphaZeroBatchEvaluator`, an optional batch-capable evaluator protocol.
- `evaluate_batch`, a fallback adapter that uses batch evaluation when available.
- `BatchedModelEvaluator` and batch support on `ModelEvaluator`.
- Batched terminal-state handling without unnecessary model calls.
- `AlphaZeroSearch.run_batch` for running multiple independent roots together.
- Batched leaf expansion across roots.
- A lightweight latency measurement helper.

It does not change:

- PUCT scoring.
- Visit-count policy construction.
- Value backup rules.
- Root noise behavior.
- Action selection semantics.

## Public Interfaces

Batch evaluation:

```python
from zero_lab.search.alpha_zero import BatchedModelEvaluator, evaluate_batch
```

Batched search:

```python
from zero_lab.search.alpha_zero import AlphaZeroSearch

results = AlphaZeroSearch(evaluator, config).run_batch((state_a, state_b))
```

Latency measurement:

```python
from zero_lab.search.alpha_zero import measure_batched_inference_latency
```

## Compatibility Guard

The batched path is required to stay behavior-compatible with the unbatched path when noise and
stochasticity are disabled.

The current tests assert that deterministic batched search and repeated single-root search produce
matching decisions and visit counts for the same roots.

This matters because batching should improve inference throughput without silently changing the
search algorithm.

## Model Batch Requirements

`ModelEvaluator.evaluate_batch` can batch non-terminal states when they share:

- `action_size`,
- canonical observation size,
- legal-action mask width.

Terminal states are evaluated locally and are not sent to the model.

If non-terminal states have incompatible model dimensions, the evaluator raises `ValueError` instead
of silently mixing incompatible game shapes.

## Search Behavior

`AlphaZeroSearch.run_batch` creates one independent search tree per root state.

For each simulation step:

1. One leaf is selected from each active root.
2. Non-terminal leaves are evaluated in a batch.
3. Terminal leaves use their game outcome directly.
4. Each value is backed up through its own root path.

The trees do not share statistics. The batch only groups inference work.

## Latency Measurement

`measure_batched_inference_latency` repeatedly calls `evaluate_batch` and reports:

- batch size,
- repeats,
- warmup count,
- total evaluations,
- total seconds,
- mean seconds,
- min and max seconds,
- per-state seconds.

This helper is intended for local smoke measurements and regression checks. It is not a replacement
for a full benchmark suite with hardware metadata, model size, and profiler traces.

## Next Work

Recommended follow-up:

- Add batched neural checkpoint agents once checkpoint loading exists.
- Add CLI or run-directory integration for latency reports.
- Add benchmark metadata for device, model size, dtype, and backend.
- Evaluate larger root batches once game and model throughput are representative.

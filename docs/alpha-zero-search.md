# AlphaZero Search

Zero Lab now includes a first AlphaZero-style PUCT search implementation.

## Scope

The current implementation provides:

- PUCT tree search over deterministic `GameState` adapters.
- Evaluator contracts for model-backed or heuristic priors and values.
- Legal-action prior normalization.
- Visit-count policy targets for training.
- Temperature-based action selection.
- A CLI smoke scenario on Tic Tac Toe.

It does not yet include:

- Root Dirichlet noise.
- Batched inference queues.
- Transposition tables.
- Tree reuse across moves.
- Native memory-managed search storage.

## Design

The search layer stays independent from concrete neural-network frameworks. It consumes an `AlphaZeroEvaluator`, which can be backed by a real model through `ModelEvaluator` or by a deterministic heuristic evaluator for tests.

This keeps the boundary clear:

- Games own legal moves, transitions, terminal states, outcomes, and observations.
- Models own policy logits and value estimates.
- Search owns selection, expansion, backup, and visit statistics.
- Training consumes visit-count policy targets, not search internals.

## Native Backend Path

The current Python implementation is deliberately correctness-first.

Future optimization work should move memory-sensitive paths to a native backend only after profiling confirms sustained bottlenecks. Likely candidates are:

- Tree node storage.
- Edge statistics arrays.
- Replay buffers.
- Batched inference queues.
- Host and device transfer plumbing.

C++ is a plausible target for this layer because it gives direct control over memory layout, allocation strategy, cache locality, and integration with future accelerator-oriented inference paths. The Python API should remain the public orchestration layer unless profiling proves otherwise.

## Verification

The current tests cover:

- Legal-action filtering in model-backed priors.
- Visit-count policy normalization.
- Temperature-based selection.
- Terminal-root handling.
- Immediate tactical win discovery in Tic Tac Toe.

The CLI smoke path is:

```bash
zero-lab search-demo --simulations 16
```

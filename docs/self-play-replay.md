# Self-Play And Replay

Zero Lab now includes a minimal AlphaZero self-play path and versioned replay records.

## Scope

The current implementation provides:

- Deterministic Tic Tac Toe self-play smoke runs.
- MCTS-backed action selection.
- Visit-count policy targets.
- Per-step value targets derived from terminal outcomes.
- Versioned episode records.
- JSONL append and read helpers.
- Replay summary reporting.
- Streaming AlphaZero sample iteration for training datasets.
- Framework-light AlphaZero batch collation and batch summary reporting.

It does not yet include:

- Multiprocess actors.
- Replay sampling windows.
- Compression.
- Reanalyze.
- Batch collation for framework-specific trainers.
- Remote replay ingestion.

## Episode Records

Each replay episode stores:

- Schema version.
- Game name.
- Terminal state.
- Outcome from the first-player perspective.
- Per-step serialized state.
- Current player.
- Selected action.
- Visit-count policy target.
- Value target from that step player's perspective.

The record is intentionally explicit so downstream training code can validate data before it reaches a model.

## Dataset Stream

`iter_alpha_zero_samples` turns one or more replay files into a lazy stream of `AlphaZeroSample`
records. Each sample contains:

- Game name.
- Serialized state.
- Current player.
- Selected action.
- Visit-count policy target.
- Value target.

The stream is intentionally file-backed and lazy. Training jobs can consume large replay shards
without materializing full episode corpora in memory.

## CLI

Generate one deterministic Tic Tac Toe episode:

```bash
zero-lab self-play-demo --simulations 4 --seed 3 --output runs/self-play/tic-tac-toe.jsonl
```

Summarize a replay file:

```bash
zero-lab replay-summary runs/self-play/tic-tac-toe.jsonl
```

Summarize AlphaZero training batches from replay:

```bash
zero-lab replay-batch-summary runs/self-play/tic-tac-toe.jsonl --batch-size 32
```

## Next Step

The next production step is a framework-specific tensor adapter for PyTorch batches and
pinned-memory transfer.

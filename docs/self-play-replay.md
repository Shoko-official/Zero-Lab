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

It does not yet include:

- Multiprocess actors.
- Replay sampling windows.
- Compression.
- Reanalyze.
- Training dataset loaders.
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

## CLI

Generate one deterministic Tic Tac Toe episode:

```bash
zero-lab self-play-demo --simulations 4 --seed 3 --output runs/self-play/tic-tac-toe.jsonl
```

Summarize a replay file:

```bash
zero-lab replay-summary runs/self-play/tic-tac-toe.jsonl
```

## Next Step

The next production step is a replay dataset layer that can stream episode steps into AlphaZero training batches without loading the full replay corpus into memory.

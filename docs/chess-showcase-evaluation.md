# Chess Showcase Evaluation

Zero Lab includes a lightweight Chess showcase evaluation flow built on the existing `ChessGame`
adapter and the fixed-seed evaluation harness.

## Scope

The current implementation provides:

- Legal Chess games generated through the Python Chess-backed adapter.
- UCI move records for every played ply.
- Final FEN preservation.
- Fixed-seed random legal move versus uniform-search baseline games.
- Color alternation between the two baselines.
- Public JSON reports with seeds, scores, game records, config, and limitations.
- A CLI command for reproducible local showcase runs.

It does not include:

- Chess model training.
- Checkpoint loading.
- Neural-network inference.
- Opening books.
- Engine adjudication.
- Elo or SPRT for the Chess showcase.

## CLI

Run the default showcase:

```bash
zero-lab chess-evaluate
```

Run a short reproducible smoke showcase:

```bash
zero-lab chess-evaluate --seed 5 --games-per-side 1 --max-plies 4 --simulations 1
```

Write the JSON report to disk:

```bash
zero-lab chess-evaluate --output runs/evaluation/chess-showcase.json
```

Options:

- `--seed`: base seed used to derive game seeds.
- `--games-per-side`: starts per baseline and color pairing.
- `--max-plies`: maximum half-moves before the game is marked unfinished.
- `--simulations`: PUCT simulations used by the uniform-search baseline.
- `--output`: optional JSON report path.

## Report Contents

The report includes:

- `game`: always `chess`.
- `config`: seed, games per side, max plies, and baseline configuration.
- `seeds`: per-game seeds in execution order.
- `scores`: wins, losses, draws, and unfinished counts by baseline.
- `games`: per-game records with agents, moves, final FEN, outcome, and termination.
- `limitations`: explicit scope boundaries for the showcase.

Each move record includes:

- ply number,
- player,
- AlphaZero action index,
- UCI move,
- FEN after the move.

## Reproducibility

The runner creates a fresh seeded RNG for each game. Seeds follow:

```text
game seed = base seed + game index
```

For each `games_per_side` unit:

1. `random_legal` plays White and `uniform_search` plays Black.
2. `uniform_search` plays White and `random_legal` plays Black.

This keeps the showcase small, deterministic, and easy to inspect.

## Legal Move Guard

The Chess showcase validates selected actions against the adapter legal-action set before applying
them. The report stores UCI moves so a reviewer can replay the game with any standard Chess tool.

The adapter also round-trips all legal moves in the starting position, including knight moves, through
the AlphaZero action encoding.

## Next Work

Recommended follow-up:

- Add checkpoint-backed Chess agents once checkpoint loading exists.
- Add persistent run directories for public report artifacts.
- Add longer showcase schedules after runtime costs are measured.
- Add statistical promotion workflows only after model-backed Chess evaluation exists.

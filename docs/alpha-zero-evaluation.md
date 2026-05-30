# AlphaZero Evaluation Harness

Zero Lab includes a first fixed-seed evaluation harness for baseline head-to-head matches.

## Scope

The current implementation provides:

- A random legal-move baseline.
- A uniform-search baseline backed by AlphaZero PUCT search and `UniformEvaluator`.
- Fixed-seed head-to-head matches.
- Side alternation so each baseline starts the same number of games.
- Built-in evaluation coverage for Tic Tac Toe and Connect Four.
- JSON-ready reports with wins, losses, draws, seeds, games, config, match records, and limitations.
- A CLI command for reproducible local smoke evaluations.

It does not yet include:

- Elo ratings.
- SPRT.
- Confidence intervals.
- Parallel match execution.
- Neural-network checkpoint loading.
- Strength benchmarking against external engines.

## Public Interfaces

Baseline agents:

```python
from zero_lab.evaluation import RandomLegalMoveAgent, UniformSearchAgent
```

Match runner and report helpers:

```python
from zero_lab.evaluation import MatchConfig, run_head_to_head, summarize_match_results
```

The default CLI pairing is:

```text
agent_one = random_legal
agent_two = uniform_search
```

The match runner alternates first-player assignment, so both agents start the same number of games
for each selected game adapter.

## CLI

Run the default evaluation suite:

```bash
zero-lab evaluate
```

Run a smaller smoke evaluation:

```bash
zero-lab evaluate --games tic_tac_toe --simulations 4 --seed 7
```

Write the JSON report to disk while also printing it:

```bash
zero-lab evaluate --output runs/evaluation/baselines.json
```

Configurable options:

- `--games`: one or more of `tic_tac_toe` and `connect_four`.
- `--seed`: base seed used to derive per-match seeds.
- `--games-per-side`: number of starts per agent, per game.
- `--max-moves`: cap for unfinished games.
- `--simulations`: PUCT simulations used by the uniform-search baseline.
- `--output`: optional JSON report path.

## JSON Report

The report is stable JSON and includes:

- `config`: seed, games per side, max moves, and baseline configuration.
- `games`: selected game names.
- `seeds`: per-match seeds in execution order.
- `scores`: wins, losses, draws, and unfinished counts by baseline name.
- `matches`: one record per played match.
- `limitations`: explicit scope boundaries for the current harness.

Example shape:

```json
{
  "config": {
    "agents": [
      {"name": "random_legal", "role": "agent_one"},
      {
        "name": "uniform_search",
        "role": "agent_two",
        "simulations": 32,
        "temperature": 0.0
      }
    ],
    "games_per_side": 1,
    "max_moves": 512,
    "seed": 0
  },
  "games": ["tic_tac_toe", "connect_four"],
  "limitations": [
    "Elo ratings are intentionally out of scope for this report.",
    "SPRT is intentionally out of scope for this report.",
    "Results cover fixed-seed baseline matches only."
  ],
  "scores": {
    "random_legal": {"draws": 0, "losses": 0, "unfinished": 0, "wins": 0},
    "uniform_search": {"draws": 0, "losses": 0, "unfinished": 0, "wins": 0}
  }
}
```

The actual report also includes the full `matches` array and the exact `seeds` used.

## Reproducibility

The runner creates a fresh `random.Random(seed)` instance for every match. Match seeds are derived
from the base seed in execution order.

For each configured game:

1. `agent_one` starts.
2. `agent_two` starts.
3. The sequence repeats until `games_per_side` starts per agent are complete.

This keeps short evaluation runs inspectable and repeatable while the harness is still small.

## Next Work

Elo and SPRT should stay in a later MR after this runner has had enough usage to validate:

- report schema stability,
- match determinism,
- baseline behavior,
- runtime cost on larger game counts,
- failure behavior for unfinished games.

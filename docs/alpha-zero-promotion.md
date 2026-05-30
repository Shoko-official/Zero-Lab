# AlphaZero Checkpoint Promotion Reports

Zero Lab now includes a first promotion-report layer for comparing an AlphaZero champion checkpoint
against a candidate checkpoint.

## Scope

The current implementation provides:

- Champion versus candidate checkpoint descriptors.
- Commit hash preservation for both checkpoints.
- Agent-backed checkpoint comparison using the evaluation harness.
- Fixed-seed match execution through `MatchConfig`.
- Approximate Elo confidence intervals for the candidate score.
- A stable JSON-ready promotion report schema.
- A conservative promotion decision based on the lower bound of the candidate interval.

It does not yet include:

- Checkpoint file loading.
- Model deserialization.
- Neural-network inference from checkpoint paths.
- SPRT.
- Multi-run promotion history storage.
- External engine adjudication.

Until the repository has a checkpoint format, callers supply the agents used to play each checkpoint.
The checkpoint descriptors preserve the identity and commit metadata that will later connect to real
checkpoint files.

## Public Interfaces

Checkpoint descriptors:

```python
from zero_lab.evaluation import AlphaZeroCheckpoint
```

Champion versus candidate comparison:

```python
from zero_lab.evaluation import compare_alpha_zero_checkpoints
```

Promotion reports:

```python
from zero_lab.evaluation import PromotionConfig, build_alpha_zero_promotion_report
```

Elo confidence intervals:

```python
from zero_lab.evaluation import estimate_elo_confidence_interval
```

## Example

```python
from zero_lab.evaluation import (
    AlphaZeroCheckpoint,
    MatchConfig,
    PromotionConfig,
    RandomLegalMoveAgent,
    UniformSearchAgent,
    build_alpha_zero_promotion_report,
    compare_alpha_zero_checkpoints,
)
from zero_lab.games.toy import TicTacToeGame

match_config = MatchConfig(seed=17, games_per_side=8)
comparison = compare_alpha_zero_checkpoints(
    champion=AlphaZeroCheckpoint(
        name="champion",
        uri="checkpoints/champion.pt",
        commit_hash="abc1234",
    ),
    candidate=AlphaZeroCheckpoint(
        name="candidate",
        uri="checkpoints/candidate.pt",
        commit_hash="def5678",
    ),
    champion_agent=RandomLegalMoveAgent(),
    candidate_agent=UniformSearchAgent(simulations=32),
    games=(TicTacToeGame(),),
    config=match_config,
)
report = build_alpha_zero_promotion_report(
    comparison,
    config=PromotionConfig(match_config=match_config),
)
print(report.to_json())
```

## Report Contents

The promotion report includes:

- `schema_version`: report schema version.
- `champion`: checkpoint name, URI, and commit hash.
- `candidate`: checkpoint name, URI, and commit hash.
- `seed_policy`: deterministic seed policy and base seed.
- `config`: match config, confidence level, seed policy, and promotion threshold.
- `results`: full match summary from the evaluation harness.
- `candidate_elo_confidence_interval`: candidate score rate, Elo estimate, and interval.
- `promotion`: conservative decision and threshold.

The default seed policy is:

```text
match seed = base seed + match index
```

## Promotion Decision

The report uses a conservative decision rule:

```text
promote when candidate_interval.lower >= promotion_elo_threshold
```

The default threshold is `0.0`, meaning the candidate must clear a non-negative lower-bound Elo
estimate to be marked as promotable.

This is intentionally stricter than checking the point estimate alone.

## Confidence Interval

`estimate_elo_confidence_interval` estimates the candidate score rate from:

```text
score = wins + 0.5 * draws
games = wins + losses + draws
```

It then converts the score rate and Wilson interval bounds into Elo differences. This is a lightweight
approximation designed for early promotion reports, not a replacement for a full statistical testing
pipeline.

## Next Work

Recommended follow-up:

- Add checkpoint loading once a checkpoint file format exists.
- Add model-backed agents for real AlphaZero inference.
- Persist promotion reports under run directories.
- Add SPRT once the runner and report schema have stabilized.
- Add promotion history comparison across multiple candidates.

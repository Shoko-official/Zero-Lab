# Evaluation Plan

## Purpose

Evaluation must determine whether the system is correct, improving, efficient, and honest about its limits.

The project should evaluate three separate things:

- Algorithmic strength.
- Learning efficiency.
- Hardware efficiency.

These dimensions must not be collapsed into one vague score.

## Evaluation Tiers

### Correctness Evaluation

Required checks:

- Environment transitions are deterministic under fixed seed.
- Legal-action masks reject every illegal move.
- Terminal states are final and cannot be expanded.
- Rewards and outcomes match domain rules.
- Symmetry transforms preserve value and map policies correctly.
- MCTS never selects an illegal action.
- Checkpoint loading preserves model output within numeric tolerance.

### Search Evaluation

Required checks:

- MCTS solves known small positions.
- Visit counts converge toward optimal actions as simulations increase.
- Root noise can be disabled for deterministic evaluation.
- Temperature schedules produce expected action distributions.
- Gumbel search is evaluated under equal simulation budgets.

### Learning Evaluation

Required checks:

- Toy environments show monotonic improvement over random baselines.
- Loss components remain finite.
- Replay sampling produces valid targets.
- Training can resume from checkpoint.
- Fixed-seed smoke runs are stable enough to catch regressions.

### Strength Evaluation

Required checks:

- Head-to-head matches against fixed checkpoints.
- Promotion tournaments against current champion.
- Baseline matches against simple agents.
- Domain-specific external engine matches for chess.
- Elo estimates with confidence intervals.
- SPRT or equivalent sequential testing for promotion.

External engines should be used for evaluation only unless a separate supervised baseline is intentionally introduced and documented.

### Hardware Evaluation

Required checks:

- Self-play games per hour.
- Search nodes per second.
- Neural inferences per second.
- Inference latency percentiles.
- Training samples per second.
- GPU utilization.
- CPU utilization.
- Host to device transfer cost.
- Replay write and read throughput.
- Memory usage under sustained load.

The main hardware score should be useful training data per hour at a fixed evaluation target, not only raw throughput.

## Benchmark Domains

### Toy Domains

Purpose:

- Catch correctness regressions.
- Validate search behavior.
- Keep CI fast.

Candidates:

- Tic Tac Toe.
- Connect Four.
- Small custom deterministic games with known optimal moves.

### Showcase Domain

Purpose:

- Demonstrate a recognizable, non-trivial system.
- Support strong external baselines.

Default:

- Chess.

Chess gives visible progress, mature notation, strong baseline engines, and well-understood evaluation pitfalls.

### Research Extension Domains

Purpose:

- Exercise MuZero in settings where a learned model matters.
- Test stochasticity, visual observations, or larger action spaces.

Candidates:

- Atari-style environments.
- Gymnasium control tasks.
- Custom planning tasks.

## Promotion Policy

A checkpoint should be promoted only when:

- It passes all correctness tests.
- It wins a statistically meaningful match against the current champion.
- It does not regress critical hardware metrics beyond the configured tolerance.
- It has a retained config, commit hash, seed policy, and hardware profile.

For unstable domains, promotion should require repeated independent runs.

## Reporting Artifacts

Each serious run should produce:

- Configuration snapshot.
- Git commit hash.
- Dependency lock or environment export.
- Hardware summary.
- Training curves.
- Evaluation match table.
- Elo estimate with confidence interval.
- Throughput and latency metrics.
- Known caveats.
- Link to retained checkpoints and replay metadata.

## Acceptance Gates By Stage

### Foundation Gate

- Unit tests pass.
- Toy game transitions are deterministic.
- Configuration round trips without loss.

### Search Gate

- MCTS solves known toy positions.
- Illegal moves are impossible.
- Fixed-seed search is reproducible.

### AlphaZero Gate

- Self-play data is valid.
- Training improves a toy agent.
- Promotion tournament works.
- Replay can be inspected after the run.

### MuZero Gate

- Latent dynamics unroll without numeric instability.
- Reward, value, and policy targets are trained.
- The model learns a toy domain without simulator access inside search.

### Hardware Gate

- A baseline profile exists before optimization.
- Optimized path improves useful throughput.
- Correctness tests pass on both baseline and optimized paths.

### Showcase Gate

- A full run can be reproduced from public setup steps.
- Metrics are reported with limits.
- Documentation explains what was measured and what was not measured.

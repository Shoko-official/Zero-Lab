# Reference Map

This map collects the research and system references that should inform implementation choices. It is not a claim that Zero Lab already matches these systems.

## Core Algorithms

### AlphaZero

Reference:

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

Why it matters:

- Defines the game-agnostic self-play recipe for known-rule environments.
- Establishes policy and value training from MCTS-improved self-play.
- Provides the baseline architecture family for the AlphaZero track.

Implementation use:

- Use as the first full algorithmic target.
- Keep domain rules in the environment, not in the model.
- Validate against small deterministic games before chess.

### MuZero

Reference:

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)

Why it matters:

- Extends tree-search planning to learned latent dynamics.
- Predicts reward, policy, and value instead of reconstructing observations.
- Makes the system relevant beyond board games with known simulators.

Implementation use:

- Add after AlphaZero replay, training, and evaluation are stable.
- Keep MuZero search separate from simulator-backed search.
- Test on toy domains before visual or stochastic environments.

### Gumbel AlphaZero And Gumbel MuZero

Reference:

- [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)

Why it matters:

- Improves policy improvement behavior when simulation budgets are small.
- Replaces some heuristic action-selection machinery with a more principled variant.

Implementation use:

- Implement behind configuration flags.
- Compare against PUCT under matched simulation budgets.
- Do not make it the default until ablations justify the change.

### Sampled MuZero

Reference:

- [Learning and Planning in Complex Action Spaces](https://arxiv.org/abs/2104.06303)

Why it matters:

- Provides a path for large or continuous action spaces.
- Keeps planning feasible by searching sampled action subsets.

Implementation use:

- Defer until the discrete-action MuZero path is stable.
- Treat as an extension track for control and large-action domains.

### EfficientZero

Reference:

- [Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210)

Why it matters:

- Focuses on sample efficiency in visual reinforcement learning.
- Adds useful ideas around self-supervised consistency and reanalyze-style improvements.

Implementation use:

- Borrow concepts carefully after base MuZero is correct.
- Measure sample efficiency separately from wall-clock throughput.

## Practical Systems

### KataGo

Reference:

- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565)

Why it matters:

- Shows how domain-aware and system-aware improvements can drastically reduce compute needs.
- Demonstrates that strong engineering can matter as much as the baseline algorithm.

Implementation use:

- Study for optimization patterns and evaluation discipline.
- Keep domain-specific improvements isolated from the generic core.

### LightZero

Reference:

- [LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios](https://proceedings.neurips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html)

Why it matters:

- Provides a modern benchmark perspective on MCTS and MuZero-style systems across domains.
- Highlights the difficulty of general sequential decision settings.

Implementation use:

- Use as a benchmark design reference.
- Avoid overfitting the architecture to chess alone.

## Hardware And Runtime

### PyTorch Compile

Reference:

- [torch.compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)

Why it matters:

- Provides a modern PyTorch compilation path for potential model speedups.
- Best evaluated after model shapes and code paths are stable.

Implementation use:

- Keep as an optional backend.
- Measure graph breaks, compile time, and runtime speed separately.

### TensorRT

Reference:

- [NVIDIA TensorRT](https://developer.nvidia.com/TensorRT)

Why it matters:

- Provides optimized inference engines, quantization support, and deployment paths for NVIDIA hardware.
- May be valuable for high-throughput batched inference once model exports are stable.

Implementation use:

- Treat as an optimization stage, not a baseline requirement.
- Require numeric tolerance checks against PyTorch outputs.

### Ray RLlib

Reference:

- [Ray RLlib documentation](https://docs.ray.io/en/master/rllib/)

Why it matters:

- Shows mature patterns for scalable and fault-tolerant reinforcement learning workloads.
- Can inform distributed orchestration even if the project uses a custom runner.

Implementation use:

- Evaluate later for distributed experiments.
- Avoid bringing a large orchestration dependency into the foundation before it is necessary.

"""Tree data structures for AlphaZero-style MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field

from zero_lab.games import GameState


@dataclass(slots=True)
class EdgeStats:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: SearchNode | None = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(slots=True)
class SearchNode:
    state: GameState
    edges: dict[int, EdgeStats] = field(default_factory=dict)

    @property
    def visit_count(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())

    @property
    def is_expanded(self) -> bool:
        return bool(self.edges)

    def expand(self, priors: dict[int, float]) -> None:
        if self.state.is_terminal:
            self.edges.clear()
            return
        self.edges = {action: EdgeStats(prior=prior) for action, prior in priors.items()}

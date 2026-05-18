"""AlphaZero PUCT search."""

from __future__ import annotations

import math
from dataclasses import dataclass

from zero_lab.games import GameState
from zero_lab.search.alpha_zero.config import MCTSSearchConfig
from zero_lab.search.alpha_zero.evaluator import AlphaZeroEvaluator
from zero_lab.search.alpha_zero.model_evaluator import as_evaluator, normalize_policy
from zero_lab.search.alpha_zero.tree import EdgeStats, SearchNode


@dataclass(frozen=True, slots=True)
class SearchResult:
    root: SearchNode
    visit_counts: dict[int, int]
    action_values: dict[int, float]
    policy: dict[int, float]

    @property
    def best_action(self) -> int:
        if not self.visit_counts:
            raise ValueError("search result has no available actions")
        return max(self.visit_counts, key=lambda action: (self.visit_counts[action], -action))


@dataclass(frozen=True, slots=True)
class AlphaZeroSearch:
    evaluator: AlphaZeroEvaluator
    config: MCTSSearchConfig = MCTSSearchConfig()

    def __init__(
        self,
        evaluator: AlphaZeroEvaluator,
        config: MCTSSearchConfig | None = None,
    ) -> None:
        object.__setattr__(self, "evaluator", as_evaluator(evaluator))
        object.__setattr__(self, "config", MCTSSearchConfig() if config is None else config)

    def run(self, root_state: GameState) -> SearchResult:
        root = SearchNode(root_state)
        if root_state.is_terminal:
            return SearchResult(root=root, visit_counts={}, action_values={}, policy={})

        self._expand(root)
        for _ in range(self.config.simulations):
            self._run_simulation(root)

        visit_counts = {
            action: edge.visit_count
            for action, edge in sorted(root.edges.items())
            if edge.visit_count > 0
        }
        action_values = {
            action: edge.value
            for action, edge in sorted(root.edges.items())
            if edge.visit_count > 0
        }
        total_visits = sum(visit_counts.values())
        policy = (
            {action: count / total_visits for action, count in visit_counts.items()}
            if total_visits > 0
            else {}
        )
        return SearchResult(
            root=root,
            visit_counts=visit_counts,
            action_values=action_values,
            policy=policy,
        )

    def _run_simulation(self, root: SearchNode) -> None:
        node = root
        path: list[EdgeStats] = []

        while node.is_expanded and not node.state.is_terminal:
            _action, edge = self._select_child(node)
            if edge.child is None:
                edge.child = SearchNode(node.state.apply(_action))
            path.append(edge)
            node = edge.child

        if node.state.is_terminal:
            outcome = node.state.outcome_for(node.state.current_player)
            value = 0.0 if outcome is None else float(outcome)
        else:
            value = self._expand(node)

        self._backpropagate(path, value)

    def _expand(self, node: SearchNode) -> float:
        evaluation = self.evaluator.evaluate(node.state)
        legal_actions = node.state.legal_actions()
        priors = normalize_policy(dict(evaluation.policy), legal_actions)
        node.expand(priors)
        return evaluation.value

    def _select_child(self, node: SearchNode) -> tuple[int, EdgeStats]:
        parent_visits = node.visit_count
        return max(
            sorted(node.edges.items()),
            key=lambda item: self._puct_score(parent_visits, item[1]),
        )

    def _puct_score(self, parent_visits: int, edge: EdgeStats) -> float:
        pb_c = (
            math.log((parent_visits + self.config.pb_c_base + 1.0) / self.config.pb_c_base)
            + self.config.pb_c_init
        )
        exploration = pb_c * edge.prior * math.sqrt(parent_visits + 1.0) / (edge.visit_count + 1)
        return edge.value + exploration

    def _backpropagate(self, path: list[EdgeStats], leaf_value: float) -> None:
        value = leaf_value
        for edge in reversed(path):
            value = -value
            edge.value_sum += value
            edge.visit_count += 1

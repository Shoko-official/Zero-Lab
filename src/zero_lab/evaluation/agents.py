"""Baseline agents for evaluation matches."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from zero_lab.games import GameState
from zero_lab.search import AlphaZeroSearch, MCTSSearchConfig, select_action_by_temperature
from zero_lab.search.alpha_zero import AlphaZeroEvaluator, UniformEvaluator


class EvaluationAgent(Protocol):
    @property
    def name(self) -> str:
        pass

    def select_action(self, state: GameState, *, rng: random.Random) -> int:
        pass


@dataclass(frozen=True, slots=True)
class RandomLegalMoveAgent:
    label: str = "random_legal"

    @property
    def name(self) -> str:
        return self.label

    def select_action(self, state: GameState, *, rng: random.Random) -> int:
        legal_actions = state.legal_actions()
        if not legal_actions:
            raise ValueError("cannot select an action without legal actions")
        return rng.choice(legal_actions)


@dataclass(frozen=True, slots=True)
class AlphaZeroSearchAgent:
    evaluator: AlphaZeroEvaluator
    simulations: int = 32
    temperature: float = 0.0
    label: str = "alpha_zero_search"

    def __post_init__(self) -> None:
        MCTSSearchConfig(simulations=self.simulations)
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")

    @property
    def name(self) -> str:
        return self.label

    def select_action(self, state: GameState, *, rng: random.Random) -> int:
        result = AlphaZeroSearch(
            self.evaluator,
            MCTSSearchConfig(simulations=self.simulations),
        ).run(state)
        return select_action_by_temperature(
            result.visit_counts,
            temperature=self.temperature,
            rng=rng,
        )


@dataclass(frozen=True, slots=True)
class UniformSearchAgent:
    simulations: int = 32
    temperature: float = 0.0
    label: str = "uniform_search"

    def __post_init__(self) -> None:
        MCTSSearchConfig(simulations=self.simulations)
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")

    @property
    def name(self) -> str:
        return self.label

    def select_action(self, state: GameState, *, rng: random.Random) -> int:
        return AlphaZeroSearchAgent(
            UniformEvaluator(),
            simulations=self.simulations,
            temperature=self.temperature,
            label=self.label,
        ).select_action(state, rng=rng)

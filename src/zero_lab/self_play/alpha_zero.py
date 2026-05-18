"""AlphaZero self-play episode generation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from zero_lab.games import GameRules, GameState
from zero_lab.replay import EpisodeRecord, EpisodeStep
from zero_lab.search import (
    AlphaZeroSearch,
    MCTSSearchConfig,
    select_action_by_temperature,
    visit_count_policy,
)
from zero_lab.search.alpha_zero import AlphaZeroEvaluator


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    max_moves: int = 512
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.max_moves, bool) or self.max_moves <= 0:
            raise ValueError("max_moves must be a positive integer")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")


@dataclass(frozen=True, slots=True)
class PendingStep:
    state: str
    current_player: int
    action: int
    policy: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AlphaZeroSelfPlay:
    evaluator: AlphaZeroEvaluator
    search_config: MCTSSearchConfig = MCTSSearchConfig()
    config: SelfPlayConfig = SelfPlayConfig()

    def play(self, game: GameRules, *, seed: int | None = None) -> EpisodeRecord:
        rng = random.Random(seed)
        state = game.reset(seed=seed)
        pending_steps: list[PendingStep] = []

        while not state.is_terminal and len(pending_steps) < self.config.max_moves:
            result = AlphaZeroSearch(self.evaluator, self.search_config).run(state)
            action = select_action_by_temperature(
                result.visit_counts,
                temperature=self.config.temperature,
                rng=rng,
            )
            pending_steps.append(
                PendingStep(
                    action=action,
                    current_player=state.current_player,
                    policy=visit_count_policy(result.visit_counts, state.action_size),
                    state=state.serialize(),
                )
            )
            state = state.apply(action)

        return self._build_record(game=game, terminal_state=state, pending_steps=pending_steps)

    def _build_record(
        self,
        *,
        game: GameRules,
        terminal_state: GameState,
        pending_steps: list[PendingStep],
    ) -> EpisodeRecord:
        outcome = terminal_state.outcome_for(1) if terminal_state.is_terminal else None
        steps = tuple(
            EpisodeStep(
                action=step.action,
                current_player=step.current_player,
                policy=step.policy,
                state=step.state,
                value_target=_value_target(terminal_state, step.current_player),
            )
            for step in pending_steps
        )
        return EpisodeRecord(
            game=game.name,
            outcome=outcome,
            steps=steps,
            terminal_state=terminal_state.serialize(),
        )


def _value_target(terminal_state: GameState, player: int) -> float:
    if not terminal_state.is_terminal:
        return 0.0
    outcome = terminal_state.outcome_for(player)
    return 0.0 if outcome is None else float(outcome)

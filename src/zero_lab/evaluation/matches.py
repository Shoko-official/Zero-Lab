"""Fixed-seed head-to-head evaluation matches."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from zero_lab.evaluation.agents import EvaluationAgent
from zero_lab.games import GameRules
from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER


@dataclass(frozen=True, slots=True)
class MatchConfig:
    seed: int = 0
    games_per_side: int = 1
    max_moves: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.games_per_side, bool) or self.games_per_side <= 0:
            raise ValueError("games_per_side must be a positive integer")
        if isinstance(self.max_moves, bool) or self.max_moves <= 0:
            raise ValueError("max_moves must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "games_per_side": self.games_per_side,
            "max_moves": self.max_moves,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    game: str
    seed: int
    first_agent: str
    second_agent: str
    moves: int
    terminal: bool
    outcome_for_first: int | None
    winner: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "first_agent": self.first_agent,
            "game": self.game,
            "moves": self.moves,
            "outcome_for_first": self.outcome_for_first,
            "second_agent": self.second_agent,
            "seed": self.seed,
            "terminal": self.terminal,
            "winner": self.winner,
        }


def run_head_to_head(
    *,
    games: Sequence[GameRules],
    first_agent: EvaluationAgent,
    second_agent: EvaluationAgent,
    config: MatchConfig | None = None,
) -> tuple[MatchResult, ...]:
    active_config = MatchConfig() if config is None else config
    results: list[MatchResult] = []
    match_index = 0
    for game in games:
        for _ in range(active_config.games_per_side):
            results.append(
                play_match(
                    game=game,
                    first_agent=first_agent,
                    second_agent=second_agent,
                    seed=active_config.seed + match_index,
                    max_moves=active_config.max_moves,
                )
            )
            match_index += 1
            results.append(
                play_match(
                    game=game,
                    first_agent=second_agent,
                    second_agent=first_agent,
                    seed=active_config.seed + match_index,
                    max_moves=active_config.max_moves,
                )
            )
            match_index += 1
    return tuple(results)


def play_match(
    *,
    game: GameRules,
    first_agent: EvaluationAgent,
    second_agent: EvaluationAgent,
    seed: int,
    max_moves: int,
) -> MatchResult:
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(max_moves, bool) or max_moves <= 0:
        raise ValueError("max_moves must be a positive integer")

    rng = random.Random(seed)
    state = game.reset(seed=seed)
    agents_by_player = {
        FIRST_PLAYER: first_agent,
        SECOND_PLAYER: second_agent,
    }
    moves = 0

    while not state.is_terminal and moves < max_moves:
        agent = agents_by_player[state.current_player]
        action = agent.select_action(state, rng=rng)
        if action not in state.legal_actions():
            raise ValueError(f"{agent.name} selected illegal action {action}")
        state = state.apply(action)
        moves += 1

    outcome = state.outcome_for(FIRST_PLAYER) if state.is_terminal else None
    winner = _winner_name(
        outcome_for_first=outcome,
        first_agent=first_agent,
        second_agent=second_agent,
    )
    return MatchResult(
        game=game.name,
        seed=seed,
        first_agent=first_agent.name,
        second_agent=second_agent.name,
        moves=moves,
        terminal=state.is_terminal,
        outcome_for_first=outcome,
        winner=winner,
    )


def _winner_name(
    *,
    outcome_for_first: int | None,
    first_agent: EvaluationAgent,
    second_agent: EvaluationAgent,
) -> str | None:
    if outcome_for_first == 1:
        return first_agent.name
    if outcome_for_first == -1:
        return second_agent.name
    return None

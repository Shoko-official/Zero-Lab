"""Checkpoint comparison helpers for AlphaZero promotion evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from zero_lab.evaluation.agents import EvaluationAgent
from zero_lab.evaluation.matches import MatchConfig, MatchResult, run_head_to_head
from zero_lab.games import GameRules, GameState


@dataclass(frozen=True, slots=True)
class AlphaZeroCheckpoint:
    name: str
    uri: str
    commit_hash: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.uri:
            raise ValueError("uri must not be empty")
        if not self.commit_hash:
            raise ValueError("commit_hash must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "commit_hash": self.commit_hash,
            "name": self.name,
            "uri": self.uri,
        }


@dataclass(frozen=True, slots=True)
class CheckpointAgent:
    checkpoint: AlphaZeroCheckpoint
    agent: EvaluationAgent

    @property
    def name(self) -> str:
        return self.checkpoint.name

    def select_action(self, state: GameState, *, rng: random.Random) -> int:
        return self.agent.select_action(state, rng=rng)


@dataclass(frozen=True, slots=True)
class CheckpointComparison:
    champion: AlphaZeroCheckpoint
    candidate: AlphaZeroCheckpoint
    results: tuple[MatchResult, ...]


def compare_alpha_zero_checkpoints(
    *,
    champion: AlphaZeroCheckpoint,
    candidate: AlphaZeroCheckpoint,
    champion_agent: EvaluationAgent,
    candidate_agent: EvaluationAgent,
    games: Sequence[GameRules],
    config: MatchConfig | None = None,
) -> CheckpointComparison:
    if champion.name == candidate.name:
        raise ValueError("champion and candidate checkpoint names must differ")

    results = run_head_to_head(
        games=games,
        first_agent=CheckpointAgent(champion, champion_agent),
        second_agent=CheckpointAgent(candidate, candidate_agent),
        config=config,
    )
    return CheckpointComparison(
        champion=champion,
        candidate=candidate,
        results=results,
    )

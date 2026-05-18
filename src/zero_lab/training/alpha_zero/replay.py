"""Replay collation for AlphaZero training batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from zero_lab.games import GameRules
from zero_lab.replay import AlphaZeroSample, iter_alpha_zero_samples
from zero_lab.training.alpha_zero.batches import AlphaZeroTrainingBatch


def build_alpha_zero_training_batch(
    samples: Sequence[AlphaZeroSample],
    *,
    games: Mapping[str, GameRules],
) -> AlphaZeroTrainingBatch:
    if not samples:
        raise ValueError("samples must not be empty")

    observations: list[tuple[int, ...]] = []
    legal_action_masks: list[tuple[bool, ...]] = []
    target_policies: list[tuple[float, ...]] = []
    target_values: list[float] = []
    selected_actions: list[int] = []
    current_players: list[int] = []
    action_size: int | None = None

    for sample in samples:
        game = games.get(sample.game)
        if game is None:
            raise ValueError(f"unknown replay game: {sample.game}")

        state = game.deserialize(sample.state)
        if state.current_player != sample.current_player:
            raise ValueError("sample current_player does not match serialized state")
        if len(sample.policy) != game.action_size:
            raise ValueError("sample policy action dimension does not match game action_size")

        if action_size is None:
            action_size = game.action_size
        elif action_size != game.action_size:
            raise ValueError("samples in one batch must share the same action_size")

        observations.append(state.canonical_observation())
        legal_action_masks.append(state.legal_action_mask())
        target_policies.append(sample.policy)
        target_values.append(sample.value_target)
        selected_actions.append(sample.action)
        current_players.append(sample.current_player)

    if action_size is None:
        raise ValueError("samples must not be empty")

    return AlphaZeroTrainingBatch.from_sequences(
        observations=observations,
        legal_action_masks=legal_action_masks,
        target_policies=target_policies,
        target_values=target_values,
        selected_actions=selected_actions,
        current_players=current_players,
        action_size=action_size,
    )


def iter_alpha_zero_training_batches(
    paths: Path | Sequence[Path],
    *,
    games: Mapping[str, GameRules],
    batch_size: int,
    drop_remainder: bool = False,
) -> Iterator[AlphaZeroTrainingBatch]:
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    batch_samples: list[AlphaZeroSample] = []
    for sample in iter_alpha_zero_samples(paths):
        batch_samples.append(sample)
        if len(batch_samples) == batch_size:
            yield build_alpha_zero_training_batch(batch_samples, games=games)
            batch_samples = []

    if batch_samples and not drop_remainder:
        yield build_alpha_zero_training_batch(batch_samples, games=games)

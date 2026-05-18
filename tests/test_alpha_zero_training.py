from __future__ import annotations

from pathlib import Path

import pytest

from zero_lab.games.toy import TicTacToeGame, TicTacToeState
from zero_lab.replay import AlphaZeroSample, EpisodeRecord, EpisodeStep, append_episode
from zero_lab.training.alpha_zero import (
    AlphaZeroBatchSummary,
    AlphaZeroTrainingBatch,
    build_alpha_zero_training_batch,
    iter_alpha_zero_training_batches,
    summarize_alpha_zero_training_batches,
)


def test_alpha_zero_training_batch_validates_targets_and_builds_model_batch() -> None:
    batch = AlphaZeroTrainingBatch.from_sequences(
        observations=((1, 0, -1), (0, 1, 0)),
        legal_action_masks=((True, False), (True, True)),
        target_policies=((1.0, 0.0), (0.25, 0.75)),
        target_values=(1, -1),
        selected_actions=(0, 1),
        current_players=(1, -1),
        action_size=2,
    )

    model_batch = batch.to_model_batch()

    assert batch.shape.batch_size == 2
    assert batch.shape.observation_size == 3
    assert batch.shape.action_size == 2
    assert model_batch.observations == batch.observations
    assert model_batch.legal_action_masks == batch.legal_action_masks
    assert model_batch.shape == batch.shape


def test_alpha_zero_training_batch_rejects_invalid_policy_target() -> None:
    with pytest.raises(ValueError, match="target_policies must sum to 1"):
        AlphaZeroTrainingBatch.from_sequences(
            observations=((1, 0, -1),),
            legal_action_masks=((True, False),),
            target_policies=((0.2, 0.2),),
            target_values=(1,),
            selected_actions=(0,),
            current_players=(1,),
            action_size=2,
        )


def test_alpha_zero_training_batch_rejects_illegal_policy_mass() -> None:
    with pytest.raises(ValueError, match="illegal actions"):
        AlphaZeroTrainingBatch.from_sequences(
            observations=((1, 0, -1),),
            legal_action_masks=((True, False),),
            target_policies=((0.5, 0.5),),
            target_values=(1,),
            selected_actions=(0,),
            current_players=(1,),
            action_size=2,
        )


def test_alpha_zero_training_batch_rejects_out_of_range_action() -> None:
    with pytest.raises(ValueError, match="selected_actions must be within"):
        AlphaZeroTrainingBatch.from_sequences(
            observations=((1, 0, -1),),
            legal_action_masks=((True, False),),
            target_policies=((1.0, 0.0),),
            target_values=(1,),
            selected_actions=(2,),
            current_players=(1,),
            action_size=2,
        )


def test_alpha_zero_training_batch_rejects_illegal_selected_action() -> None:
    with pytest.raises(ValueError, match="selected_actions must be legal"):
        AlphaZeroTrainingBatch.from_sequences(
            observations=((1, 0, -1),),
            legal_action_masks=((True, False),),
            target_policies=((1.0, 0.0),),
            target_values=(1,),
            selected_actions=(1,),
            current_players=(1,),
            action_size=2,
        )


def test_alpha_zero_training_batch_rejects_invalid_current_player() -> None:
    with pytest.raises(ValueError, match="current_players must contain"):
        AlphaZeroTrainingBatch.from_sequences(
            observations=((1, 0, -1),),
            legal_action_masks=((True, False),),
            target_policies=((1.0, 0.0),),
            target_values=(1,),
            selected_actions=(0,),
            current_players=(0,),
            action_size=2,
        )


def test_build_alpha_zero_training_batch_decodes_replay_samples() -> None:
    state = TicTacToeState(board=(1, 0, 0, 0, -1, 0, 0, 0, 0), current_player=1)
    sample = AlphaZeroSample(
        action=1,
        current_player=1,
        game="tic_tac_toe",
        policy=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        state=state.serialize(),
        value_target=1.0,
    )

    batch = build_alpha_zero_training_batch(
        (sample,),
        games={"tic_tac_toe": TicTacToeGame()},
    )

    assert batch.observations == ((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0),)
    assert batch.legal_action_masks == ((False, True, True, True, False, True, True, True, True),)
    assert batch.target_policies == (sample.policy,)
    assert batch.target_values == (1.0,)
    assert batch.selected_actions == (1,)


def test_iter_alpha_zero_training_batches_streams_replay_file(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    first_state = TicTacToeState()
    second_state = first_state.apply(0)
    append_episode(
        path,
        EpisodeRecord(
            game="tic_tac_toe",
            outcome=1,
            steps=(
                EpisodeStep(
                    action=0,
                    current_player=1,
                    policy=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    state=first_state.serialize(),
                    value_target=1.0,
                ),
                EpisodeStep(
                    action=4,
                    current_player=-1,
                    policy=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                    state=second_state.serialize(),
                    value_target=-1.0,
                ),
            ),
            terminal_state=second_state.apply(4).serialize(),
        ),
    )

    batches = list(
        iter_alpha_zero_training_batches(
            path,
            games={"tic_tac_toe": TicTacToeGame()},
            batch_size=1,
        )
    )

    assert len(batches) == 2
    assert batches[0].selected_actions == (0,)
    assert batches[1].selected_actions == (4,)


def test_iter_alpha_zero_training_batches_can_drop_remainder(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    state = TicTacToeState()
    append_episode(
        path,
        EpisodeRecord(
            game="tic_tac_toe",
            outcome=1,
            steps=(
                EpisodeStep(
                    action=0,
                    current_player=1,
                    policy=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    state=state.serialize(),
                    value_target=1.0,
                ),
            ),
            terminal_state=state.apply(0).serialize(),
        ),
    )

    batches = list(
        iter_alpha_zero_training_batches(
            path,
            games={"tic_tac_toe": TicTacToeGame()},
            batch_size=2,
            drop_remainder=True,
        )
    )

    assert batches == []


def test_build_alpha_zero_training_batch_rejects_unknown_game() -> None:
    sample = AlphaZeroSample(
        action=0,
        current_player=1,
        game="unknown",
        policy=(1.0,),
        state="{}",
        value_target=0.0,
    )

    with pytest.raises(ValueError, match="unknown replay game"):
        build_alpha_zero_training_batch((sample,), games={})


def test_build_alpha_zero_training_batch_rejects_mismatched_current_player() -> None:
    state = TicTacToeState(current_player=1)
    sample = AlphaZeroSample(
        action=0,
        current_player=-1,
        game="tic_tac_toe",
        policy=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        state=state.serialize(),
        value_target=0.0,
    )

    with pytest.raises(ValueError, match="current_player"):
        build_alpha_zero_training_batch((sample,), games={"tic_tac_toe": TicTacToeGame()})


def test_summarize_alpha_zero_training_batches_counts_full_and_partial_batches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episodes.jsonl"
    state = TicTacToeState()
    append_episode(
        path,
        EpisodeRecord(
            game="tic_tac_toe",
            outcome=1,
            steps=(
                EpisodeStep(
                    action=0,
                    current_player=1,
                    policy=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    state=state.serialize(),
                    value_target=1.0,
                ),
            ),
            terminal_state=state.apply(0).serialize(),
        ),
    )

    summary = summarize_alpha_zero_training_batches(
        path,
        games={"tic_tac_toe": TicTacToeGame()},
        batch_size=2,
    )

    assert summary == AlphaZeroBatchSummary(
        source_samples=1,
        emitted_samples=1,
        batches=1,
        batch_size=2,
        drop_remainder=False,
        remainder_samples=0,
        action_sizes=(9,),
        observation_sizes=(9,),
    )
    assert summary.to_dict() == {
        "action_sizes": [9],
        "batch_size": 2,
        "batches": 1,
        "drop_remainder": False,
        "emitted_samples": 1,
        "observation_sizes": [9],
        "remainder_samples": 0,
        "source_samples": 1,
    }


def test_summarize_alpha_zero_training_batches_counts_dropped_remainder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episodes.jsonl"
    state = TicTacToeState()
    append_episode(
        path,
        EpisodeRecord(
            game="tic_tac_toe",
            outcome=1,
            steps=(
                EpisodeStep(
                    action=0,
                    current_player=1,
                    policy=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    state=state.serialize(),
                    value_target=1.0,
                ),
            ),
            terminal_state=state.apply(0).serialize(),
        ),
    )

    summary = summarize_alpha_zero_training_batches(
        path,
        games={"tic_tac_toe": TicTacToeGame()},
        batch_size=2,
        drop_remainder=True,
    )

    assert summary.source_samples == 1
    assert summary.emitted_samples == 0
    assert summary.batches == 0
    assert summary.remainder_samples == 1

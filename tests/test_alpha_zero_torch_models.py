from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zero_lab.games.toy import TicTacToeGame, TicTacToeState
from zero_lab.replay import EpisodeRecord, EpisodeStep, append_episode
from zero_lab.training.alpha_zero import (
    AlphaZeroTrainerConfig,
    LinearAlphaZeroModel,
    MLPAlphaZeroModel,
    load_linear_alpha_zero_evaluator_checkpoint,
    train_alpha_zero_from_replay,
)


def test_linear_alpha_zero_model_returns_policy_logits_and_values() -> None:
    model = LinearAlphaZeroModel(observation_size=9, action_size=9)
    observations = torch.zeros((2, 9), dtype=torch.float32)

    policy_logits, values = model(observations)

    assert policy_logits.shape == torch.Size((2, 9))
    assert values.shape == torch.Size((2,))


def test_mlp_alpha_zero_model_returns_policy_logits_and_values() -> None:
    model = MLPAlphaZeroModel(observation_size=9, action_size=9, hidden_size=16)
    observations = torch.zeros((2, 9), dtype=torch.float32)

    policy_logits, values = model(observations)

    assert policy_logits.shape == torch.Size((2, 9))
    assert values.shape == torch.Size((2,))


def test_linear_alpha_zero_model_validates_sizes() -> None:
    with pytest.raises(ValueError, match="observation_size"):
        LinearAlphaZeroModel(observation_size=0, action_size=9)
    with pytest.raises(ValueError, match="action_size"):
        LinearAlphaZeroModel(observation_size=9, action_size=0)


def test_mlp_alpha_zero_model_validates_sizes() -> None:
    with pytest.raises(ValueError, match="observation_size"):
        MLPAlphaZeroModel(observation_size=0, action_size=9, hidden_size=16)
    with pytest.raises(ValueError, match="action_size"):
        MLPAlphaZeroModel(observation_size=9, action_size=0, hidden_size=16)
    with pytest.raises(ValueError, match="hidden_size"):
        MLPAlphaZeroModel(observation_size=9, action_size=9, hidden_size=0)


def test_load_linear_alpha_zero_evaluator_checkpoint_evaluates_restored_model(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "episodes.jsonl"
    checkpoint_path = tmp_path / "checkpoints" / "alpha-zero.pt"
    write_replay(replay_path)
    model = LinearAlphaZeroModel(observation_size=9, action_size=9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = train_alpha_zero_from_replay(
        replay_path,
        games={"tic_tac_toe": TicTacToeGame()},
        model=model,
        optimizer=optimizer,
        config=AlphaZeroTrainerConfig(
            batch_size=1,
            max_steps=1,
            checkpoint_path=checkpoint_path,
        ),
    )

    loaded = load_linear_alpha_zero_evaluator_checkpoint(
        checkpoint_path,
        observation_size=9,
        action_size=9,
    )
    evaluation = loaded.evaluator.evaluate(TicTacToeState())

    assert loaded.metadata.steps == result.steps
    assert loaded.metadata.samples == result.samples
    assert set(evaluation.policy) == set(TicTacToeState().legal_actions())
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)


def write_replay(path: Path) -> None:
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

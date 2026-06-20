from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zero_lab.games.toy import TicTacToeGame, TicTacToeState
from zero_lab.replay import EpisodeRecord, EpisodeStep, append_episode
from zero_lab.training.alpha_zero import (
    AlphaZeroCheckpointMetadata,
    AlphaZeroTrainerConfig,
    load_alpha_zero_checkpoint,
    load_alpha_zero_model_checkpoint,
    train_alpha_zero_from_replay,
)


class TinyAlphaZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = torch.nn.Linear(9, 9)
        self.value = torch.nn.Linear(9, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(observations), self.value(observations).squeeze(dim=1)


def test_train_alpha_zero_from_replay_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    replay_path = write_replay(tmp_path / "episodes.jsonl")
    checkpoint_path = tmp_path / "checkpoints" / "alpha-zero.pt"
    model = TinyAlphaZeroModel()
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

    assert result.steps == 1
    assert result.samples == 1
    assert result.checkpoint_path == checkpoint_path
    assert result.last_total_loss > 0.0
    assert checkpoint_path.exists()
    assert result.to_dict()["checkpoint_path"] == str(checkpoint_path)


def test_train_alpha_zero_from_replay_resumes_checkpoint(tmp_path: Path) -> None:
    replay_path = write_replay(tmp_path / "episodes.jsonl")
    checkpoint_path = tmp_path / "checkpoints" / "alpha-zero.pt"
    first_model = TinyAlphaZeroModel()
    first_optimizer = torch.optim.SGD(first_model.parameters(), lr=0.1)
    train_alpha_zero_from_replay(
        replay_path,
        games={"tic_tac_toe": TicTacToeGame()},
        model=first_model,
        optimizer=first_optimizer,
        config=AlphaZeroTrainerConfig(
            batch_size=1,
            max_steps=1,
            checkpoint_path=checkpoint_path,
        ),
    )

    second_model = TinyAlphaZeroModel()
    second_optimizer = torch.optim.SGD(second_model.parameters(), lr=0.1)
    result = train_alpha_zero_from_replay(
        replay_path,
        games={"tic_tac_toe": TicTacToeGame()},
        model=second_model,
        optimizer=second_optimizer,
        config=AlphaZeroTrainerConfig(
            batch_size=1,
            max_steps=1,
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path,
        ),
    )

    assert result.steps == 2
    assert result.samples == 2
    loaded_model = TinyAlphaZeroModel()
    loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.1)
    metadata = load_alpha_zero_checkpoint(
        checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
    )
    assert metadata == AlphaZeroCheckpointMetadata(
        steps=2,
        samples=2,
        last_total_loss=result.last_total_loss,
        last_policy_loss=result.last_policy_loss,
        last_value_loss=result.last_value_loss,
    )
    assert metadata.to_dict()["steps"] == 2


def test_load_alpha_zero_model_checkpoint_restores_model_without_optimizer(tmp_path: Path) -> None:
    replay_path = write_replay(tmp_path / "episodes.jsonl")
    checkpoint_path = tmp_path / "checkpoints" / "alpha-zero.pt"
    trained_model = TinyAlphaZeroModel()
    optimizer = torch.optim.SGD(trained_model.parameters(), lr=0.1)
    result = train_alpha_zero_from_replay(
        replay_path,
        games={"tic_tac_toe": TicTacToeGame()},
        model=trained_model,
        optimizer=optimizer,
        config=AlphaZeroTrainerConfig(
            batch_size=1,
            max_steps=1,
            checkpoint_path=checkpoint_path,
        ),
    )
    restored_model = TinyAlphaZeroModel()

    metadata = load_alpha_zero_model_checkpoint(checkpoint_path, model=restored_model)

    assert metadata == AlphaZeroCheckpointMetadata(
        steps=result.steps,
        samples=result.samples,
        last_total_loss=result.last_total_loss,
        last_policy_loss=result.last_policy_loss,
        last_value_loss=result.last_value_loss,
    )
    for trained_parameter, restored_parameter in zip(
        trained_model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.equal(trained_parameter, restored_parameter)


def test_train_alpha_zero_from_replay_rejects_empty_replay(tmp_path: Path) -> None:
    replay_path = tmp_path / "empty.jsonl"
    replay_path.write_text("", encoding="utf-8")
    model = TinyAlphaZeroModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="no training batches"):
        train_alpha_zero_from_replay(
            replay_path,
            games={"tic_tac_toe": TicTacToeGame()},
            model=model,
            optimizer=optimizer,
            config=AlphaZeroTrainerConfig(batch_size=1, max_steps=1),
        )


def write_replay(path: Path) -> Path:
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
    return path

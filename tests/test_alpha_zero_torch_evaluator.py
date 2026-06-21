from __future__ import annotations

import pytest
import torch

from zero_lab.games.toy import TicTacToeState
from zero_lab.training.alpha_zero import TorchAlphaZeroEvaluator


class CenterBiasedModule(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros((observations.shape[0], 9), dtype=observations.dtype)
        logits[:, 4] = 8.0
        values = torch.full((observations.shape[0],), 0.25, dtype=observations.dtype)
        return logits, values


class CountingModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        logits = torch.zeros((observations.shape[0], 9), dtype=observations.dtype)
        values = torch.zeros((observations.shape[0],), dtype=observations.dtype)
        return logits, values


class BadShapeModule(torch.nn.Module):
    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros((observations.shape[0], 8), dtype=observations.dtype)
        values = torch.zeros((observations.shape[0],), dtype=observations.dtype)
        return logits, values


def test_torch_alpha_zero_evaluator_filters_policy_to_legal_actions() -> None:
    state = TicTacToeState().apply(4)

    evaluation = TorchAlphaZeroEvaluator(CenterBiasedModule()).evaluate(state)

    assert 4 not in evaluation.policy
    assert set(evaluation.policy) == set(state.legal_actions())
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)
    assert evaluation.value == pytest.approx(0.25)


def test_torch_alpha_zero_evaluator_skips_terminal_states() -> None:
    state = TicTacToeState()
    for action in (0, 3, 1, 4, 2):
        state = state.apply(action)
    model = CountingModule()

    evaluation = TorchAlphaZeroEvaluator(model).evaluate(state)

    assert model.calls == 0
    assert evaluation.policy == {}
    assert evaluation.value == pytest.approx(-1.0)


def test_torch_alpha_zero_evaluator_restores_training_mode() -> None:
    model = CenterBiasedModule()
    model.train()

    TorchAlphaZeroEvaluator(model).evaluate(TicTacToeState())

    assert model.training is True


def test_torch_alpha_zero_evaluator_rejects_bad_policy_shape() -> None:
    with pytest.raises(ValueError, match="policy logits shape"):
        TorchAlphaZeroEvaluator(BadShapeModule()).evaluate(TicTacToeState())

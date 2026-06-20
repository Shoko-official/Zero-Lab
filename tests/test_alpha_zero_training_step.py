from __future__ import annotations

import pytest
import torch

from zero_lab.training.alpha_zero import (
    AlphaZeroLossConfig,
    AlphaZeroTrainStepConfig,
    TorchAlphaZeroTrainingBatch,
    alpha_zero_loss,
    as_torch_alpha_zero_training_batch,
    run_alpha_zero_training_step,
)
from zero_lab.training.alpha_zero.batches import AlphaZeroTrainingBatch


class TinyAlphaZeroModel(torch.nn.Module):
    def __init__(self, *, column_values: bool = False) -> None:
        super().__init__()
        self.policy = torch.nn.Linear(3, 2)
        self.value = torch.nn.Linear(3, 1)
        self.column_values = column_values

        with torch.no_grad():
            self.policy.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.3],
                        [-0.3, 0.4, 0.1],
                    ],
                    dtype=torch.float32,
                )
            )
            self.policy.bias.copy_(torch.tensor([0.05, -0.05], dtype=torch.float32))
            self.value.weight.copy_(torch.tensor([[0.1, -0.2, 0.25]], dtype=torch.float32))
            self.value.bias.zero_()

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.value(observations)
        if self.column_values:
            return self.policy(observations), values
        return self.policy(observations), values.squeeze(dim=1)


class InvalidOutputModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations + self.bias


def make_torch_batch() -> TorchAlphaZeroTrainingBatch:
    batch = AlphaZeroTrainingBatch.from_sequences(
        observations=((1, 0, -1), (0, 1, 0)),
        legal_action_masks=((True, False), (True, True)),
        target_policies=((1.0, 0.0), (0.25, 0.75)),
        target_values=(1, -1),
        selected_actions=(0, 1),
        current_players=(1, -1),
        action_size=2,
    )
    return as_torch_alpha_zero_training_batch(batch)


def test_alpha_zero_train_step_config_rejects_invalid_gradient_clip() -> None:
    with pytest.raises(ValueError, match="max_grad_norm"):
        AlphaZeroTrainStepConfig(max_grad_norm=0.0)


def test_run_alpha_zero_training_step_updates_model_parameters() -> None:
    model = TinyAlphaZeroModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())

    result = run_alpha_zero_training_step(model, optimizer, make_torch_batch())

    after = tuple(parameter.detach() for parameter in model.parameters())
    assert any(
        not torch.allclose(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(before, after, strict=True)
    )
    assert result.batch_size == 2
    assert result.action_size == 2
    assert result.total_loss > 0.0
    assert result.policy_loss > 0.0
    assert result.value_loss > 0.0
    assert result.grad_norm is None


def test_run_alpha_zero_training_step_uses_loss_config() -> None:
    model = TinyAlphaZeroModel()
    batch = make_torch_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss_config = AlphaZeroLossConfig(policy_weight=2.0, value_weight=0.5, value_loss="huber")
    logits, values = model(batch.observations)
    expected = alpha_zero_loss(logits, values, batch, config=loss_config)

    result = run_alpha_zero_training_step(
        model,
        optimizer,
        batch,
        config=AlphaZeroTrainStepConfig(loss=loss_config),
    )

    assert result.total_loss == pytest.approx(float(expected.total_loss.detach().item()))
    assert result.policy_loss == pytest.approx(float(expected.policy_loss.detach().item()))
    assert result.value_loss == pytest.approx(float(expected.value_loss.detach().item()))


def test_run_alpha_zero_training_step_clips_gradients() -> None:
    model = TinyAlphaZeroModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = run_alpha_zero_training_step(
        model,
        optimizer,
        make_torch_batch(),
        config=AlphaZeroTrainStepConfig(max_grad_norm=0.05),
    )

    assert result.grad_norm is not None
    assert result.grad_norm > 0.05
    assert _current_grad_norm(model) <= 0.05 + 1e-6


def test_run_alpha_zero_training_step_accepts_column_value_output() -> None:
    model = TinyAlphaZeroModel(column_values=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    result = run_alpha_zero_training_step(model, optimizer, make_torch_batch())

    assert result.batch_size == 2
    assert result.total_loss > 0.0


def test_run_alpha_zero_training_step_rejects_invalid_model_output() -> None:
    model = InvalidOutputModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="model output"):
        run_alpha_zero_training_step(model, optimizer, make_torch_batch())


def _current_grad_norm(model: torch.nn.Module) -> float:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())

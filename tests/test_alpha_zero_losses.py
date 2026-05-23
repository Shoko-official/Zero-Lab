from __future__ import annotations

import pytest
import torch

from zero_lab.training.alpha_zero import (
    AlphaZeroLossConfig,
    TorchAlphaZeroTrainingBatch,
    alpha_zero_loss,
    alpha_zero_policy_loss,
    alpha_zero_value_loss,
    as_torch_alpha_zero_training_batch,
)
from zero_lab.training.alpha_zero.batches import AlphaZeroTrainingBatch


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


def test_alpha_zero_loss_config_defaults_to_balanced_mse() -> None:
    config = AlphaZeroLossConfig()

    assert config.policy_weight == 1.0
    assert config.value_weight == 1.0
    assert config.value_loss == "mse"
    assert config.huber_delta == 1.0


def test_alpha_zero_loss_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="policy_weight"):
        AlphaZeroLossConfig(policy_weight=-1.0)


def test_alpha_zero_loss_config_rejects_invalid_huber_delta() -> None:
    with pytest.raises(ValueError, match="huber_delta"):
        AlphaZeroLossConfig(value_loss="huber", huber_delta=0.0)


def test_alpha_zero_policy_loss_matches_soft_target_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float32)

    loss = alpha_zero_policy_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

    assert torch.allclose(loss, expected)


def test_alpha_zero_policy_loss_rejects_wrong_shape() -> None:
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="shape"):
        alpha_zero_policy_loss(logits, targets)


def test_alpha_zero_policy_loss_rejects_invalid_probability_target() -> None:
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[0.2, 0.2]], dtype=torch.float32)

    with pytest.raises(ValueError, match="sum to 1"):
        alpha_zero_policy_loss(logits, targets)


def test_alpha_zero_policy_loss_rejects_negative_probability_target() -> None:
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[1.1, -0.1]], dtype=torch.float32)

    with pytest.raises(ValueError, match="non-negative"):
        alpha_zero_policy_loss(logits, targets)


def test_alpha_zero_value_loss_matches_mse() -> None:
    predicted = torch.tensor([1.0, -0.5], dtype=torch.float32)
    target = torch.tensor([0.0, -1.0], dtype=torch.float32)

    loss = alpha_zero_value_loss(predicted, target)
    expected = torch.nn.functional.mse_loss(predicted, target, reduction="mean")

    assert torch.allclose(loss, expected)


def test_alpha_zero_value_loss_matches_huber() -> None:
    predicted = torch.tensor([2.0, -0.5], dtype=torch.float32)
    target = torch.tensor([0.0, -1.0], dtype=torch.float32)

    loss = alpha_zero_value_loss(predicted, target, loss_kind="huber", huber_delta=1.0)
    expected = torch.nn.functional.huber_loss(predicted, target, reduction="mean", delta=1.0)

    assert torch.allclose(loss, expected)


def test_alpha_zero_value_loss_rejects_wrong_shape() -> None:
    predicted = torch.tensor([[1.0]], dtype=torch.float32)
    target = torch.tensor([1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="rank-1"):
        alpha_zero_value_loss(predicted, target)


def test_alpha_zero_loss_combines_weighted_components() -> None:
    batch = make_torch_batch()
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    values = torch.tensor([0.5, -0.25], dtype=torch.float32)
    config = AlphaZeroLossConfig(policy_weight=2.0, value_weight=0.5)

    losses = alpha_zero_loss(logits, values, batch, config=config)

    expected_policy = alpha_zero_policy_loss(logits, batch.target_policies)
    expected_value = alpha_zero_value_loss(values, batch.target_values)
    assert torch.allclose(losses.policy_loss, expected_policy)
    assert torch.allclose(losses.value_loss, expected_value)
    assert torch.allclose(losses.total_loss, 2.0 * expected_policy + 0.5 * expected_value)

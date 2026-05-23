from __future__ import annotations

import pytest
import torch

from zero_lab.training.alpha_zero import AlphaZeroLossConfig, alpha_zero_policy_loss


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

from __future__ import annotations

import pytest

from zero_lab.training.alpha_zero import AlphaZeroLossConfig


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

from __future__ import annotations

import pytest

from zero_lab.training.alpha_zero import AlphaZeroTrainingBatch


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

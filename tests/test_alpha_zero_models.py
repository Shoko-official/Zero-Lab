from __future__ import annotations

import pytest

from zero_lab.models.alpha_zero import AlphaZeroBatch, AlphaZeroOutput


def test_alpha_zero_batch_validates_shapes() -> None:
    batch = AlphaZeroBatch.from_sequences(
        observations=((1, 0, -1), (0, 1, 0)),
        legal_action_masks=((True, False), (False, True)),
        action_size=2,
    )

    assert batch.shape.batch_size == 2
    assert batch.shape.observation_size == 3
    assert batch.shape.action_size == 2


def test_alpha_zero_batch_rejects_empty_policy_mask() -> None:
    with pytest.raises(ValueError, match="at least one legal action"):
        AlphaZeroBatch.from_sequences(
            observations=((1, 0, -1),),
            legal_action_masks=((False, False),),
            action_size=2,
        )


def test_alpha_zero_output_validates_policy_and_value_shapes() -> None:
    batch = AlphaZeroBatch.from_sequences(
        observations=((1, 0, -1),),
        legal_action_masks=((True, False),),
        action_size=2,
    )

    output = AlphaZeroOutput.from_sequences(
        policy_logits=((0.2, -0.1),),
        values=(0.5,),
        batch=batch,
    )

    assert output.policy_value.values == (0.5,)


def test_alpha_zero_output_rejects_wrong_action_dimension() -> None:
    batch = AlphaZeroBatch.from_sequences(
        observations=((1, 0, -1),),
        legal_action_masks=((True, False),),
        action_size=2,
    )

    with pytest.raises(ValueError, match="action dimension"):
        AlphaZeroOutput.from_sequences(
            policy_logits=((0.2,),),
            values=(0.5,),
            batch=batch,
        )

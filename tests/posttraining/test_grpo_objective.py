from __future__ import annotations

import torch

from trainer.grpo.objective import (
    centered_group_advantages,
    clipped_grpo_loss,
    completion_reward,
    extract_json_object,
    plackett_luce_logprob,
)


def test_completion_reward_orders_invalid_family_and_exact() -> None:
    gold = '{"tool_name":"select_coarse_clips","arguments":{"selected_coarse_indices":[1]}}'
    invalid = completion_reward("not-json", gold)
    wrong = completion_reward('{"tool_name":"stop_coarse_retrieval","arguments":{}}', gold)
    family = completion_reward('{"tool_name":"select_coarse_clips","arguments":{"selected_coarse_indices":[2]}}', gold)
    exact = completion_reward(gold, gold)
    assert invalid["reward"] < wrong["reward"] < family["reward"] < exact["reward"]


def test_extract_json_object_repairs_truncated_suffix_only() -> None:
    truncated = (
        '{"schema_version":"video-skills/l2-retrieval-action-v0.1",'
        '"tool_name":"select_coarse_clips","arguments":{"select":[3]}'
    )
    assert extract_json_object(truncated) == {
        "schema_version": "video-skills/l2-retrieval-action-v0.1",
        "tool_name": "select_coarse_clips",
        "arguments": {"select": [3]},
    }
    assert extract_json_object('{"tool_name":"select') is None


def test_group_advantages_preserve_ties_and_center() -> None:
    values = centered_group_advantages([0.0, 0.5, 0.5, 1.0])
    assert values[1] == values[2]
    assert abs(sum(values)) < 1e-6
    assert centered_group_advantages([1.0, 1.0]) == [0.0, 0.0]


def test_k3_kl_is_nonnegative_and_zero_at_reference() -> None:
    new = torch.tensor([-1.0, -2.0], requires_grad=True)
    old = new.detach().clone()
    ref = new.detach().clone()
    loss, _, kl = clipped_grpo_loss(new, old, ref, 1.0)
    assert kl.item() == 0.0
    assert torch.isfinite(loss)

    shifted = torch.tensor([-0.4, -2.8], requires_grad=True)
    _, _, shifted_kl = clipped_grpo_loss(shifted, old, ref, 0.0)
    assert shifted_kl.item() > 0.0


def test_positive_advantage_increases_sample_logprob() -> None:
    new = torch.tensor([-2.0], requires_grad=True)
    old = new.detach().clone()
    ref = new.detach().clone()
    loss, _, _ = clipped_grpo_loss(new, old, ref, 1.0, kl_coef=0.0)
    loss.backward()
    # Gradient descent subtracts a negative gradient, increasing log-probability.
    assert new.grad is not None and new.grad.item() < 0.0


def test_plackett_luce_ordered_set_probability_and_gradient() -> None:
    scores = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
    logprob = plackett_luce_logprob(scores, [0, 1], temperature=1.0)
    expected = torch.log_softmax(scores, dim=0)[0] + torch.log_softmax(scores[1:], dim=0)[0]
    assert torch.allclose(logprob, expected)
    assert logprob.item() <= 0.0
    (-logprob).backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_plackett_luce_rejects_invalid_sets() -> None:
    import pytest

    scores = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="unique"):
        plackett_luce_logprob(scores, [0, 0])
    with pytest.raises(ValueError, match="outside"):
        plackett_luce_logprob(scores, [2])
    with pytest.raises(ValueError, match="positive"):
        plackett_luce_logprob(scores, [0], temperature=0.0)

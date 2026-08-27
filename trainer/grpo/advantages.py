"""Group-relative advantages and KL-regularized GRPO surrogate (CPU-safe)."""

from __future__ import annotations

import math
from typing import Sequence

from trainer.reward.verified_reward import VerifiedRewardBreakdown, group_rank_advantages


def assign_group_advantages(scores: Sequence[VerifiedRewardBreakdown]) -> list[float]:
    return group_rank_advantages(scores)


def kl_penalty(logprob: float, ref_logprob: float) -> float:
    """Token/sequence-level KL proxy: π logπ - π logπ_ref ≈ logπ - logπ_ref when evaluated under π."""
    return float(logprob) - float(ref_logprob)


def grpo_surrogate_loss(
    *,
    advantages: Sequence[float],
    logprobs: Sequence[float],
    ref_logprobs: Sequence[float] | None = None,
    kl_coef: float = 0.05,
) -> dict[str, float]:
    """Compute mean GRPO surrogate: -A * logπ + β KL(π || π_ref).

    This is the training objective wrapper for a plugged-in LoRA student.
    Values are CPU floats for smoke / unit tests.
    """
    if not advantages:
        return {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "n": 0}
    if len(advantages) != len(logprobs):
        raise ValueError("advantages and logprobs length mismatch")
    refs = list(ref_logprobs) if ref_logprobs is not None else list(logprobs)
    if len(refs) != len(logprobs):
        raise ValueError("ref_logprobs length mismatch")

    policy_terms = []
    kl_terms = []
    for adv, lp, rlp in zip(advantages, logprobs, refs):
        policy_terms.append(-float(adv) * float(lp))
        kl_terms.append(kl_penalty(float(lp), float(rlp)))
    n = len(policy_terms)
    policy_loss = sum(policy_terms) / n
    kl = sum(kl_terms) / n
    loss = policy_loss + float(kl_coef) * kl
    if not math.isfinite(loss):
        raise ValueError(f"non-finite GRPO loss: {loss}")
    return {
        "loss": float(loss),
        "policy_loss": float(policy_loss),
        "kl": float(kl),
        "n": float(n),
    }


def return_to_go_credits(
    step_progress_deltas: Sequence[int],
    *,
    terminal_success: bool,
) -> list[float]:
    """Assign return-to-go style credits from per-step milestone deltas.

    Terminal success remains lexicographically dominant in ranking; this helper only
    attributes process credit along a trajectory for diagnostics / optional shaping
    ablations that still respect the frozen rank_key for advantages.
    """
    n = len(step_progress_deltas)
    if n == 0:
        return []
    suffix = [0] * n
    running = 0
    for i in range(n - 1, -1, -1):
        running += max(int(step_progress_deltas[i]), 0)
        suffix[i] = running
    terminal_bonus = 1.0 if terminal_success else 0.0
    return [float(v) + terminal_bonus for v in suffix]

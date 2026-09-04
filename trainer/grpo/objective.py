"""CPU-testable pieces of the on-policy specialist GRPO objective."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence


def _repair_truncated_json_object(text: str) -> str | None:
    """Best-effort repair for generation cut exactly at a JSON suffix.

    Model actions are short JSON objects. When max_new_tokens cuts the final
    braces, the useful action is often still fully present. Repair only the
    conservative case where bracket/string structure can be balanced by adding
    suffix characters; malformed interior JSON still fails json.loads below.
    """
    if not text:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                return None
            stack.pop()
    if in_string or escaped:
        return None
    return text + "".join(reversed(stack))


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the outer JSON object from a model completion."""
    start = str(text or "").find("{")
    end = str(text or "").rfind("}")
    if start < 0:
        return None
    raw = str(text)[start : end + 1] if end > start else str(text)[start:]
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        repaired = _repair_truncated_json_object(raw)
        if repaired is None:
            return None
        try:
            value = json.loads(repaired)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return value if isinstance(value, dict) else None


def action_signature(payload: dict[str, Any] | None) -> tuple[str, str, str]:
    """Normalize L2 and Repair action ownership without inspecting gold metadata."""
    if not payload:
        return "", "", ""
    nested = payload.get("action") if isinstance(payload.get("action"), dict) else {}
    return (
        str(payload.get("tool_name") or nested.get("action_type") or ""),
        str(payload.get("round_type") or ""),
        str(payload.get("target_policy") or ""),
    )


def completion_reward(completion: str, gold_text: str) -> dict[str, Any]:
    """Deterministic action reward for SFT-aligned specialist prompts.

    Exact action/argument equality is the terminal success signal.  Valid JSON
    and the correct action family provide bounded partial credit so a group can
    still learn before exact matches appear.
    """
    predicted = extract_json_object(completion)
    gold = extract_json_object(gold_text)
    if gold is None:
        raise ValueError("gold assistant message is not a JSON object")
    if predicted is None:
        return {"reward": -1.0, "json_valid": False, "action_match": False, "exact_match": False}
    exact = predicted == gold
    signature_match = action_signature(predicted) == action_signature(gold)
    if exact:
        reward = 1.0
    elif signature_match:
        reward = 0.5
    else:
        reward = 0.0
    return {
        "reward": reward,
        "json_valid": True,
        "action_match": signature_match,
        "exact_match": exact,
    }


def centered_group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """GRPO group normalization with ties preserved.

    The previous rank implementation assigned different advantages to equal
    rewards.  Mean/std normalization gives tied samples identical credit.
    """
    if not rewards:
        return []
    mean = sum(float(value) for value in rewards) / len(rewards)
    variance = sum((float(value) - mean) ** 2 for value in rewards) / len(rewards)
    if variance <= eps * eps:
        return [0.0 for _ in rewards]
    scale = math.sqrt(variance + eps)
    return [(float(value) - mean) / scale for value in rewards]


def clipped_grpo_loss(
    new_logprobs: Any,
    old_logprobs: Any,
    ref_logprobs: Any,
    advantage: float,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
) -> tuple[Any, Any, Any]:
    """Token-level clipped GRPO loss with the non-negative k3 KL estimator."""
    import torch

    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != ref_logprobs.shape:
        raise ValueError("new/old/reference logprob shapes must match")
    if new_logprobs.numel() == 0:
        raise ValueError("completion has no scored tokens")
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    adv = torch.as_tensor(float(advantage), dtype=new_logprobs.dtype, device=new_logprobs.device)
    policy_loss = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()

    # Schulman k3 estimator.  Let x = log(pi_ref/pi); exp(x)-x-1 >= 0.
    log_ref_ratio = ref_logprobs - new_logprobs
    kl = (torch.exp(log_ref_ratio) - log_ref_ratio - 1.0).mean()
    loss = policy_loss + float(kl_coef) * kl
    return loss, policy_loss, kl


def plackett_luce_logprob(
    scores: Any,
    ordered_indices: Sequence[int],
    *,
    temperature: float = 1.0,
) -> Any:
    """Log probability of an ordered sample without replacement.

    Gumbel top-k samples an ordering from the Plackett--Luce distribution.  The
    returned scalar keeps gradients to every candidate score, so a pointwise
    relevance adapter can be optimized from a terminal set reward without
    pretending that its JSON rendering was the sampled policy action.
    """
    import torch

    if not torch.is_tensor(scores) or scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional torch tensor")
    if scores.numel() == 0:
        raise ValueError("scores must not be empty")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    selected = [int(index) for index in ordered_indices]
    if not selected:
        raise ValueError("ordered_indices must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("ordered_indices must be unique")
    if min(selected) < 0 or max(selected) >= scores.numel():
        raise ValueError("ordered index outside score vector")

    scaled = scores / float(temperature)
    available = list(range(scores.numel()))
    result = scores.new_zeros(())
    for index in selected:
        denominator = torch.logsumexp(scaled[available], dim=0)
        result = result + scaled[index] - denominator
        available.remove(index)
    return result

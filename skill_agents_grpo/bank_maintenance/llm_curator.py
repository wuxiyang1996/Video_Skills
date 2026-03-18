"""
LLM curator filter for bank maintenance candidate actions.

The algorithmic pipeline in ``run_bank_maintenance()`` proposes candidate
mutations (refine, merge, split, materialize, promote). This module adds
a single-turn LLM filter that reviews candidates and returns approve /
veto / defer decisions.

GRPO wrapping: ``enable_curator_grpo()`` activates the GRPO wrapper on
``filter_candidates()``. G samples are generated, evaluated via
``curator_reward()``, and the best is returned. The maintenance pipeline
is unaware of the wrapping.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_CURATOR_PROMPT_TEMPLATE = """\
You are a skill bank maintenance curator. Review the proposed actions and decide \
whether to approve, veto, or defer each one.

## Bank Summary
Total skills: {n_skills}
Mean pass rate: {mean_pass_rate:.2f}
Skills with low pass rate (<0.60): {n_low_pass}

## Proposed Actions
{actions_text}

For each action, respond with a JSON object:
{{"decisions": [{{"idx": 0, "verdict": "approve|veto|defer", "reason": "brief reason"}}, ...]}}

Action types: SPLIT, MERGE, REFINE, MATERIALIZE, PROMOTE.

Guidelines:
- APPROVE actions that improve bank quality with clear evidence.
- VETO actions where the evidence is contradictory or the risk outweighs benefit.
- DEFER when evidence is insufficient (fewer than 5 instances, marginal metrics).
- For MATERIALIZE: approve if the recurring pattern is distinct from existing skills.
- For PROMOTE: approve if the proto-skill has a reasonable pass rate and enough instances.
- Prefer conservative decisions — a deferred action can be reconsidered later.
"""


def _format_action(idx: int, action: Dict[str, Any]) -> str:
    """Format one candidate action for the prompt."""
    action_type = action.get("type", "unknown")
    skill_id = action.get("skill_id", "?")
    parts = [f"  Action {idx}: {action_type.upper()} on {skill_id}"]

    if "trigger" in action:
        parts.append(f"    Trigger: {action['trigger']}")
    if "pass_rate" in action:
        parts.append(f"    Pass rate: {action['pass_rate']:.2f}")
    if "n_instances" in action:
        parts.append(f"    Instances: {action['n_instances']}")
    if "details" in action:
        details = action["details"]
        if isinstance(details, dict):
            for k, v in list(details.items())[:5]:
                parts.append(f"    {k}: {v}")
        else:
            parts.append(f"    Details: {str(details)[:200]}")

    return "\n".join(parts)


def _get_curator_ask_fn() -> Optional[Callable[..., str]]:
    """Return a CURATOR-routed ask function, or fallback to ask_model."""
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.CURATOR),
            )
    except Exception:
        pass
    from API_func import ask_model
    return wrap_ask_for_reasoning_models(ask_model)


def _build_curator_prompt(
    candidates: List[Dict[str, Any]],
    bank_summary: Dict[str, Any],
) -> str:
    """Build the CURATOR prompt from candidates and bank summary."""
    actions_text = "\n\n".join(
        _format_action(i, c) for i, c in enumerate(candidates)
    )
    return _CURATOR_PROMPT_TEMPLATE.format(
        n_skills=bank_summary.get("n_skills", 0),
        mean_pass_rate=bank_summary.get("mean_pass_rate", 0.0),
        n_low_pass=bank_summary.get("n_low_pass", 0),
        actions_text=actions_text or "(no actions proposed)",
    )


def make_bank_summary(bank: Any) -> Dict[str, Any]:
    """Extract a summary dict from a SkillBankMVP for the curator prompt."""
    skills = []
    if hasattr(bank, "skills"):
        skills = list(bank.skills.values()) if isinstance(bank.skills, dict) else bank.skills
    elif hasattr(bank, "list_skills"):
        skills = bank.list_skills()

    n_skills = len(skills)
    pass_rates = []
    for s in skills:
        if hasattr(s, "contract") and s.contract:
            pr = getattr(s, "pass_rate", None)
            if pr is not None:
                pass_rates.append(pr)

    mean_pr = sum(pass_rates) / max(len(pass_rates), 1) if pass_rates else 0.0
    n_low = sum(1 for pr in pass_rates if pr < 0.60)

    return {
        "n_skills": n_skills,
        "mean_pass_rate": mean_pr,
        "n_low_pass": n_low,
    }


def filter_candidates(
    candidates: List[Dict[str, Any]],
    bank: Any,
    *,
    bank_summary: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Use the CURATOR adapter to filter bank maintenance candidates.

    Returns ``{"decisions": [{"idx": 0, "verdict": "approve", "reason": "..."}, ...]}``
    or None if the adapter is unavailable.

    When GRPO wrapping is active (via ``enable_curator_grpo``), the
    wrapper intercepts this call automatically.
    """
    import time as _time
    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

    if not candidates:
        return {"decisions": []}

    ask_fn = _get_curator_ask_fn()
    if ask_fn is None:
        return None

    summary = bank_summary or make_bank_summary(bank)
    prompt = _build_curator_prompt(candidates, summary)

    try:
        t0 = _time.time()
        raw = ask_fn(prompt, temperature=temperature)
        elapsed = _time.time() - t0
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = None
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])

        record_io(ColdStartRecord(
            module="bank_curator",
            function="filter_candidates",
            prompt=prompt,
            response=raw or "",
            parsed=parsed,
            model="",
            temperature=temperature,
            elapsed_s=round(elapsed, 3),
            extra={"n_candidates": len(candidates)},
            error=None if parsed and "decisions" in parsed else "parse_failed",
        ))

        if parsed and "decisions" in parsed:
            return parsed
    except Exception as exc:
        logger.debug("CURATOR adapter call failed: %s", exc)

    return None


def apply_curator_decisions(
    candidates: List[Dict[str, Any]],
    decisions: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter candidates by curator decisions, returning only approved ones.

    If decisions is None (curator unavailable), returns all candidates
    (algorithmic default).
    """
    if decisions is None:
        return candidates

    decision_list = decisions.get("decisions", [])
    approved_indices = {
        d["idx"] for d in decision_list
        if d.get("verdict") == "approve" and "idx" in d
    }

    if not approved_indices and not decision_list:
        return candidates

    return [c for i, c in enumerate(candidates) if i in approved_indices]


# ── GRPO integration ──────────────────────────────────────────────────

_grpo_original_fn: Optional[Callable] = None


def enable_curator_grpo(
    buffer: Any,
    group_size: int = 4,
    temperature: float = 0.7,
    compute_quality_fn: Optional[Any] = None,
    execute_fn: Optional[Any] = None,
) -> None:
    """Activate GRPO wrapping on ``filter_candidates``.

    Parameters
    ----------
    buffer : GRPOBuffer
    group_size : int
    temperature : float
    compute_quality_fn : callable
        ``compute_quality_fn(bank) -> float``
    execute_fn : callable
        ``execute_fn(actions, bank) -> None``
    """
    import skill_agents_grpo.bank_maintenance.llm_curator as _mod
    from skill_agents_grpo.grpo.rewards import curator_reward
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction
    from functools import partial

    global _grpo_original_fn

    if _grpo_original_fn is not None:
        logger.warning("Curator GRPO already enabled — skipping")
        return

    _grpo_original_fn = _mod.filter_candidates

    bound_reward = partial(
        curator_reward,
        compute_quality_fn=compute_quality_fn,
        execute_fn=execute_fn,
    )

    def _prompt_extractor(
        candidates: List[Dict[str, Any]],
        bank: Any,
        *,
        bank_summary: Optional[Dict[str, Any]] = None,
        **kw: Any,
    ) -> str:
        summary = bank_summary or make_bank_summary(bank)
        return _build_curator_prompt(candidates, summary)

    def _metadata_extractor(
        candidates: List[Dict[str, Any]],
        *a: Any,
        **kw: Any,
    ) -> Dict[str, Any]:
        return {"n_candidates": len(candidates)}

    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.CURATOR,
        reward_fn=bound_reward,
        buffer=buffer,
        group_size=group_size,
        temperature=temperature,
        prompt_extractor=_prompt_extractor,
        metadata_extractor=_metadata_extractor,
    )

    _mod.filter_candidates = wrapper.wrap(_grpo_original_fn)
    logger.info("Curator GRPO enabled: G=%d, temp=%.2f", group_size, temperature)


def disable_curator_grpo() -> None:
    """Deactivate GRPO wrapping, restore original function."""
    import skill_agents_grpo.bank_maintenance.llm_curator as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.filter_candidates = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Curator GRPO disabled")

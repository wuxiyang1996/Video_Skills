"""
LLM-assisted contract learning via the CONTRACT LoRA adapter.

Provides ``llm_summarize_contract`` which uses the trained CONTRACT adapter
to generate effect summaries from segment observations, enriching the
purely algorithmic frequency-based contract learning in Stage 3.

When the multi-LoRA model is not configured, falls back gracefully so
the algorithmic path continues to work.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_CONTRACT_PROMPT_TEMPLATE = """\
You are analyzing skill effects from game trajectory segments.

Skill: {skill_id}
Number of instances: {n_instances}

Representative segment observations:
{segment_observations}

State predicates at segment start: {predicates_start}
State predicates at segment end: {predicates_end}

Summarize the effects of this skill as a JSON object:
{{"eff_add": ["predicates that become true"], "eff_del": ["predicates that become false"], "description": "one-line description"}}"""


def _get_contract_ask_fn() -> Optional[Callable[..., str]]:
    """Return a CONTRACT-routed ask function, or None if unavailable.

    The returned callable is wrapped for reasoning-model compatibility
    (Qwen3 ``/no_think``, think-tag stripping).
    """
    from skill_agents._llm_compat import wrap_ask_for_reasoning_models

    try:
        from skill_agents.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.CONTRACT),
            )
    except Exception:
        pass
    return None


def llm_summarize_contract(
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
) -> Optional[Dict[str, Any]]:
    """Use the CONTRACT adapter to generate an effect summary for a skill.

    Returns a dict ``{"eff_add": [...], "eff_del": [...], "description": "..."}``
    or None if the adapter is unavailable or the call fails.
    """
    ask_fn = _get_contract_ask_fn()
    if ask_fn is None:
        return None

    prompt = _CONTRACT_PROMPT_TEMPLATE.format(
        skill_id=skill_id,
        n_instances=n_instances,
        segment_observations="; ".join(segment_observations[:5]),
        predicates_start=json.dumps(sorted(predicates_start)),
        predicates_end=json.dumps(sorted(predicates_end)),
    )

    try:
        raw = ask_fn(prompt, temperature=0.1)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as exc:
        logger.debug("CONTRACT adapter call failed for %s: %s", skill_id, exc)

    return None

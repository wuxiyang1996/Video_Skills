"""
LLM-assisted contract learning via the CONTRACT LoRA adapter.

Provides ``llm_summarize_contract`` which uses the trained CONTRACT adapter
to generate effect summaries from segment observations, enriching the
purely algorithmic frequency-based contract learning in Stage 3.

When GRPO is active, calls are intercepted by ``GRPOCallWrapper``:
G samples are generated, evaluated via ``contract_reward()``, and the
best is returned. The EM pipeline is unaware of the wrapping.

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
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.CONTRACT),
            )
    except Exception:
        pass
    return None


def _build_contract_prompt(
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
) -> str:
    """Build the CONTRACT prompt from arguments (exposed for GRPO prompt extraction)."""
    return _CONTRACT_PROMPT_TEMPLATE.format(
        skill_id=skill_id,
        n_instances=n_instances,
        segment_observations="; ".join(segment_observations[:5]),
        predicates_start=json.dumps(sorted(predicates_start)),
        predicates_end=json.dumps(sorted(predicates_end)),
    )


def llm_summarize_contract(
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
    *,
    temperature: float = 0.1,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Use the CONTRACT adapter to generate an effect summary for a skill.

    Returns a dict ``{"eff_add": [...], "eff_del": [...], "description": "..."}``
    or None if the adapter is unavailable or the call fails.

    When GRPO wrapping is active (via ``enable_contract_grpo``), the
    wrapper intercepts this call, generates G samples at higher temperature,
    evaluates each with ``contract_reward()``, stores data for training,
    and returns the best.
    """
    ask_fn = _get_contract_ask_fn()
    if ask_fn is None:
        return None

    prompt = _build_contract_prompt(
        skill_id, segment_observations, predicates_start, predicates_end, n_instances,
    )

    try:
        raw = ask_fn(prompt, temperature=temperature)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as exc:
        logger.debug("CONTRACT adapter call failed for %s: %s", skill_id, exc)

    return None


# ── GRPO integration ──────────────────────────────────────────────────

_grpo_original_fn: Optional[Callable] = None


def enable_contract_grpo(
    buffer: Any,
    group_size: int = 4,
    temperature: float = 0.7,
    consensus_add: Optional[Set[str]] = None,
    consensus_del: Optional[Set[str]] = None,
    holdout_instances: Optional[list] = None,
    verify_config: Optional[Any] = None,
) -> None:
    """Activate GRPO wrapping on ``llm_summarize_contract``.

    After calling this, every ``llm_summarize_contract()`` invocation
    generates G samples, evaluates with ``contract_reward()``, stores
    data in *buffer*, and returns the best.

    Parameters
    ----------
    buffer : GRPOBuffer
    group_size : int
    temperature : float
    consensus_add, consensus_del : sets
        Frequency consensus for union-merge in reward computation.
    holdout_instances : list[SegmentRecord]
    verify_config : Stage3MVPConfig
    """
    import skill_agents_grpo.stage3_mvp.llm_contract as _mod
    from skill_agents_grpo.grpo.rewards import contract_reward
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction
    from functools import partial

    global _grpo_original_fn

    if _grpo_original_fn is not None:
        logger.warning("Contract GRPO already enabled — skipping re-enable")
        return

    _grpo_original_fn = _mod.llm_summarize_contract

    # Bind verification context into the reward function
    bound_reward = partial(
        contract_reward,
        consensus_add=consensus_add,
        consensus_del=consensus_del,
        holdout_instances=holdout_instances,
        verify_config=verify_config,
    )

    def _prompt_extractor(
        skill_id: str,
        segment_observations: List[str],
        predicates_start: Set[str],
        predicates_end: Set[str],
        n_instances: int = 0,
        **kw: Any,
    ) -> str:
        return _build_contract_prompt(
            skill_id, segment_observations, predicates_start, predicates_end, n_instances,
        )

    def _metadata_extractor(skill_id: str, *a: Any, **kw: Any) -> Dict[str, Any]:
        return {"skill_id": skill_id}

    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.CONTRACT,
        reward_fn=bound_reward,
        buffer=buffer,
        group_size=group_size,
        temperature=temperature,
        prompt_extractor=_prompt_extractor,
        metadata_extractor=_metadata_extractor,
    )

    _mod.llm_summarize_contract = wrapper.wrap(_grpo_original_fn)
    logger.info("Contract GRPO enabled: G=%d, temp=%.2f", group_size, temperature)


def disable_contract_grpo() -> None:
    """Deactivate GRPO wrapping, restore original function."""
    import skill_agents_grpo.stage3_mvp.llm_contract as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.llm_summarize_contract = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Contract GRPO disabled")

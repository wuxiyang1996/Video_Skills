"""Live Motif-gated rollout_fn for GRPO collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
from atomic_skills.skill_executor import SkillExecutor
from atomic_skills.skill_model_client import SkillModelClient
from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key

RolloutFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

# Diversity stack for GRPO K-samples (kept lean for wall-clock):
# - generate: LLM rank/priors over MCQ options (1 call)
# - score: rule all + LLM rescore top-2 only (≤2 calls; enforced in planner)
# - compare: LLM pick among scored hypotheses (1 call)
_GRPO_LLM_SKILLS = {
    "generate_answer_hypotheses",
    "score_hypothesis_support",
    "compare_hypotheses",
}


def _grpo_skill_backend_config() -> SkillBackendConfig:
    """RULE for schema/retrieval scaffolding; LLM for answer-critical steps."""
    rule_only = {
        "parse_question_target",
        "propose_evidence_roles",
        "compose_evidence_chain",
        "detect_missing_role",
        "segment_video_or_select_clip",
        "link_graph_relation",
        "retrieve_by_event",
        "retrieve_by_entity",
        "retrieve_by_time",
        "retrieve_by_relation",
        "retrieve_evidence_for_hypothesis",
        "search_counterevidence",
        "verify_claim_support",
        "commit_answer",
    }
    return SkillBackendConfig(
        default_mode=SkillBackendMode.RULE,
        llm_skills=set(_GRPO_LLM_SKILLS),
        rule_only_skills=rule_only,
    )


def _pick_forced_motif_id(motif_bank_path: str, seed: int) -> str | None:
    """Rotate forced motif across ACTIVE bank entries for K-sample diversity."""
    try:
        from motif.bank import MotifBank

        bank = MotifBank.load_jsonl(motif_bank_path)
        ids = [r.motif_id for r in bank.active_records()] or list(bank.motif_ids)
    except Exception:
        return None
    if not ids:
        return None
    return str(ids[int(seed) % len(ids)])


def make_motif_gated_rollout_fn(
    *,
    motif_bank_path: str | Path,
    planner_model: str = "openai/gpt-oss-120b",
    skill_model: str = "qwen/qwen3.5-9b",
    keys_py: str = "/fs/gamma-projects/vlm-robot/keys.py",
    api_key: str | None = None,
    timeout_s: int = 180,
    skill_timeout_s: int = 90,
    skill_temperature: float = 0.8,
    with_skill_executor: bool = True,
    rotate_motifs: bool = True,
    force_explore: bool = True,
    explore_top_k: int = 3,
    motif_candidate_sink_path: str | Path | None = None,
) -> RolloutFn:
    """Build a Motif-first rollout function for GRPO live collection.

    Motif expand still supplies the skill sequence prior, but answer-critical
    skills run through an LLM SkillExecutor with temperature>0 so K samples
    diverge. Motif ids can rotate by ``grpo_seed``.
    """
    key = api_key or load_openrouter_api_key(keys_py_path=keys_py)
    client = OpenRouterClient(
        model=planner_model,
        api_key=key,
        max_tokens=1800,
        temperature=0.0,  # planner only used on motif fallback
        reasoning={"effort": "minimal", "exclude": True},
        timeout_s=timeout_s,
    )
    bank = str(motif_bank_path)
    sink = str(motif_candidate_sink_path) if motif_candidate_sink_path else None

    skill_executor: SkillExecutor | None = None
    skill_llm: SkillModelClient | None = None
    if with_skill_executor:
        skill_llm = SkillModelClient(
            model=skill_model,
            api_key=key,
            max_tokens=768,
            temperature=float(skill_temperature),
            timeout_s=int(skill_timeout_s),
        )
        skill_executor = SkillExecutor(
            llm_client=skill_llm,
            vlm_client=None,
            config=_grpo_skill_backend_config(),
        )
        # Planner reads these for seed-rotated compare explore.
        skill_executor.grpo_force_explore = bool(force_explore)  # type: ignore[attr-defined]
        skill_executor.grpo_explore_top_k = int(explore_top_k)  # type: ignore[attr-defined]
        skill_executor.grpo_force_answer_path = True  # type: ignore[attr-defined]
        # Keep L2/explore labels; L1 top-1 override was collapsing K-samples.
        skill_executor.grpo_disable_l1_override = True  # type: ignore[attr-defined]

    def _rollout(example: dict[str, Any], clue: dict[str, Any]) -> dict[str, Any]:
        meta = dict(example.get("metadata") or {})
        meta["motif_enabled"] = True
        meta["motif_bank_path"] = bank
        meta["grpo_force_explore"] = bool(force_explore)
        meta["grpo_explore_top_k"] = int(explore_top_k)
        meta["grpo_force_answer_path"] = True
        meta["grpo_disable_l1_override"] = True
        if sink:
            meta["motif_candidate_sink_path"] = sink
        seed = int(meta.get("grpo_seed") or 0)
        forced = None
        if rotate_motifs:
            forced = _pick_forced_motif_id(bank, seed)
            if forced:
                meta["forced_motif_id"] = forced
        if skill_llm is not None:
            skill_llm.temperature = float(skill_temperature)
            skill_llm.seed = seed
        if skill_executor is not None:
            skill_executor.grpo_force_explore = bool(force_explore)  # type: ignore[attr-defined]
            skill_executor.grpo_explore_top_k = int(explore_top_k)  # type: ignore[attr-defined]
            skill_executor.grpo_force_answer_path = True  # type: ignore[attr-defined]
            skill_executor.grpo_disable_l1_override = True  # type: ignore[attr-defined]
        example = {**example, "metadata": meta}
        return build_llm_reasoning_rollout(
            example,
            clue,
            client=client,
            skill_executor=skill_executor,
            motif_enabled=True,
            motif_bank_path=bank,
            forced_motif_id=forced,
            motif_candidate_sink_path=sink,
        )

    return _rollout

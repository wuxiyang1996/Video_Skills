"""
LLM preference teacher: the LLM provides skill rankings (not numeric scores).

The LLM is asked to **rank** candidate skills for a segment, and to **rank**
likely next-skills given a previous skill.  These rankings are converted to
pairwise preferences and stored for training.

The LLM never produces numeric scores.  Numeric scores come from the
``PreferenceScorer`` in ``preference.py``, which is *trained* on these
collected preferences.

Two collection modes:
  1. **Proactive** — ``collect_segment_preferences``: for every segment in a
     trajectory, ask the LLM to rank skills.  Used for initial cold-start.
  2. **Active** — ``collect_uncertain_preferences``: only query segments where
     the current scorer is uncertain (low margin).  Used in the iterative loop.

GRPO wrapping: ``enable_segment_grpo()`` wraps ``collect_segment_preferences``
at the batch level (per episode). G complete preference sets are generated,
each evaluated via scorer-rebuild + decode, and the best is returned.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from skill_agents_grpo.infer_segmentation.config import LLMTeacherConfig

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _get_ask_model():
    """Lazy import to avoid pulling in API dependencies at module load.

    Prefers the LoRA segment adapter when available, otherwise falls back
    to the API-based ``ask_model``.  The returned callable is wrapped for
    reasoning-model compatibility (Qwen3 ``/no_think``, think-tag stripping).
    """
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(llm.as_ask_fn(SkillFunction.SEGMENT))
    except Exception:
        pass
    from API_func import ask_model
    return wrap_ask_for_reasoning_models(ask_model)


def _wrap_ask_with_semaphore(ask: Callable, max_concurrent: int):
    """Wrap ask so at most max_concurrent calls run at once (e.g. for local GPU)."""
    sem = threading.Semaphore(max_concurrent)

    def limited_ask(*args, **kwargs):
        with sem:
            return ask(*args, **kwargs)

    return limited_ask


def _parse_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON from an LLM response that may contain extra text."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ── Ranking prompts ──────────────────────────────────────────────────

def _build_segment_ranking_prompt(
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    segment_start: int,
    segment_end: int,
    predicates_start: Optional[dict] = None,
    predicates_end: Optional[dict] = None,
) -> str:
    """
    Ask the LLM to rank skills for one segment (best fit first).

    The LLM considers behavior fit, duration plausibility, and state-change
    consistency holistically — but returns a ranking, not scores.
    """
    obs_str = str(list(observations))
    act_str = str(list(actions))
    length = len(observations)
    skills_str = ", ".join(f'"{s}"' for s in skill_names)
    preds_s = str(predicates_start) if predicates_start else "N/A"
    preds_e = str(predicates_end) if predicates_end else "N/A"

    return (
        f"You are an expert at recognizing skills in agent trajectories.\n"
        f"\n"
        f"A trajectory segment spans timesteps {segment_start} to {segment_end} "
        f"(length {length}).\n"
        f"\n"
        f"Observations:\n{obs_str}\n"
        f"\n"
        f"Actions:\n{act_str}\n"
        f"\n"
        f"State at segment start: {preds_s}\n"
        f"State at segment end:   {preds_e}\n"
        f"\n"
        f"Candidate skills: [{skills_str}]\n"
        f"\n"
        f"Rank ALL candidate skills from best fit to worst fit for this "
        f"segment.  Consider:\n"
        f"  - Do the actions match what this skill would produce?\n"
        f"  - Is the segment length reasonable for this skill?\n"
        f"  - Is the state change consistent with this skill's purpose?\n"
        f"\n"
        f"Return ONLY a JSON object (no extra text):\n"
        f'{{"ranking": ["best_skill", "second_best", ...], '
        f'"reasoning": "brief explanation"}}\n'
    )


def _build_transition_ranking_prompt(
    prev_skill: str,
    skill_names: List[str],
) -> str:
    """
    Ask the LLM to rank which skills most naturally follow prev_skill.
    """
    skills_str = ", ".join(f'"{s}"' for s in skill_names)

    return (
        f"You are an expert at modeling skill sequences in agent trajectories.\n"
        f"\n"
        f"The agent just finished executing skill: \"{prev_skill}\"\n"
        f"\n"
        f"Candidate next skills: [{skills_str}]\n"
        f"\n"
        f"Rank ALL candidate skills from most likely to follow to least "
        f"likely.  Consider natural task ordering, common behavior patterns, "
        f"and logical dependencies between skills.\n"
        f"\n"
        f"Return ONLY a JSON object (no extra text):\n"
        f'{{"ranking": ["most_likely", "second", ...], '
        f'"reasoning": "brief explanation"}}\n'
    )


def _build_pairwise_prompt(
    observations: Sequence,
    actions: Sequence,
    skill_a: str,
    skill_b: str,
    segment_start: int,
    segment_end: int,
) -> str:
    """
    Ask the LLM which of two skills better fits this segment.
    Used for active learning on uncertain segments.
    """
    obs_str = str(list(observations))
    act_str = str(list(actions))

    return (
        f"You are an expert at recognizing skills in agent trajectories.\n"
        f"\n"
        f"Segment: timesteps {segment_start} to {segment_end}\n"
        f"Observations:\n{obs_str}\n"
        f"Actions:\n{act_str}\n"
        f"\n"
        f"Which skill better explains this segment?\n"
        f"  A: \"{skill_a}\"\n"
        f"  B: \"{skill_b}\"\n"
        f"\n"
        f"Return ONLY a JSON object:\n"
        f'{{"choice": "A" or "B", "evidence": "brief explanation"}}\n'
    )


# ── Ranking → pairwise conversion ────────────────────────────────────

def ranking_to_pairwise(
    ranking: List[str],
    segment_start: int,
    segment_end: int,
    source: str = "llm",
    evidence: str = "",
) -> list:
    """
    Convert a ranked list [best, ..., worst] into pairwise preferences.

    For a ranking [A, B, C] produces: A≻B, A≻C, B≻C.
    """
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            pairs.append(PreferenceExample(
                segment_start=segment_start,
                segment_end=segment_end,
                skill_win=ranking[i],
                skill_lose=ranking[j],
                evidence=evidence,
                source=source,
            ))
    return pairs


# ── Public API ────────────────────────────────────────────────────────

def _collect_one_segment_prefs(
    start: int,
    end: int,
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    predicates: Optional[List[Optional[dict]]],
    cfg: LLMTeacherConfig,
    ask,
) -> list:
    """Get pairwise preferences for a single segment (used by batch workers)."""
    seg_obs = observations[start: end + 1]
    seg_act = actions[start: end + 1]
    p_start = predicates[start] if predicates and start < len(predicates) else None
    p_end = predicates[end] if predicates and end < len(predicates) else None
    prompt = _build_segment_ranking_prompt(
        seg_obs, seg_act, skill_names, start, end, p_start, p_end,
    )
    response = ask(
        prompt, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    parsed = _parse_json_from_response(response)
    if not parsed or "ranking" not in parsed:
        return []
    ranking = [s for s in parsed["ranking"] if s in skill_names]
    evidence = parsed.get("reasoning", "")
    if len(ranking) < 2:
        return []
    return ranking_to_pairwise(ranking, start, end, "llm", evidence)


def collect_segment_preferences(
    segments: List[Tuple[int, int]],
    observations: Sequence,
    actions: Sequence,
    skill_names: List[str],
    predicates: Optional[List[Optional[dict]]] = None,
    config: Optional[LLMTeacherConfig] = None,
) -> list:
    """
    Proactive preference collection: ask the LLM to rank skills for
    each segment.  Used for cold-start before any scorer is trained.

    When config.max_workers > 1, LLM calls are run in parallel to reduce
    wall-clock time.

    Parameters
    ----------
    segments : list[(start, end)]
        Segment boundaries (inclusive).
    observations, actions : Sequence
        Full trajectory data.
    skill_names : list[str]
        Candidate skills to rank.
    predicates : list[dict], optional
        Per-timestep predicates.
    config : LLMTeacherConfig, optional

    Returns
    -------
    list[PreferenceExample]
        Pairwise preferences derived from LLM rankings.
    """
    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    max_workers = getattr(cfg, "max_workers", None)
    if max_workers is None or max_workers <= 1:
        # Sequential
        all_prefs = []
        for start, end in segments:
            prefs = _collect_one_segment_prefs(
                start, end, observations, actions, skill_names, predicates, cfg, ask,
            )
            all_prefs.extend(prefs)
        return all_prefs

    # Parallel batch; cap concurrent LLM calls if set (e.g. local GPU)
    max_concurrent = getattr(cfg, "max_concurrent_llm_calls", None)
    if max_concurrent is not None and max_concurrent > 0:
        ask = _wrap_ask_with_semaphore(ask, max_concurrent)
    all_prefs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_one_segment_prefs,
                start, end, observations, actions, skill_names, predicates, cfg, ask,
            ): (start, end)
            for start, end in segments
        }
        for future in as_completed(futures):
            try:
                prefs = future.result()
                all_prefs.extend(prefs)
            except Exception:
                pass  # skip failed segments
    return all_prefs


def _collect_one_transition_prefs(
    prev_skill: str,
    skill_names: List[str],
    cfg: LLMTeacherConfig,
    ask,
) -> list:
    """Get transition preferences for one prev_skill (used by batch workers)."""
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    prompt = _build_transition_ranking_prompt(prev_skill, skill_names)
    response = ask(
        prompt, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    parsed = _parse_json_from_response(response)
    if not parsed or "ranking" not in parsed:
        return []
    ranking = [s for s in parsed["ranking"] if s in skill_names]
    evidence = parsed.get("reasoning", "")
    prefs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            prefs.append(PreferenceExample(
                segment_start=-1,
                segment_end=-1,
                skill_win=f"{prev_skill}->{ranking[i]}",
                skill_lose=f"{prev_skill}->{ranking[j]}",
                evidence=evidence,
                source="llm",
            ))
    return prefs


def collect_transition_preferences(
    skill_names: List[str],
    config: Optional[LLMTeacherConfig] = None,
) -> list:
    """
    Ask the LLM to rank transition likelihoods for each skill.

    For each skill as prev_skill, collects pairwise preferences on
    which next-skill is more likely. When config.max_workers > 1,
    LLM calls run in parallel.
    """
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    max_workers = getattr(cfg, "max_workers", None)
    if max_workers is None or max_workers <= 1:
        all_prefs = []
        for prev_skill in skill_names:
            all_prefs.extend(
                _collect_one_transition_prefs(prev_skill, skill_names, cfg, ask),
            )
        return all_prefs

    max_concurrent = getattr(cfg, "max_concurrent_llm_calls", None)
    if max_concurrent is not None and max_concurrent > 0:
        ask = _wrap_ask_with_semaphore(ask, max_concurrent)
    all_prefs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_collect_one_transition_prefs, ps, skill_names, cfg, ask): ps
            for ps in skill_names
        }
        for future in as_completed(futures):
            try:
                all_prefs.extend(future.result())
            except Exception:
                pass
    return all_prefs


def _collect_one_uncertain_pref(
    seg,
    observations: Sequence,
    actions: Sequence,
    cfg: LLMTeacherConfig,
    ask,
) -> Optional[object]:
    """Get one A/B preference for an uncertain segment (used by batch workers)."""
    from skill_agents_grpo.infer_segmentation.preference import PreferenceExample

    if len(seg.candidates) < 2:
        return None
    skill_a = seg.candidates[0].skill
    skill_b = seg.candidates[1].skill
    seg_obs = observations[seg.start: seg.end + 1]
    seg_act = actions[seg.start: seg.end + 1]
    prompt = _build_pairwise_prompt(
        seg_obs, seg_act, skill_a, skill_b, seg.start, seg.end,
    )
    response = ask(
        prompt, model=cfg.model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )
    parsed = _parse_json_from_response(response)
    if not parsed:
        return None
    choice = parsed.get("choice", "").upper().strip()
    evidence = parsed.get("evidence", "")
    if choice == "A":
        win, lose = skill_a, skill_b
    elif choice == "B":
        win, lose = skill_b, skill_a
    else:
        return None
    return PreferenceExample(
        segment_start=seg.start,
        segment_end=seg.end,
        skill_win=win,
        skill_lose=lose,
        evidence=evidence,
        source="llm",
    )


# ── Naming new skills ─────────────────────────────────────────────────

def _build_skill_naming_prompt(
    observation_slices: List[Sequence],
    eff_add: Optional[Sequence[str]] = None,
    eff_del: Optional[Sequence[str]] = None,
    eff_event: Optional[Sequence[str]] = None,
) -> str:
    """Build a prompt for the LLM to suggest a human-readable name for a new skill."""
    parts = [
        "You are a game-AI skill naming expert.",
        "",
        "Below are trajectory segments grouped as the same skill.",
        "Generate a concrete, actionable skill name and description.",
        "",
        "Segment observations (state summaries):",
    ]
    for i, obs_slice in enumerate(observation_slices[:3]):
        parts.append(f"  Segment {i + 1}: {str(list(obs_slice))[:800]}")
    if eff_add or eff_del or eff_event:
        parts.append("")
        parts.append("Learned effects of this skill:")
        if eff_add:
            parts.append(f"  adds: {list(eff_add)}")
        if eff_del:
            parts.append(f"  deletes: {list(eff_del)}")
        if eff_event:
            parts.append(f"  events: {list(eff_event)}")
    parts.extend([
        "",
        "RULES:",
        "- The NAME must be a concrete imperative verb phrase (2-6 words) describing",
        "  the actual game action. Reference game objects, mechanics, or goals.",
        "- The DESCRIPTION must be specific: describe WHAT the skill does and WHEN",
        "  to invoke it. Avoid generic phrases like 'applies effects' or 'does skill'.",
        "- Use snake_case or short phrases (e.g. \"pick_up_onion\", \"clear_bottom_row\").",
        "",
        "Return ONLY a JSON object (no extra text):",
        '{"name": "short skill name", "description": "one line description"}',
    ])
    return "\n".join(parts)


def suggest_skill_name(
    observation_slices: List[Sequence],
    eff_add: Optional[Sequence[str]] = None,
    eff_del: Optional[Sequence[str]] = None,
    eff_event: Optional[Sequence[str]] = None,
    config: Optional[LLMTeacherConfig] = None,
) -> Optional[Dict[str, str]]:
    """
    Ask the LLM to suggest a human-readable name and description for a new skill.

    Used when materializing __NEW__ segments into a real skill so the bank
    has a readable label (e.g. \"pick_up_onion\") instead of only an ID.

    Parameters
    ----------
    observation_slices : list of observation sequences
        One or more segment observation slices (e.g. from cluster segments).
    eff_add, eff_del, eff_event : optional
        Effect literals learned for this skill (from contract).
    config : LLMTeacherConfig, optional
        Model and sampling settings.

    Returns
    -------
    dict with "name" and "description" keys, or None if the LLM response could not be parsed.
    """
    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    prompt = _build_skill_naming_prompt(
        observation_slices,
        eff_add=eff_add,
        eff_del=eff_del,
        eff_event=eff_event,
    )
    response = ask(
        prompt,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    parsed = _parse_json_from_response(response)
    if not parsed or "name" not in parsed:
        return None
    name = (parsed.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": (parsed.get("description") or "").strip() or None,
    }


def collect_uncertain_preferences(
    result,
    observations: Sequence,
    actions: Sequence,
    margin_threshold: float = 1.0,
    max_queries: int = 10,
    config: Optional[LLMTeacherConfig] = None,
) -> list:
    """
    Active learning: query the LLM only on uncertain segments (low margin).

    For each uncertain segment, asks the LLM to choose between the top-2
    skill candidates.     When config.max_workers > 1, LLM calls run in parallel.
    """
    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    uncertain = result.uncertain_segments(margin_threshold)
    uncertain.sort(key=lambda s: s.margin)
    to_query = [seg for seg in uncertain[:max_queries] if len(seg.candidates) >= 2]
    if not to_query:
        return []

    max_workers = getattr(cfg, "max_workers", None)
    if max_workers is None or max_workers <= 1:
        all_prefs = []
        for seg in to_query:
            pref = _collect_one_uncertain_pref(seg, observations, actions, cfg, ask)
            if pref is not None:
                all_prefs.append(pref)
        return all_prefs

    max_concurrent = getattr(cfg, "max_concurrent_llm_calls", None)
    if max_concurrent is not None and max_concurrent > 0:
        ask = _wrap_ask_with_semaphore(ask, max_concurrent)
    all_prefs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_collect_one_uncertain_pref, seg, observations, actions, cfg, ask): seg
            for seg in to_query
        }
        for future in as_completed(futures):
            try:
                pref = future.result()
                if pref is not None:
                    all_prefs.append(pref)
            except Exception:
                pass
    return all_prefs


# ── GRPO integration ──────────────────────────────────────────────────

_grpo_original_fn: Optional[Callable] = None


def enable_segment_grpo(
    buffer: "Any",
    group_size: int = 4,
    temperature: float = 0.7,
    scorer_factory: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
) -> None:
    """Activate GRPO wrapping on ``collect_segment_preferences``.

    Wraps at the **batch level** (per episode): each GRPO sample is a
    complete set of preferences for all segments. The reward requires
    rebuilding the scorer and running the decoder per sample.

    Parameters
    ----------
    buffer : GRPOBuffer
    group_size : int
    temperature : float
    scorer_factory : callable
        ``scorer_factory(preference_list) -> SegmentScorer``
    decode_fn : callable
        ``decode_fn(scorer, segments, obs, actions, skill_names, preds) -> result``
    """
    import skill_agents_grpo.infer_segmentation.llm_teacher as _mod
    from skill_agents_grpo.grpo.rewards import segmentation_reward
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction
    from functools import partial

    global _grpo_original_fn

    if _grpo_original_fn is not None:
        logger.warning("Segment GRPO already enabled — skipping")
        return

    _grpo_original_fn = _mod.collect_segment_preferences

    bound_reward = partial(
        segmentation_reward,
        scorer_factory=scorer_factory,
        decode_fn=decode_fn,
    )

    def _prompt_extractor(
        segments: List[Tuple[int, int]],
        observations: Sequence,
        actions: Sequence,
        skill_names: List[str],
        predicates: Optional[List[Optional[dict]]] = None,
        config: Optional[LLMTeacherConfig] = None,
        **kw: Any,
    ) -> str:
        parts = [f"Segment preferences for {len(segments)} segments"]
        parts.append(f"Skills: {skill_names}")
        return "\n".join(parts)

    def _metadata_extractor(
        segments: List[Tuple[int, int]],
        *a: Any,
        **kw: Any,
    ) -> Dict[str, Any]:
        return {"n_segments": len(segments)}

    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.SEGMENT,
        reward_fn=bound_reward,
        buffer=buffer,
        group_size=group_size,
        temperature=temperature,
        prompt_extractor=_prompt_extractor,
        metadata_extractor=_metadata_extractor,
    )

    _mod.collect_segment_preferences = wrapper.wrap(_grpo_original_fn)
    logger.info("Segment GRPO enabled: G=%d, temp=%.2f", group_size, temperature)


def disable_segment_grpo() -> None:
    """Deactivate GRPO wrapping, restore original function."""
    import skill_agents_grpo.infer_segmentation.llm_teacher as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.collect_segment_preferences = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Segment GRPO disabled")

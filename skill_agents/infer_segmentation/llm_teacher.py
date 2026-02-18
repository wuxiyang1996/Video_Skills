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
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from skill_agents.infer_segmentation.config import LLMTeacherConfig

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _get_ask_model():
    """Lazy import to avoid pulling in API dependencies at module load."""
    from API_func import ask_model
    return ask_model


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
    from skill_agents.infer_segmentation.preference import PreferenceExample

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
    all_prefs = []

    for start, end in segments:
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

        if parsed and "ranking" in parsed:
            ranking = [s for s in parsed["ranking"] if s in skill_names]
            evidence = parsed.get("reasoning", "")
            if len(ranking) >= 2:
                pairs = ranking_to_pairwise(ranking, start, end, "llm", evidence)
                all_prefs.extend(pairs)

    return all_prefs


def collect_transition_preferences(
    skill_names: List[str],
    config: Optional[LLMTeacherConfig] = None,
) -> list:
    """
    Ask the LLM to rank transition likelihoods for each skill.

    For each skill as prev_skill, collects pairwise preferences on
    which next-skill is more likely.

    Returns
    -------
    list[PreferenceExample]
        Pairwise preferences with segment_start=segment_end=-1
        (sentinel for transition prefs).
    """
    from skill_agents.infer_segmentation.preference import PreferenceExample

    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    all_prefs = []

    for prev_skill in skill_names:
        prompt = _build_transition_ranking_prompt(prev_skill, skill_names)
        response = ask(
            prompt, model=cfg.model,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        )
        parsed = _parse_json_from_response(response)

        if parsed and "ranking" in parsed:
            ranking = [s for s in parsed["ranking"] if s in skill_names]
            evidence = parsed.get("reasoning", "")
            for i in range(len(ranking)):
                for j in range(i + 1, len(ranking)):
                    all_prefs.append(PreferenceExample(
                        segment_start=-1,
                        segment_end=-1,
                        skill_win=f"{prev_skill}->{ranking[i]}",
                        skill_lose=f"{prev_skill}->{ranking[j]}",
                        evidence=evidence,
                        source="llm",
                    ))

    return all_prefs


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
    skill candidates.

    Parameters
    ----------
    result : SegmentationResult
        Output from a previous decoding pass.

    Returns
    -------
    list[PreferenceExample]
    """
    from skill_agents.infer_segmentation.preference import PreferenceExample

    cfg = config or LLMTeacherConfig()
    ask = _get_ask_model()
    all_prefs = []

    uncertain = result.uncertain_segments(margin_threshold)
    uncertain.sort(key=lambda s: s.margin)

    for seg in uncertain[:max_queries]:
        if len(seg.candidates) < 2:
            continue

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

        if parsed:
            choice = parsed.get("choice", "").upper().strip()
            evidence = parsed.get("evidence", "")
            if choice == "A":
                win, lose = skill_a, skill_b
            elif choice == "B":
                win, lose = skill_b, skill_a
            else:
                continue
            all_prefs.append(PreferenceExample(
                segment_start=seg.start,
                segment_end=seg.end,
                skill_win=win,
                skill_lose=lose,
                evidence=evidence,
                source="llm",
            ))

    return all_prefs

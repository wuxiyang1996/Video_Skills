"""
Episode adapter: bridge between Stage 1 (boundary_proposal) and Stage 2 (InferSegmentation).

Pipeline:
  1. Stage 1 proposes candidate boundaries C.
  2. LLM teacher ranks skills for each candidate segment → pairwise preferences.
  3. Train a PreferenceScorer from those preferences (Bradley-Terry).
  4. Decode with the trained scorer (DP or beam) → skill sequence + diagnostics.
  5. On uncertain segments, query the LLM for more preferences → retrain → re-decode.
  6. Return SubTask_Experience segments with learned skill labels.

The LLM never produces numeric scores.  It only provides rankings/preferences.
All numeric scoring comes from the trained PreferenceScorer.

Usage:
    from skill_agents.infer_segmentation import infer_and_segment

    result, sub_episodes, store = infer_and_segment(
        episode,
        skill_names=["move", "attack", "gather", "craft"],
        env_name="avalon",
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from skill_agents.infer_segmentation.config import SegmentationConfig
from skill_agents.infer_segmentation.scorer import SegmentScorer
from skill_agents.infer_segmentation.dp_decoder import viterbi_decode
from skill_agents.infer_segmentation.beam_decoder import beam_decode
from skill_agents.infer_segmentation.diagnostics import SegmentationResult
from skill_agents.infer_segmentation.preference import (
    PreferenceStore,
    PreferenceScorer,
    generate_preference_queries,
)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _extract_obs_actions(experiences: list) -> Tuple[list, list]:
    """Pull observations and actions from Experience objects."""
    observations = []
    actions = []
    for exp in experiences:
        obs = getattr(exp, "summary_state", None) or getattr(exp, "summary", None) or getattr(exp, "state", None)
        observations.append(obs)
        actions.append(getattr(exp, "action", None))
    return observations, actions


def _extract_predicates(experiences: list) -> List[Optional[dict]]:
    """Build per-timestep predicate dicts from experiences.

    When an experience carries an ``intentions`` string with a ``[TAG]``
    prefix, we decompose it into ``tag_<tag>`` one-hot keys and a
    ``<tag>_completed`` flag so that ``_effects_compat_score`` can match
    intention-extracted contracts (Strategy B).
    """
    from skill_agents.boundary_proposal.signal_extractors import parse_intention_tag

    predicates = []
    for exp in experiences:
        preds: dict = {}

        if getattr(exp, "sub_tasks", None) is not None:
            preds["sub_task"] = exp.sub_tasks

        intent = getattr(exp, "intentions", None)
        if intent is not None:
            preds["intention"] = intent
            tag = parse_intention_tag(intent)
            if tag != "UNKNOWN":
                preds[f"tag_{tag.lower()}"] = 1.0
                preds[f"{tag.lower()}_completed"] = float(
                    getattr(exp, "done", False)
                )

        if getattr(exp, "done", None) is not None:
            preds["done"] = exp.done

        predicates.append(preds if preds else None)
    return predicates


def _build_scorer_from_preferences(
    skill_names: List[str],
    store: PreferenceStore,
    config: SegmentationConfig,
    compat_fn=None,
) -> SegmentScorer:
    """Train a PreferenceScorer and wrap it in a SegmentScorer."""
    pref_scorer = PreferenceScorer(
        skill_names=skill_names,
        lr=config.preference.learning_rate,
    )
    if len(store) > 0:
        pref_scorer.train(store, epochs=config.preference.training_epochs)

    return SegmentScorer(
        skill_names=skill_names,
        config=config,
        behavior_fit_fn=pref_scorer.behavior_fit,
        transition_fn=pref_scorer.transition_prior,
        compat_fn=compat_fn,
    )


def _decode(
    candidates: List[int],
    T: int,
    scorer: SegmentScorer,
    observations: Sequence,
    actions: Sequence,
    predicates: Optional[List[Optional[dict]]],
    config: SegmentationConfig,
) -> SegmentationResult:
    """Run DP or beam decoder."""
    if config.method == "beam":
        return beam_decode(candidates, T, scorer, observations, actions, predicates, config)
    return viterbi_decode(candidates, T, scorer, observations, actions, predicates, config)


def _segments_to_sub_episodes(result, experiences: list, task, outcome_length: int) -> list:
    """Build SubTask_Experience list from segmentation result and episode experiences."""
    from data_structure.experience import SubTask_Experience
    T = len(experiences)
    sub_episodes = []
    for seg in result.segments:
        start, end = seg.start, min(seg.end + 1, T)
        segment_exps = experiences[start:end]
        outcome_start = end
        outcome_end = min(end + outcome_length, T)
        outcome_exps = (
            experiences[outcome_start:outcome_end]
            if outcome_start < outcome_end
            else None
        )
        sub_ep = SubTask_Experience(
            sub_task=seg.assigned_skill,
            final_goal=task,
            experiences=segment_exps,
            outcome=outcome_exps,
            seg_id=getattr(seg, "seg_id", None),
        )
        sub_episodes.append(sub_ep)
    return sub_episodes


# ── Low-level API ────────────────────────────────────────────────────

def infer_segmentation(
    candidates: List[int],
    T: int,
    skill_names: List[str],
    observations: Sequence,
    actions: Sequence,
    predicates: Optional[List[Optional[dict]]] = None,
    config: Optional[SegmentationConfig] = None,
    scorer: Optional[SegmentScorer] = None,
    behavior_fit_fn=None,
    transition_fn=None,
    duration_stats=None,
    compat_fn=None,
) -> SegmentationResult:
    """
    Core InferSegmentation: run DP or beam search over candidates.

    If ``scorer`` is provided, uses it directly.
    Otherwise builds a SegmentScorer from the given functions.
    """
    cfg = config or SegmentationConfig()

    if scorer is not None:
        active_scorer = scorer
    else:
        active_scorer = SegmentScorer(
            skill_names=skill_names,
            config=cfg,
            behavior_fit_fn=behavior_fit_fn,
            transition_fn=transition_fn,
            duration_stats=duration_stats,
            compat_fn=compat_fn,
        )

    return _decode(candidates, T, active_scorer, observations, actions, predicates, cfg)


# ── High-level API ───────────────────────────────────────────────────

def infer_and_segment(
    episode,
    skill_names: List[str],
    env_name: str = "generic",
    config: Optional[SegmentationConfig] = None,
    proposal_config=None,
    embedder=None,
    surprisal=None,
    outcome_length: int = 5,
    preference_store: Optional[PreferenceStore] = None,
    extractor_kwargs=None,
    compat_fn=None,
) -> Tuple[SegmentationResult, list, PreferenceStore]:
    """
    Inference pipeline with LLM (e.g. GPT-5):
      1. Stage 1 → candidate boundaries C.
      2. LLM teacher ranks skills for candidate segments → preferences.
      3. Train PreferenceScorer from preferences.
      4. Decode with trained scorer.
      5. Query LLM on uncertain segments → more preferences → retrain.
      6. Return segments + diagnostics + preference store.

    Use this when you want to interface with an LLM at inference time.

    Parameters
    ----------
    episode : Episode
        From data_structure.experience.
    skill_names : list[str]
        Known skill labels.
    env_name : str
        Passed to Stage 1 boundary proposal.
    config : SegmentationConfig, optional
    proposal_config : ProposalConfig, optional
    embedder : TextEmbedderBase, optional
    surprisal : np.ndarray, optional
    outcome_length : int
        Look-ahead steps for outcome window.
    preference_store : PreferenceStore, optional
        Existing preferences to bootstrap from.
    extractor_kwargs : dict, optional

    Returns
    -------
    (SegmentationResult, list[SubTask_Experience], PreferenceStore)
    """
    from data_structure.experience import SubTask_Experience
    from skill_agents.boundary_proposal import (
        propose_from_episode,
        candidate_centers_only,
    )
    from skill_agents.infer_segmentation.llm_teacher import (
        collect_segment_preferences,
        collect_transition_preferences,
        collect_uncertain_preferences,
    )

    cfg = config or SegmentationConfig()
    experiences = episode.experiences
    T = len(experiences)
    if T == 0:
        return SegmentationResult(), [], PreferenceStore()

    # Stage 1: boundary proposal
    boundary_candidates = propose_from_episode(
        episode,
        env_name=env_name,
        config=proposal_config,
        embedder=embedder,
        surprisal=surprisal,
        extractor_kwargs=extractor_kwargs,
    )
    centers = candidate_centers_only(boundary_candidates)

    observations, actions = _extract_obs_actions(experiences)
    predicates = _extract_predicates(experiences)

    # Build segment list from boundaries
    cut_indices = sorted(set([0] + centers + [T - 1]))
    segments = []
    for idx in range(len(cut_indices) - 1):
        segments.append((cut_indices[idx], cut_indices[idx + 1]))
    if not segments:
        segments = [(0, T - 1)]

    store = preference_store or PreferenceStore()

    # ── Cold-start: collect preferences from LLM teacher ────────────
    if len(store) == 0:
        segment_prefs = collect_segment_preferences(
            segments, observations, actions, skill_names,
            predicates=predicates, config=cfg.llm_teacher,
        )
        store.add_batch(segment_prefs)

        if cfg.preference.collect_transitions:
            transition_prefs = collect_transition_preferences(
                skill_names, config=cfg.llm_teacher,
            )
            store.add_batch(transition_prefs)

    # ── Train scorer and decode ─────────────────────────────────────
    scorer = _build_scorer_from_preferences(skill_names, store, cfg, compat_fn=compat_fn)
    result = _decode(centers, T, scorer, observations, actions, predicates, cfg)

    # ── Active learning iterations ──────────────────────────────────
    for iteration in range(cfg.preference.num_iterations):
        new_prefs = collect_uncertain_preferences(
            result, observations, actions,
            margin_threshold=cfg.preference.margin_threshold,
            max_queries=cfg.preference.max_queries_per_iter,
            config=cfg.llm_teacher,
        )
        if not new_prefs:
            break

        store.add_batch(new_prefs)
        scorer = _build_scorer_from_preferences(skill_names, store, cfg, compat_fn=compat_fn)
        result = _decode(centers, T, scorer, observations, actions, predicates, cfg)

    sub_episodes = _segments_to_sub_episodes(result, experiences, episode.task, outcome_length)
    return result, sub_episodes, store


def infer_and_segment_offline(
    episode,
    skill_names: List[str],
    env_name: str = "generic",
    config: Optional[SegmentationConfig] = None,
    proposal_config=None,
    embedder=None,
    surprisal=None,
    outcome_length: int = 5,
    behavior_fit_fn=None,
    transition_fn=None,
    duration_stats=None,
    extractor_kwargs=None,
    compat_fn=None,
) -> Tuple[SegmentationResult, list]:
    """
    Offline pipeline (no LLM): decode using provided scoring functions only.

    No LLM (e.g. GPT-5) is called. Use when (1) you have a pre-trained
    PreferenceScorer and want inference without API, (2) training/decoding
    from pre-collected preferences, or (3) testing without API keys.
    """
    from data_structure.experience import SubTask_Experience
    from skill_agents.boundary_proposal import (
        propose_from_episode,
        candidate_centers_only,
    )

    cfg = config or SegmentationConfig()
    experiences = episode.experiences
    T = len(experiences)
    if T == 0:
        return SegmentationResult(), []

    boundary_candidates = propose_from_episode(
        episode,
        env_name=env_name,
        config=proposal_config,
        embedder=embedder,
        surprisal=surprisal,
        extractor_kwargs=extractor_kwargs,
    )
    centers = candidate_centers_only(boundary_candidates)

    observations, actions = _extract_obs_actions(experiences)
    predicates = _extract_predicates(experiences)

    result = infer_segmentation(
        candidates=centers,
        T=T,
        skill_names=skill_names,
        observations=observations,
        actions=actions,
        predicates=predicates,
        config=cfg,
        behavior_fit_fn=behavior_fit_fn,
        transition_fn=transition_fn,
        duration_stats=duration_stats,
        compat_fn=compat_fn,
    )
    sub_episodes = _segments_to_sub_episodes(result, experiences, episode.task, outcome_length)
    return result, sub_episodes

"""
Sub-episode quality evaluation, drift detection, and protocol synthesis.

Stage 4.5 in the SkillBankAgent pipeline: after contract learning (Stage 3)
and before bank maintenance (Stage 4), this module inspects sub-episode
**pointers** (``SubEpisodeRef``) attached to each skill and makes
aggregate / update / drop decisions.

SubEpisodeRef carries only lightweight metadata (summary, intention_tags,
cumulative_reward, outcome) — the actual Experience data stays in rollout
storage.  All quality scoring here operates on those cached fields without
needing to load full trajectories.

Quality scoring dimensions:
  - outcome_reward:  normalized cumulative reward of the segment
  - follow_through:  did the skill's success_criteria get met?
  - consistency:     does the tag sequence match the skill's expected pattern?
  - compactness:     segment length vs expected_duration (penalty for too long/short)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from skill_agents_grpo.stage3_mvp.schemas import (
    Protocol,
    Skill,
    SubEpisodeRef,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────────────────────────────

def _tag_consistency(
    intention_tags: List[str],
    expected_pattern: List[str],
) -> float:
    """Measure how well the observed intention tags match the expected pattern.

    Returns 1.0 for perfect alignment, 0.0 for no overlap.  When no
    expected_pattern is set, returns a neutral 0.5.
    """
    if not expected_pattern:
        return 0.5
    if not intention_tags:
        return 0.0
    expected_set = set(expected_pattern)
    observed_set = set(intention_tags)
    if not expected_set:
        return 0.5
    overlap = len(expected_set & observed_set)
    return overlap / len(expected_set)


def score_sub_episode(
    sub_ep: SubEpisodeRef,
    skill: Skill,
    *,
    reward_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Compute a composite quality score in [0, 1] for a single sub-episode.

    Dimensions:
      - outcome_reward: min-max normalized cumulative reward (weight 0.3)
      - follow_through: 1.0 if success, 0.5 if partial, 0.0 if failure (weight 0.3)
      - compactness: 1 - |len - expected_duration| / expected_duration (weight 0.2)
      - consistency: tag-sequence alignment with expected pattern (weight 0.2)
    """
    # Outcome reward (normalized)
    r_min, r_max = reward_range
    r_range = r_max - r_min if r_max > r_min else 1.0
    outcome_reward = max(0.0, min(1.0, (sub_ep.cumulative_reward - r_min) / r_range))

    # Follow-through
    follow_map = {"success": 1.0, "partial": 0.5, "failure": 0.0}
    follow_through = follow_map.get(sub_ep.outcome, 0.5)

    # Compactness (uses SubEpisodeRef.length property)
    expected = max(1, skill.protocol.expected_duration)
    compactness = max(0.0, 1.0 - abs(sub_ep.length - expected) / expected)

    # Consistency (uses intention_tags stored on the pointer)
    consistency = _tag_consistency(
        sub_ep.intention_tags,
        skill.expected_tag_pattern,
    )

    w_reward, w_follow, w_compact, w_consist = 0.3, 0.3, 0.2, 0.2
    score = (
        w_reward * outcome_reward
        + w_follow * follow_through
        + w_compact * compactness
        + w_consist * consistency
    )
    return max(0.0, min(1.0, score))


def score_all_sub_episodes(
    skill: Skill,
    reward_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Score all sub-episodes of a skill in place."""
    if not skill.sub_episodes:
        return

    if reward_range is None:
        rewards = [se.cumulative_reward for se in skill.sub_episodes]
        r_min = min(rewards) if rewards else 0.0
        r_max = max(rewards) if rewards else 1.0
        reward_range = (r_min, r_max)

    for se in skill.sub_episodes:
        se.quality_score = score_sub_episode(se, skill, reward_range=reward_range)


# ─────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────

def protocol_drift_detected(
    new_sub_eps: List[SubEpisodeRef],
    skill: Skill,
    drift_threshold: float = 0.2,
) -> bool:
    """Detect whether new sub-episodes show significantly different patterns
    from the existing protocol.

    Heuristic: if the success rate of new sub-episodes differs from
    the overall success rate by more than *drift_threshold*, drift is detected.
    """
    if not new_sub_eps or not skill.sub_episodes:
        return False

    overall_sr = skill.success_rate
    new_successes = sum(1 for se in new_sub_eps if se.outcome == "success")
    new_sr = new_successes / len(new_sub_eps)

    return abs(new_sr - overall_sr) > drift_threshold


# ─────────────────────────────────────────────────────────────────────
# Aggregate / Update / Drop decisions
# ─────────────────────────────────────────────────────────────────────

def run_quality_check(
    skill: Skill,
    *,
    drop_threshold: float = 0.2,
    min_aggregate_count: int = 3,
    min_viable_count: int = 2,
) -> Dict:
    """Run the full quality-check pipeline for one skill.

    Steps:
      1. Score all sub-episodes
      2. DROP bottom-quality sub-episodes
      3. AGGREGATE: mark skill for protocol update if enough high-quality exist
      4. RETIRE if too few sub-episodes remain

    Returns a dict summarizing the actions taken.
    """
    score_all_sub_episodes(skill)

    result = {
        "skill_id": skill.skill_id,
        "before_count": len(skill.sub_episodes),
        "dropped": 0,
        "needs_protocol_update": False,
        "retired": False,
    }

    # DROP: remove bottom-quality sub-episodes
    if skill.sub_episodes:
        scores = sorted(se.quality_score for se in skill.sub_episodes)
        adaptive_threshold = max(
            drop_threshold,
            scores[len(scores) // 5] if len(scores) >= 5 else 0.0,
        )
        before = len(skill.sub_episodes)
        skill.sub_episodes = [
            se for se in skill.sub_episodes
            if se.quality_score >= adaptive_threshold
        ]
        result["dropped"] = before - len(skill.sub_episodes)

    # AGGREGATE: check if enough high-quality sub-episodes exist
    high_quality = [se for se in skill.sub_episodes if se.quality_score >= 0.6]
    if len(high_quality) >= min_aggregate_count:
        result["needs_protocol_update"] = True

    # RETIRE if too few remain
    if len(skill.sub_episodes) < min_viable_count:
        skill.retired = True
        result["retired"] = True

    # Update instance count
    skill.n_instances = len(skill.sub_episodes)
    result["after_count"] = len(skill.sub_episodes)

    return result


def run_quality_check_batch(
    skills: List[Skill],
    **kwargs,
) -> List[Dict]:
    """Run quality check on a batch of skills."""
    results = []
    for skill in skills:
        if skill.retired:
            continue
        r = run_quality_check(skill, **kwargs)
        results.append(r)
        if r["dropped"] > 0:
            logger.info(
                "Skill %s: dropped %d low-quality sub-episodes (%d remaining)",
                skill.skill_id, r["dropped"], r["after_count"],
            )
        if r["retired"]:
            logger.info("Skill %s: retired (too few sub-episodes)", skill.skill_id)
    return results

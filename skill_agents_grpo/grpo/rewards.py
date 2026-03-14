"""
Reward functions for each GRPO-wrapped stage.

Each reward function has the signature expected by ``GRPOCallWrapper``:
    ``reward_fn(sample_output, *original_args, **original_kwargs) -> float``

All reward computations are CPU-only — no additional LLM inference.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Stage 3: CONTRACT reward ─────────────────────────────────────────

def contract_reward(
    llm_output: Optional[Dict[str, Any]],
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
    *,
    consensus_add: Optional[Set[str]] = None,
    consensus_del: Optional[Set[str]] = None,
    holdout_instances: Optional[list] = None,
    verify_config: Optional[Any] = None,
    **kwargs: Any,
) -> float:
    """Reward for a contract LLM output based on holdout verification.

    If holdout data isn't provided, falls back to a simple structural
    score based on whether the output is valid JSON with non-empty fields.

    Parameters
    ----------
    llm_output : dict or None
        The parsed contract from ``llm_summarize_contract()``.
    consensus_add, consensus_del : sets
        Frequency-based consensus effects (for union-merge).
    holdout_instances : list[SegmentRecord]
        Held-out segment records for verification.
    verify_config : Stage3MVPConfig
        Configuration for ``verify_effects_contract()``.
    """
    if llm_output is None:
        return 0.0

    eff_add = llm_output.get("eff_add", [])
    eff_del = llm_output.get("eff_del", [])

    if not eff_add and not eff_del:
        return 0.05  # valid JSON but empty — slightly above None

    # Full verification path (when holdout data available)
    if holdout_instances is not None and verify_config is not None:
        return _contract_reward_with_verification(
            llm_output, skill_id, consensus_add, consensus_del,
            holdout_instances, verify_config,
        )

    # Fallback: structural reward
    n_literals = len(eff_add) + len(eff_del)
    sparsity_bonus = max(0.0, 1.0 - n_literals / 50.0) * 0.2
    return 0.3 + sparsity_bonus


def _contract_reward_with_verification(
    llm_output: Dict[str, Any],
    skill_id: str,
    consensus_add: Optional[Set[str]],
    consensus_del: Optional[Set[str]],
    holdout_instances: list,
    verify_config: Any,
) -> float:
    """Compute reward using actual ``verify_effects_contract()``."""
    from skill_agents_grpo.stage3_mvp.contract_verify import verify_effects_contract
    from skill_agents_grpo.stage3_mvp.schemas import SkillEffectsContract

    llm_add = set(llm_output.get("eff_add", []))
    llm_del = set(llm_output.get("eff_del", []))

    # Union-merge with frequency consensus
    merged_add = (consensus_add or set()) | llm_add
    merged_del = (consensus_del or set()) | llm_del

    contract = SkillEffectsContract(
        skill_id=skill_id,
        eff_add=merged_add,
        eff_del=merged_del,
    )

    report = verify_effects_contract(contract, holdout_instances, verify_config)

    # Weighted reward components
    r_pass = report.overall_pass_rate  # [0, 1]

    all_sr = list(report.eff_add_success_rate.values()) + \
             list(report.eff_del_success_rate.values())
    r_literal = sum(all_sr) / max(len(all_sr), 1) if all_sr else 0.0

    n_literals = contract.total_literals
    budget = 20
    r_sparsity = max(0.0, 1.0 - max(0, n_literals - budget) / budget)

    n_covered = sum(
        1 for inst in holdout_instances
        if _instance_passes(contract, inst, verify_config)
    )
    r_coverage = n_covered / max(len(holdout_instances), 1)

    return (
        0.50 * r_pass
        + 0.25 * r_literal
        + 0.15 * r_sparsity
        + 0.10 * r_coverage
    )


def _instance_passes(contract: Any, instance: Any, config: Any) -> bool:
    """Check if a single instance passes the contract."""
    fails = 0
    for p in contract.eff_add:
        if p not in instance.eff_add and p not in instance.B_end:
            fails += 1
    for p in contract.eff_del:
        if p not in instance.eff_del and p in instance.B_end:
            fails += 1
    total = contract.total_literals
    if total == 0:
        return True
    return (total - fails) / total >= config.instance_pass_literal_frac


# ── Stage 4: CURATOR reward ──────────────────────────────────────────

def curator_reward(
    decisions: Optional[Dict[str, Any]],
    candidates: list,
    bank: Any,
    *args: Any,
    compute_quality_fn: Optional[Any] = None,
    execute_fn: Optional[Any] = None,
    **kwargs: Any,
) -> float:
    """Reward for curator filter decisions.

    Compares bank quality after filtered execution vs. full execution.
    Falls back to a heuristic if bank/execution functions aren't available.

    Parameters
    ----------
    decisions : dict
        ``{"decisions": [{"idx": 0, "verdict": "approve", "reason": "..."}, ...]}``
    candidates : list
        Candidate bank mutations (from the algorithmic proposer).
    bank : SkillBankMVP
        Current skill bank state.
    compute_quality_fn : callable, optional
        ``compute_quality_fn(bank) -> float`` — bank quality metric.
    execute_fn : callable, optional
        ``execute_fn(actions, bank) -> bank`` — execute actions on a bank copy.
    """
    if decisions is None:
        return 0.0

    decision_list = decisions.get("decisions", [])
    if not decision_list:
        return 0.05

    n_approve = sum(1 for d in decision_list if d.get("verdict") == "approve")
    n_veto = sum(1 for d in decision_list if d.get("verdict") == "veto")
    n_defer = sum(1 for d in decision_list if d.get("verdict") == "defer")
    n_total = len(decision_list)

    # Full quality comparison (when bank operations available)
    if compute_quality_fn is not None and execute_fn is not None:
        return _curator_reward_with_quality(
            decisions, candidates, bank, compute_quality_fn, execute_fn,
        )

    # Heuristic fallback: reward conservative but non-trivial filtering
    if n_total == 0:
        return 0.0

    approve_rate = n_approve / n_total
    has_reasons = all(d.get("reason") for d in decision_list)
    reason_bonus = 0.1 if has_reasons else 0.0

    # Prefer moderate filtering (not all-approve, not all-veto)
    diversity = 1.0 - abs(approve_rate - 0.5) * 2.0
    conservative_bonus = 0.1 * (n_defer / n_total) if n_defer > 0 else 0.0

    return 0.3 + 0.3 * diversity + conservative_bonus + reason_bonus


def _curator_reward_with_quality(
    decisions: Dict[str, Any],
    candidates: list,
    bank: Any,
    compute_quality_fn: Any,
    execute_fn: Any,
) -> float:
    """Compute quality delta: q(filtered) - q(all)."""
    decision_list = decisions.get("decisions", [])
    approved_indices = {
        d["idx"] for d in decision_list
        if d.get("verdict") == "approve" and "idx" in d
    }

    filtered_actions = [c for i, c in enumerate(candidates) if i in approved_indices]

    bank_filtered = deepcopy(bank)
    execute_fn(filtered_actions, bank_filtered)
    q_filtered = compute_quality_fn(bank_filtered)

    bank_all = deepcopy(bank)
    execute_fn(candidates, bank_all)
    q_all = compute_quality_fn(bank_all)

    quality_delta = q_filtered - q_all  # positive = filtering helped

    n_total = len(decision_list)
    n_defer = sum(1 for d in decision_list if d.get("verdict") == "defer")
    conservative_bonus = 0.1 * (n_defer / max(n_total, 1))

    # Combine: quality delta is primary, conservative bonus is secondary
    return max(0.0, 0.5 + quality_delta) + conservative_bonus


# ── Stage 2: SEGMENT reward ──────────────────────────────────────────

def segmentation_reward(
    preference_list: list,
    segments: list,
    observations: Any,
    actions: Any,
    skill_names: List[str],
    *args: Any,
    scorer_factory: Optional[Any] = None,
    decode_fn: Optional[Any] = None,
    predicates: Optional[list] = None,
    **kwargs: Any,
) -> float:
    """Reward for a batch of segment preferences.

    When scorer/decode functions are available, rebuilds the scorer
    from the preferences, runs the decoder, and evaluates diagnostics.
    Otherwise falls back to a preference-count heuristic.

    Parameters
    ----------
    preference_list : list[PreferenceExample]
        Pairwise preferences produced by one GRPO sample.
    segments : list[(start, end)]
        Segment boundaries.
    scorer_factory : callable, optional
        Builds a SegmentScorer from preferences.
    decode_fn : callable, optional
        Runs Viterbi/beam decode with the scorer.
    """
    if not preference_list:
        return 0.0

    # Full decode path
    if scorer_factory is not None and decode_fn is not None:
        return _segmentation_reward_with_decode(
            preference_list, segments, observations, actions,
            skill_names, scorer_factory, decode_fn, predicates,
        )

    # Heuristic: more preferences from more segments = better coverage
    segment_coverage = len({(p.segment_start, p.segment_end) for p in preference_list})
    coverage_ratio = segment_coverage / max(len(segments), 1)

    # Preferences per segment (more = more discriminative rankings)
    prefs_per_seg = len(preference_list) / max(segment_coverage, 1)
    depth_score = min(1.0, prefs_per_seg / 6.0)  # 6 prefs = 4-way ranking

    return 0.4 * coverage_ratio + 0.4 * depth_score + 0.2


def _segmentation_reward_with_decode(
    preference_list: list,
    segments: list,
    observations: Any,
    actions: Any,
    skill_names: List[str],
    scorer_factory: Any,
    decode_fn: Any,
    predicates: Optional[list],
) -> float:
    """Build scorer from preferences, decode, evaluate diagnostics."""
    try:
        scorer = scorer_factory(preference_list)
        result = decode_fn(scorer, segments, observations, actions, skill_names, predicates)
    except Exception:
        logger.debug("Segmentation reward decode failed", exc_info=True)
        return 0.1

    # Extract diagnostics from result
    if hasattr(result, "segments") and result.segments:
        margins = []
        n_new = 0
        n_total = len(result.segments)

        for seg in result.segments:
            if hasattr(seg, "margin"):
                margins.append(seg.margin)
            if hasattr(seg, "skill") and seg.skill == "__NEW__":
                n_new += 1

        r_margin = (sum(margins) / len(margins) / 5.0) if margins else 0.0
        r_new_penalty = -0.5 * (n_new / max(n_total, 1))
        n_confident = sum(1 for m in margins if m > 1.0)
        r_confident = n_confident / max(n_total, 1)

        return max(0.0, 0.35 * r_margin + 0.35 * r_confident + 0.20 + 0.10 * r_new_penalty)

    return 0.2  # decode succeeded but no segment diagnostics

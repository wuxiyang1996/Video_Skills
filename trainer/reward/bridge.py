"""Bridge runtime verifier + hidden evaluator + milestones → verified reward."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dataset_clip_wrapper.verification.runtime_verifier import verify_rollout

from .milestone_ledger import MilestoneLedger, ledger_from_events
from .semantic_judge import (
    JudgeFn,
    SemanticJudgeResult,
    aggregate_dual_views,
    credit_allowed,
    mock_semantic_judge,
)
from .verified_reward import VerifiedRewardBreakdown, score_verified_rollout


def _labels_equal(pred: Any, gold: Any) -> bool:
    if pred is None or gold is None:
        return False
    if isinstance(pred, Mapping):
        pred_l = pred.get("label") or pred.get("text") or pred
    else:
        pred_l = pred
    if isinstance(gold, Mapping):
        gold_l = gold.get("label") or gold.get("text") or gold
    else:
        gold_l = gold
    return str(pred_l).strip().lower() == str(gold_l).strip().lower()


def hidden_terminal_eval(
    rollout: Mapping[str, Any],
    *,
    gold_answer: Any = None,
    unanswerable: bool | None = None,
) -> dict[str, Any]:
    """Hidden evaluator only — never passed to the policy surface."""
    final_answer = rollout.get("final_answer")
    acceptance = str(rollout.get("acceptance_status") or "")
    abstained = bool(
        rollout.get("abstained")
        or acceptance.startswith("abstain")
        or str((final_answer or {}).get("label") if isinstance(final_answer, Mapping) else "")
        .lower()
        in {"abstain", "unanswerable"}
    )
    if unanswerable is None:
        unanswerable = bool(rollout.get("unanswerable") or (rollout.get("question") or {}).get("unanswerable"))
    answer_correct = False
    if gold_answer is not None and not abstained:
        answer_correct = _labels_equal(final_answer, gold_answer)
    elif gold_answer is None and "answer_correct" in rollout:
        answer_correct = bool(rollout.get("answer_correct"))
    return {
        "answer_correct": bool(answer_correct),
        "unanswerable": bool(unanswerable),
        "abstained": bool(abstained),
        # Keep gold out of any policy-facing dict by marking this hidden.
        "_hidden": True,
    }


def policy_safe_rollout_view(rollout: Mapping[str, Any]) -> dict[str, Any]:
    """Strip hidden reward / gold fields before returning traces to the policy path."""
    banned = {
        "gold_answer",
        "official_answer",
        "answer_correct",
        "hidden_eval",
        "judge_rationale",
        "teacher_probs",
        "reward",
        "rank_key",
        "verified_reward",
    }
    out: dict[str, Any] = {}
    for key, value in rollout.items():
        if key in banned or key.startswith("_hidden"):
            continue
        if key == "metadata" and isinstance(value, Mapping):
            meta = {
                mk: mv
                for mk, mv in value.items()
                if mk not in banned and not str(mk).startswith("_hidden")
            }
            out[key] = meta
        else:
            out[key] = value
    return out


def _cost_from_rollout(rollout: Mapping[str, Any]) -> dict[str, int]:
    meta = rollout.get("metadata") or {}
    costs = meta.get("costs") or rollout.get("costs") or {}
    return {
        "clip_reads": int(costs.get("clip_reads") or meta.get("clip_reads") or 0),
        "tool_calls": int(
            costs.get("tool_calls")
            or len(meta.get("executed_skill_ids") or [])
            or meta.get("tool_calls")
            or 0
        ),
        "tokens": int(costs.get("tokens") or meta.get("tokens") or 0),
        "repair_rounds": int(costs.get("repair_rounds") or meta.get("repair_rounds") or 0),
    }


def maybe_judge_claim(
    *,
    question_text: str,
    claim: str,
    evidence: Sequence[Mapping[str, Any]],
    judge_fn: JudgeFn | None,
    n_views: int = 2,
) -> SemanticJudgeResult | None:
    if judge_fn is None:
        return None
    views = []
    for i in range(max(int(n_views), 1)):
        views.append(
            judge_fn(
                {
                    "question_text": question_text,
                    "claim": claim,
                    "evidence": list(evidence),
                    "view_id": str(i),
                }
            )
        )
    return aggregate_dual_views(views) if len(views) > 1 else views[0]


def build_ledger_from_rollout(
    rollout: Mapping[str, Any],
    *,
    judge_fn: JudgeFn | None = None,
) -> tuple[MilestoneLedger, bool]:
    """Return (ledger, blocked_strong_commit)."""
    meta = rollout.get("metadata") or {}
    events = list(meta.get("milestone_events") or rollout.get("milestone_events") or [])
    final_used = list(meta.get("final_used_milestone_keys") or rollout.get("final_used_milestone_keys") or [])
    contradicted = list(meta.get("contradicted_milestone_keys") or [])

    blocked_strong = False
    # Optional semantic gate on a commit claim.
    claim = meta.get("commit_claim") or rollout.get("commit_claim")
    evidence = meta.get("commit_evidence") or rollout.get("commit_evidence") or []
    if claim and judge_fn is not None:
        question_text = str(((rollout.get("question") or {}).get("question_text")) or "")
        judged = maybe_judge_claim(
            question_text=question_text,
            claim=str(claim),
            evidence=list(evidence) if isinstance(evidence, Sequence) else [],
            judge_fn=judge_fn,
        )
        if judged is not None:
            if judged.verdict == "contradicted":
                blocked_strong = True
                contradicted.append("verify:commit_claim")
            elif not credit_allowed(judged):
                blocked_strong = judged.verdict != "supported"
            elif credit_allowed(judged):
                events.append(
                    {
                        "kind": "verify",
                        "key": "commit_claim",
                        "step_index": len(events),
                        "grounded": True,
                        "detail": {"refs": list(judged.grounded_refs)},
                    }
                )
                final_used.append("verify:commit_claim")

    ledger = ledger_from_events(
        events,
        final_used_keys=final_used or None,
        contradicted_keys=contradicted or None,
    )
    # Motif lifecycle must never create milestones.
    motif = meta.get("motif_online") or {}
    if motif.get("candidate_mined") or motif.get("mined_motif_id"):
        pass  # explicitly ignored
    return ledger, blocked_strong


def score_rollout_trace(
    rollout: Mapping[str, Any],
    *,
    clue_graph: Mapping[str, Any] | None = None,
    gold_answer: Any = None,
    unanswerable: bool | None = None,
    judge_fn: JudgeFn | None = None,
    skill_allowed: bool = True,
    within_hard_budget: bool = True,
    non_diagnostic_visual_ok: bool | None = None,
) -> VerifiedRewardBreakdown:
    """Score a full rollout using deterministic verifier + hidden eval + milestones."""
    graph = dict(clue_graph or (rollout.get("metadata") or {}).get("clue_memory_graph") or {})
    verifier = verify_rollout(graph, dict(rollout), mode=str(rollout.get("eval_mode") or "video_only"))
    summary = verifier.get("verifier_summary") or {}
    hidden = hidden_terminal_eval(rollout, gold_answer=gold_answer, unanswerable=unanswerable)
    ledger, blocked_strong = build_ledger_from_rollout(rollout, judge_fn=judge_fn)
    costs = _cost_from_rollout(rollout)

    acceptance = str(rollout.get("acceptance_status") or verifier.get("acceptance_status") or "")
    # Promote verifier "accepted" to strong only when commit evidence + no block.
    if acceptance == "accepted" and summary.get("all_commits_have_evidence") and not blocked_strong:
        acceptance = "accepted_strong"
    elif acceptance == "accepted":
        acceptance = "accepted_weak"

    if non_diagnostic_visual_ok is None:
        non_diagnostic_visual_ok = bool(summary.get("retrieval_not_used_as_support", True))

    return score_verified_rollout(
        answer_correct=hidden["answer_correct"],
        acceptance_status=acceptance,
        schema_valid=bool(summary.get("schema_valid", True)),
        skill_allowed=skill_allowed,
        refs_exist=bool(summary.get("evidence_refs_exist", True)),
        no_hidden_leakage=bool(summary.get("no_hidden_supervision_leakage", True)),
        streaming_visibility_ok=bool(summary.get("streaming_visibility_ok", True)),
        within_hard_budget=within_hard_budget,
        unanswerable=hidden["unanswerable"],
        abstained=hidden["abstained"],
        commit_evidence_ok=bool(summary.get("all_commits_have_evidence", False)),
        non_diagnostic_visual_ok=bool(non_diagnostic_visual_ok),
        claim_support_hard_ok=bool(summary.get("retrieval_not_used_as_support", False))
        and not blocked_strong,
        verified_atomic_progress=ledger,
        blocked_strong_commit=blocked_strong,
        **costs,
    )


# Convenience default for dry-runs.
DEFAULT_MOCK_JUDGE: JudgeFn = mock_semantic_judge

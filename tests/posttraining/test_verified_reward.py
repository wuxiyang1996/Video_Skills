from __future__ import annotations

from trainer.reward import (
    aggregate_dual_views,
    assert_judge_prompt_safe,
    group_rank_advantages,
    ledger_from_events,
    mock_semantic_judge,
    score_verified_rollout,
)
from trainer.reward.semantic_judge import SemanticJudgeResult


def test_infeasible_is_worst() -> None:
    bad = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        refs_exist=False,
        no_hidden_leakage=False,
    )
    good = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        claim_support_hard_ok=True,
    )
    assert bad.rank_key < good.rank_key
    assert bad.hard_failures


def test_wrong_answer_cannot_beat_correct_via_cost() -> None:
    cheap_wrong = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        clip_reads=0,
        tool_calls=1,
    )
    expensive_correct = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        clip_reads=20,
        tool_calls=30,
        tokens=5000,
        repair_rounds=2,
    )
    assert cheap_wrong.rank_key < expensive_correct.rank_key


def test_partial_progress_cannot_beat_terminal_success() -> None:
    partial_only = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_weak",
        verified_atomic_progress=(4, 4, 3, 3, 2),
        commit_evidence_ok=True,
        claim_support_hard_ok=True,
    )
    correct = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        verified_atomic_progress=(0, 0, 0, 0, 0),
        clip_reads=50,
    )
    assert partial_only.rank_key < correct.rank_key


def test_duplicate_milestones_and_unused_revoked() -> None:
    ledger = ledger_from_events(
        [
            {"kind": "retrieval", "key": "r1", "step_index": 0, "grounded": True},
            {"kind": "retrieval", "key": "r1", "step_index": 1, "grounded": True},
            {"kind": "inference", "key": "e1", "step_index": 2, "grounded": True},
            {"kind": "compose", "key": "c1", "step_index": 3, "grounded": True},
        ],
        final_used_keys=["retrieval:r1"],
        contradicted_keys=["inference:e1"],
    )
    # duplicate retrieval collapsed; unused compose revoked; contradicted inference revoked
    assert ledger.progress_counts()["retrieval"] == 1
    assert ledger.progress_counts()["inference"] == 0
    assert ledger.progress_counts()["compose"] == 0


def test_accepted_weak_not_terminal_success() -> None:
    weak = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_weak",
        commit_evidence_ok=True,
        claim_support_hard_ok=True,
    )
    assert not weak.terminal_success


def test_abstain_only_when_unanswerable() -> None:
    on_answerable = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_strong",
        abstained=True,
        unanswerable=False,
    )
    on_unanswerable = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_strong",
        abstained=True,
        unanswerable=True,
    )
    assert not on_answerable.terminal_success
    assert on_unanswerable.terminal_success


def test_evidence_tiebreak_only_when_success_ties() -> None:
    low_ev = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
    )
    high_ev = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        non_diagnostic_visual_ok=True,
        claim_support_hard_ok=True,
    )
    assert high_ev.rank_key > low_ev.rank_key


def test_progress_tiebreak_among_equal_terminal() -> None:
    low = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_weak",
        verified_atomic_progress=(1, 0, 0, 0, 0),
    )
    high = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_weak",
        verified_atomic_progress=(2, 0, 0, 0, 0),
    )
    assert high.rank_key > low.rank_key


def test_group_rank_preserves_order() -> None:
    scores = [
        score_verified_rollout(answer_correct=False, acceptance_status="accepted_weak"),
        score_verified_rollout(
            answer_correct=True,
            acceptance_status="accepted_strong",
            commit_evidence_ok=True,
            claim_support_hard_ok=True,
        ),
        score_verified_rollout(
            answer_correct=True,
            acceptance_status="accepted_strong",
            commit_evidence_ok=True,
            clip_reads=100,
        ),
    ]
    adv = group_rank_advantages(scores)
    assert adv[1] > adv[0]


def test_judge_forbidden_fields_and_dual_view_disagreement() -> None:
    try:
        assert_judge_prompt_safe({"gold_answer": "A"})
        raised = False
    except ValueError:
        raised = True
    assert raised

    a = SemanticJudgeResult(verdict="supported", relation_valid=True, question_relevant=True)
    b = SemanticJudgeResult(verdict="contradicted", relation_valid=False, question_relevant=True)
    agg = aggregate_dual_views([a, b])
    assert agg.verdict == "insufficient"

    ok = mock_semantic_judge(
        {
            "claim": "person pushed cup before fall",
            "evidence": [{"ref": "e1", "text": "push"}, {"ref": "e2", "text": "fall"}],
        }
    )
    assert ok.verdict == "supported"

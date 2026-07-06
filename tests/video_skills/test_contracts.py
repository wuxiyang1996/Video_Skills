"""Tests for the canonical runtime contracts (§2A)."""
from __future__ import annotations

import pytest

from video_skills.contracts import (
    SCHEMA_VERSION,
    AtomicStepResult,
    EvidenceBundle,
    EvidenceRef,
    HopGoal,
    HopRecord,
    QuestionAnalysis,
    ReasoningTrace,
    RetrievalQuery,
    VerificationCheck,
    VerificationResult,
    most_severe_next_action,
    new_id,
    validate_atomic_step,
)


def _make_bundle(*, refs=None, contradictions=None) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs or [],
        query=RetrievalQuery(query_id=new_id("rq"), text="t"),
        contradictions=contradictions or [],
    )


def test_schema_version_present_on_all_objects():
    objs = [
        EvidenceBundle(bundle_id="b", refs=[], query=RetrievalQuery(query_id="q", text="t")),
        HopGoal(hop_id="h", parent_question_id="q", goal_text="g", target_claim_type="claim"),
        VerificationResult(passed=True),
        QuestionAnalysis(question_id="q", question_text="t", question_type="free"),
        HopRecord(hop_goal=HopGoal(hop_id="h", parent_question_id="q", goal_text="g", target_claim_type="claim")),
    ]
    for o in objs:
        assert getattr(o, "schema_version", None) == SCHEMA_VERSION


def test_most_severe_next_action_picks_worst():
    assert most_severe_next_action([]) == "continue"
    assert most_severe_next_action(["continue", "broaden", "retry"]) == "broaden"
    assert most_severe_next_action(["continue", "abstain", "retry"]) == "abstain"
    assert most_severe_next_action(["switch_skill", "broaden"]) == "switch_skill"


def test_validate_atomic_step_rejects_unknown_output_type():
    step = AtomicStepResult(
        step_id="s", hop_id="h", skill_id="x",
        inputs={}, output={}, output_type="bogus",
        evidence=None,
        verification=VerificationResult(passed=True),
        confidence=0.0,
    )
    violations = validate_atomic_step(step)
    assert any("output_type" in v for v in violations)


def test_validate_atomic_step_claim_without_evidence_or_abstain():
    step = AtomicStepResult(
        step_id="s", hop_id="h", skill_id="x",
        inputs={}, output={"claim": True}, output_type="claim",
        evidence=None,
        verification=VerificationResult(passed=True, next_action="continue"),
        confidence=0.5,
    )
    violations = validate_atomic_step(step)
    assert any("no evidence" in v for v in violations)


def test_validate_atomic_step_claim_with_abstain_is_ok():
    step = AtomicStepResult(
        step_id="s", hop_id="h", skill_id="x",
        inputs={}, output={"claim": True}, output_type="claim",
        evidence=None,
        verification=VerificationResult(passed=False, next_action="abstain"),
        confidence=0.0,
    )
    assert validate_atomic_step(step) == []


def test_validate_atomic_step_claim_with_nonempty_bundle_is_ok():
    bundle = _make_bundle(refs=[EvidenceRef(ref_id="r", modality="frame")])
    step = AtomicStepResult(
        step_id="s", hop_id="h", skill_id="x",
        inputs={}, output={"claim": True}, output_type="claim",
        evidence=bundle,
        verification=VerificationResult(passed=True),
        confidence=0.8,
    )
    assert validate_atomic_step(step) == []


def test_reasoning_trace_append_hop_updates_cost_and_skill_set():
    qa = QuestionAnalysis(question_id="q", question_text="t", question_type="free")
    trace = ReasoningTrace(trace_id="t", question_id="q", question_analysis=qa)
    hop = HopRecord(
        hop_goal=HopGoal(
            hop_id="h", parent_question_id="q",
            goal_text="g", target_claim_type="claim",
        ),
        cost={"atomic_steps": 3, "retrieval_calls": 1, "broaden_levels": 0, "latency_ms": 12},
        steps=[
            AtomicStepResult(
                step_id="s1", hop_id="h", skill_id="atom.x",
                inputs={}, output={}, output_type="meta",
                evidence=None,
                verification=VerificationResult(passed=True),
                confidence=1.0,
            ),
            AtomicStepResult(
                step_id="s2", hop_id="h", skill_id="atom.y",
                inputs={}, output={}, output_type="meta",
                evidence=None,
                verification=VerificationResult(passed=True),
                confidence=1.0,
            ),
        ],
    )
    trace.append_hop(hop)
    assert trace.cost["hops"] == 1
    assert trace.cost["atomic_steps"] == 3
    assert trace.cost["retrieval_calls"] == 1
    assert trace.bank_skill_ids_used == ["atom.x", "atom.y"]


def test_evidence_bundle_is_empty():
    assert _make_bundle().is_empty()
    assert not _make_bundle(refs=[EvidenceRef(ref_id="r", modality="frame")]).is_empty()


def test_verification_check_default_score_one():
    c = VerificationCheck(name="claim_evidence_alignment", passed=True)
    assert c.score == 1.0

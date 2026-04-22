"""Tests for the Verifier subsystem (§2C)."""
from __future__ import annotations

import pytest

from video_skills.contracts import (
    AtomicStepResult,
    EvidenceBundle,
    EvidenceRef,
    HopGoal,
    HopRecord,
    QuestionAnalysis,
    ReasoningTrace,
    RetrievalQuery,
    VerificationResult,
    new_id,
)
from video_skills.verifier import CHECK_NAMES, Verifier, VerifierConfig


def _bundle(*, refs=None, contradictions=None) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs or [],
        query=RetrievalQuery(query_id=new_id("rq"), text="t"),
        contradictions=contradictions or [],
    )


def _claim_step(*, evidence=None, output=None, output_type="claim") -> AtomicStepResult:
    return AtomicStepResult(
        step_id=new_id("s"), hop_id="h", skill_id="atom.x",
        inputs={}, output=output or {"claim": True},
        output_type=output_type,
        evidence=evidence,
        verification=VerificationResult(passed=True),
        confidence=0.8,
    )


def test_check_names_match_design_catalog():
    expected = {
        "claim_evidence_alignment",
        "evidence_sufficiency",
        "counterevidence",
        "temporal_consistency",
        "perspective_consistency",
        "entity_consistency",
    }
    assert set(CHECK_NAMES) == expected


def test_verify_step_no_evidence_for_claim_routes_to_broaden_or_abstain():
    v = Verifier()
    step = _claim_step(evidence=None)
    result = v.verify_step(step)
    assert not result.passed
    assert result.next_action in ("broaden", "abstain")


def test_verify_step_with_supporting_evidence_passes():
    v = Verifier()
    bundle = _bundle(refs=[
        EvidenceRef(ref_id="r1", modality="memory_node", text="alice picks up key", confidence=0.9),
        EvidenceRef(ref_id="r2", modality="memory_node", text="key on table", confidence=0.7),
    ])
    step = _claim_step(evidence=bundle, output={"claim": True, "text": "alice"})
    # Evidence text contains "alice" — should pass alignment
    result = v.verify_step(step)
    assert result.score > 0.0


def test_verify_step_counterevidence_outscoring_support_fails():
    v = Verifier()
    bundle = _bundle(
        refs=[EvidenceRef(ref_id="r", modality="memory_node", confidence=0.4, text="alice")],
        contradictions=[EvidenceRef(ref_id="c", modality="memory_node", confidence=0.95)],
    )
    step = _claim_step(evidence=bundle, output={"text": "alice"})
    result = v.verify_step(step)
    counter = next((c for c in result.checks if c.name == "counterevidence"), None)
    assert counter is not None
    assert counter.passed is False


def test_threshold_action_maps_score_to_next_action():
    v = Verifier(VerifierConfig(support_threshold=0.6, abstain_threshold=0.35))
    assert v._threshold_action(0.9) == "continue"
    assert v._threshold_action(0.4) == "broaden"
    assert v._threshold_action(0.1) == "abstain"


def test_verify_hop_with_perspective_anchor_requires_evidence_match():
    v = Verifier()
    hop_goal = HopGoal(
        hop_id="h", parent_question_id="q", goal_text="g",
        target_claim_type="belief", perspective_anchor="alice",
    )
    bundle_with_alice = _bundle(refs=[
        EvidenceRef(ref_id="r1", modality="memory_node", entities=["alice"], confidence=0.9),
    ])
    step = _claim_step(evidence=bundle_with_alice)
    hop = HopRecord(hop_goal=hop_goal, steps=[step])
    result = v.verify_hop(hop)
    perspective = next((c for c in result.checks if c.name == "perspective_consistency"), None)
    assert perspective is not None
    assert perspective.passed is True


def test_verify_hop_perspective_fails_when_evidence_missing_anchor():
    v = Verifier()
    hop_goal = HopGoal(
        hop_id="h", parent_question_id="q", goal_text="g",
        target_claim_type="belief", perspective_anchor="alice",
    )
    bundle_no_alice = _bundle(refs=[
        EvidenceRef(ref_id="r", modality="memory_node", entities=["bob"], confidence=0.9),
    ])
    step = _claim_step(evidence=bundle_no_alice)
    hop = HopRecord(hop_goal=hop_goal, steps=[step])
    result = v.verify_hop(hop)
    perspective = next((c for c in result.checks if c.name == "perspective_consistency"), None)
    assert perspective is not None
    assert perspective.passed is False
    assert result.next_action != "continue"


def test_decide_abstain_returns_blocking_checks():
    v = Verifier()
    qa = QuestionAnalysis(question_id="q", question_text="t", question_type="free")
    trace = ReasoningTrace(trace_id="t", question_id="q", question_analysis=qa)
    # Create a hop whose verification has a failing check
    hop = HopRecord(
        hop_goal=HopGoal(
            hop_id="h", parent_question_id="q", goal_text="g",
            target_claim_type="belief", perspective_anchor="alice",
        ),
        steps=[_claim_step(evidence=_bundle())],
    )
    hop.hop_verification = v.verify_hop(hop)
    trace.append_hop(hop)
    trace.final_verification = v.verify_final(trace)
    decision = v.decide_abstain(trace)
    assert decision.abstain is True
    assert isinstance(decision.blocking_checks, list)

"""Commit policy on multiple choice.

An abstention scores as wrong on these benchmarks, so it is strictly dominated
by committing the best hypothesis.  On the cached rollouts 95.7% of abstentions
already carried an option_label at the commit step and were refused there for
claim_not_verified.  always_commit_mcq lets the label through while keeping the
verification outcome visible; the default is unchanged.
"""

from atomic_skills.reasoning_graph_assembly.skills import commit_answer

OPTIONS = [{"label": "A", "text": "victim"}, {"label": "B", "text": "murderer"}]
CHAIN = {"evidence_refs": [], "items": []}


def _unverified(label="B"):
    return {"text": "the man is the murderer", "option_label": label, "claim_status": "insufficient"}


def test_default_still_refuses_an_unverified_claim() -> None:
    result = commit_answer(_unverified(), options=OPTIONS, answer_format="multiple_choice", support_chain=CHAIN)
    assert not result.ok
    assert result.failure_code == "claim_not_verified"


def test_always_commit_emits_the_candidate_label() -> None:
    result = commit_answer(
        _unverified("B"), options=OPTIONS, answer_format="multiple_choice",
        support_chain=CHAIN, decision_policy={"always_commit_mcq": True},
    )
    assert result.ok
    assert result.outputs["final_answer"] == "B"
    assert result.outputs["committed_unverified"] is True


def test_always_commit_marks_low_confidence() -> None:
    result = commit_answer(
        _unverified(), options=OPTIONS, answer_format="multiple_choice",
        support_chain=CHAIN, decision_policy={"always_commit_mcq": True},
    )
    assert result.outputs["confidence"] == 0.2


def test_always_commit_does_not_touch_verified_claims() -> None:
    verified = {"text": "the man is the murderer", "option_label": "B", "claim_status": "verified", "confidence": 0.9}
    result = commit_answer(
        verified, options=OPTIONS, answer_format="multiple_choice",
        support_chain={"evidence_refs": ["clip:1"], "items": []},
        decision_policy={"always_commit_mcq": True},
    )
    assert result.ok
    assert result.outputs["committed_unverified"] is False
    assert result.outputs["confidence"] == 0.9


def test_always_commit_is_multiple_choice_only() -> None:
    # Free text has no finite label set to guess over; the gate stays.
    result = commit_answer(
        _unverified(), options=None, answer_format="free_text",
        support_chain=CHAIN, decision_policy={"always_commit_mcq": True},
    )
    assert not result.ok
    assert result.failure_code == "claim_not_verified"


def test_always_commit_falls_back_to_lexical_option_match_without_a_label() -> None:
    claim = {"text": "he is the murderer", "claim_status": "insufficient"}
    result = commit_answer(
        claim, options=OPTIONS, answer_format="multiple_choice",
        support_chain=CHAIN, decision_policy={"always_commit_mcq": True},
    )
    assert result.ok
    assert result.outputs["final_answer"] == "B"

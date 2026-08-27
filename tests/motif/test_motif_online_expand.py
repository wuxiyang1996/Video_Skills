from __future__ import annotations

from motif import MotifBank, MotifLifecycleStatus, MotifQueryEngine, MotifRecord
from motif.online_expand import expand_motif_record, expand_skill_sequence_to_plan


def test_expand_skill_sequence_valid() -> None:
    result = expand_skill_sequence_to_plan(
        [
            "parse_question_target",
            "generate_answer_hypotheses",
            "retrieve_evidence_for_hypothesis",
            "score_hypothesis_support",
            "compare_hypotheses",
            "verify_claim_support",
            "commit_answer",
        ]
    )
    assert result.expansion_valid
    assert len(result.reasoning_plan) == 7
    assert result.reasoning_plan[0]["skill_id"] == "parse_question_target"
    assert result.reasoning_plan[-1]["skill_id"] == "commit_answer"
    assert result.reasoning_plan[1]["depends_on"] == ["m1"]
    # score must bind hypotheses from generate (m2), not from retrieve (m3)
    score_args = result.reasoning_plan[3]["args"]
    assert score_args["hypothesis"] == "$step.m2.hypotheses"
    assert score_args["support_evidence"] == "$step.m3"
    commit_args = result.reasoning_plan[6]["args"]
    assert commit_args["verified_claim"] == "$step.m6.verified_claim"


def test_expand_truncates_after_first_commit() -> None:
    result = expand_skill_sequence_to_plan(
        [
            "parse_question_target",
            "commit_answer",
            "retrieve_by_event",
            "commit_answer",
        ]
    )
    assert result.expansion_valid
    assert [s["skill_id"] for s in result.reasoning_plan] == [
        "parse_question_target",
        "commit_answer",
    ]


def test_expand_rejects_unknown_skill() -> None:
    result = expand_skill_sequence_to_plan(["parse_question_target", "not_a_real_skill"])
    assert not result.expansion_valid
    assert result.fallback_reason == "unknown_skill_id"


def test_expand_motif_record_and_query_select(tmp_path) -> None:
    record = MotifRecord(
        motif_id="motif_mcq_bridge",
        name="MCQ Bridge",
        description="Hypothesis compare then commit for multiple choice.",
        status=MotifLifecycleStatus.ACTIVE,
        trigger_signature={"task_family": "causal", "answer_format": "multiple_choice"},
        l2_template={
            "skill_sequence": [
                "parse_question_target",
                "generate_answer_hypotheses",
                "compare_hypotheses",
                "commit_answer",
            ],
            "compressed_skill_sequence": [
                "parse_question_target",
                "generate_answer_hypotheses",
                "compare_hypotheses",
                "commit_answer",
            ],
        },
    )
    bank = MotifBank([record])
    path = tmp_path / "motifs.jsonl"
    bank.save_jsonl(path)

    loaded = MotifBank.load_jsonl(path)
    expansion = expand_motif_record(loaded.require("motif_mcq_bridge"))
    assert expansion.expansion_valid
    assert expansion.skill_sequence[0] == "parse_question_target"

    selections = MotifQueryEngine(loaded).select(
        query="MCQ Bridge hypothesis compare commit multiple choice",
        task_family="causal",
        dataset="cg_bench",
        top_k=1,
    )
    assert selections
    assert selections[0].motif_id == "motif_mcq_bridge"

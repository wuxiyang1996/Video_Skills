from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_clip_wrapper.l2_reasoning_graph import reasoning_planner as rp
from motif.bank import MotifBank
from motif.dual_loop import (
    build_candidate_from_repaired_sequence,
    maybe_mine_candidate_after_verified,
    select_repair_motif,
)
from motif.retrieval import MotifQueryEngine
from motif.schemas import MotifLifecycleStatus
from trainer.verified_reward import score_verified_rollout


BANK = Path("motif/fixtures/dual_loop_motif_bank.jsonl")


class _EmptyPlanClient:
    def chat_json(self, messages, response_format=None):
        return {"reasoning_plan": [], "notes": "empty"}

    def chat(self, *args, **kwargs):
        return "{}"


def test_accelerate_pool_hides_shadow_and_candidate() -> None:
    bank = MotifBank.load_jsonl(BANK)
    engine = MotifQueryEngine(bank)
    accel = engine.select(query="causal multiple_choice", task_family="causal", phase="accelerate", top_k=5)
    ids = {item.motif_id for item in accel}
    assert "motif_accel_mcq" in ids
    assert "motif_repair_gap_retrieve" not in ids


def test_repair_pool_can_select_shadow() -> None:
    result = select_repair_motif(
        bank_path=BANK,
        question={"question_text": "Why did the cup move after the cut?"},
        task_family="causal",
        dataset="cg_bench",
        faults=[{"fault_type": "missing_evidence", "failure_code": "no_evidence_match", "skill_id": "retrieve_evidence_for_hypothesis"}],
        exclude_motif_ids=["motif_accel_mcq"],
        top_k=3,
    )
    assert result.meta_updates["repair_retrieval_attempted"] is True
    assert result.used_repair_motif is True
    assert result.selected_motif_id == "motif_repair_gap_retrieve"
    assert result.meta_updates["repair_expansion_valid"] is True
    assert "retrieve_by_event" in result.skill_sequence
    assert result.reasoning_plan


def test_mine_only_after_verified_repair_success(tmp_path: Path) -> None:
    seq = [
        "parse_question_target",
        "retrieve_by_event",
        "generate_answer_hypotheses",
        "verify_claim_support",
        "commit_answer",
    ]
    example = {
        "example_id": "e1",
        "dataset": "cg_bench",
        "task_family": "causal",
        "question": {"answer_format": "multiple_choice"},
    }
    denied = maybe_mine_candidate_after_verified(
        downstream_verified_success=False,
        repair_contributed=True,
        skill_sequence=seq,
        example=example,
    )
    assert denied.mined is False

    denied2 = maybe_mine_candidate_after_verified(
        downstream_verified_success=True,
        repair_contributed=False,
        skill_sequence=seq,
        example=example,
    )
    assert denied2.mined is False

    sink = tmp_path / "candidates.jsonl"
    mined = maybe_mine_candidate_after_verified(
        downstream_verified_success=True,
        repair_contributed=True,
        skill_sequence=seq,
        example=example,
        faults=[{"fault_type": "missing_evidence"}],
        repair_motif_id="motif_repair_gap_retrieve",
        candidate_sink_path=sink,
    )
    assert mined.mined is True
    assert mined.motif_id
    bank = MotifBank.load_jsonl(sink)
    record = bank.require(mined.motif_id)
    assert record.status == MotifLifecycleStatus.CANDIDATE
    assert record.l2_template["skill_sequence"] == seq


def test_candidate_not_auto_promoted_and_reward_ignores_lifecycle() -> None:
    record = build_candidate_from_repaired_sequence(
        skill_sequence=["parse_question_target", "commit_answer"],
        example={"example_id": "x", "dataset": "d", "task_family": "causal"},
        repair_motif_id="m",
    )
    assert record.status == MotifLifecycleStatus.CANDIDATE
    # GRPO-ready: motif lifecycle must not change lexicographic reward ordering.
    a = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        claim_support_hard_ok=True,
        tool_calls=1,
    )
    b = score_verified_rollout(
        answer_correct=False,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        claim_support_hard_ok=True,
        non_diagnostic_visual_ok=True,
        tool_calls=0,
    )
    assert a.rank_key > b.rank_key
    # Same outcomes stay equal; candidate_mined is not a reward input.
    a2 = score_verified_rollout(
        answer_correct=True,
        acceptance_status="accepted_strong",
        commit_evidence_ok=True,
        claim_support_hard_ok=True,
        tool_calls=1,
    )
    assert a.rank_key == a2.rank_key


def test_planner_repair_motif_phase_and_candidate_mine(tmp_path: Path, monkeypatch: Any) -> None:
    """Force first-plan failures, then succeed on repair_* steps and mine CANDIDATE."""

    def fake_execute(*, reasoning_plan, clue_memory_graph, question, skill_executor=None):
        del clue_memory_graph, question, skill_executor
        is_repair = any(
            str(step.get("step_id") or "").startswith("repair_")
            or step.get("from_repair_motif")
            for step in reasoning_plan
        )
        if not is_repair:
            return (
                [
                    {
                        "step_id": "r1",
                        "skill_id": "retrieve_evidence_for_hypothesis",
                        "ok": False,
                        "failure_code": "no_evidence_match",
                        "messages": ["forced fail"],
                    }
                ],
                {},
            )
        # Emit enough ok steps so fail_ratio after the forced first failure stays <= 0.2
        # (required for accepted_strong / dual-loop mine gate).
        ok_trace = []
        outputs: dict[str, Any] = {}
        for step in reasoning_plan:
            sid = str(step.get("step_id") or "s")
            skill = str(step.get("skill_id") or "parse_question_target")
            item = {
                "step_id": sid,
                "skill_id": skill,
                "ok": True,
                "evidence_refs": ["obs:1", "obs:2", "obs:3", "obs:4"],
                "confidence": 0.9,
            }
            ok_trace.append(item)
            if skill == "commit_answer":
                outputs[sid] = {
                    "final_answer": {"label": "A", "text": "x"},
                    "answer_support_chain": {
                        "evidence_refs": ["obs:1", "obs:2", "obs:3", "obs:4"]
                    },
                    "evidence_refs": ["obs:1", "obs:2", "obs:3", "obs:4"],
                    "confidence": 0.9,
                }
            else:
                outputs[sid] = {"evidence_refs": ["obs:1", "obs:2", "obs:3", "obs:4"]}
        return ok_trace, outputs

    monkeypatch.setattr(rp, "execute_reasoning_plan", fake_execute)
    monkeypatch.setattr(
        rp,
        "attempt_repair",
        lambda *a, **k: {"attempted": False, "repaired_count": 0},
        raising=False,
    )
    # attempt_repair is imported inside the function; patch fault_repair module.
    import dataset_clip_wrapper.l2_reasoning_graph.fault_repair as fr

    monkeypatch.setattr(
        fr,
        "attempt_repair",
        lambda *a, **k: {"attempted": False, "repaired_count": 0},
    )

    sink = tmp_path / "mined.jsonl"
    example = {
        "example_id": "itest:dual",
        "dataset": "cg_bench",
        "task_family": "causal",
        "available_inputs": {"mode": "video_only"},
        "question": {
            "question_text": "Why did the object change after the cut?",
            "options": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
            "answer_format": "multiple_choice",
        },
        "metadata": {
            "answerability_diagnostic": {},
            "motif_candidate_sink_path": str(sink),
        },
    }
    def _obs(i: int) -> dict:
        return {
            "node_id": f"obs:{i}",
            "node_type": "observation",
            "text": f"observation {i}",
            "clip_id": f"c{i}",
            "time_span": {"start_s": float(i), "end_s": float(i + 1)},
            "modality": "visual",
            "provenance": {"created_by": "test"},
            "layer": "clue_memory",
        }

    clue = {
        "schema_version": "video-skills/clue-memory-v0",
        "graph_id": "g1",
        "video_id": "v1",
        "layer": "clue_memory",
        "nodes": [_obs(i) for i in range(1, 5)],
        "edges": [],
    }
    rollout = rp.build_llm_reasoning_rollout(
        example,
        clue,
        client=_EmptyPlanClient(),
        skill_executor=None,
        motif_enabled=True,
        motif_bank_path=str(BANK),
        forced_motif_id="motif_accel_mcq",
        motif_candidate_sink_path=sink,
    )
    motif = rollout["metadata"]["motif_online"]
    assert motif["motif_retrieval_attempted"] is True
    assert motif["repair_retrieval_attempted"] is True
    assert motif["repair_selected_motif_id"] == "motif_repair_gap_retrieve"
    assert motif["repair_expansion_valid"] is True
    assert motif["motif_phase"] == "repair"
    assert motif.get("repair_selected_motif_id")
    assert motif["downstream_verified_success"] is True, (
        rollout.get("acceptance_status"),
        (rollout.get("metadata") or {}).get("runtime_verifier"),
    )
    assert motif["candidate_mined"] is True
    assert sink.exists()
    mined_bank = MotifBank.load_jsonl(sink)
    assert any(r.status == MotifLifecycleStatus.CANDIDATE for r in mined_bank.records)

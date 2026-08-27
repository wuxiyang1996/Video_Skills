from __future__ import annotations

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout


class _BoomClient:
    def chat_json(self, *args, **kwargs):
        raise AssertionError("LLM planner must not run when motif expansion succeeds")

    def chat(self, *args, **kwargs):
        raise AssertionError("LLM planner must not run when motif expansion succeeds")


class _EmptyPlanClient:
    def chat_json(self, messages, response_format=None):
        return {"reasoning_plan": [], "notes": "empty"}

    def chat(self, *args, **kwargs):
        return "{}"


def _example() -> dict:
    return {
        "example_id": "itest:1",
        "dataset": "cg_bench",
        "task_family": "causal",
        "available_inputs": {"mode": "video_only"},
        "question": {
            "question_text": "Why did the object change after the cut?",
            "options": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
            "answer_format": "multiple_choice",
            "answer": "A",
        },
        "metadata": {"answerability_diagnostic": {}},
    }


def _clue() -> dict:
    return {
        "schema_version": "video-skills/clue-memory-v0",
        "graph_id": "g1",
        "video_id": "v1",
        "nodes": [
            {
                "node_id": "obs:1",
                "node_type": "observation",
                "text": "a cup moves",
                "clip_id": "c1",
                "time_span": {"start_s": 0, "end_s": 2},
                "modality": "visual",
                "provenance": {"created_by": "test"},
            }
        ],
        "edges": [],
    }


def test_motif_expand_skips_llm_planner() -> None:
    rollout = build_llm_reasoning_rollout(
        _example(),
        _clue(),
        client=_BoomClient(),
        skill_executor=None,
        motif_enabled=True,
        motif_bank_path="motif/fixtures/demo_motif_bank.jsonl",
        forced_motif_id="motif_mcq_bridge",
    )
    motif = rollout["metadata"]["motif_online"]
    assert motif["motif_retrieval_attempted"] is True
    assert motif["expansion_valid"] is True
    assert rollout["metadata"]["llm_plan"]["planner"] == "motif_expanded_skill_sequence"


def test_missing_bank_falls_back_without_abort() -> None:
    rollout = build_llm_reasoning_rollout(
        _example(),
        _clue(),
        client=_EmptyPlanClient(),
        skill_executor=None,
        motif_enabled=True,
        motif_bank_path="/tmp/missing_motif_bank.jsonl",
    )
    motif = rollout["metadata"]["motif_online"]
    assert motif["motif_retrieval_attempted"] is True
    assert motif["fallback_reason"] == "motif_bank_missing"
    assert motif["expansion_valid"] is False

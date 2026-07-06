#!/usr/bin/env python3
"""Smoke test for compact expert-demo -> ReasoningTrace/SFT export."""

from __future__ import annotations

import json

from dataset_clip_wrapper.training.trace_adapter import build_training_exports


def _demo(status: str, *, training: bool, abstain: bool) -> dict:
    return {
        "demo_id": f"expert_demo:toy:1:{status}",
        "demo_type": "direct_strong" if training else "abstain_needs_more_evidence",
        "dataset": "toy",
        "example_id": "toy:1",
        "video_regime": "short",
        "task_family": "visual_qa",
        "visible_demo_inputs": {
            "mode": "video_only",
            "video": {"video_id": "v1", "path": "video.mp4"},
            "question": {
                "question_text": "What color is the cup?",
                "options": [{"label": "A", "text": "red"}, {"label": "B", "text": "blue"}],
            },
        },
        "hidden_supervision": {"sources": ["official_answer"], "available_for_inference": False},
        "l1": {
            "graph_id": "g1",
            "training_view": "compact",
            "compact_evidence_nodes": [
                {
                    "ref": "n1",
                    "role": "used_by_l2_or_repair",
                    "node_type": "observation",
                    "time_span": {"start_s": 1.0, "end_s": 2.0},
                    "text": "A red cup is visible on the table.",
                }
            ],
        },
        "l2": {
            "final_acceptance_status": status,
            "final_repair_applied": False,
            "verifier_reason": "strict evidence check",
            "l2_status": {
                "final_answer": {"label": "A", "text": "red"},
                "support_refs": ["n1"],
                "support_ref_count": 1,
            },
            "trajectory": {
                "rounds": [
                    {
                        "round_type": "initial_l2_reasoning",
                        "goal": "Identify the cup color.",
                        "observation_summary": {"support_refs": ["n1"]},
                    }
                ]
            },
        },
        "quality_flags": {
            "training_candidate": training,
            "abstain_candidate": abstain,
            "no_gold_keys_in_visible_inputs": True,
        },
    }


def test_trace_adapter_exports() -> None:
    demos = [
        _demo("accepted_strong", training=True, abstain=False),
        _demo("needs_more_evidence", training=False, abstain=True),
    ]
    traces, chats, summary = build_training_exports(demos)
    assert summary["exported_traces"] == 2
    assert summary["accepted_traces"] == 1
    assert summary["abstain_traces"] == 1
    assert traces[0]["final_verification"]["passed"] is True
    assert traces[1]["abstain"]["abstain"] is True
    assert traces[0]["final_evidence"]["refs"][0]["ref_id"] == "n1"
    prompt = chats[0]["messages"][1]["content"]
    assert "hidden_supervision" not in prompt
    assert "official_answer" not in prompt
    json.loads(chats[0]["messages"][2]["content"])


if __name__ == "__main__":
    test_trace_adapter_exports()
    print("trace adapter smoke test passed")

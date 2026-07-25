import json

from dataset_clip_wrapper.training.l2_specialist_sft_adapter import _positive_expansions
from dataset_clip_wrapper.training.motif_evidence_sft_adapter import build_motif_evidence_exports
from scripts.sft_pilot.build_specialist_sft_v3 import (
    _apply_l1_skill_weights,
    _l1_quality_reason,
    _sanitize_repair_row,
)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_l2_positive_trace_expands_to_set_atomic_rank_and_stop():
    source = {
        "transition_id": "cg_bench:1::l2_retrieval::0",
        "state_t": {
            "dataset": "cg_bench",
            "example_id": "cg_bench:1",
            "question": {"question_text": "find the red car"},
            "l1_coarse_summary_catalog": [
                {"coarse_index": 0, "scene_description": "blue car " * 30, "observable_facts": ["blue"] * 3},
                {"coarse_index": 1, "scene_description": "red car " * 30, "observable_facts": ["red"] * 3},
                {"coarse_index": 2, "scene_description": "red truck"},
            ],
        },
        "action_t": {
            "arguments": {"selected_coarse_indices": [1], "rationale_short": "visible red car"}
        },
    }

    rows = _positive_expansions([source], hard_negatives_per_selected=2)

    assert len(rows) == 12
    assert {row["task"] for row in rows} == {
        "select_coarse_set",
        "select_next_coarse_clip",
        "rank_coarse_candidates",
        "rank_coarse_candidates_listwise",
        "decide_retrieval_stop",
    }
    assert sum(row["task"] == "rank_coarse_candidates" for row in rows) == 2
    assert sum(row["task"] == "select_coarse_set" for row in rows) == 4
    assert sum(row["task"] == "select_next_coarse_clip" for row in rows) == 3
    assert sum(row["task"] == "decide_retrieval_stop" for row in rows) == 2
    assert "remaining_teacher_selection_count" not in json.dumps(rows)
    assert all("answer" not in json.dumps(row["state_t"]).lower() for row in rows)
    catalogs = [
        candidate
        for row in rows
        for candidate in (
            row["state_t"].get("l1_coarse_summary_catalog")
            or row["state_t"].get("candidate_coarse_summaries")
            or []
        )
    ]
    assert all(len(candidate["scene_description"]) <= 80 for candidate in catalogs)
    assert all(len(candidate["observable_facts"]) <= 1 for candidate in catalogs)


def test_motif_audit_resolves_nodes_and_exposes_gate_observations(tmp_path):
    bank = tmp_path / "bank.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    _write_jsonl(bank, [{
        "motif_id": "motif:1",
        "status": "candidate",
        "evidence_refs": [
            {
                "dataset": "cg_bench",
                "example_id": "cg_bench:1",
                "l1_node_ids": ["node:1"],
                "l2_node_ids": [],
                "verifier_passed": True,
                "evidence_valid": True,
                "no_hidden_leakage": True,
            },
            {
                "dataset": "cg_bench",
                "example_id": "cg_bench:1",
                "l1_node_ids": ["node:1"],
                "l2_node_ids": [],
                "verifier_passed": False,
                "evidence_valid": False,
                "no_hidden_leakage": True,
            },
        ],
    }])
    _write_jsonl(rollouts, [{
        "dataset": "cg_bench",
        "example_id": "cg_bench:1",
        "metadata": {
            "clue_memory_graph": {
                "nodes": [{"node_id": "node:1", "node_type": "observation", "text": "A red car is visible."}]
            },
            "reasoning_rollout": {"nodes": []},
        },
    }])

    transitions, chats, report = build_motif_evidence_exports(bank, [rollouts], max_refs_per_motif=4)

    assert len(transitions) == len(chats) == 2
    assert report["verdict_counts"] == {"accept_ref": 1, "reject_ref": 1}
    prompts = [json.loads(chat["messages"][1]["content"])["state_t"] for chat in chats]
    serialized_refs = json.dumps([prompt["evidence_ref"] for prompt in prompts])
    assert "verifier_passed" not in serialized_refs
    assert "evidence_valid" not in serialized_refs
    assert "no_hidden_leakage" not in serialized_refs
    assert prompts[0]["audit_observations"] != prompts[1]["audit_observations"]
    assert all(set(prompt["audit_observations"]) == {
        "runtime_verifier_passed", "evidence_validation_passed", "leakage_scan_passed"
    } for prompt in prompts)
    assert "A red car is visible." in json.dumps(prompts)


def test_repair_sanitizer_removes_precomputed_decision_and_paths():
    user = {
        "task": "choose_next_controller_action",
        "state_t": {
            "visible_demo_inputs": {
                "repair_mode": "reroute",
                "strategy": "copy_me",
                "selection_mode": "exploratory_probe",
                "video": {
                    "video_id": "v1",
                    "primary_path": "/fs/private/video.mp4",
                    "derived_clips": [{"path": "/fs/private/clip.mp4"}],
                },
            },
            "l1_compact": {"compact_evidence_nodes": [{"node_id": str(i)} for i in range(30)]},
        },
    }
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(user)},
            {"role": "assistant", "content": json.dumps({"action": {"repair_mode": "reroute"}})},
        ]
    }

    sanitized = _sanitize_repair_row(row)
    state = json.loads(sanitized["messages"][1]["content"])["state_t"]
    assert "repair_mode" not in state["visible_demo_inputs"]
    assert "selection_mode" not in state["visible_demo_inputs"]
    assert "/fs/" not in sanitized["messages"][1]["content"]
    assert len(state["l1_compact"]["compact_evidence_nodes"]) == 20


def test_l1_quality_gate_rejects_invisible_or_malformed_endpoints():
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps({"state_t": {"visible": "node:a"}})},
            {"role": "assistant", "content": json.dumps({
                "tool_name": "neighbor_vlm_l1_create_edge",
                "arguments": {"edge": {"src": "node:a", "dst": "node:hidden"}},
            })},
        ]
    }

    assert _l1_quality_reason(row) == "l1_invisible_edge_endpoint"


def test_l1_skill_weights_keep_all_rows_and_balance_family_loss():
    rows = [
        {"metadata": {"skill_id": "create_node"}},
        {"metadata": {"skill_id": "create_node"}},
        {"metadata": {"controller": "l1_patch"}},
    ]

    counts = _apply_l1_skill_weights(rows)

    assert counts == {"create_node": 2, "l1_patch": 1}
    assert len(rows) == 3
    assert [row["metadata"]["source_family_weight"] for row in rows] == [0.5, 0.5, 1.0]
    assert [row["metadata"]["task"] for row in rows] == ["create_node", "create_node", "l1_patch"]

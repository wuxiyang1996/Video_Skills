import json

from dataset_clip_wrapper.training.motif_sft_adapter import build_motif_exports
from dataset_clip_wrapper.training.l2_retrieval_sft_adapter import build_l2_retrieval_exports
from dataset_clip_wrapper.training.verifier_sft_adapter import build_verifier_exports
from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import execute_reasoning_plan
from dataset_clip_wrapper.training.l1_builder_sft_adapter import _skill_balanced_cap
from dataset_clip_wrapper.verification.runtime_verifier import verify_rollout


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_accepted_weak_is_not_positive_verifier_supervision(tmp_path):
    demos = tmp_path / "expert_demos.jsonl"
    _write_jsonl(demos, [{
        "demo_id": "demo:weak",
        "dataset": "cg_bench",
        "example_id": "cg_bench:1",
        "l1": {"compact_evidence_nodes": []},
        "l2": {"trajectory": {"rounds": [{
            "verifier_signal": {
                "status": "accepted_weak",
                "verified_evidence_pack": {
                    "claim_text": "candidate",
                    "final_label": "A",
                    "support_refs": ["obs:1"],
                },
            },
        }]}},
    }])

    transitions, _, report = build_verifier_exports(None, demos)

    arguments = transitions[0]["action_t"]["arguments"]
    assert arguments["decision"] == "insufficient"
    assert arguments["failure_code"] == "weak_evidence_not_positive"
    assert report["decision_counts"] == {"insufficient": 1}


def test_motif_export_applies_evidence_gates(tmp_path):
    bank = tmp_path / "motifs.jsonl"

    def motif(motif_id, status, refs):
        return {
            "motif_id": motif_id,
            "status": status,
            "motif_type": "l1_evidence_profile",
            "support": {"support_count": len(refs)},
            "evidence_refs": refs,
        }

    valid = {"verifier_passed": True, "evidence_valid": True, "no_hidden_leakage": True}
    leaked = {"verifier_passed": True, "evidence_valid": True, "no_hidden_leakage": False}
    _write_jsonl(bank, [
        motif("candidate", "candidate", [valid, valid]),
        motif("rejected", "candidate", [valid, leaked]),
        motif("shadow", "shadow", [valid]),
    ])

    transitions, _, report = build_motif_exports(bank)

    statuses = [row["action_t"]["arguments"]["status"] for row in transitions]
    assert statuses == ["candidate", "rejected", "shadow"]
    assert report["status_counts"] == {"candidate": 1, "rejected": 1, "shadow": 1}
    assert report["gate_failure_counts"]["hidden_leakage_or_unknown"] == 1


def test_l2_retrieval_export_hides_gold_and_requires_strong_or_resolved_correct_final(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    base = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:1",
        "question": {
            "question_text": "What color is the vehicle?",
            "options": [{"label": "A", "text": "White"}],
            "answer": {"label": "A", "text": "White"},
        },
        "metadata": {
            "coarse_clip_schemas": [{"time_span": {"start_s": 0, "end_s": 30}, "scene_description": "White van"}],
            "clip_schemas": [{"clip_id": "fine:0"}],
            "perception": {"retrieval": {
                "ok": True,
                "mode": "gpt_oss_atomic_select_coarse",
                "selected_coarse_indices": [0],
                "topk": 2,
            }},
            "reasoning_rollout": {
                "acceptance_status": "accepted_strong",
                "final_answer": {"label": "A"},
            },
        },
    }
    weak = json.loads(json.dumps(base))
    weak["example_id"] = "cg_bench:2"
    weak["metadata"]["reasoning_rollout"]["acceptance_status"] = "accepted_weak"
    weak_resolved = json.loads(json.dumps(weak))
    weak_resolved["example_id"] = "cg_bench:3"
    _write_jsonl(rollouts, [base, weak, weak_resolved])
    repairs = tmp_path / "repairs.jsonl"
    _write_jsonl(repairs, [{
        "example_id": "cg_bench:1",
        "repair_status": "resolved_strong",
        "repair_needed_after_round": False,
    }, {
        "example_id": "cg_bench:3",
        "repair_status": "resolved_strong",
    }])

    transitions, chats, report = build_l2_retrieval_exports([rollouts], repair_results_paths=[repairs])

    assert len(transitions) == len(chats) == 2
    assert "answer" not in transitions[0]["state_t"]["question"]
    assert transitions[1]["reward_proxy_t"]["initial_accepted_strong"] == 0.0
    assert transitions[1]["reward_proxy_t"]["downstream_resolved_strong"] == 1.0
    assert report["prompt_forbidden_key_hits"] == 0
    assert report["excluded_counts"] == {"final_not_strong_or_resolved": 1}


def test_l2_retrieval_catalog_is_bounded_but_keeps_all_indices(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    schemas = [{
        "time_span": {"start_s": index * 30, "end_s": (index + 1) * 30},
        "scene_description": "scene " + ("x" * 1000),
        "observable_facts": [{"text": "fact " + ("y" * 1000)} for _ in range(10)],
        "events": [{"text": "event " + ("z" * 1000)} for _ in range(10)],
        "searchable_phrases": ["phrase " + ("p" * 1000) for _ in range(10)],
    } for index in range(100)]
    _write_jsonl(rollouts, [{
        "dataset": "cg_bench",
        "example_id": "cg_bench:large",
        "question": {"question_text": "What happens?", "answer": {"label": "A"}},
        "metadata": {
            "coarse_clip_schemas": schemas,
            "clip_schemas": [{"clip_id": "fine:0"}],
            "perception": {"retrieval": {
                "ok": True,
                "mode": "gpt_oss_atomic_select_coarse",
                "selected_coarse_indices": [2, 90],
                "topk": 2,
            }},
            "reasoning_rollout": {
                "acceptance_status": "accepted_strong",
                "final_answer": {"label": "A"},
            },
        },
    }])

    repairs = tmp_path / "repairs.jsonl"
    _write_jsonl(repairs, [{"example_id": "cg_bench:large", "repair_status": "resolved_strong"}])
    transitions, chats, _ = build_l2_retrieval_exports([rollouts], repair_results_paths=[repairs])
    catalog = transitions[0]["state_t"]["l1_coarse_summary_catalog"]

    assert len(catalog) == 100
    assert {row["coarse_index"] for row in catalog} == set(range(100))
    assert len(chats[0]["messages"][1]["content"]) < 64000
    assert len(catalog[2]["observable_facts"][0]) > len(catalog[3]["observable_facts"][0])


def test_missing_chain_evidence_never_injects_unknown_ref():
    trace, _ = execute_reasoning_plan(
        reasoning_plan=[{
            "step_id": "chain",
            "skill_id": "compose_evidence_chain",
            "args": {"role_labeled_evidence": []},
            "depends_on": [],
        }],
        clue_memory_graph={"schema_version": "test", "nodes": [], "edges": []},
        question={"question_text": "What happened?"},
    )

    assert trace[0]["ok"] is False
    assert trace[0]["failure_code"] == "missing_role_labeled_evidence"
    assert trace[0]["evidence_refs"] == []


def test_verifier_catalog_resolves_original_l1_refs_from_repair_plan(tmp_path):
    source = tmp_path / "examples.jsonl"
    _write_jsonl(source, [{
        "example_id": "cg_bench:1",
        "metadata": {"clue_memory_graph": {"nodes": [{
            "node_id": "evidence:1",
            "node_type": "observation",
            "text": "A white vehicle enters the frame.",
        }]}},
    }])
    stage = tmp_path / "stages" / "cg_bench_1"
    stage.mkdir(parents=True)
    (stage / "repair_01_plan.json").write_text(json.dumps({"source_path": str(source)}), encoding="utf-8")
    (stage / "repair_03_l1_patch.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    (stage / "repair_04_l2_verifier.json").write_text(json.dumps({
        "dataset": "cg_bench",
        "example_id": "cg_bench:1",
        "option_verifications": [{
            "option_label": "A",
            "option_text": "White",
            "positive_refs": ["evidence:1"],
            "negative_refs": [],
            "verifier_decision": "supported",
            "ok": True,
        }],
    }), encoding="utf-8")

    transitions, _, report = build_verifier_exports(tmp_path / "stages")

    catalog = transitions[0]["state_t"]["proposed_evidence_pack"]["evidence_catalog"]
    assert catalog[0]["text"] == "A white vehicle enters the frame."
    assert report["skipped_unresolved_evidence_refs"] == 0


def test_l1_per_example_cap_round_robins_skills():
    rows = [
        {"action_t": {"tool_name": skill}, "transition_id": f"{skill}:{index}"}
        for skill in ("node", "edge")
        for index in range(5)
    ]

    capped = _skill_balanced_cap(rows, 4)

    assert [row["action_t"]["tool_name"] for row in capped] == ["edge", "node", "edge", "node"]


def test_runtime_verifier_flattens_nested_model_refs():
    clue_graph = {
        "graph_id": "g1",
        "layer": "clue_memory",
        "nodes": [{"node_id": "obs:1", "source_type": "vlm_clip"}],
    }
    rollout = {
        "rollout_id": "r1",
        "layer": "reasoning",
        "clue_memory_ref": {"graph_id": "g1"},
        "nodes": [{"node_id": "n1", "evidence_refs": [["obs:1"]]}],
        "claims": [{"claim_id": "c1", "claim_status": "verified", "supported_by_refs": [{"node_id": "obs:1"}]}],
        "answer_support_chain": [{"evidence_refs": [{"evidence_ref": "obs:1"}]}],
    }

    result = verify_rollout(clue_graph, rollout)

    assert result["verifier_summary"]["evidence_refs_exist"] is True
    assert result["failure_reasons"] == []

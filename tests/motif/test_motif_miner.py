from pathlib import Path

from dataset_clip_wrapper.motifs.miner import mine_paths


def test_mine_paths_extracts_l1_l2_motifs(tmp_path: Path) -> None:
    input_path = tmp_path / "l1_l2.jsonl"
    row = {
        "dataset": "video_holmes",
        "example_id": "vh_1",
        "task_family": "temporal_comparison",
        "metadata": {
            "video_regime": "short",
            "clue_memory_graph": {
                "nodes": [
                    {"node_id": "evidence.observation:1", "node_type": "observation", "modality": "visual"},
                    {"node_id": "evidence.event:1", "node_type": "event", "modality": "visual"},
                ],
                "edges": [{"edge_type": "temporal_next"}],
            },
            "reasoning_rollout": {
                "question": {"answer_format": "multiple_choice"},
                "nodes": [
                    {"node_id": "skill:1", "skill_id": "parse_question_target", "evidence_refs": []},
                    {"node_id": "skill:2", "skill_id": "generate_answer_hypotheses", "evidence_refs": []},
                    {"node_id": "skill:3", "skill_id": "retrieve_evidence_for_hypothesis", "evidence_refs": ["e1"]},
                    {"node_id": "skill:4", "skill_id": "retrieve_evidence_for_hypothesis", "evidence_refs": ["e2"]},
                    {"node_id": "skill:5", "skill_id": "score_hypothesis_support", "evidence_refs": ["e1"]},
                    {"node_id": "skill:6", "skill_id": "score_hypothesis_support", "evidence_refs": ["e2"]},
                    {"node_id": "skill:7", "skill_id": "compare_hypotheses", "evidence_refs": ["e1", "e2"]},
                    {"node_id": "skill:8", "skill_id": "verify_claim_support", "evidence_refs": ["e1"]},
                    {"node_id": "skill:9", "skill_id": "commit_answer", "evidence_refs": ["e1"]},
                ],
                "edges": [{"edge_type": "data"}],
                "claims": [{"supported_by_refs": ["e1"]}],
                "final_answer": {"label": "A"},
                "verifier_summary": {
                    "no_hidden_supervision_leakage": True,
                    "no_old_video_fact_leakage": True,
                },
                "acceptance_status": "accepted_weak",
            },
        },
    }
    input_path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")

    result = mine_paths([input_path], min_support=1)

    assert result.input_rows == 1
    assert result.rows_with_l1 == 1
    assert result.rows_with_l2 == 1
    assert result.bank.summary()["motif_count"] >= 4
    assert "l2_hypothesis_fanout" in result.motif_type_counts

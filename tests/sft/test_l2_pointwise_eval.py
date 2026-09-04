import hashlib
import json

from dataset_clip_wrapper.training.evaluate_l2_pointwise_adapter import (
    evaluation_input_provenance,
    rank_results,
)


def test_rank_results_uses_scores_and_reports_visual_baseline() -> None:
    rows = [
        {"example_id": "a", "candidate_index": 1, "retrieval_rank": 1, "gold_relevant": False, "score": -2.0},
        {"example_id": "a", "candidate_index": 2, "retrieval_rank": 3, "gold_relevant": True, "score": 3.0},
        {"example_id": "a", "candidate_index": 3, "retrieval_rank": 2, "gold_relevant": False, "score": 1.0},
    ]
    results, metrics = rank_results(rows)
    assert results[0]["predicted"] == [2, 3]
    assert results[0]["retrieval_predicted"] == [1, 3]
    assert metrics["pointwise_top2"]["hit_rate"] == 1.0
    assert metrics["visual_retrieval_top2"]["hit_rate"] == 0.0


def test_rank_results_preserves_gold_outside_candidate_pool() -> None:
    rows = [
        {"example_id": "a", "candidate_index": 1, "retrieval_rank": 1, "gold_relevant": False, "gold_indices": [9], "score": 1.0},
        {"example_id": "a", "candidate_index": 2, "retrieval_rank": 2, "gold_relevant": False, "gold_indices": [9], "score": 0.0},
    ]
    results, metrics = rank_results(rows)
    assert results[0]["gold"] == [9]
    assert metrics["pointwise_top2"]["mean_recall"] == 0.0


def test_pointwise_report_records_frozen_input_provenance(tmp_path) -> None:
    rows = [{
        "schema_version": "video-skills/l2-dataset-eval-chat-v0.2",
        "metadata": {
            "dataset": "video_holmes",
            "split_role": "heldout_test",
            "source_example_id": "video_holmes:test:q1",
        },
    }]
    path = tmp_path / "heldout.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    report = evaluation_input_provenance(rows, path)
    assert report["evaluation_jsonl_sha256"] == hashlib.sha256(payload.encode()).hexdigest()
    assert report["input_split_roles"] == ["heldout_test"]
    assert report["input_datasets"] == ["video_holmes"]
    assert report["input_examples"] == 1

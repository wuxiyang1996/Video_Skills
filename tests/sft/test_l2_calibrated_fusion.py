import numpy as np

from dataset_clip_wrapper.training.evaluate_l2_calibrated_fusion import joined_rows, rank_metrics


def test_join_and_rank_fusion_rows() -> None:
    pointwise = {"results": [{"example_id": "x", "ranking": [
        {"candidate_index": 1, "score": -2.0, "retrieval_rank": 1},
        {"candidate_index": 2, "score": 3.0, "retrieval_rank": 2},
    ]}]}
    visual = {"results": [{"example_id": "x", "gold": [2], "ranking": [
        {"candidate_index": 1, "score": 0.9},
        {"candidate_index": 2, "score": 0.8},
    ]}]}
    rows = joined_rows(pointwise, visual)
    assert [row["gold"] for row in rows] == [False, True]
    results, metrics = rank_metrics(rows, np.asarray([0.1, 0.9]))
    assert results[0]["predicted"][0] == 2
    assert metrics["hit_rate"] == 1.0


def test_rank_fusion_preserves_gold_outside_candidate_pool() -> None:
    pointwise = {"results": [{"example_id": "x", "ranking": [
        {"candidate_index": 1, "score": 0.0, "retrieval_rank": 1},
    ]}]}
    visual = {"results": [{"example_id": "x", "gold": [9], "ranking": [
        {"candidate_index": 1, "score": 0.5},
    ]}]}
    rows = joined_rows(pointwise, visual)
    results, metrics = rank_metrics(rows, np.asarray([1.0]))
    assert results[0]["gold"] == [9]
    assert metrics["mean_recall"] == 0.0

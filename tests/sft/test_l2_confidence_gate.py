from dataset_clip_wrapper.training.evaluate_l2_confidence_gate import apply_gate


def test_visual_route_requires_margin_threshold() -> None:
    pointwise = {"results": [{"example_id": "x", "gold": [2], "predicted": [2, 3]}]}
    visual = {"results": [{
        "example_id": "x", "gold": [2], "predicted": [1, 4],
        "ranking": [{"candidate_index": 1, "score": 0.8}, {"candidate_index": 4, "score": 0.7}],
    }]}
    rows, metrics = apply_gate(pointwise, visual, 0.2)
    assert rows[0]["route"] == "pointwise"
    assert metrics["hit_rate"] == 1.0
    rows, metrics = apply_gate(pointwise, visual, 0.05)
    assert rows[0]["route"] == "visual"
    assert metrics["hit_rate"] == 0.0

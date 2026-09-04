from dataset_clip_wrapper.training.evaluate_l2_retrieval_adapter import retrieval_scores, selected_indices


def test_selected_indices_and_retrieval_scores() -> None:
    payload = {"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": [2, 2, 3]}}
    assert selected_indices(payload) == [2, 3]
    scores = retrieval_scores([2, 3], [1, 2])
    assert scores == {"precision": 0.5, "recall": 0.5, "hit": True, "exact": False}

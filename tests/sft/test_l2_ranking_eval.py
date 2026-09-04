from dataset_clip_wrapper.training.evaluate_l2_ranking_adapter import chosen_index


def test_chosen_index_requires_ranking_tool_and_integer_index() -> None:
    assert chosen_index({
        "tool_name": "choose_better_coarse_candidate", "arguments": {"coarse_index": "7"}
    }) == 7
    assert chosen_index({"tool_name": "select_coarse_clips", "arguments": {"coarse_index": 7}}) is None
    assert chosen_index({"tool_name": "choose_best_coarse_candidate", "arguments": {}}) is None

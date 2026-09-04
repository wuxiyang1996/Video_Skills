from dataset_clip_wrapper.training.train_lora_sft import _retrieval_argument_match, _select_generation_rows


def _row(task: str, index: int, *, core: bool = False) -> dict:
    return {
        "transition_id": f"{task}:{index}",
        "metadata": {"task": task, "is_core": core},
        "messages": [{"role": "assistant", "content": "{}"}],
    }


def test_generation_sampling_round_robins_tasks_and_prioritizes_core() -> None:
    rows = [_row("a", index, core=index == 2) for index in range(4)]
    rows += [_row("b", index) for index in range(4)]
    selected = _select_generation_rows(rows, 6)
    assert [row["metadata"]["task"] for row in selected] == ["a", "b", "a", "b", "a", "b"]
    assert selected[0]["transition_id"] == "a:2"


def test_retrieval_argument_match_checks_indices_not_only_tool_name() -> None:
    gold = {"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": [2, 3]}}
    assert _retrieval_argument_match(
        {"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": [3, 2]}}, gold
    ) is True
    assert _retrieval_argument_match(
        {"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": [4]}}, gold
    ) is False

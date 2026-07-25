import json

from dataset_clip_wrapper.training.train_lora_sft import (
    _extract_json_object,
    _representative_subset,
    _row_task,
    _row_weight,
)


def _row(record_id, controller, size):
    return {
        "transition_id": record_id,
        "controller": controller,
        "messages": [{"content": "x" * size}],
    }


def test_representative_subset_keeps_longest_row_per_controller():
    rows = [
        _row("a-short", "a", 1),
        _row("a-long", "a", 100),
        _row("b-short", "b", 2),
        _row("b-long", "b", 200),
    ]

    selected = _representative_subset(rows, 2, seed=42)

    assert {row["transition_id"] for row in selected} == {"a-long", "b-long"}


def test_extract_json_object_handles_reasoning_wrapper():
    payload = _extract_json_object("thinking...\n" + json.dumps({"action": "ok"}) + "\nfinished")

    assert payload == {"action": "ok"}


def test_task_and_weight_are_read_from_chat_metadata():
    row = {
        "metadata": {"controller": "l2_controller", "task": "rank_candidates", "source_family_weight": 0.25}
    }

    assert _row_task(row) == "rank_candidates"
    assert _row_weight(row) == 0.25


def test_representative_subset_stratifies_metadata_tasks():
    rows = [
        {**_row("rank", None, 10), "metadata": {"task": "rank"}},
        {**_row("recover", None, 10), "metadata": {"task": "recover"}},
    ]

    assert len(_representative_subset(rows, 2, seed=42)) == 2

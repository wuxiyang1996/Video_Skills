from __future__ import annotations

import json

import pytest

from dataset_clip_wrapper.training.sft_common import contains_forbidden_prompt_key
from trainer.build_l2_dataset_dev_eval import build_dataset_dev_rows


def _entry(index: int, start: float, text: str) -> dict:
    return {
        "clip_id": f"c{index}",
        "time_span": {"start_s": start, "end_s": start + 4},
        "scene_description": text,
        "observable_facts": [],
        "events": [],
    }


def test_dev_builder_keeps_labels_evaluator_side() -> None:
    example = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:1",
        "question": {"question_id": "1", "question_text": "what happened?", "answer": {"label": "A"}},
        "metadata": {"coarse_clip_schemas": [
            _entry(0, 0, "irrelevant"), _entry(1, 10, "clue"), _entry(2, 20, "other")
        ]},
    }
    rows, report = build_dataset_dev_rows(
        [example], {"cg_bench:1": {"clue_spans": [{"start_s": 11, "end_s": 12}]}}
    )
    assert report["source_examples"] == {"cg_bench": 1}
    assert len(rows) == 3
    assert rows[0]["metadata"]["gold_indices"] == [1]
    for row in rows:
        prompt = json.loads(row["messages"][1]["content"])
        assert not contains_forbidden_prompt_key(prompt)
        assert "process_supervision" not in row["messages"][1]["content"]


def test_dev_builder_keeps_hard_example_when_gold_is_outside_prefix() -> None:
    example = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:2",
        "question": {"question_id": "2", "question_text": "what happened?"},
        "metadata": {"coarse_clip_schemas": [
            _entry(0, 0, "irrelevant"),
            _entry(1, 10, "irrelevant"),
            _entry(2, 20, "clue"),
        ]},
    }
    rows, report = build_dataset_dev_rows(
        [example],
        {"cg_bench:2": {"clue_spans": [{"start_s": 21, "end_s": 22}]}},
        max_candidates=2,
    )
    assert report["source_examples"] == {"cg_bench": 1}
    assert len(rows) == 2
    assert rows[0]["metadata"]["gold_indices"] == [2]
    assert rows[0]["metadata"]["gold_in_visible_prefix"] is False
    assert all(row["metadata"]["candidate_relevant"] is False for row in rows)


def test_dev_builder_uses_fixed_candidate_manifest_order_without_labels() -> None:
    example = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:3",
        "question": {"question_id": "3", "question_text": "what happened?"},
        "metadata": {"coarse_clip_schemas": [
            _entry(0, 0, "first"), _entry(1, 10, "clue"), _entry(2, 20, "third")
        ]},
    }
    rows, report = build_dataset_dev_rows(
        [example],
        {"cg_bench:3": {"clue_spans": [{"start_s": 11, "end_s": 12}]}},
        candidate_indices_by_example={"cg_bench:3": [2, 0]},
    )
    assert [row["metadata"]["candidate_index"] for row in rows] == [2, 0]
    assert [row["metadata"]["retrieval_rank"] for row in rows] == [1, 2]
    assert rows[0]["metadata"]["gold_indices"] == [1]
    assert report["candidate_selection_mode"] == "fixed_candidate_manifest"


def test_dev_builder_can_freeze_label_independent_prompt_payload() -> None:
    example = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:4",
        "question": {"question_id": "4", "question_text": "what happened?"},
        "metadata": {"coarse_clip_schemas": [_entry(0, 0, "irrelevant"), _entry(1, 10, "clue")]},
    }
    frozen_user = {
        "task": "score_coarse_candidate_relevance",
        "state_t": {
            "candidate_retrieval": None,
            "candidate_coarse_summary": {
                "coarse_index": 1,
                "retrieval_rank": 1,
                "visual_teacher_reranker": {"rank": 2, "score": 0.25},
            },
        },
    }
    reference = {
        ("cg_bench:4", 1): {
            "messages": [
                {"role": "system", "content": "frozen system"},
                {"role": "user", "content": json.dumps(frozen_user)},
                {"role": "assistant", "content": "must not be copied"},
            ]
        }
    }
    rows, report = build_dataset_dev_rows(
        [example],
        {"cg_bench:4": {"clue_spans": [{"start_s": 11, "end_s": 12}]}},
        candidate_indices_by_example={"cg_bench:4": [1]},
        prompt_reference_by_candidate=reference,
    )
    assert rows[0]["messages"][:2] == reference[("cg_bench:4", 1)]["messages"][:2]
    assert json.loads(rows[0]["messages"][2]["content"])["arguments"]["relevant"] is True
    assert report["prompt_payload_mode"] == "frozen_reference"


def test_builder_stamps_explicit_heldout_role() -> None:
    example = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:test:1",
        "question": {"question_id": "test:1", "question_text": "what happened?"},
        "metadata": {"coarse_clip_schemas": [
            _entry(0, 0, "irrelevant"), _entry(1, 10, "clue")
        ]},
    }
    rows, report = build_dataset_dev_rows(
        [example],
        {"cg_bench:test:1": {"clue_spans": [{"start_s": 11, "end_s": 12}]}},
        split_role="heldout_test",
    )
    assert report["split_role"] == "heldout_test"
    assert report["schema_version"] == "video-skills/l2-dataset-eval-build-v0.2"
    assert {row["metadata"]["split_role"] for row in rows} == {"heldout_test"}
    assert {row["schema_version"] for row in rows} == {
        "video-skills/l2-dataset-eval-chat-v0.2"
    }


def test_builder_rejects_cross_split_prompt_reference() -> None:
    example = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:test:2",
        "question": {"question_id": "test:2", "question_text": "what happened?"},
        "metadata": {"coarse_clip_schemas": [
            _entry(0, 0, "irrelevant"), _entry(1, 10, "clue")
        ]},
    }
    reference = {
        ("cg_bench:test:2", 1): {
            "metadata": {"split_role": "dev_tune"},
            "messages": [
                {"role": "system", "content": "frozen"},
                {"role": "user", "content": json.dumps({
                    "task": "score_coarse_candidate_relevance",
                    "state_t": {"candidate_coarse_summary": {"coarse_index": 1}},
                })},
            ],
        }
    }
    with pytest.raises(ValueError, match="split mismatch"):
        build_dataset_dev_rows(
            [example],
            {"cg_bench:test:2": {"clue_spans": [{"start_s": 11, "end_s": 12}]}},
            candidate_indices_by_example={"cg_bench:test:2": [1]},
            prompt_reference_by_candidate=reference,
            split_role="heldout_test",
        )

from __future__ import annotations

import json

from dataset_clip_wrapper.training.sft_common import contains_forbidden_prompt_key
from trainer.build_l2_dataset_opd import (
    VH_OPD_TEACHER_VERSION,
    build_dataset_opd_rows,
    candidate_teacher_score,
)


def _entry(index: int, start: float, text: str) -> dict:
    return {
        "clip_id": f"c{index}",
        "time_span": {"start_s": start, "end_s": start + 4},
        "scene_description": text,
        "observable_facts": [],
        "events": [],
    }


def test_video_holmes_teacher_uses_inference_and_relationship_support() -> None:
    example = {
        "dataset": "video_holmes",
        "example_id": "video_holmes:train:vid:q1",
        "question": {"question_text": "Why does the woman attack?", "answer": {"text": "predator"}},
        "metadata": {},
    }
    supervision = {
        "inference_spans": [{"start_s": 10, "end_s": 11}],
        "inference_texts": ["glowing eyes"],
        "relationship_texts": ["woman attacks as a predator"],
        "segment_texts": ["the woman approaches"],
    }
    positive = candidate_teacher_score(example, _entry(0, 9, "The woman has glowing eyes and attacks."), supervision)
    negative = candidate_teacher_score(example, _entry(1, 30, "An empty room."), supervision)
    assert positive > negative
    assert positive >= 0.6


def test_builds_balanced_train_only_rows_without_hidden_labels_in_prompt() -> None:
    example = {
        "dataset": "video_holmes",
        "example_id": "video_holmes:train:vid:q1",
        "question": {"question_text": "Why?", "answer": {"label": "A", "text": "attack"}},
        "metadata": {"coarse_clip_schemas": [], "clip_schemas": [
            _entry(0, 9, "glowing eyes attack"),
            _entry(1, 30, "empty room"),
            _entry(2, 40, "a table"),
        ]},
    }
    index = {"video_holmes:vid": {
        "segment_spans": [{"start_s": 0, "end_s": 50}],
        "inference_spans": [{"start_s": 10, "end_s": 11}],
        "segment_texts": ["setup"],
        "inference_texts": ["glowing eyes"],
        "relationship_texts": ["woman attacks"],
    }}
    rows, report = build_dataset_opd_rows([example], index, positives_per_example=1, negatives_per_example=2)
    assert report["source_examples"] == {"video_holmes": 1}
    assert report["video_holmes_teacher_contract"] == VH_OPD_TEACHER_VERSION
    assert report["relationship_support_contract"] == "structured-concept-overlap-v2"
    assert len(rows) == 3
    assert sum(row["state"]["sample_weight"] for row in rows) == 1.0
    prompts = " ".join(row["state"]["messages"][1]["content"] for row in rows)
    assert "inference_spans" not in prompts
    assert "relationship_texts" not in prompts
    for row in rows:
        payload = json.loads(row["state"]["messages"][1]["content"])
        assert not contains_forbidden_prompt_key(payload)
        assert "answer" not in payload["state_t"]["question"]


def test_video_holmes_partial_support_is_not_hard_labeled_negative() -> None:
    example = {
        "dataset": "video_holmes",
        "example_id": "video_holmes:train:vid:q2",
        "question": {"question_text": "Why?", "answer": {"text": "attack"}},
        "metadata": {"coarse_clip_schemas": [], "clip_schemas": [
            _entry(0, 9, "glowing eyes attack"),
            _entry(1, 20, "woman attacks"),
            _entry(2, 30, "empty room"),
        ]},
    }
    index = {"video_holmes:vid": {
        "inference_spans": [{"start_s": 10, "end_s": 11}],
        "inference_texts": ["glowing eyes"],
        "relationship_texts": ["woman attacks"],
        "segment_texts": [],
    }}
    rows, _ = build_dataset_opd_rows(
        [example], index, positives_per_example=1, negatives_per_example=1,
        min_video_holmes_score=0.5, max_video_holmes_negative_score=0.05,
    )
    selected = {row["state"]["candidate_index"] for row in rows}
    assert selected == {0, 2}

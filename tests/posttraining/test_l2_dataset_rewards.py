from __future__ import annotations

import json
from pathlib import Path

from trainer.grpo.l2_dataset_rewards import (
    RELATIONSHIP_SUPPORT_VERSION,
    is_placeholder_annotation,
    lexical_support,
    load_dataset_reward_supervision,
    supervision_key,
    temporal_retrieval_metrics,
)


def test_structured_relationship_support_matches_visual_combat_concepts() -> None:
    selected = [{
        "scene_description": "Two men face each other.",
        "visual_social_cues": [{"description": "The man is in a combat stance."}],
        "events": [{"description": "A man fires a glowing projectile."}],
    }]
    support = lexical_support(
        selected, ["Big-sized man and small-sized man", "Two people are fighting in the arena."]
    )
    assert RELATIONSHIP_SUPPORT_VERSION == "structured-concept-overlap-v2"
    assert support >= 0.25


def test_temporal_retrieval_metrics_distinguish_hit_and_miss() -> None:
    gold = [{"start_s": 10, "end_s": 20}, {"start_s": 30, "end_s": 35}]
    hit = temporal_retrieval_metrics([{"start_s": 12, "end_s": 16}], gold)
    miss = temporal_retrieval_metrics([{"start_s": 40, "end_s": 45}], gold)
    assert hit["recall"] == 0.5
    assert hit["precision"] == 1.0
    assert hit["mean_best_iou"] > miss["mean_best_iou"]


def test_loads_cg_and_video_holmes_hidden_labels_evaluator_side(tmp_path: Path) -> None:
    cg = tmp_path / "CG-Bench"
    cg.mkdir()
    (cg / "cgbench.json").write_text(
        json.dumps([{"qid": 7, "clue_intervals": [[3, 8]]}]), encoding="utf-8"
    )
    vh = tmp_path / "Video-Holmes" / "Benchmark" / "annotation_training"
    vh.mkdir(parents=True)
    (vh / "vid.json").write_text(json.dumps([{
        "SegmentDescription": [{"TimeRange": "00:00-00:10", "Description": "setup"}],
        "InferenceScenes": [{"Time": "00:07", "Clue": "key clue"}],
        "KeyRelationships": [{"Combination": "A and B", "Reason": "A caused B"}],
    }]), encoding="utf-8")
    index = load_dataset_reward_supervision(tmp_path)
    assert index["cg_bench:7"]["clue_spans"] == [{"start_s": 3.0, "end_s": 8.0}]
    assert index["video_holmes:vid"]["inference_spans"] == [{"start_s": 7.0, "end_s": 8.0}]
    example = {"dataset": "video_holmes", "example_id": "video_holmes:train:vid:q1"}
    assert supervision_key(example) == "video_holmes:vid"


def test_video_holmes_placeholder_rows_do_not_create_fake_zero_second_gold(tmp_path: Path) -> None:
    vh = tmp_path / "Video-Holmes" / "Benchmark" / "annotations"
    vh.mkdir(parents=True)
    (vh / "vid.json").write_text(json.dumps([{
        "Segment Description": [
            {"TimeRange": "00:00-00:05", "Description": "Fill in the segment"},
            {"TimeRange": "00:10-00:20", "Description": "The man hides the knife."},
        ],
        "Inference Shots": [
            {"Time": "00:00", "Clue": "Fill in the hint clues", "Conclusion": "Fill in the inference conclusion"},
            {"Time": "00:15", "Clue": "The knife is hidden", "Conclusion": "Fill in the inference conclusion"},
        ],
        "Key Relationships": [
            {"Combination": "Fill in character combination 2", "Relation": "The same person", "Reason": "Fill in the reason"},
        ],
    }]), encoding="utf-8")
    row = load_dataset_reward_supervision(tmp_path)["video_holmes:vid"]
    assert row["segment_spans"] == [{"start_s": 10.0, "end_s": 20.0}]
    assert row["inference_spans"] == [{"start_s": 15.0, "end_s": 16.0}]
    assert row["inference_texts"] == ["The knife is hidden"]
    assert row["relationship_texts"] == ["The same person"]
    assert row["annotation_quality"]["dropped_placeholder_inference_rows"] == 1
    assert is_placeholder_annotation("Fill in the hint clues")
    assert not is_placeholder_annotation("The murderer and the victim")

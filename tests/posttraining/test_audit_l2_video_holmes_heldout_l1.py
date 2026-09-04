from pathlib import Path

from scripts.eval.audit_l2_video_holmes_heldout_l1 import audit


def _manifest() -> dict:
    return {
        "manifest_hash": "manifest-sha",
        "videos": [
            {
                "dataset": "video_holmes",
                "video_id": "vh-test-1",
                "role": "heldout_test",
                "official_split": "test",
            }
        ],
    }


def _row() -> dict:
    return {
        "dataset": "video_holmes",
        "split": "test",
        "example_id": "video_holmes:test:vh-test-1:q1",
        "video": {"video_id": "vh-test-1"},
        "hidden_supervision": {
            "available_for_training": True,
            "available_for_inference": False,
        },
        "metadata": {
            "l1_perception_protocol": "no-redundant-covered-tail-v1",
            "clip_schemas": [
                {
                    "clip_id": "clip:vh-test-1:fine:0001",
                    "producer": "qwen_clip_schema",
                    "model": "Qwen/Qwen3.5-9B",
                    "llm_usage": {"sampled_frame_count": 4, "max_tokens": 1600},
                }
            ],
            "clue_memory_graph": {
                "index_stats": {"fine_clip_count": 1, "perception_clip_count": 1},
                "perception": {"clip_schema_model": "Qwen/Qwen3.5-9B"},
            }
        },
        "evidence_candidates": [
            {
                "source_type": "caption_span",
                "trust_level": "model_labeled",
                "text": "A visible video caption.",
            }
        ],
    }


def test_heldout_l1_audit_accepts_exact_video_only_coverage(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    report = audit(_manifest(), [(path, _row())], expected_count=1)
    assert report["passed"] is True
    assert report["checks"]["no_hidden_gold_in_visible_candidates"] is True


def test_heldout_l1_audit_rejects_visible_gold(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["evidence_candidates"][0] = {
        "source_type": "inference_shot",
        "trust_level": "gold",
    }
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["no_hidden_gold_in_visible_candidates"] is False


def test_heldout_l1_audit_rejects_missing_video(tmp_path: Path) -> None:
    report = audit(_manifest(), [], expected_count=1)
    assert report["passed"] is False
    assert report["missing_video_ids"] == ["vh-test-1"]


def test_heldout_l1_audit_rejects_temporal_only_catalog(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["evidence_candidates"] = [
        {"source_type": "video_segment", "trust_level": "derived"}
    ]
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["every_video_has_model_caption_spans"] is False


def test_heldout_l1_audit_rejects_partial_perception(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["metadata"]["clue_memory_graph"]["index_stats"]["fine_clip_count"] = 2
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["every_fine_clip_has_perception_schema"] is False


def test_heldout_l1_audit_rejects_failed_or_non_qwen_schema(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["metadata"]["clip_schemas"][0]["model_error"] = "timeout"
    row["metadata"]["clip_schemas"][0]["producer"] = "fallback"
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["all_perception_schemas_are_valid_qwen_outputs"] is False
    assert report["invalid_perception_schemas"] == ["vh-test-1"]


def test_heldout_l1_audit_rejects_mixed_perception_config(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["metadata"]["clip_schemas"][0]["llm_usage"]["sampled_frame_count"] = 5
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["uniform_frozen_perception_config"] is False
    assert report["inconsistent_perception_configs"] == ["vh-test-1"]


def test_heldout_l1_audit_accepts_marked_six_frame_anchor_repass(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    schema = row["metadata"]["clip_schemas"][0]
    schema["schema_attempt_context"] = "query_time_anchor_repass"
    schema["request_frames"] = 6
    schema["llm_usage"]["sampled_frame_count"] = 6
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is True


def test_heldout_l1_audit_rejects_unmarked_six_frame_schema(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["metadata"]["clip_schemas"][0]["llm_usage"]["sampled_frame_count"] = 6
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["uniform_frozen_perception_config"] is False


def test_heldout_l1_audit_rejects_old_perception_protocol(tmp_path: Path) -> None:
    path = tmp_path / "04_l1_example.json"
    path.write_text("{}")
    row = _row()
    row["metadata"]["l1_perception_protocol"] = "legacy-tail-windows"
    report = audit(_manifest(), [(path, row)], expected_count=1)
    assert report["passed"] is False
    assert report["checks"]["uniform_frozen_perception_protocol"] is False
    assert report["inconsistent_perception_protocols"] == ["vh-test-1"]

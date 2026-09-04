import json

from dataset_clip_wrapper.runners.run_staged_llm_pipeline import _cached_clip_schema_error_count


def _schema(clip_id: str, sampled_frames: int, *, anchor: bool = False) -> dict:
    row = {
        "clip_id": clip_id,
        "producer": "qwen_clip_schema",
        "llm_usage": {"sampled_frame_count": sampled_frames},
    }
    if anchor:
        row["schema_attempt_context"] = "query_time_anchor_repass"
        row["request_frames"] = 6
    return row


def test_cached_schema_audit_marks_short_anchor_for_retry(tmp_path) -> None:
    (tmp_path / "01_perception_spans.json").write_text(
        json.dumps({"derived_clips": [{"clip_id": "clip:test:fine:0001"}]}),
        encoding="utf-8",
    )
    (tmp_path / "02_clip_schemas.jsonl").write_text(
        json.dumps(_schema("clip:test:fine:0001", 4)) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "02b_anchor_clip_schemas.jsonl").write_text(
        json.dumps(_schema("clip:test:fine:0001", 5, anchor=True)) + "\n",
        encoding="utf-8",
    )

    assert _cached_clip_schema_error_count(tmp_path) == 1


def test_cached_schema_audit_accepts_exact_anchor_frame_count(tmp_path) -> None:
    (tmp_path / "01_perception_spans.json").write_text(
        json.dumps({"derived_clips": [{"clip_id": "clip:test:fine:0001"}]}),
        encoding="utf-8",
    )
    (tmp_path / "02_clip_schemas.jsonl").write_text(
        json.dumps(_schema("clip:test:fine:0001", 4)) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "02b_anchor_clip_schemas.jsonl").write_text(
        json.dumps(_schema("clip:test:fine:0001", 6, anchor=True)) + "\n",
        encoding="utf-8",
    )

    assert _cached_clip_schema_error_count(tmp_path) == 0

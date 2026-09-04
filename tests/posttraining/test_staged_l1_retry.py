import json
from types import SimpleNamespace

from dataset_clip_wrapper.runners.run_staged_llm_pipeline import (
    _cached_clip_schema_error_count,
    _produce_or_resume_clip_schemas,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cached_clip_schema_error_count_covers_all_schema_stages(tmp_path):
    _write_jsonl(
        tmp_path / "02_clip_schemas.jsonl",
        [
            {"clip_id": "fine-good", "summary": "ok"},
            {"clip_id": "fine-bad", "model_error": "timeout"},
        ],
    )
    _write_jsonl(
        tmp_path / "00b_coarse_clip_schemas.jsonl",
        [{"clip_id": "coarse-bad", "model_error": "http 500"}],
    )
    _write_jsonl(
        tmp_path / "02b_anchor_clip_schemas.jsonl",
        [{"clip_id": "anchor-good", "model_error": None}],
    )
    _write_jsonl(
        tmp_path / "unrelated.jsonl",
        [{"model_error": "must not count"}],
    )

    assert _cached_clip_schema_error_count(tmp_path) == 2


def test_cached_clip_schema_error_count_is_zero_without_cache(tmp_path):
    assert _cached_clip_schema_error_count(tmp_path) == 0


def test_cached_clip_schema_error_count_detects_missing_and_duplicate_ids(tmp_path):
    (tmp_path / "01_perception_spans.json").write_text(
        json.dumps({"derived_clips": [{"clip_id": "a"}, {"clip_id": "b"}]}),
        encoding="utf-8",
    )
    _write_jsonl(
        tmp_path / "02_clip_schemas.jsonl",
        [{"clip_id": "a"}, {"clip_id": "a"}],
    )

    # Missing b plus one duplicate a.
    assert _cached_clip_schema_error_count(tmp_path) == 2


def test_cached_clip_schema_error_count_rejects_zero_frame_qwen_success(tmp_path):
    (tmp_path / "01_perception_spans.json").write_text(
        json.dumps({"derived_clips": [{"clip_id": "tail"}]}),
        encoding="utf-8",
    )
    _write_jsonl(
        tmp_path / "02_clip_schemas.jsonl",
        [{
            "clip_id": "tail",
            "producer": "qwen_clip_schema",
            "llm_usage": {"sampled_frame_count": 0},
        }],
    )

    assert _cached_clip_schema_error_count(tmp_path) == 1


def test_retry_removes_zero_frame_tail_outside_recomputed_spans(tmp_path):
    stage_path = tmp_path / "02_clip_schemas.jsonl"
    _write_jsonl(
        stage_path,
        [
            {
                "clip_id": "kept",
                "producer": "qwen_clip_schema",
                "llm_usage": {"sampled_frame_count": 4},
            },
            {
                "clip_id": "obsolete-tail",
                "producer": "qwen_clip_schema",
                "llm_usage": {"sampled_frame_count": 0},
            },
        ],
    )
    config = SimpleNamespace(
        clip_schema=SimpleNamespace(max_clips=999, backend="qwen")
    )

    rows = _produce_or_resume_clip_schemas(
        item=None,
        config=config,
        spans=[None],
        derived_clips=[{"clip_id": "kept"}],
        visible_segments=[],
        stage_path=stage_path,
        force=False,
        retry_failed=True,
        retry_non_backbone=False,
        fill_missing=True,
        workers=1,
    )

    assert [row["clip_id"] for row in rows] == ["kept"]
    assert [json.loads(line)["clip_id"] for line in stage_path.read_text().splitlines()] == ["kept"]


def test_cached_clip_schema_error_count_rebuilds_legacy_final_protocol(tmp_path):
    (tmp_path / "01_perception_spans.json").write_text(
        json.dumps({"derived_clips": [{"clip_id": "a"}]}), encoding="utf-8"
    )
    _write_jsonl(tmp_path / "02_clip_schemas.jsonl", [{"clip_id": "a"}])
    (tmp_path / "04_l1_example.json").write_text(
        json.dumps({"metadata": {"l1_perception_protocol": "legacy"}}),
        encoding="utf-8",
    )

    assert _cached_clip_schema_error_count(tmp_path) == 1

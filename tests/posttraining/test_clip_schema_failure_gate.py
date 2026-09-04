"""Run-level accounting for degraded clip schemas.

``build_clip_schema`` returns a schema-shaped placeholder carrying ``model_error``
when every retry fails, so a backend that rejects 100% of requests still produces
structurally valid output that passes every downstream check.  The only defence is
counting the placeholders.
"""

import json

from dataset_clip_wrapper.runners.run_staged_llm_pipeline import (
    _build_parser,
    _clip_schema_failure_counts,
)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_counts_placeholders_against_the_full_cached_catalog(tmp_path) -> None:
    _write(tmp_path / "02_clip_schemas.jsonl", [
        {"clip_id": "a", "scene_description": "a person walks in"},
        {"clip_id": "b", "scene_description": "clip schema generation failed", "model_error": "422"},
        {"clip_id": "c", "scene_description": "a door closes"},
    ])
    assert _clip_schema_failure_counts(tmp_path) == (1, 3)


def test_counts_span_multiple_stage_files(tmp_path) -> None:
    _write(tmp_path / "02_clip_schemas.jsonl", [{"clip_id": "a", "model_error": "422"}])
    _write(tmp_path / "05_anchor_clip_schemas.jsonl", [{"clip_id": "b"}])
    assert _clip_schema_failure_counts(tmp_path) == (1, 2)


def test_missing_stage_dir_is_not_a_failure(tmp_path) -> None:
    assert _clip_schema_failure_counts(tmp_path / "absent") == (0, 0)


def test_gate_defaults_are_calibrated_to_measured_lanes() -> None:
    args = _build_parser().parse_args(["--dataset", "cg_bench"])
    # Final gate: healthy lane measured 0.9% final, degraded one 22.7%.
    assert args.max_clip_schema_failure_rate == 0.01
    # Mid-run abort: healthy peaked at ~6% cumulative, the dead lane sat at 100%.
    assert args.abort_clip_schema_failure_rate == 0.50
    assert args.clip_schema_failure_min_sample == 200


def test_thresholds_separate_the_three_measured_lanes() -> None:
    """Rolling rate alone cannot work: failures are front-loaded when healthy and
    back-loaded when degraded, so the early rate is inverted against the truth."""
    lanes = {                 # (early cumulative, final cumulative)
        "healthy":  (0.056, 0.009),
        "degraded": (0.003, 0.227),
        "dead":     (1.000, 1.000),
    }
    abort, final = 0.50, 0.01
    aborted = {k: early > abort for k, (early, _) in lanes.items()}
    failed = {k: fin > final for k, (_, fin) in lanes.items()}
    assert aborted == {"healthy": False, "degraded": False, "dead": True}
    assert failed == {"healthy": False, "degraded": True, "dead": True}


def test_gate_is_overridable() -> None:
    args = _build_parser().parse_args(
        ["--dataset", "cg_bench", "--max-clip-schema-failure-rate", "1.0"]
    )
    assert args.max_clip_schema_failure_rate == 1.0

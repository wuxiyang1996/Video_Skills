"""Tests for split manifest freezing and SFT warm-up quality gates."""

from __future__ import annotations

import json
from pathlib import Path

from dataset_clip_wrapper.training.build_split_manifest import build_split_manifest
from dataset_clip_wrapper.training.evaluate_lora_sft_gates import (
    evaluate_lora_report,
    majority_action_baseline,
)
from dataset_clip_wrapper.training.evaluate_sft_package_gates import (
    decide_package_gates,
    evaluate_split_file,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _chat_row(*, dataset: str, video_id: str, tool_name: str, decision: str | None = None) -> dict:
    args = {"decision": decision} if decision else {"selected_coarse_indices": [0]}
    assistant = {
        "schema_version": "test",
        "tool_name": tool_name,
        "arguments": args,
    }
    user = {
        "task": "test",
        "state_t": {
            "dataset": dataset,
            "video_id": video_id,
            "video_state": {"video_id": video_id},
            "example_id": f"{dataset}:1",
        },
    }
    return {
        "transition_id": f"{dataset}:{video_id}::0",
        "dataset": dataset,
        "metadata": {"dataset": dataset, "video_id": video_id, "task": tool_name},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps(user)},
            {"role": "assistant", "content": json.dumps(assistant)},
        ],
    }


def test_build_split_manifest_respects_video_holmes_official_splits():
    root = Path("/fs/gamma-projects/vlm-robot/datasets")
    if not (root / "Video-Holmes" / "Benchmark" / "train_Video-Holmes.json").exists():
        return
    if not (root / "CG-Bench" / "cgbench.json").exists():
        return
    manifest = build_split_manifest(root, salt="unit-test-salt")
    vh_test = [row for row in manifest["videos"] if row["dataset"] == "video_holmes" and row["official_split"] == "test"]
    vh_train = [row for row in manifest["videos"] if row["dataset"] == "video_holmes" and row["official_split"] == "train"]
    assert vh_test
    assert all(row["role"] == "heldout_test" for row in vh_test)
    assert vh_train
    assert all(row["role"] != "heldout_test" for row in vh_train)
    assert manifest["manifest_hash"]
    # Deterministic under same salt.
    again = build_split_manifest(root, salt="unit-test-salt")
    assert again["manifest_hash"] == manifest["manifest_hash"]


def test_package_gate_detects_eval_only_and_forbidden_keys(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    good = _chat_row(dataset="cg_bench", video_id="vidA", tool_name="select_coarse_clips")
    bad_eval = _chat_row(dataset="vrbench", video_id="vidB", tool_name="select_coarse_clips")
    leak = _chat_row(dataset="cg_bench", video_id="vidC", tool_name="select_coarse_clips")
    leak_user = json.loads(leak["messages"][1]["content"])
    leak_user["state_t"]["gold_answer"] = "A"
    leak["messages"][1]["content"] = json.dumps(leak_user)
    _write_jsonl(path, [good, bad_eval, leak])

    report = evaluate_split_file(path, specialist="l2", split_name="train")
    assert report["eval_only_rows"] == 1
    assert report["prompt_forbidden_key_hits"] == 1
    assert report["assistant_json_parse_rate"] == 1.0

    decision = decide_package_gates([report], require_split_manifest=False)
    assert decision["passed"] is False
    assert any("eval_only_rows" in item for item in decision["failures"])
    assert any("prompt_forbidden_key_hits" in item for item in decision["failures"])


def test_majority_baseline_and_lora_gate_require_beating_majority(tmp_path: Path):
    path = tmp_path / "dev.jsonl"
    rows = [
        _chat_row(dataset="cg_bench", video_id="v1", tool_name="select_coarse_clips"),
        _chat_row(dataset="cg_bench", video_id="v2", tool_name="select_coarse_clips"),
        _chat_row(dataset="cg_bench", video_id="v3", tool_name="reject_commit_and_retrieve_more"),
    ]
    _write_jsonl(path, rows)
    majority = majority_action_baseline(path)
    assert majority["majority_family"] == "select_coarse_clips"
    assert majority["action_match_rate"] == 2 / 3

    # LoRA that only matches majority rate fails.
    fail = evaluate_lora_report(
        specialist="l2",
        lora_generation_report={"examples": 3, "json_valid_rate": 1.0, "action_match_rate": 2 / 3},
        majority_baseline=majority,
        thresholds={"min_json_valid_rate": 0.95, "min_action_match_rate": 0.5, "require_beat_majority_action_match": True},
    )
    assert fail["passed"] is False

    ok = evaluate_lora_report(
        specialist="l2",
        lora_generation_report={"examples": 3, "json_valid_rate": 1.0, "action_match_rate": 1.0},
        base_generation_report={"json_valid_rate": 0.3, "action_match_rate": 0.2},
        majority_baseline=majority,
        thresholds={"min_json_valid_rate": 0.95, "min_action_match_rate": 0.5, "require_beat_majority_action_match": True},
    )
    assert ok["passed"] is True


def test_single_family_majority_beat_is_skipped():
    majority = {
        "n_rows": 4,
        "majority_family": "emit_verifier_decision",
        "majority_rate": 1.0,
        "json_valid_rate": 1.0,
        "action_match_rate": 1.0,
        "family_counts": {"emit_verifier_decision": 4},
    }
    result = evaluate_lora_report(
        specialist="verifier",
        lora_generation_report={"examples": 4, "json_valid_rate": 1.0, "action_match_rate": 1.0},
        base_generation_report={"json_valid_rate": 1.0, "action_match_rate": 0.0},
        majority_baseline=majority,
        thresholds={
            "min_json_valid_rate": 0.95,
            "min_action_match_rate": 0.5,
            "require_beat_majority_action_match": True,
        },
    )
    assert result["passed"] is True
    assert any("majority baseline skipped" in w for w in result["warnings"])

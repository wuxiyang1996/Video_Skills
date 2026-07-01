#!/usr/bin/env python3
"""Smoke test for dataset clip wrappers without API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.pipeline import iter_canonical_examples
from dataset_clip_wrapper.schemas import BackboneConfig, RuntimeMode, VideoRegime, WrapperConfig


def _check_example(example: dict) -> list[str]:
    errors: list[str] = []
    required = ["schema_version", "example_id", "dataset", "video", "question", "evidence_candidates", "evidence_index"]
    for key in required:
        if key not in example:
            errors.append(f"missing key: {key}")
    clips = example.get("video", {}).get("derived_clips", [])
    if not clips:
        errors.append("no derived_clips")
    policy = example.get("evidence_index", {}).get("clip_policy", {})
    if not policy.get("strategy"):
        errors.append("missing clip_policy.strategy")
    return errors


def main() -> int:
    dataset_root = "/fs/gamma-projects/vlm-robot/datasets"
    cases = [
        ("video_holmes", VideoRegime.SHORT, "train"),
        ("cg_bench", VideoRegime.LONG, "train"),
        ("vrbench", VideoRegime.LONG, "train"),
        ("siv_bench", VideoRegime.SHORT, "train"),
    ]
    report = []
    for dataset, regime, split in cases:
        config = WrapperConfig(
            dataset_root=dataset_root,
            dataset=dataset,  # type: ignore[arg-type]
            regime=regime,
            mode=RuntimeMode.EXPERT_DEMO,
            split=split,
            limit=1,
            backbone=BackboneConfig(name="annotation_only"),
            run_backbone=False,
        )
        example = next(iter_canonical_examples(config))
        errors = _check_example(example)
        report.append(
            {
                "dataset": dataset,
                "regime": regime.value,
                "example_id": example["example_id"],
                "clip_count": len(example["video"]["derived_clips"]),
                "evidence_count": len(example["evidence_candidates"]),
                "strategy": example["evidence_index"]["clip_policy"]["strategy"],
                "passed": not errors,
                "errors": errors,
            }
        )

    streaming_config = WrapperConfig(
        dataset_root=dataset_root,
        dataset="video_holmes",
        regime=VideoRegime.STREAMING,
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
        backbone=BackboneConfig(name="annotation_only"),
        run_backbone=False,
    )
    streaming = next(iter_canonical_examples(streaming_config))
    max_end = max(c["source_span"]["end_s"] for c in streaming["video"]["derived_clips"])
    obs_end = streaming["evidence_index"]["clip_policy"].get("observation_end_s")
    report.append(
        {
            "dataset": "video_holmes",
            "regime": "streaming",
            "clip_count": len(streaming["video"]["derived_clips"]),
            "observation_end_s": obs_end,
            "max_clip_end_s": max_end,
            "passed": obs_end is not None and max_end <= obs_end + 1e-6,
            "errors": [],
        }
    )

    print(json.dumps(report, indent=2))
    return 0 if all(item["passed"] for item in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())

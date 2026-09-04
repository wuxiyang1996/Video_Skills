#!/usr/bin/env python3
"""Audit cleaned Video-Holmes evaluator supervision by frozen video split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trainer.grpo.l2_dataset_rewards import (
    VH_PLACEHOLDER_FILTER_VERSION,
    load_dataset_reward_supervision,
)
from trainer.split_filter import load_split_manifest


def audit(dataset_root: Path, split_manifest: Path) -> dict[str, Any]:
    supervision = load_dataset_reward_supervision(dataset_root)
    manifest = load_split_manifest(split_manifest)
    roles = {
        str(row.get("video_id") or ""): str(row.get("role") or "unknown")
        for row in manifest.get("videos") or []
        if row.get("dataset") == "video_holmes"
    }
    totals: Counter[str] = Counter()
    affected_by_role: Counter[str] = Counter()
    ids_by_role: dict[str, list[str]] = defaultdict(list)
    empty_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    video_count = 0
    for key, row in sorted(supervision.items()):
        if not key.startswith("video_holmes:"):
            continue
        video_count += 1
        video_id = key.split(":", 1)[1]
        role = roles.get(video_id, "unknown")
        quality = row.get("annotation_quality") or {}
        drops = {
            name: int(value)
            for name, value in quality.items()
            if str(name).startswith("dropped_")
        }
        totals.update(drops)
        if any(drops.values()):
            affected_by_role[role] += 1
            ids_by_role[role].append(video_id)
        if not row.get("inference_spans"):
            empty_by_role[role]["empty_inference_spans"] += 1
        if not row.get("relationship_texts"):
            empty_by_role[role]["empty_relationship_texts"] += 1
    return {
        "schema_version": "video-skills/video-holmes-supervision-audit-v0.1",
        "placeholder_filter": VH_PLACEHOLDER_FILTER_VERSION,
        "video_count": video_count,
        "dropped_placeholder_totals": dict(sorted(totals.items())),
        "affected_videos": sum(affected_by_role.values()),
        "affected_videos_by_split_role": dict(sorted(affected_by_role.items())),
        "affected_video_ids_by_split_role": {
            role: sorted(ids_) for role, ids_ in sorted(ids_by_role.items())
        },
        "empty_supervision_by_split_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(empty_by_role.items())
        },
        "split_manifest": str(split_manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset_root, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Produce paper-facing evidence that post-training roles are video-exclusive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


VALID_ROLES = {"sft_seed", "opd_pool", "grpo_pool", "dev_tune", "heldout_test"}


def audit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    videos = [row for row in manifest.get("videos") or [] if isinstance(row, Mapping)]
    keys = [str(row.get("key") or "") for row in videos]
    roles_by_key: dict[str, set[str]] = defaultdict(set)
    for row in videos:
        roles_by_key[str(row.get("key") or "")].add(str(row.get("role") or ""))
    vh_test = [
        row for row in videos
        if row.get("dataset") == "video_holmes" and row.get("official_split") == "test"
    ]
    vh_train = [
        row for row in videos
        if row.get("dataset") == "video_holmes" and row.get("official_split") == "train"
    ]
    digest = hashlib.sha256(
        json.dumps(
            {
                "salt": manifest.get("salt"),
                "assignment": manifest.get("assignment"),
                "videos": [
                    (row.get("key"), row.get("role"), row.get("n_questions")) for row in videos
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    role_counts = Counter(str(row.get("role") or "") for row in videos)
    dataset_role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in videos:
        dataset_role_counts[str(row.get("dataset") or "")][str(row.get("role") or "")] += 1
    checks = {
        "schema_version": manifest.get("schema_version") == "video-skills/split-manifest-v1",
        "manifest_content_hash": digest == manifest.get("manifest_hash"),
        "all_video_keys_nonempty": all(keys),
        "video_keys_unique": len(keys) == len(set(keys)),
        "one_role_per_video": all(len(roles) == 1 for roles in roles_by_key.values()),
        "roles_valid": all(str(row.get("role") or "") in VALID_ROLES for row in videos),
        "vh_official_test_heldout_only": bool(vh_test)
        and all(row.get("role") == "heldout_test" for row in vh_test),
        "vh_official_train_never_heldout": bool(vh_train)
        and all(row.get("role") != "heldout_test" for row in vh_train),
        "summary_video_count_matches": int((manifest.get("summary") or {}).get("n_videos") or -1)
        == len(videos),
        "summary_role_counts_match": (manifest.get("summary") or {}).get("role_video_counts")
        == dict(role_counts),
    }
    return {
        "schema_version": "video-skills/l2-split-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_hash": manifest.get("manifest_hash"),
        "videos": len(videos),
        "questions": sum(int(row.get("n_questions") or 0) for row in videos),
        "role_video_counts": dict(role_counts),
        "dataset_role_video_counts": {
            dataset: dict(counts) for dataset, counts in sorted(dataset_role_counts.items())
        },
        "video_holmes_official_test_videos": len(vh_test),
        "video_holmes_official_train_videos": len(vh_train),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

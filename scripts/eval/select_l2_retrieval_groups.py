#!/usr/bin/env python3
"""Select evidence-hit, reward-variance groups from retrieval-only mining."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def select_groups(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    eligible = [
        row for row in rows
        if bool(row.get("reward_variance"))
        and int(row.get("process_supported_samples") or 0) > 0
        and int(row.get("format_compliant_samples") or 0) > 0
    ]
    example_ids = list(dict.fromkeys(str(row.get("example_id") or "") for row in eligible))
    example_ids = [value for value in example_ids if value]
    datasets = Counter(str(row.get("dataset") or "unknown") for row in eligible)
    report = {
        "schema_version": "video-skills/l2-retrieval-group-selection-v0.1",
        "groups_seen": len(rows),
        "groups_eligible": len(eligible),
        "eligible_group_rate": len(eligible) / max(1, len(rows)),
        "unique_examples_selected": len(example_ids),
        "eligible_groups_by_dataset": dict(datasets),
        "criteria": {
            "reward_variance": True,
            "min_process_supported_samples": 1,
            "min_format_compliant_samples": 1,
        },
    }
    return example_ids, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    example_ids, report = select_groups(rows)
    args.allowlist.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.allowlist.write_text("".join(f"{value}\n" for value in example_ids), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if example_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())

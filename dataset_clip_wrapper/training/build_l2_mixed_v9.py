#!/usr/bin/env python3
"""Build a balanced L2 package that retains selection and pointwise abilities."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .sft_common import read_jsonl, write_json, write_jsonl


def mix_rows(
    selection_rows: list[dict[str, Any]], pointwise_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    counts: Counter[str] = Counter()
    source_weights: defaultdict[str, float] = defaultdict(float)
    for lane, rows in (("selection_ranking", selection_rows), ("pointwise", pointwise_rows)):
        for source in rows:
            row = dict(source)
            metadata = dict(row.get("metadata") or {})
            metadata["mixed_v9_lane"] = lane
            metadata["source_family_weight"] = 0.5 * float(
                metadata.get("source_family_weight", 1.0)
            )
            row["metadata"] = metadata
            output.append(row)
            counts[lane] += 1
            source_weights[str(metadata.get("source_example_id") or "")] += float(
                metadata["source_family_weight"]
            )
    return output, {
        "rows": len(output),
        "lane_rows": dict(counts),
        "source_examples": len(source_weights),
        "source_weight_min": min(source_weights.values(), default=0.0),
        "source_weight_max": max(source_weights.values(), default=0.0),
        "source_weight_sum": sum(source_weights.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pointwise-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"schema_version": "video-skills/l2-mixed-v9-report-v0.1"}
    outputs = {}
    for split in ("train", "dev"):
        outputs[split], report[split] = mix_rows(
            read_jsonl(args.selection_root / f"{split}.jsonl"),
            read_jsonl(args.pointwise_root / f"{split}.jsonl"),
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", outputs["train"])
    write_jsonl(args.output_dir / "dev.jsonl", outputs["dev"])
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

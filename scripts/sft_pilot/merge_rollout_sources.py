#!/usr/bin/env python3
"""Merge rollout JSONL sources, keeping the last row for each example ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--input-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    paths = [path for path in args.inputs if path.exists()]
    for root in args.input_roots:
        paths.extend(sorted(root.glob("**/examples.jsonl")))
    paths = list(dict.fromkeys(paths))

    chosen: dict[str, dict] = {}
    source_rows = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source_rows += 1
                row = json.loads(line)
                example_id = str(row.get("example_id") or "")
                if example_id:
                    chosen[example_id] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example_id in sorted(chosen):
            handle.write(json.dumps(chosen[example_id], ensure_ascii=False) + "\n")

    report = {
        "source_paths": [str(path) for path in paths],
        "source_rows": source_rows,
        "unique_examples": len(chosen),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

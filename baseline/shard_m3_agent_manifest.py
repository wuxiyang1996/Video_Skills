#!/usr/bin/env python3
"""Split M3 memorization inputs into deterministic graph shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for shard_index in range(args.num_shards):
        selected = [row for index, row in enumerate(rows) if index % args.num_shards == shard_index]
        path = args.output_dir / f"memorization_{shard_index:04d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts.append(len(selected))
    summary = {"graphs": len(rows), "num_shards": args.num_shards, "counts": counts}
    (args.output_dir / "shard_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge disjoint L2 visual candidate-evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_l2_candidate_retrieval import _aggregate
from .sft_common import write_json


def merge_reports(payloads: list[dict]) -> dict:
    if not payloads:
        raise ValueError("No reports to merge")
    results = [row for payload in payloads for row in payload["results"]]
    ids = [str(row["example_id"]) for row in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Reports contain duplicate example IDs")
    first = payloads[0]
    for payload in payloads[1:]:
        for key in ("model", "num_frames_per_coarse_window", "max_frame_side", "fine_window_sec", "fine_stride_sec"):
            if payload.get(key) != first.get(key):
                raise ValueError(f"Shard configuration mismatch: {key}")
    return {
        **{key: value for key, value in first.items() if key not in {"summary", "boundary_hybrid_summary", "results", "num_shards", "shard_index"}},
        "merged_shards": len(payloads),
        "summary": _aggregate(results),
        "boundary_hybrid_summary": {
            "examples": len(results),
            "hit_at_32": sum(row["boundary_hybrid_at_32"]["hit"] for row in results) / len(results),
            "recall_at_32": sum(row["boundary_hybrid_at_32"]["recall"] for row in results) / len(results),
        },
        "results": sorted(results, key=lambda row: str(row["example_id"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    output = merge_reports(payloads)
    write_json(args.output, output)
    print(json.dumps({"summary": output["summary"], "boundary_hybrid_summary": output["boundary_hybrid_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

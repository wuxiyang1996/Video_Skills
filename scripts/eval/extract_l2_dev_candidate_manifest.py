#!/usr/bin/env python3
"""Freeze the label-independent candidate set embedded in an L2 eval report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    examples: dict[str, list[int]] = {}
    for row in report.get("results") or []:
        example_id = str(row.get("example_id") or "")
        ranking = row.get("ranking") or []
        ordered = sorted(
            ranking,
            key=lambda item: (int(item.get("retrieval_rank") or 0), int(item["candidate_index"])),
        )
        indices = [int(item["candidate_index"]) for item in ordered]
        if not example_id or not indices or len(indices) != len(set(indices)):
            raise ValueError(f"invalid candidate rows for {example_id or '<missing>'}")
        examples[example_id] = indices
    payload = {
        "schema_version": "video-skills/l2-dev-candidate-manifest-v0.1",
        "selection_contract": "legacy-label-independent-retrieval-rank-v1",
        "source_report": str(args.report),
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "examples": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

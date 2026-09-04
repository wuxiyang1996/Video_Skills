#!/usr/bin/env python3
"""Keep L2 chats whose source example is absent from an existing candidate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sft_common import read_json, read_jsonl, write_json, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--existing-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    scored = {
        str(row["example_id"])
        for row in (read_json(args.existing_report).get("results") or [])
    }
    source = read_jsonl(args.input_jsonl)
    output = [
        row for row in source
        if str((row.get("metadata") or {}).get("source_example_id") or "") not in scored
    ]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, output)
    summary = {
        "input_rows": len(source),
        "existing_scored_examples": len(scored),
        "output_rows": len(output),
        "output_core_examples": len({
            str((row.get("metadata") or {}).get("source_example_id") or "")
            for row in output
            if (row.get("metadata") or {}).get("task") == "select_coarse_set"
            and (row.get("metadata") or {}).get("is_core") is True
        }),
    }
    write_json(args.report, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

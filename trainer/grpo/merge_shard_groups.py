"""Merge per-shard ``grpo_groups.jsonl`` files after fan-out collect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.grpo.quality import summarize_group_quality


def merge_shard_groups(shard_root: str | Path) -> list[dict[str, Any]]:
    root = Path(shard_root)
    paths = sorted(root.glob("shard_*/grpo_groups.jsonl"))
    if not paths:
        # Also accept raw shards if filtered file missing.
        paths = sorted(root.glob("shard_*/grpo_groups_raw.jsonl"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                gid = str(row.get("group_id") or row.get("example_id") or "")
                if gid and gid in seen:
                    continue
                if gid:
                    seen.add(gid)
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default="")
    args = parser.parse_args(argv)

    rows = merge_shard_groups(args.shard_root)
    if not rows:
        raise SystemExit(f"No shard groups under {args.shard_root}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    quality = summarize_group_quality(rows)
    summary = {
        "n_groups": len(rows),
        "groups_path": str(out),
        "shard_root": str(args.shard_root),
        "mean_terminal_success": quality["mean_terminal_success"],
        "quality": quality,
    }
    summary_path = Path(args.summary_out) if args.summary_out else out.parent / "collect_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

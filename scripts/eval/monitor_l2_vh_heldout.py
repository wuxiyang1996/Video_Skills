#!/usr/bin/env python3
"""Monitor resumable Video-Holmes heldout L1 generation with its frozen contract."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "no-redundant-covered-tail-v1"
MODEL = "Qwen/Qwen3.5-9B"


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _schema_valid(schema: Mapping[str, Any]) -> bool:
    usage = schema.get("llm_usage") or {}
    if not isinstance(usage, Mapping):
        return False
    sampled = int(usage.get("sampled_frame_count") or 0)
    anchor = schema.get("schema_attempt_context") == "query_time_anchor_repass"
    frames_ok = (
        sampled == 6 and int(schema.get("request_frames") or 0) == 6
        if anchor
        else 1 <= sampled <= 4
    )
    return bool(
        not schema.get("model_error")
        and schema.get("producer") == "qwen_clip_schema"
        and schema.get("model") == MODEL
        and frames_ok
        and int(usage.get("max_tokens") or 0) == 1600
    )


def snapshot(root: Path, filesystem_path: Path, expected_schemas: int) -> dict[str, Any]:
    primary_valid = primary_errors = primary_zero = parse_errors = 0
    anchor_valid = anchor_errors = 0
    stages = list(root.glob("start_*/stages/*"))
    for stage in stages:
        for path, is_anchor in (
            (stage / "02_clip_schemas.jsonl", False),
            (stage / "02b_anchor_clip_schemas.jsonl", True),
        ):
            try:
                rows = _rows(path)
            except (OSError, json.JSONDecodeError):
                parse_errors += 1
                continue
            for row in rows:
                usage = row.get("llm_usage") or {}
                zero = (
                    row.get("producer") == "qwen_clip_schema"
                    and int(usage.get("sampled_frame_count") or 0) <= 0
                )
                if row.get("model_error"):
                    if is_anchor:
                        anchor_errors += 1
                    else:
                        primary_errors += 1
                elif zero:
                    if is_anchor:
                        anchor_errors += 1
                    else:
                        primary_zero += 1
                elif is_anchor and not _schema_valid(row):
                    anchor_errors += 1
                elif is_anchor:
                    anchor_valid += 1
                else:
                    primary_valid += 1

    finals = list(root.glob("start_*/stages/*/04_l1_example.json"))
    contract_valid = 0
    for path in finals:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        metadata = row.get("metadata") or {}
        schemas = metadata.get("clip_schemas") or []
        stats = ((metadata.get("clue_memory_graph") or {}).get("index_stats") or {})
        fine_count = int(stats.get("fine_clip_count") or 0)
        contract_valid += int(
            metadata.get("l1_perception_protocol") == PROTOCOL
            and fine_count > 0
            and len(schemas) == fine_count
            and all(isinstance(schema, Mapping) and _schema_valid(schema) for schema in schemas)
        )

    disk = shutil.disk_usage(filesystem_path)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage_count": len(stages),
        "primary_valid_schemas": primary_valid,
        "expected_primary_schemas": expected_schemas,
        "primary_progress": round(primary_valid / expected_schemas, 6),
        "primary_model_errors": primary_errors,
        "primary_zero_frame": primary_zero,
        "anchor_valid_schemas": anchor_valid,
        "anchor_errors": anchor_errors,
        "parse_errors": parse_errors,
        "raw_l1": len(finals),
        "contract_valid_l1": contract_valid,
        "disk_free_gib": round(disk.free / 1024**3, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--filesystem-path", type=Path, default=Path("."))
    parser.add_argument("--expected-schemas", type=int, default=16589)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        print(
            json.dumps(snapshot(args.root, args.filesystem_path, args.expected_schemas), sort_keys=True),
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())

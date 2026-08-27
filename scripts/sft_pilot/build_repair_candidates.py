#!/usr/bin/env python3
"""Build prioritized repair candidate ids from a quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_done(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.name == "repair_report.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("reports") or []:
                if isinstance(row, dict) and row.get("example_id"):
                    done.add(str(row["example_id"]))
        elif path.is_dir():
            for report_path in path.glob("**/repair_05_report.json"):
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                if payload.get("example_id"):
                    done.add(str(payload["example_id"]))
    return done


def _bucket(row: dict[str, Any]) -> int:
    l2 = row.get("L2_status") if isinstance(row.get("L2_status"), dict) else {}
    l1 = row.get("L1_quality") if isinstance(row.get("L1_quality"), dict) else {}
    status = l2.get("acceptance_status")
    correct = l2.get("correct_eval_only")
    grade = l1.get("grade")
    if status == "accepted_weak" and correct is True:
        return 0
    if status == "rejected" and grade in {"medium", "high"}:
        return 1
    if status in {"accepted_weak", "accepted_strong"} and correct is False:
        return 2
    if grade in {"medium", "high"}:
        return 3
    return 9


def build_candidates(
    quality_report: Path,
    output: Path,
    *,
    datasets: set[str],
    done_paths: list[Path],
    limit: int,
) -> dict[str, Any]:
    payload = json.loads(quality_report.read_text(encoding="utf-8"))
    done = _load_done(done_paths)
    rows = payload.get("reports") if isinstance(payload, dict) else payload
    buckets: dict[int, list[tuple[str, dict[str, Any]]]] = {i: [] for i in range(10)}
    for row in rows or []:
        if not isinstance(row, dict) or not row.get("repair_needed"):
            continue
        dataset = str(row.get("dataset") or "")
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in done:
            continue
        if datasets and dataset not in datasets:
            continue
        b = _bucket(row)
        if b < 9:
            buckets[b].append((example_id, row))
    chosen: list[str] = []
    seen: set[str] = set()
    bucket_counts: dict[str, int] = {}
    for bucket_id in sorted(buckets):
        for example_id, _row in buckets[bucket_id]:
            if example_id in seen:
                continue
            chosen.append(example_id)
            seen.add(example_id)
            bucket_counts[str(bucket_id)] = bucket_counts.get(str(bucket_id), 0) + 1
            if limit and len(chosen) >= limit:
                break
        if limit and len(chosen) >= limit:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")
    summary = {
        "quality_report": str(quality_report),
        "output": str(output),
        "datasets": sorted(datasets),
        "done_examples": len(done),
        "candidate_count": len(chosen),
        "bucket_counts": bucket_counts,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--done-path", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_candidates(
        args.quality_report,
        args.output,
        datasets=set(args.dataset),
        done_paths=args.done_path,
        limit=args.limit,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

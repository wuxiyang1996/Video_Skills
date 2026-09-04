#!/usr/bin/env python3
"""Summarize terminal_sample JSON events into dataset-aware gate metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_terminal_samples(paths: Iterable[Path]) -> list[dict[str, Any]]:
    samples = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "terminal_sample":
                samples.append(payload)
    return samples


def minimum_verified_terminal_success(sample: dict[str, Any]) -> bool:
    if bool(sample.get("terminal_success")):
        return True
    diagnostic = sample.get("rollout_diagnostic") or {}
    return bool(
        sample.get("acceptance_status") in {"accepted_weak", "accepted_bridge"}
        and sample.get("answer_correct")
        and sample.get("verifier_passed")
        and sample.get("process_supported")
        and sample.get("format_budget_compliant", True)
        and int(diagnostic.get("min_support_refs") or 0) > 0
        and int(diagnostic.get("support_ref_count") or 0)
        >= int(diagnostic.get("min_support_refs") or 0)
        and int(diagnostic.get("trace_fail") or 0) == 0
    )


def summarize(
    samples: list[dict[str, Any]], *, reclassify_minimum_verified: bool = False
) -> dict[str, Any]:
    if reclassify_minimum_verified:
        samples = [
            {**row, "terminal_success": minimum_verified_terminal_success(row)}
            for row in samples
        ]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_dataset[str(sample.get("dataset") or "unknown")].append(sample)
    dataset_metrics = {}
    for dataset, rows in sorted(by_dataset.items()):
        by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_group[int(row.get("group") or 0)].append(row)
        trainable = 0
        for group_rows in by_group.values():
            rewards = [float(row.get("reward") or 0.0) for row in group_rows]
            if any(bool(row.get("terminal_success")) for row in group_rows) and len(set(rewards)) > 1:
                trainable += 1
        status = Counter(str(row.get("acceptance_status") or "unknown") for row in rows)
        component_names = sorted({name for row in rows for name in (row.get("reward_components") or {})})
        dataset_metrics[dataset] = {
            "groups_seen": len(by_group),
            "groups_trainable": trainable,
            "trainable_group_rate": trainable / max(1, len(by_group)),
            "samples": len(rows),
            "terminal_successes": sum(bool(row.get("terminal_success")) for row in rows),
            "terminal_success_rate": sum(bool(row.get("terminal_success")) for row in rows) / max(1, len(rows)),
            "valid_retrieval_action_rate": sum(
                str(row.get("acceptance_status") or "") != "invalid_retrieval_action" for row in rows
            ) / max(1, len(rows)),
            "answer_accuracy": sum(bool(row.get("answer_correct")) for row in rows) / max(1, len(rows)),
            "verifier_pass_rate": sum(bool(row.get("verifier_passed")) for row in rows) / max(1, len(rows)),
            "acceptance_status_counts": dict(status),
            "mean_reward_components": {
                name: sum(float((row.get("reward_components") or {}).get(name, 0.0)) for row in rows)
                / max(1, len(rows))
                for name in component_names
            },
        }
    total = sum(len(rows) for rows in by_dataset.values())
    successes = sum(bool(row.get("terminal_success")) for row in samples)
    return {
        "schema_version": "video-skills/l2-terminal-sample-summary-v0.1",
        "samples": total,
        "terminal_successes": successes,
        "terminal_success_rate": successes / max(1, total),
        "reclassified_minimum_verified": bool(reclassify_minimum_verified),
        "dataset_metrics": dataset_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reclassify-minimum-verified", action="store_true")
    args = parser.parse_args()
    report = summarize(
        read_terminal_samples(args.logs),
        reclassify_minimum_verified=args.reclassify_minimum_verified,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["samples"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

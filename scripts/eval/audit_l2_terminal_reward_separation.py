#!/usr/bin/env python3
"""Audit empirical separation of terminal success, wrong answer, and insufficient evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "terminal_sample":
                yield row


def category(row: dict[str, Any]) -> str:
    if bool(row.get("terminal_success")):
        return "correct_verified"
    if bool(row.get("process_supported")) and bool(row.get("answer_correct")):
        return "correct_uncommitted_or_rejected"
    if bool(row.get("process_supported")):
        return "incorrect_or_rejected"
    return "evidence_insufficient"


def audit(
    events: Iterable[dict[str, Any]],
    *,
    min_samples: int,
    terminal_reward_contracts: list[str] | None = None,
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in events:
        values[str(row.get("dataset") or "unknown")][category(row)].append(float(row["reward"]))
    dataset_metrics = {}
    checks = {}
    all_categories: set[str] = set()
    for dataset, buckets in sorted(values.items()):
        all_categories.update(buckets)
        summary = {
            name: {
                "samples": len(rewards),
                "mean_reward": sum(rewards) / len(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
            }
            for name, rewards in sorted(buckets.items())
        }
        dataset_metrics[dataset] = {
            "samples": sum(len(rewards) for rewards in buckets.values()),
            "categories": summary,
        }
        checks[f"{dataset}:min_samples"] = dataset_metrics[dataset]["samples"] >= min_samples
        checks[f"{dataset}:has_success"] = bool(buckets.get("correct_verified"))
        checks[f"{dataset}:has_insufficient"] = bool(buckets.get("evidence_insufficient"))
        if buckets.get("correct_verified") and buckets.get("evidence_insufficient"):
            checks[f"{dataset}:success_above_insufficient"] = (
                summary["correct_verified"]["mean_reward"]
                > summary["evidence_insufficient"]["mean_reward"]
            )
        if buckets.get("incorrect_or_rejected"):
            checks[f"{dataset}:incorrect_below_success"] = (
                summary["incorrect_or_rejected"]["mean_reward"]
                < summary["correct_verified"]["mean_reward"]
            )
            checks[f"{dataset}:incorrect_distinct_from_insufficient"] = (
                summary["incorrect_or_rejected"]["mean_reward"]
                != summary["evidence_insufficient"]["mean_reward"]
            )
        if buckets.get("correct_uncommitted_or_rejected"):
            checks[f"{dataset}:uncommitted_below_success"] = (
                summary["correct_uncommitted_or_rejected"]["mean_reward"]
                < summary["correct_verified"]["mean_reward"]
            )
    checks["all_three_outcomes_observed"] = {
        "correct_verified", "incorrect_or_rejected", "evidence_insufficient"
    }.issubset(all_categories)
    contracts = terminal_reward_contracts or []
    if terminal_reward_contracts is not None:
        checks["terminal_reward_contract_present"] = bool(contracts) and all(contracts)
        checks["terminal_reward_contract_consistent"] = (
            bool(contracts) and len(set(contracts)) == 1
        )
    return {
        "schema_version": "video-skills/l2-terminal-reward-separation-audit-v1",
        "passed": bool(dataset_metrics) and all(checks.values()),
        "checks": checks,
        "terminal_reward_contract": contracts[0] if contracts and len(set(contracts)) == 1 else None,
        "dataset_metrics": dataset_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-log", action="append", type=Path, required=True)
    parser.add_argument("--terminal-report", action="append", type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contracts = None
    if args.terminal_report is not None:
        contracts = [
            str(json.loads(path.read_text(encoding="utf-8")).get("terminal_reward_contract") or "")
            for path in args.terminal_report
        ]
    report = audit(
        rows(args.sample_log),
        min_samples=args.min_samples,
        terminal_reward_contracts=contracts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed gate for a completed paper L2 GRPO training pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = ("cg_bench", "video_holmes")
EXPECTED_TERMINAL_REWARD_CONTRACT = (
    "dataset-aware-terminal-reward:structured-concept-overlap-v2:"
    "verified-query-finalizer-repair-v1"
)


def audit_training_pool(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("dataset_metrics") or {}
    pool_filters = report.get("pool_filters") or {}
    rows = {dataset: metrics.get(dataset) or {} for dataset in DATASETS}
    groups = {dataset: int(rows[dataset].get("groups_seen") or 0) for dataset in DATASETS}
    checks = {
        "trained_artifact": report.get("artifact_status") == "trained"
        and bool((report.get("trained_adapter_outputs") or {}).get("default")),
        "balanced_50_to_100_groups_per_dataset": (
            50 <= groups["cg_bench"] <= 100
            and groups["cg_bench"] == groups["video_holmes"]
            and int(report.get("groups_seen") or 0) == sum(groups.values())
        ),
        "cg_trainable_group_rate_at_least_25pct": float(
            rows["cg_bench"].get("trainable_group_rate") or 0.0
        ) >= 0.25,
        "vh_trainable_group_rate_at_least_25pct": float(
            rows["video_holmes"].get("trainable_group_rate") or 0.0
        ) >= 0.25,
        "optimizer_updated_both_datasets": all(
            int(rows[dataset].get("groups_trained") or 0) > 0 for dataset in DATASETS
        ),
        "current_terminal_reward_contract": (
            report.get("terminal_reward_contract") == EXPECTED_TERMINAL_REWARD_CONTRACT
        ),
        "frozen_exact_balanced_training_pool": (
            report.get("split_role") == "grpo_pool"
            and pool_filters.get("exact_mined_group_allowlist") is True
            and pool_filters.get("preserve_allowlist_order") is True
            and pool_filters.get("dataset_balanced_sampling") is True
            and bool(pool_filters.get("example_id_allowlist_sha256"))
            and bool(report.get("split_manifest_sha256"))
        ),
    }
    return {
        "schema_version": "video-skills/l2-grpo-training-pool-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "terminal_reward_contract": report.get("terminal_reward_contract"),
        "expected_terminal_reward_contract": EXPECTED_TERMINAL_REWARD_CONTRACT,
        "dataset_metrics": {
            dataset: {
                "groups_seen": groups[dataset],
                "groups_trainable": int(rows[dataset].get("groups_trainable") or 0),
                "groups_trained": int(rows[dataset].get("groups_trained") or 0),
                "trainable_group_rate": float(rows[dataset].get("trainable_group_rate") or 0.0),
            }
            for dataset in DATASETS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.training_report.read_text(encoding="utf-8"))
        gate = audit_training_pool(report)
    except (OSError, json.JSONDecodeError) as error:
        gate = {
            "schema_version": "video-skills/l2-grpo-training-pool-gate-v1",
            "passed": False,
            "checks": {"training_report_valid": False},
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    gate["training_report"] = str(args.training_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

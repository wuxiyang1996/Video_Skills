#!/usr/bin/env python3
"""Fail-closed release gate before any official L2 heldout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = ("cg_bench", "video_holmes")


def _selection_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy mining and terminal-consensus selection reports."""
    if report.get("schema_version") == "video-skills/l2-terminal-consensus-group-selection-v1":
        metrics = report.get("dataset_metrics") or {}
        groups = {
            dataset: int((metrics.get(dataset) or {}).get("selected") or 0)
            for dataset in DATASETS
        }
        return {
            "passed": report.get("passed") is True,
            "groups": groups,
            "row_count": sum(groups.values()),
            "reported_row_count": 2 * int(report.get("target_per_dataset") or 0),
            "balanced": len(set(groups.values())) == 1 and all(groups.values()),
            "ordering_contract": report.get("ordering_contract"),
            "training_pool_only": report.get(
                "selection_uses_training_pool_terminal_outcomes_only"
            ) is True,
        }
    selection = report.get("allowlist_selection") or {}
    groups = selection.get("groups_by_dataset") or {}
    return {
        "passed": bool(selection),
        "groups": {dataset: int(groups.get(dataset) or 0) for dataset in DATASETS},
        "row_count": int(selection.get("groups") or 0),
        "reported_row_count": sum(int(groups.get(dataset) or 0) for dataset in DATASETS),
        "balanced": selection.get("balanced_datasets") is True,
        "ordering_contract": selection.get("ordering_contract"),
        "training_pool_only": True,
    }


def audit_pretest_release(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mining = artifacts["mining"]
    selection = _selection_summary(mining)
    groups = selection["groups"]
    aggregate = artifacts["three_seed_aggregate"]
    aggregate_contracts = [
        ((seed.get("contracts") or {}).get("terminal_reward_contract"))
        for seed in aggregate.get("seeds") or []
    ]
    reward_contract = artifacts["reward_separation"].get("terminal_reward_contract")
    opd_terminal_contract = (
        (artifacts["opd_terminal_selection"].get("selected") or {}).get(
            "terminal_reward_contract"
        )
    )

    checks = {
        "split_manifest_video_exclusive": artifacts["split_audit"].get("passed") is True,
        "terminal_reward_separates_outcomes": artifacts["reward_separation"].get("passed") is True,
        "same_terminal_reward_contract": bool(reward_contract)
        and reward_contract == opd_terminal_contract
        and len(aggregate_contracts) == 3
        and all(contract == reward_contract for contract in aggregate_contracts),
        "opd_pointwise_checkpoint_selected": artifacts["opd_selection"].get("passed") is True
        and bool(artifacts["opd_selection"].get("selected")),
        "opd_terminal_checkpoint_qualified": artifacts["opd_terminal_selection"].get("passed") is True
        and bool(artifacts["opd_terminal_selection"].get("selected")),
        "selection_report_passed": selection["passed"],
        "selection_training_pool_only": selection["training_pool_only"],
        "mining_balanced": selection["balanced"] is True
        and groups.get("cg_bench") == groups.get("video_holmes"),
        "mining_50_to_100_groups_per_dataset": all(
            50 <= int(groups.get(dataset) or 0) <= 100 for dataset in DATASETS
        ),
        "mining_row_count_consistent": selection["row_count"]
        == selection["reported_row_count"],
        "mining_round_robin_order": selection["ordering_contract"]
        == "dataset-round-robin-v1",
        "pilot_pointwise_gate": artifacts["pilot_pointwise_gate"].get("passed") is True,
        "pilot_cg_terminal_gate": artifacts["pilot_cg_gate"].get("passed") is True,
        "pilot_vh_terminal_gate": artifacts["pilot_vh_gate"].get("passed") is True,
        "three_seed_aggregate": aggregate.get("passed") is True,
        "exactly_three_seeds": aggregate.get("seed_count") == 3
        and len(aggregate.get("seeds") or []) == 3,
        "same_training_contracts": aggregate.get("same_training_contracts") is True,
        "all_seed_gates_passed": len(aggregate.get("seeds") or []) == 3
        and all(seed.get("passed") is True for seed in aggregate.get("seeds") or []),
    }
    return {
        "schema_version": "video-skills/l2-paper-pretest-release-v1",
        "passed": all(checks.values()),
        "checks": checks,
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_invalid": str(path), "error": str(exc)}
    return value if isinstance(value, dict) else {"_invalid": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "split_audit", "reward_separation", "opd_selection", "opd_terminal_selection",
        "mining", "pilot_pointwise_gate", "pilot_cg_gate", "pilot_vh_gate",
        "three_seed_aggregate",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = (
        "split_audit", "reward_separation", "opd_selection", "opd_terminal_selection",
        "mining", "pilot_pointwise_gate", "pilot_cg_gate", "pilot_vh_gate",
        "three_seed_aggregate",
    )
    paths = {name: getattr(args, name) for name in names}
    report = audit_pretest_release({name: _load(path) for name, path in paths.items()})
    report["artifacts"] = {name: str(path) for name, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

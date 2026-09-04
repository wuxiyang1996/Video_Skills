#!/usr/bin/env python3
"""Audit dataset-local GRPO reward components and group normalization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from trainer.grpo.objective import centered_group_advantages


REQUIRED_COMPONENTS = {
    "cg_bench": {"clue_recall", "clue_mean_best_iou", "evidence_precision"},
    "video_holmes": {
        "segment_recall",
        "segment_precision",
        "inference_shot_recall",
        "relationship_support",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}")
            rows.append(row)
    return rows


def _dataset_from_example_id(example_id: str) -> str:
    if example_id.startswith("cg_bench:"):
        return "cg_bench"
    if example_id.startswith("video_holmes:"):
        return "video_holmes"
    return ""


def _close_lists(left: Iterable[float], right: Iterable[float], *, atol: float = 1e-8) -> bool:
    lhs = [float(value) for value in left]
    rhs = [float(value) for value in right]
    return len(lhs) == len(rhs) and all(math.isclose(a, b, abs_tol=atol, rel_tol=0.0) for a, b in zip(lhs, rhs))


def audit_seed(name: str, report: Mapping[str, Any], metrics: list[Mapping[str, Any]]) -> dict[str, Any]:
    dataset_metrics = report.get("dataset_metrics") or {}
    datasets = [_dataset_from_example_id(str(row.get("example_id") or "")) for row in metrics]
    recomputed = [
        centered_group_advantages([float(value) for value in (row.get("rewards") or [])])
        for row in metrics
    ]
    stored = [[float(value) for value in (row.get("advantages") or [])] for row in metrics]
    group_counts = {dataset: datasets.count(dataset) for dataset in REQUIRED_COMPONENTS}
    report_seen_counts = {
        dataset: int((dataset_metrics.get(dataset) or {}).get("groups_seen") or 0)
        for dataset in REQUIRED_COMPONENTS
    }
    report_trained_counts = {
        dataset: int((dataset_metrics.get(dataset) or {}).get("groups_trained") or 0)
        for dataset in REQUIRED_COMPONENTS
    }
    component_keys = {
        dataset: set(((dataset_metrics.get(dataset) or {}).get("mean_reward_components") or {}).keys())
        for dataset in REQUIRED_COMPONENTS
    }
    trainable_rates = {
        dataset: float((dataset_metrics.get(dataset) or {}).get("trainable_group_rate") or 0.0)
        for dataset in REQUIRED_COMPONENTS
    }
    checks = {
        "only_expected_datasets": bool(metrics) and set(datasets) == set(REQUIRED_COMPONENTS),
        "one_dataset_per_group": all(bool(dataset) for dataset in datasets),
        "dataset_balanced_input_group_counts": len(set(report_seen_counts.values())) == 1,
        "stored_update_groups_match_report": group_counts == report_trained_counts,
        "dataset_specific_components_reported": all(
            REQUIRED_COMPONENTS[dataset] <= component_keys[dataset] for dataset in REQUIRED_COMPONENTS
        ),
        "stored_advantages_match_mean_std_normalization": all(
            _close_lists(expected, actual) for expected, actual in zip(recomputed, stored)
        ),
        "all_groups_have_multiple_rollouts": all(len(row.get("rewards") or []) >= 2 for row in metrics),
        "trainable_group_rate_at_least_25pct": all(rate >= 0.25 for rate in trainable_rates.values()),
    }
    return {
        "name": name,
        "passed": all(checks.values()),
        "checks": checks,
        "stored_update_group_counts": group_counts,
        "report_seen_group_counts": report_seen_counts,
        "report_trained_group_counts": report_trained_counts,
        "trainable_group_rates": trainable_rates,
        "reward_component_keys": {dataset: sorted(keys) for dataset, keys in component_keys.items()},
    }


def audit_runs(specs: Iterable[tuple[str, Path, Path]]) -> dict[str, Any]:
    seeds = []
    for name, report_path, metrics_path in specs:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError(f"expected JSON object in {report_path}")
        seed = audit_seed(name, report, _read_jsonl(metrics_path))
        seed["report"] = str(report_path.resolve())
        seed["metrics"] = str(metrics_path.resolve())
        seeds.append(seed)
    checks = {
        "at_least_one_run": bool(seeds),
        "all_runs_passed": bool(seeds) and all(seed["passed"] for seed in seeds),
        "same_normalization_contract": True,
    }
    return {
        "schema_version": "video-skills/l2-reward-normalization-audit-v1",
        "passed": all(checks.values()),
        "normalization_contract": "dataset-homogeneous-group-mean-std-v1",
        "checks": checks,
        "runs": seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME|TERMINAL_GRPO_REPORT|TERMINAL_METRICS_JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs: list[tuple[str, Path, Path]] = []
    for value in args.run:
        parts = value.split("|", 2)
        if len(parts) != 3 or not all(parts):
            parser.error(f"invalid --run value: {value}")
        specs.append((parts[0], Path(parts[1]), Path(parts[2])))
    result = audit_runs(specs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

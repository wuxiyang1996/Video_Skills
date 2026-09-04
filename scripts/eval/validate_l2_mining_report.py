#!/usr/bin/env python3
"""Fail closed unless a mined GRPO pool matches the training policy contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trainer.artifact_hash import adapter_weight_sha256


def parse_routes(values: Sequence[str]) -> dict[str, Path]:
    routes: dict[str, Path] = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not dataset.strip() or not path.strip():
            raise ValueError(f"dataset adapter must use DATASET=PATH: {value!r}")
        routes[dataset.strip()] = Path(path.strip())
    return routes


def validate_report(
    report: Mapping[str, Any],
    *,
    source_adapter: Path,
    dataset_adapters: Mapping[str, Path],
    controller_action_contract: str,
    relationship_support_contract: str,
    generation_temperature: float,
    pointwise_temperature: float | None,
    pointwise_datasets: Sequence[str],
    min_eligible_per_dataset: int,
    min_eligible_group_rate: float,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["split_role_grpo_pool"] = report.get("split_role") == "grpo_pool"
    checks["source_adapter_hash"] = report.get(
        "source_adapter_weight_sha256"
    ) == adapter_weight_sha256(source_adapter)
    mined_routes = report.get("dataset_adapter_backends") or {}
    for dataset, path in dataset_adapters.items():
        checks[f"dataset_adapter_hash:{dataset}"] = (
            (mined_routes.get(dataset) or {}).get("adapter_weight_sha256")
            == adapter_weight_sha256(path)
        )
    checks["controller_action_contract"] = (
        report.get("controller_action_contract") == controller_action_contract
    )
    checks["relationship_support_contract"] = (
        report.get("relationship_support_contract") == relationship_support_contract
    )
    protocol = report.get("sampling_protocol") or {}
    checks["generation_temperature"] = abs(
        float(protocol.get("generation_temperature", float("inf")))
        - float(generation_temperature)
    ) < 1e-9
    if pointwise_temperature is not None:
        checks["pointwise_temperature"] = abs(
            float(protocol.get("pointwise_temperature", float("inf")))
            - float(pointwise_temperature)
        ) < 1e-9
    checks["pointwise_datasets"] = sorted(report.get("pointwise_action_datasets") or []) == sorted(
        str(value) for value in pointwise_datasets
    )
    for dataset in ("cg_bench", "video_holmes"):
        count = int(
            (((report.get("dataset_metrics") or {}).get(dataset) or {}).get("groups_eligible"))
            or 0
        )
        checks[f"eligible_groups:{dataset}"] = count >= int(min_eligible_per_dataset)
        rate = float(
            (((report.get("dataset_metrics") or {}).get(dataset) or {}).get("eligible_group_rate"))
            or 0.0
        )
        checks[f"eligible_group_rate:{dataset}"] = rate >= float(min_eligible_group_rate)
    selection = report.get("allowlist_selection") or {}
    selected_counts = selection.get("groups_by_dataset") or {}
    cg_selected = int(selected_counts.get("cg_bench") or 0)
    vh_selected = int(selected_counts.get("video_holmes") or 0)
    selection_cap = int(selection.get("max_groups_per_dataset") or 0)
    checks["allowlist_balanced_contract"] = (
        selection.get("balanced_datasets") is True
        and selection.get("ordering_contract") == "dataset-round-robin-v1"
    )
    checks["allowlist_balanced_counts"] = (
        cg_selected == vh_selected
        and cg_selected >= int(min_eligible_per_dataset)
        and (selection_cap <= 0 or cg_selected <= selection_cap)
        and int(selection.get("balanced_target_per_dataset") or 0) == cg_selected
        and int(selection.get("groups") or 0) == cg_selected + vh_selected
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--dataset-adapter", action="append", default=[])
    parser.add_argument("--controller-action-contract", required=True)
    parser.add_argument("--relationship-support-contract", required=True)
    parser.add_argument("--generation-temperature", type=float, required=True)
    parser.add_argument("--pointwise-temperature", type=float)
    parser.add_argument("--pointwise-datasets", default="")
    parser.add_argument("--min-eligible-per-dataset", type=int, required=True)
    parser.add_argument("--min-eligible-group-rate", type=float, default=0.25)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = validate_report(
        report,
        source_adapter=args.source_adapter,
        dataset_adapters=parse_routes(args.dataset_adapter),
        controller_action_contract=args.controller_action_contract,
        relationship_support_contract=args.relationship_support_contract,
        generation_temperature=args.generation_temperature,
        pointwise_temperature=args.pointwise_temperature,
        pointwise_datasets=[value for value in args.pointwise_datasets.split(",") if value],
        min_eligible_per_dataset=args.min_eligible_per_dataset,
        min_eligible_group_rate=args.min_eligible_group_rate,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 25


if __name__ == "__main__":
    raise SystemExit(main())

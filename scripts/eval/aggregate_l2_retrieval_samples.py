#!/usr/bin/env python3
"""Aggregate retrieval-only sample logs with the current evidence-hit rules."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def process_supported(row: dict[str, Any]) -> bool:
    dataset = str(row.get("dataset") or "")
    components = row.get("reward_components") or {}
    if dataset == "cg_bench":
        return float(components.get("clue_recall") or 0.0) > 0.0
    if dataset == "video_holmes":
        inference = float(components.get("inference_shot_recall") or 0.0) > 0.0
        relationship = float(components.get("relationship_support") or 0.0) >= 0.25
        question_type = str(row.get("question_type") or "SR").upper()
        if question_type in {"MHR", "IMC"}:
            return inference
        # SR and legacy rows without question_type use the conservative dual-evidence gate.
        return inference and relationship
    return bool(row.get("process_supported"))


def aggregate(rows: Iterable[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event") != "terminal_sample":
            continue
        key = (
            str(row.get("dataset") or "unknown"),
            str(row.get("example_id") or ""),
            int(row.get("repeat_index") or 0),
        )
        if key[1]:
            groups[key].append(row)

    group_rows = []
    for (dataset, example_id, repeat_index), samples in groups.items():
        rewards = [float(row.get("reward") or 0.0) for row in samples]
        process_hits = sum(process_supported(row) for row in samples)
        format_hits = sum(bool(row.get("format_budget_compliant")) for row in samples)
        group_rows.append({
            "dataset": dataset,
            "example_id": example_id,
            "repeat_index": repeat_index,
            "samples": len(samples),
            "reward_variance": len({round(value, 8) for value in rewards}) > 1,
            "process_supported_samples": process_hits,
            "format_compliant_samples": format_hits,
        })

    eligible = [
        row for row in group_rows
        if row["reward_variance"]
        and row["process_supported_samples"] > 0
        and row["format_compliant_samples"] > 0
    ]
    example_ids = list(dict.fromkeys(row["example_id"] for row in eligible))
    seen_by_dataset = Counter(row["dataset"] for row in group_rows)
    eligible_by_dataset = Counter(row["dataset"] for row in eligible)
    dataset_metrics = {
        dataset: {
            "groups_seen": seen,
            "groups_eligible": eligible_by_dataset[dataset],
            "eligible_group_rate": eligible_by_dataset[dataset] / max(1, seen),
        }
        for dataset, seen in sorted(seen_by_dataset.items())
    }
    report = {
        "schema_version": "video-skills/l2-retrieval-sample-aggregation-v0.1",
        "groups_seen": len(group_rows),
        "groups_eligible": len(eligible),
        "eligible_group_rate": len(eligible) / max(1, len(group_rows)),
        "unique_examples_selected": len(example_ids),
        "dataset_metrics": dataset_metrics,
        "criteria": {
            "reward_variance": True,
            "min_process_supported_samples": 1,
            "min_format_compliant_samples": 1,
            "cg_process": "clue_recall>0",
            "vh_sr_process": "inference_shot_recall>0 AND relationship_support>=0.25",
            "vh_mhr_imc_process": "inference_shot_recall>0",
        },
        "eligible_groups": eligible,
    }
    return example_ids, report


def read_json_events(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def select_allowlist_groups(
    eligible_groups: list[dict[str, Any]],
    *,
    max_groups_per_dataset: int | None,
    balanced_datasets: bool,
) -> tuple[list[dict[str, Any]], Counter[str], int | None]:
    """Select a deterministic allowlist, optionally equal and round-robin by dataset."""
    if not balanced_datasets:
        selected: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in eligible_groups:
            dataset = str(row["dataset"])
            if max_groups_per_dataset is not None and counts[dataset] >= max_groups_per_dataset:
                continue
            selected.append(row)
            counts[dataset] += 1
        return selected, counts, None

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_groups:
        buckets[str(row["dataset"])].append(row)
    if not buckets:
        return [], Counter(), 0
    target = min(len(rows) for rows in buckets.values())
    if max_groups_per_dataset is not None:
        target = min(target, max_groups_per_dataset)
    selected = [buckets[dataset][index] for index in range(target) for dataset in sorted(buckets)]
    return selected, Counter({dataset: target for dataset in buckets}), target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-log", action="append", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--run-report",
        type=Path,
        help="Runner report whose frozen policy/reward contracts are copied into the mining report.",
    )
    parser.add_argument(
        "--exact-group-allowlist",
        action="store_true",
        help="Write example_id<TAB>repeat_index rows instead of deduplicated example IDs.",
    )
    parser.add_argument(
        "--max-groups-per-dataset",
        type=int,
        help="Cap the emitted allowlist per dataset while retaining full mining statistics.",
    )
    parser.add_argument(
        "--balanced-datasets",
        action="store_true",
        help=(
            "Emit the same number of groups for every dataset, using the smallest "
            "eligible bucket (after the optional cap), in round-robin order."
        ),
    )
    args = parser.parse_args()
    example_ids, report = aggregate(read_json_events(args.sample_log))
    if args.run_report:
        run_report = json.loads(args.run_report.read_text(encoding="utf-8"))
        for field in (
            "source_adapter",
            "source_adapter_weight_sha256",
            "dataset_adapter_backends",
            "controller_action_contract",
            "pointwise_action_datasets",
            "sampling_protocol",
            "relationship_support_contract",
            "boundary_anchor_index0",
            "split_role",
        ):
            report[field] = run_report.get(field)
        report["source_run_report"] = str(args.run_report)
    args.allowlist.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.max_groups_per_dataset is not None and args.max_groups_per_dataset <= 0:
        parser.error("--max-groups-per-dataset must be positive")
    selected_groups, selected_counts, balanced_target = select_allowlist_groups(
        report["eligible_groups"],
        max_groups_per_dataset=args.max_groups_per_dataset,
        balanced_datasets=bool(args.balanced_datasets),
    )
    report["allowlist_selection"] = {
        "max_groups_per_dataset": args.max_groups_per_dataset,
        "balanced_datasets": bool(args.balanced_datasets),
        "balanced_target_per_dataset": balanced_target,
        "ordering_contract": (
            "dataset-round-robin-v1" if args.balanced_datasets else "eligible-log-order-v1"
        ),
        "groups_by_dataset": dict(selected_counts),
        "groups": len(selected_groups),
    }
    allowlist_values = (
        [f"{row['example_id']}\t{row['repeat_index']}" for row in selected_groups]
        if args.exact_group_allowlist
        else list(dict.fromkeys(row["example_id"] for row in selected_groups))
    )
    args.allowlist.write_text(
        "".join(f"{value}\n" for value in allowlist_values), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "eligible_groups"}, indent=2))
    return 0 if example_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())

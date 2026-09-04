#!/usr/bin/env python3
"""Select balanced terminal-capable GRPO groups from three training-pool probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("cg_bench", "video_holmes")


def failure_report(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "video-skills/l2-terminal-consensus-group-selection-v1",
        "passed": False,
        "checks": {"inputs_valid_and_complete": False},
        "error": {"type": type(error).__name__, "message": str(error)},
        "selection_uses_training_pool_terminal_outcomes_only": True,
    }


def _summarize_seed(
    rows: list[dict[str, Any]], *, samples_per_group: int
) -> dict[tuple[str, str, int], dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset") or ""),
            str(row.get("example_id") or ""),
            int(row.get("repeat_index") or 0),
        )
        grouped[key].append(row)
    result = {}
    for key, samples in grouped.items():
        if len(samples) != samples_per_group:
            raise ValueError(f"incomplete group {key}: {len(samples)} != {samples_per_group}")
        rewards = {round(float(row.get("reward") or 0.0), 8) for row in samples}
        successes = sum(bool(row.get("terminal_success")) for row in samples)
        result[key] = {
            "group": int(samples[0].get("group") or 0),
            "successes": successes,
            "reward_variance": len(rewards) > 1,
            "process_hit": any(bool(row.get("process_supported")) for row in samples),
            "format_compliant": all(bool(row.get("format_budget_compliant", True)) for row in samples),
            "trainable": bool(successes) and len(rewards) > 1,
        }
    return result


def select_terminal_consensus_groups(
    seed_rows: dict[int, list[dict[str, Any]]],
    *,
    samples_per_group: int = 8,
    target_per_dataset: int = 50,
    min_predicted_trainable_rate: float = 0.25,
) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    if len(seed_rows) != 3:
        raise ValueError("exactly three seed probes are required")
    summaries = {
        seed: _summarize_seed(rows, samples_per_group=samples_per_group)
        for seed, rows in sorted(seed_rows.items())
    }
    key_sets = [set(value) for value in summaries.values()]
    if not key_sets[0] or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("seed probes must contain the same complete group keys")
    seeds = sorted(summaries)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in key_sets[0]:
        dataset, example_id, repeat_index = key
        values = [summaries[seed][key] for seed in seeds]
        total_successes = sum(int(value["successes"]) for value in values)
        pooled_p = total_successes / (len(seeds) * samples_per_group)
        predicted = 1.0 - (1.0 - pooled_p) ** samples_per_group - pooled_p ** samples_per_group
        candidates[dataset].append({
            "key": key,
            "example_id": example_id,
            "repeat_index": repeat_index,
            "source_order": min(int(value["group"]) for value in values),
            "seed_trainable_hits": sum(bool(value["trainable"]) for value in values),
            "reward_variance_seed_count": sum(bool(value["reward_variance"]) for value in values),
            "process_hit_seed_count": sum(bool(value["process_hit"]) for value in values),
            "format_seed_count": sum(bool(value["format_compliant"]) for value in values),
            "terminal_successes": total_successes,
            "predicted_trainable_probability": predicted,
            "trainable_by_seed": {str(seed): bool(summaries[seed][key]["trainable"]) for seed in seeds},
        })

    selected_by_dataset = {}
    for dataset in DATASETS:
        ranked = sorted(
            candidates.get(dataset) or [],
            key=lambda row: (
                -int(row["seed_trainable_hits"]),
                -float(row["predicted_trainable_probability"]),
                -int(row["process_hit_seed_count"]),
                -int(row["reward_variance_seed_count"]),
                int(row["source_order"]),
                str(row["example_id"]),
                int(row["repeat_index"]),
            ),
        )
        selected_by_dataset[dataset] = ranked[:target_per_dataset]

    allowlist: list[tuple[str, int]] = []
    for index in range(target_per_dataset):
        for dataset in DATASETS:
            rows = selected_by_dataset.get(dataset) or []
            if index < len(rows):
                allowlist.append((str(rows[index]["example_id"]), int(rows[index]["repeat_index"])))

    dataset_reports = {}
    for dataset in DATASETS:
        rows = selected_by_dataset.get(dataset) or []
        dataset_reports[dataset] = {
            "candidates": len(candidates.get(dataset) or []),
            "selected": len(rows),
            "terminal_capable_union": sum(int(row["seed_trainable_hits"]) > 0 for row in rows),
            "terminal_consensus2": sum(int(row["seed_trainable_hits"]) >= 2 for row in rows),
            "mean_predicted_trainable_rate": (
                sum(float(row["predicted_trainable_probability"]) for row in rows) / max(1, len(rows))
            ),
            "observed_trainable_rate_by_seed": {
                str(seed): sum(bool(row["trainable_by_seed"][str(seed)]) for row in rows) / max(1, len(rows))
                for seed in seeds
            },
        }
    checks = {
        "target_between_50_and_100": 50 <= target_per_dataset <= 100,
        "enough_candidates": all(
            len(candidates.get(dataset) or []) >= target_per_dataset for dataset in DATASETS
        ),
        "exact_dataset_balance": all(
            len(selected_by_dataset.get(dataset) or []) == target_per_dataset for dataset in DATASETS
        ),
        "predicted_trainable_rate_at_least_threshold": all(
            dataset_reports[dataset]["mean_predicted_trainable_rate"]
            >= min_predicted_trainable_rate
            for dataset in DATASETS
        ),
        "observed_trainable_rate_each_seed_at_least_threshold": all(
            float(dataset_reports[dataset]["observed_trainable_rate_by_seed"][str(seed)])
            >= min_predicted_trainable_rate
            for dataset in DATASETS
            for seed in seeds
        ),
        "round_robin_rows_complete": len(allowlist) == 2 * target_per_dataset,
    }
    report = {
        "schema_version": "video-skills/l2-terminal-consensus-group-selection-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seeds": seeds,
        "samples_per_group": samples_per_group,
        "target_per_dataset": target_per_dataset,
        "min_predicted_trainable_rate": min_predicted_trainable_rate,
        "ordering_contract": "dataset-round-robin-v1",
        "dataset_metrics": dataset_reports,
        "selection_uses_training_pool_terminal_outcomes_only": True,
    }
    return allowlist, report


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", required=True, help="SEED|TERMINAL_SAMPLES_JSONL")
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--target-per-dataset", type=int, default=50)
    parser.add_argument("--min-predicted-trainable-rate", type=float, default=0.25)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--source-provenance", type=Path,
        help="Optional passed train-only replay/mining report that produced the sample logs.",
    )
    args = parser.parse_args()
    try:
        seed_rows = {}
        for raw in args.seed:
            seed_text, path_text = raw.split("|", 1)
            seed_rows[int(seed_text)] = _read(Path(path_text))
        allowlist, report = select_terminal_consensus_groups(
            seed_rows,
            samples_per_group=args.samples_per_group,
            target_per_dataset=args.target_per_dataset,
            min_predicted_trainable_rate=args.min_predicted_trainable_rate,
        )
        if args.source_provenance is not None:
            provenance = json.loads(args.source_provenance.read_text(encoding="utf-8"))
            if provenance.get("passed") is not True:
                raise ValueError(f"source provenance did not pass: {args.source_provenance}")
            report["source_provenance"] = {
                "report": str(args.source_provenance),
                "sha256": hashlib.sha256(args.source_provenance.read_bytes()).hexdigest(),
                "schema_version": provenance.get("schema_version"),
                "selection_uses_training_pool_only": provenance.get(
                    "selection_uses_training_pool_only"
                ) is True,
            }
            report["checks"]["source_provenance_train_only"] = (
                report["source_provenance"]["selection_uses_training_pool_only"] is True
            )
            report["passed"] = all(report["checks"].values())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        allowlist, report = [], failure_report(error)
    args.allowlist.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    allowlist_text = "".join(
        f"{example_id}\t{repeat_index}\n" for example_id, repeat_index in allowlist
    )
    args.allowlist.write_text(allowlist_text, encoding="utf-8")
    report["allowlist_artifact"] = {
        "path": str(args.allowlist),
        "rows": len(allowlist),
        "sha256": hashlib.sha256(allowlist_text.encode("utf-8")).hexdigest(),
    }
    if report.get("passed") is True:
        expected_rows = 2 * int(report["target_per_dataset"])
        report["checks"]["allowlist_artifact_complete"] = (
            len(allowlist) == expected_rows and bool(report["allowlist_artifact"]["sha256"])
        )
        report["passed"] = all(report["checks"].values())
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired video-level bootstrap intervals for the frozen heldout matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


MODELS = ("sft", "opd_alpha075", "grpo_seed42", "grpo_seed43", "grpo_seed44")
GRPO_MODELS = ("grpo_seed42", "grpo_seed43", "grpo_seed44")
DATASET_SPECS = {
    "cg_bench": {
        "topk_key": "pointwise_top2",
        "metrics": (
            "mean_recall",
            "hit_rate",
            "clue_recall",
            "evidence_precision",
            "clue_mean_best_iou",
        ),
    },
    "video_holmes": {
        "topk_key": "pointwise_top4",
        "metrics": (
            "mean_recall",
            "hit_rate",
            "segment_recall",
            "segment_precision",
            "inference_shot_recall",
            "relationship_support",
        ),
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _example_value(row: dict[str, Any], metric: str) -> float:
    if metric == "mean_recall":
        return float(row["metrics"]["recall"])
    if metric == "hit_rate":
        return float(bool(row["metrics"]["hit"]))
    return float(row["process_metrics"][metric])


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _interval(differences: list[float], samples: int, rng: random.Random) -> dict[str, Any]:
    count = len(differences)
    observed = sum(differences) / count
    boot = []
    for _ in range(samples):
        boot.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    boot.sort()
    low = _percentile(boot, 0.025)
    high = _percentile(boot, 0.975)
    return {
        "gain": observed,
        "gain_percentage_points": 100.0 * observed,
        "ci95": [low, high],
        "ci95_percentage_points": [100.0 * low, 100.0 * high],
        "ci95_excludes_zero": low > 0.0 or high < 0.0,
        "direction": "positive" if observed > 0.0 else "negative" if observed < 0.0 else "zero",
    }


def analyze(root: Path, samples: int, seed: int) -> dict[str, Any]:
    paths: dict[str, dict[str, Path]] = {}
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASET_SPECS:
        paths[dataset] = {}
        reports[dataset] = {}
        for model in MODELS:
            path = root / "results" / model / dataset / "eval_report.json"
            paths[dataset][model] = path
            reports[dataset][model] = _load(path)

    output: dict[str, Any] = {
        "schema_version": "video-skills/l2-heldout-paired-bootstrap-v1",
        "passed": True,
        "contract": {
            "resampling_unit": "heldout_video",
            "paired_across_models": True,
            "grpo_estimator": "per-video mean over seeds 42, 43, and 44",
            "confidence_interval": "two-sided percentile bootstrap 95%",
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "captures_training_seed_uncertainty": False,
            "performance_is_not_an_integrity_gate": True,
        },
        "sources": {},
        "datasets": {},
    }
    rng = random.Random(seed)
    for dataset, spec in DATASET_SPECS.items():
        dataset_reports = reports[dataset]
        reference = dataset_reports["sft"]
        reference_ids = [str(row["example_id"]) for row in reference["results"]]
        checks = {
            "nonempty_unique_example_ids": bool(reference_ids)
            and len(reference_ids) == len(set(reference_ids)),
            "same_example_ids_and_order": all(
                [str(row["example_id"]) for row in dataset_reports[model]["results"]]
                == reference_ids
                for model in MODELS
            ),
            "same_frozen_input_hash": len(
                {dataset_reports[model].get("evaluation_jsonl_sha256") for model in MODELS}
            )
            == 1,
            "dataset_identity": all(
                dataset_reports[model].get("input_datasets") == [dataset] for model in MODELS
            ),
        }
        if not all(checks.values()):
            output["passed"] = False
            output["datasets"][dataset] = {"checks": checks, "comparisons": {}}
            continue

        model_rows = {
            model: {str(row["example_id"]): row for row in dataset_reports[model]["results"]}
            for model in MODELS
        }
        comparisons: dict[str, Any] = {}
        for comparison, baseline in (
            ("opd_vs_sft", "sft"),
            ("grpo_mean_vs_sft", "sft"),
            ("grpo_mean_vs_opd", "opd_alpha075"),
        ):
            comparisons[comparison] = {}
            for metric in spec["metrics"]:
                differences = []
                for example_id in reference_ids:
                    baseline_value = _example_value(model_rows[baseline][example_id], metric)
                    if comparison == "opd_vs_sft":
                        treatment_value = _example_value(
                            model_rows["opd_alpha075"][example_id], metric
                        )
                    else:
                        treatment_value = sum(
                            _example_value(model_rows[model][example_id], metric)
                            for model in GRPO_MODELS
                        ) / len(GRPO_MODELS)
                    differences.append(treatment_value - baseline_value)
                comparisons[comparison][metric] = _interval(differences, samples, rng)

        output["datasets"][dataset] = {
            "examples": len(reference_ids),
            "checks": checks,
            "comparisons": comparisons,
        }
        output["sources"][dataset] = {
            model: {
                "path": str(paths[dataset][model].resolve()),
                "sha256": _sha256(paths[dataset][model]),
                "adapter_weight_sha256": dataset_reports[model].get("adapter_weight_sha256"),
            }
            for model in MODELS
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    report = analyze(args.heldout_root, args.bootstrap_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

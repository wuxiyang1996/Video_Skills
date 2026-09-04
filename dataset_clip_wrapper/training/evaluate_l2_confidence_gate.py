#!/usr/bin/env python3
"""Select a train-calibrated high-margin visual fallback for L2 top-2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .evaluate_l2_retrieval_adapter import retrieval_scores
from .sft_common import read_json, write_json


THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, math.inf)


def apply_gate(pointwise: dict[str, Any], visual: dict[str, Any], threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    visual_by_id = {str(row["example_id"]): row for row in visual.get("results") or []}
    results = []
    for point in pointwise.get("results") or []:
        example_id = str(point["example_id"])
        vision = visual_by_id.get(example_id)
        if vision is None:
            raise ValueError(f"Missing visual result: {example_id}")
        ranking = list(vision.get("ranking") or [])
        margin = float(ranking[0]["score"]) - float(ranking[1]["score"]) if len(ranking) >= 2 else math.inf
        use_visual = margin >= threshold
        predicted = list(vision["predicted"] if use_visual else point["predicted"])
        gold = [int(value) for value in point["gold"]]
        results.append({
            "example_id": example_id, "gold": gold, "predicted": predicted,
            "route": "visual" if use_visual else "pointwise", "visual_margin": margin,
            "metrics": retrieval_scores(predicted, gold),
        })
    values = [row["metrics"] for row in results]
    metrics = {
        "examples": len(values),
        "mean_precision": sum(float(row["precision"]) for row in values) / max(1, len(values)),
        "mean_recall": sum(float(row["recall"]) for row in values) / max(1, len(values)),
        "hit_rate": sum(bool(row["hit"]) for row in values) / max(1, len(values)),
        "exact_rate": sum(bool(row["exact"]) for row in values) / max(1, len(values)),
        "visual_routes": sum(row["route"] == "visual" for row in results),
    }
    return results, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pointwise", type=Path, required=True)
    parser.add_argument("--train-visual", type=Path, required=True)
    parser.add_argument("--dev-pointwise", type=Path, required=True)
    parser.add_argument("--dev-visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    train_pointwise, train_visual = read_json(args.train_pointwise), read_json(args.train_visual)
    calibration = []
    for threshold in THRESHOLDS:
        _, metrics = apply_gate(train_pointwise, train_visual, threshold)
        calibration.append({"threshold": threshold, **metrics})
    selected = max(
        calibration,
        key=lambda row: ((float(row["mean_recall"]) + float(row["hit_rate"])) / 2.0, float(row["threshold"])),
    )
    results, metrics = apply_gate(
        read_json(args.dev_pointwise), read_json(args.dev_visual), float(selected["threshold"])
    )
    report = {
        "schema_version": "video-skills/l2-confidence-gate-eval-v0.1",
        "train_only_calibration": True, "calibration": calibration,
        "selected_threshold": selected["threshold"], "metrics": metrics, "results": results,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

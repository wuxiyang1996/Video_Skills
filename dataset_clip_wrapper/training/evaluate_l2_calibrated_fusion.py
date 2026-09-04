#!/usr/bin/env python3
"""Fit train-only grouped calibration for text, vision, and retrieval L2 scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_l2_retrieval_adapter import retrieval_scores
from .sft_common import read_json, write_json


C_VALUES = (0.01, 0.1, 1.0, 10.0)
FEATURE_NAMES = ("pointwise", "visual_logit", "reciprocal_rank", "pointwise_x_visual")


def _visual_logit(score: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(score)))
    return math.log(value / (1.0 - value))


def joined_rows(
    pointwise: dict[str, Any],
    visual: dict[str, Any],
    *,
    allow_missing_pointwise: bool = False,
) -> list[dict[str, Any]]:
    point_by_example = {str(row["example_id"]): row for row in pointwise.get("results") or []}
    output = []
    for visual_row in visual.get("results") or []:
        example_id = str(visual_row["example_id"])
        point_row = point_by_example.get(example_id)
        if point_row is None:
            if allow_missing_pointwise:
                continue
            raise ValueError(f"Missing pointwise result: {example_id}")
        point_by_index = {int(row["candidate_index"]): row for row in point_row.get("ranking") or []}
        gold = {int(value) for value in visual_row.get("gold") or []}
        for candidate in visual_row.get("ranking") or []:
            index = int(candidate["candidate_index"])
            point = point_by_index.get(index)
            if point is None:
                # Training pointwise SFT intentionally retains only a bounded
                # hard-negative set. Calibration uses the score intersection;
                # every supervised gold is retained by the data builder.
                continue
            point_score = float(point["score"])
            visual_score = _visual_logit(float(candidate["score"]))
            rank = int(point["retrieval_rank"])
            output.append({
                "example_id": example_id, "candidate_index": index,
                "gold": index in gold, "gold_indices": sorted(gold),
                "features": [point_score, visual_score, 1.0 / rank, point_score * visual_score],
            })
    return output


def _fold(example_id: str, folds: int = 5) -> int:
    return int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:8], 16) % folds


def rank_metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for row, score in zip(rows, scores.tolist(), strict=True):
        grouped.setdefault(str(row["example_id"]), []).append((float(score), row))
    results = []
    for example_id, candidates in sorted(grouped.items()):
        ranked = sorted(candidates, key=lambda item: (-item[0], int(item[1]["candidate_index"])))
        predicted = [int(row["candidate_index"]) for _, row in ranked[:2]]
        gold = sorted({
            int(value)
            for _, row in candidates
            for value in row.get("gold_indices") or []
        })
        results.append({"example_id": example_id, "gold": gold, "predicted": predicted, "metrics": retrieval_scores(predicted, gold)})
    values = [row["metrics"] for row in results]
    metrics = {
        "examples": len(values),
        "mean_precision": sum(float(row["precision"]) for row in values) / max(1, len(values)),
        "mean_recall": sum(float(row["recall"]) for row in values) / max(1, len(values)),
        "hit_rate": sum(bool(row["hit"]) for row in values) / max(1, len(values)),
        "exact_rate": sum(bool(row["exact"]) for row in values) / max(1, len(values)),
    }
    return results, metrics


def fit_calibrator(train_rows: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]], float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([row["features"] for row in train_rows], dtype=np.float64)
    y = np.asarray([bool(row["gold"]) for row in train_rows], dtype=np.int64)
    cv = []
    for c_value in C_VALUES:
        all_results = []
        for fold in range(5):
            train_mask = np.asarray([_fold(str(row["example_id"])) != fold for row in train_rows])
            test_mask = ~train_mask
            if not test_mask.any():
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=42),
            )
            model.fit(x[train_mask], y[train_mask])
            fold_rows = [row for row, keep in zip(train_rows, test_mask.tolist(), strict=True) if keep]
            fold_scores = model.predict_proba(x[test_mask])[:, 1]
            fold_results, _ = rank_metrics(fold_rows, fold_scores)
            all_results.extend(fold_results)
        flat_rows = []
        flat_scores = []
        # Re-aggregate already strict fold predictions without leaking candidates
        # across examples. Each example belongs to exactly one deterministic fold.
        for row in all_results:
            metric = row["metrics"]
            flat_rows.append(metric)
            flat_scores.append((float(metric["recall"]), bool(metric["hit"])))
        mean_recall = sum(value[0] for value in flat_scores) / max(1, len(flat_scores))
        hit_rate = sum(value[1] for value in flat_scores) / max(1, len(flat_scores))
        cv.append({"C": c_value, "examples": len(flat_scores), "mean_recall": mean_recall, "hit_rate": hit_rate})
    selected = max(cv, key=lambda row: ((row["mean_recall"] + row["hit_rate"]) / 2.0, -row["C"]))
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(selected["C"]), class_weight="balanced", max_iter=2000, random_state=42),
    )
    final.fit(x, y)
    return final, cv, float(selected["C"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pointwise", type=Path, required=True)
    parser.add_argument("--train-visual", type=Path, required=True)
    parser.add_argument("--dev-pointwise", type=Path, required=True)
    parser.add_argument("--dev-visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    train_pointwise = read_json(args.train_pointwise)
    train_visual = read_json(args.train_visual)
    train_rows = joined_rows(
        train_pointwise,
        train_visual,
        allow_missing_pointwise=True,
    )
    dev_rows = joined_rows(read_json(args.dev_pointwise), read_json(args.dev_visual))
    train_pointwise_ids = {
        str(row["example_id"]) for row in train_pointwise.get("results") or []
    }
    train_visual_ids = {
        str(row["example_id"]) for row in train_visual.get("results") or []
    }
    model, cv, selected_c = fit_calibrator(train_rows)
    dev_x = np.asarray([row["features"] for row in dev_rows], dtype=np.float64)
    results, metrics = rank_metrics(dev_rows, model.predict_proba(dev_x)[:, 1])
    logistic = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    report = {
        "schema_version": "video-skills/l2-calibrated-fusion-eval-v0.1",
        "train_only_calibration": True, "feature_names": FEATURE_NAMES,
        "train_common_examples": len(train_pointwise_ids & train_visual_ids),
        "train_visual_only_examples_excluded": sorted(train_visual_ids - train_pointwise_ids),
        "cv": cv, "selected_C": selected_c,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficients": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0]),
        "metrics": metrics, "results": results,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

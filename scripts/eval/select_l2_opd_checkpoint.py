#!/usr/bin/env python3
"""Select the smallest OPD trust-region alpha satisfying frozen CG/VH dev gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from trainer.artifact_hash import adapter_weight_sha256


def metrics(report: Mapping[str, Any], *, dataset: str) -> dict[str, float]:
    payload = report.get("metrics") or {}
    pointwise_name = "pointwise_top2" if dataset == "cg_bench" else "pointwise_top4"
    pointwise = payload.get(pointwise_name) or {}
    process = (((payload.get("dataset_metrics") or {}).get(dataset) or {}).get("process_metrics")) or {}
    return {
        "mean_recall": float(pointwise.get("mean_recall") or 0.0),
        "hit_rate": float(pointwise.get("hit_rate") or 0.0),
        **{str(key): float(value) for key, value in process.items()},
    }


def parse_candidate(value: str) -> dict[str, Any]:
    parts = value.split("|", 4)
    if len(parts) != 5:
        raise ValueError("candidate must be NAME|ALPHA|ADAPTER|CG_REPORT|VH_REPORT")
    name, alpha, adapter, cg_report, vh_report = parts
    return {
        "name": name,
        "alpha": float(alpha),
        "adapter": Path(adapter),
        "cg_report": Path(cg_report),
        "vh_report": Path(vh_report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-cg-report", type=Path, required=True)
    parser.add_argument("--sft-vh-report", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline_cg = metrics(json.loads(args.sft_cg_report.read_text()), dataset="cg_bench")
    baseline_vh = metrics(json.loads(args.sft_vh_report.read_text()), dataset="video_holmes")
    rows = []
    for spec in sorted((parse_candidate(value) for value in args.candidate), key=lambda row: row["alpha"]):
        cg = metrics(json.loads(spec["cg_report"].read_text()), dataset="cg_bench")
        vh = metrics(json.loads(spec["vh_report"].read_text()), dataset="video_holmes")
        checks = {
            "cg_recall_not_below_sft": cg["mean_recall"] + 1e-12 >= baseline_cg["mean_recall"],
            "cg_hit_not_below_sft": cg["hit_rate"] + 1e-12 >= baseline_cg["hit_rate"],
            "cg_recall_at_least_60pct": cg["mean_recall"] + 1e-12 >= 0.60,
            "vh_segment_strict_gain": vh.get("segment_recall", 0.0) > baseline_vh.get("segment_recall", 0.0),
            "vh_inference_strict_gain": vh.get("inference_shot_recall", 0.0) > baseline_vh.get("inference_shot_recall", 0.0),
            "vh_relationship_strict_gain": vh.get("relationship_support", 0.0) > baseline_vh.get("relationship_support", 0.0),
        }
        rows.append({
            "name": spec["name"],
            "alpha": spec["alpha"],
            "adapter": str(spec["adapter"]),
            "adapter_weight_sha256": adapter_weight_sha256(spec["adapter"]),
            "cg_report": str(spec["cg_report"]),
            "vh_report": str(spec["vh_report"]),
            "cg_metrics": cg,
            "vh_metrics": vh,
            "gains": {
                "cg_mean_recall": cg["mean_recall"] - baseline_cg["mean_recall"],
                "cg_hit_rate": cg["hit_rate"] - baseline_cg["hit_rate"],
                "vh_segment_recall": vh.get("segment_recall", 0.0) - baseline_vh.get("segment_recall", 0.0),
                "vh_inference_shot_recall": vh.get("inference_shot_recall", 0.0) - baseline_vh.get("inference_shot_recall", 0.0),
                "vh_relationship_support": vh.get("relationship_support", 0.0) - baseline_vh.get("relationship_support", 0.0),
            },
            "checks": checks,
            "passed": all(checks.values()),
        })
    selected = next((row for row in rows if row["passed"]), None)
    report = {
        "schema_version": "video-skills/l2-opd-checkpoint-selection-v1",
        "passed": selected is not None,
        "selection_rule": "smallest-alpha-passing-cg-preservation-and-vh-three-process-strict-gains",
        "sft_cg_report": str(args.sft_cg_report),
        "sft_vh_report": str(args.sft_vh_report),
        "baseline_cg_metrics": baseline_cg,
        "baseline_vh_metrics": baseline_vh,
        "candidates": rows,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

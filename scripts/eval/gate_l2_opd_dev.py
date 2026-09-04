#!/usr/bin/env python3
"""Gate a CG/VH OPD checkpoint against the same SFT baseline dev sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pointwise(report: dict[str, Any]) -> dict[str, float]:
    top_k = int(report.get("top_k") or 2)
    return report["metrics"][f"pointwise_top{top_k}"]


def _process(report: dict[str, Any], dataset: str) -> dict[str, float]:
    """Return process metrics when available in the report schema.

    The frozen CG14 baseline predates dataset-aware process metrics.  Missing
    metrics must not crash the gate or be silently treated as zero; comparisons
    are added only when both sides expose the same evaluator output.
    """
    return (
        report.get("metrics", {})
        .get("dataset_metrics", {})
        .get(dataset, {})
        .get("process_metrics", {})
    )


def gate_reports(
    *,
    sft_cg: dict[str, Any],
    opd_cg: dict[str, Any],
    sft_vh: dict[str, Any],
    opd_vh: dict[str, Any],
    sft_cg_process: dict[str, Any] | None = None,
    opd_cg_process: dict[str, Any] | None = None,
    min_cg_recall: float = 0.60,
    require_cg_strict_gain: bool = False,
    require_vh_strict_gain: bool = True,
    vh_process_primary: bool = False,
) -> dict[str, Any]:
    cg_base, cg_new = _pointwise(sft_cg), _pointwise(opd_cg)
    vh_base, vh_new = _pointwise(sft_vh), _pointwise(opd_vh)
    cg_process_base = _process(sft_cg_process or sft_cg, "cg_bench")
    cg_process_new = _process(opd_cg_process or opd_cg, "cg_bench")
    vh_process_base, vh_process_new = _process(sft_vh, "video_holmes"), _process(opd_vh, "video_holmes")
    checks: dict[str, bool] = {
        "cg_same_top_k_contract": int(sft_cg.get("top_k") or 2)
        == int(opd_cg.get("top_k") or 2),
        "vh_same_top_k_contract": int(sft_vh.get("top_k") or 2)
        == int(opd_vh.get("top_k") or 2),
        "cg_recall_meets_60pct": float(cg_new["mean_recall"]) >= min_cg_recall,
        "cg_recall_not_below_sft": float(cg_new["mean_recall"]) >= float(cg_base["mean_recall"]),
        "cg_hit_not_below_sft": float(cg_new["hit_rate"]) >= float(cg_base["hit_rate"]),
    }
    if not vh_process_primary:
        checks["vh_recall_not_below_sft"] = float(vh_new["mean_recall"]) >= float(
            vh_base["mean_recall"]
        )
        checks["vh_hit_not_below_sft"] = float(vh_new["hit_rate"]) >= float(
            vh_base["hit_rate"]
        )
    if "boundary_anchor_index0" in sft_cg and "boundary_anchor_index0" in opd_cg:
        checks["cg_same_boundary_anchor_contract"] = bool(
            sft_cg["boundary_anchor_index0"]
        ) == bool(opd_cg["boundary_anchor_index0"])
    if "boundary_anchor_index0" in sft_vh and "boundary_anchor_index0" in opd_vh:
        checks["vh_same_boundary_anchor_contract"] = bool(
            sft_vh["boundary_anchor_index0"]
        ) == bool(opd_vh["boundary_anchor_index0"])
    gains: dict[str, float] = {
        "cg_mean_recall": float(cg_new["mean_recall"]) - float(cg_base["mean_recall"]),
        "cg_hit_rate": float(cg_new["hit_rate"]) - float(cg_base["hit_rate"]),
        "vh_mean_recall": float(vh_new["mean_recall"]) - float(vh_base["mean_recall"]),
        "vh_hit_rate": float(vh_new["hit_rate"]) - float(vh_base["hit_rate"]),
    }
    process_specs = (
        ("cg", "clue_recall", cg_process_base, cg_process_new),
        ("cg", "clue_mean_best_iou", cg_process_base, cg_process_new),
        ("vh", "segment_recall", vh_process_base, vh_process_new),
        ("vh", "inference_shot_recall", vh_process_base, vh_process_new),
        ("vh", "relationship_support", vh_process_base, vh_process_new),
    )
    compared_process_metrics: list[str] = []
    for prefix, metric, base_process, new_process in process_specs:
        if metric not in base_process or metric not in new_process:
            continue
        check_name = f"{prefix}_{metric}_not_below_sft"
        gain_name = f"{prefix}_{metric}"
        checks[check_name] = float(new_process[metric]) >= float(base_process[metric])
        gains[gain_name] = float(new_process[metric]) - float(base_process[metric])
        compared_process_metrics.append(gain_name)
    if require_vh_strict_gain:
        checks["vh_strict_gain_exists_for_low_sample_opd"] = any(
            value > 1e-12 for key, value in gains.items() if key.startswith("vh_")
        )
    if require_cg_strict_gain:
        checks["cg_strict_gain_exists"] = any(
            value > 1e-12 for key, value in gains.items() if key.startswith("cg_")
        )
    return {
        "schema_version": "video-skills/l2-opd-dev-gate-v0.1",
        "passed": all(checks.values()),
        "checks": checks,
        "gains": gains,
        "compared_process_metrics": compared_process_metrics,
        "thresholds": {
            "min_cg_recall": min_cg_recall,
            "require_cg_strict_gain": bool(require_cg_strict_gain),
            "require_vh_strict_gain": bool(require_vh_strict_gain),
            "vh_process_primary": bool(vh_process_primary),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-cg", type=Path, required=True)
    parser.add_argument("--opd-cg", type=Path, required=True)
    parser.add_argument("--sft-vh", type=Path, required=True)
    parser.add_argument("--opd-vh", type=Path, required=True)
    parser.add_argument("--sft-cg-process", type=Path)
    parser.add_argument("--opd-cg-process", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cg-recall", type=float, default=0.60)
    parser.add_argument("--require-cg-strict-gain", action="store_true")
    parser.add_argument(
        "--allow-vh-no-strict-gain",
        action="store_true",
        help=(
            "Use for a post-OPD checkpoint: require every frozen VH metric to "
            "be no worse than the OPD reference, but do not require another "
            "strict gain. OPD-vs-SFT gates keep strict gain enabled."
        ),
    )
    parser.add_argument(
        "--vh-process-primary",
        action="store_true",
        help=(
            "For Video-Holmes OPD, gate the declared segment/inference/relationship "
            "evidence components instead of generic candidate_relevant recall/hit. "
            "Headline metrics remain reported in gains."
        ),
    )
    args = parser.parse_args()
    report = gate_reports(
        sft_cg=_load(args.sft_cg), opd_cg=_load(args.opd_cg),
        sft_vh=_load(args.sft_vh), opd_vh=_load(args.opd_vh),
        sft_cg_process=_load(args.sft_cg_process) if args.sft_cg_process else None,
        opd_cg_process=_load(args.opd_cg_process) if args.opd_cg_process else None,
        min_cg_recall=args.min_cg_recall,
        require_cg_strict_gain=args.require_cg_strict_gain,
        require_vh_strict_gain=not args.allow_vh_no_strict_gain,
        vh_process_primary=args.vh_process_primary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

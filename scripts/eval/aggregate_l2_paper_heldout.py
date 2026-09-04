#!/usr/bin/env python3
"""Aggregate the frozen SFT/OPD/three-seed GRPO heldout pointwise matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.eval.select_l2_opd_checkpoint import metrics as pointwise_metrics


EXPECTED_MODELS = ("sft", "opd_alpha075", "grpo_seed42", "grpo_seed43", "grpo_seed44")
GRPO_MODELS = ("grpo_seed42", "grpo_seed43", "grpo_seed44")


def expected_hashes_from_references(
    sft: dict[str, Any], opd: dict[str, Any], three_seed: dict[str, Any]
) -> dict[str, str]:
    hashes = {
        "sft": str(sft.get("adapter_weight_sha256") or ""),
        "opd_alpha075": str((opd.get("selected") or {}).get("adapter_weight_sha256") or ""),
    }
    for seed in three_seed.get("seeds") or []:
        name = f"grpo_seed{int(seed.get('seed'))}"
        hashes[name] = str((seed.get("trained_adapter") or {}).get("adapter_weight_sha256") or "")
    return hashes


def _metric_values(report: dict[str, Any], dataset: str) -> dict[str, float]:
    values = pointwise_metrics(report, dataset=dataset)
    names = ["mean_recall", "hit_rate"]
    names += (
        ["clue_recall", "evidence_precision", "clue_mean_best_iou"]
        if dataset == "cg_bench"
        else ["segment_recall", "segment_precision", "inference_shot_recall", "relationship_support"]
    )
    return {name: float(values.get(name) or 0.0) for name in names}


def aggregate_heldout(
    reports: dict[str, dict[str, dict[str, Any]]], expected_hashes: dict[str, str]
) -> dict[str, Any]:
    model_rows = []
    global_checks = {
        "exact_model_matrix": set(reports) == set(EXPECTED_MODELS),
        "all_expected_hashes_present": all(expected_hashes.get(name) for name in EXPECTED_MODELS),
    }
    for model in EXPECTED_MODELS:
        pair = reports.get(model) or {}
        cg = pair.get("cg_bench") or {}
        vh = pair.get("video_holmes") or {}
        cg_hash = str(cg.get("adapter_weight_sha256") or "")
        vh_hash = str(vh.get("adapter_weight_sha256") or "")
        checks = {
            "reports_present": bool(cg) and bool(vh),
            "same_adapter_hash_across_datasets": bool(cg_hash) and cg_hash == vh_hash,
            "adapter_hash_matches_training_chain": bool(expected_hashes.get(model))
            and cg_hash == expected_hashes.get(model),
            "heldout_split_role": cg.get("input_split_roles") == ["heldout_test"]
            and vh.get("input_split_roles") == ["heldout_test"],
            "dataset_identity": cg.get("input_datasets") == ["cg_bench"]
            and vh.get("input_datasets") == ["video_holmes"],
            "frozen_topk_protocol": cg.get("top_k") == 2 and vh.get("top_k") == 4,
            "frozen_boundary_protocol": cg.get("boundary_anchor_index0") is True
            and vh.get("boundary_anchor_index0") is False,
            "input_rows_accounted": int(cg.get("candidate_rows") or 0) > 0
            and cg.get("candidate_rows") == cg.get("input_rows")
            and int(vh.get("candidate_rows") or 0) > 0
            and vh.get("candidate_rows") == vh.get("input_rows"),
            "input_hashes_present": bool(cg.get("evaluation_jsonl_sha256"))
            and bool(vh.get("evaluation_jsonl_sha256")),
        }
        model_rows.append({
            "model": model,
            "adapter_weight_sha256": cg_hash,
            "checks": checks,
            "passed": all(checks.values()),
            "metrics": {
                "cg_bench": _metric_values(cg, "cg_bench") if cg else {},
                "video_holmes": _metric_values(vh, "video_holmes") if vh else {},
            },
        })

    for dataset in ("cg_bench", "video_holmes"):
        payloads = [(reports.get(model) or {}).get(dataset) or {} for model in EXPECTED_MODELS]
        global_checks[f"same_{dataset}_input_hash"] = bool(payloads[0].get("evaluation_jsonl_sha256")) and len({
            value.get("evaluation_jsonl_sha256") for value in payloads
        }) == 1
        global_checks[f"same_{dataset}_example_count"] = int(payloads[0].get("input_examples") or 0) > 0 and len({
            value.get("input_examples") for value in payloads
        }) == 1

    by_name = {row["model"]: row for row in model_rows}
    grpo_summary: dict[str, Any] = {}
    gains: dict[str, Any] = {}
    for dataset in ("cg_bench", "video_holmes"):
        grpo_summary[dataset] = {}
        gains[dataset] = {}
        metric_names = sorted(by_name[GRPO_MODELS[0]]["metrics"][dataset])
        for metric in metric_names:
            values = [by_name[name]["metrics"][dataset][metric] for name in GRPO_MODELS]
            mean = statistics.fmean(values)
            grpo_summary[dataset][metric] = {
                "mean": mean,
                "std": statistics.pstdev(values),
                "values": values,
            }
            gains[dataset][metric] = {
                "vs_sft": mean - by_name["sft"]["metrics"][dataset][metric],
                "vs_opd": mean - by_name["opd_alpha075"]["metrics"][dataset][metric],
            }

    passed = all(global_checks.values()) and len(model_rows) == 5 and all(
        row["passed"] for row in model_rows
    )
    return {
        "schema_version": "video-skills/l2-paper-heldout-pointwise-aggregate-v1",
        "passed": passed,
        "integrity_checks": global_checks,
        "models": model_rows,
        "grpo_three_seed": grpo_summary,
        "grpo_gains": gains,
        "performance_is_not_a_release_gate": True,
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME|CG_REPORT|VH_REPORT")
    parser.add_argument("--sft-reference", type=Path, required=True)
    parser.add_argument("--opd-selection", type=Path, required=True)
    parser.add_argument("--three-seed-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for raw in args.model:
        parts = raw.split("|", 2)
        if len(parts) != 3:
            parser.error(f"invalid --model: {raw!r}")
        name, cg, vh = parts
        reports[name] = {"cg_bench": _load(Path(cg)), "video_holmes": _load(Path(vh))}
    expected = expected_hashes_from_references(
        _load(args.sft_reference), _load(args.opd_selection), _load(args.three_seed_aggregate)
    )
    report = aggregate_heldout(reports, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

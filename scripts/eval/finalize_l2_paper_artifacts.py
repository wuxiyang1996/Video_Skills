#!/usr/bin/env python3
"""Build paper tables and a hash-pinned reproducibility manifest fail-closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_NAMES = (
    "split_audit",
    "reward_separation",
    "reward_normalization",
    "opd_selection",
    "grpo_aggregate",
    "pretest_gate",
    "vh_l1_audit",
    "heldout_aggregate",
)
MODEL_ORDER = ("sft", "opd_alpha075", "grpo_seed42", "grpo_seed43", "grpo_seed44")
DATASETS = ("cg_bench", "video_holmes")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paper_artifacts(
    sources: dict[str, tuple[Path, dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payloads = {name: payload for name, (_, payload) in sources.items()}
    heldout = payloads["heldout_aggregate"]
    grpo = payloads["grpo_aggregate"]
    models = {row.get("model"): row for row in heldout.get("models") or []}
    source_checks = {name: payload.get("passed") is True for name, payload in payloads.items()}
    checks = {
        **{f"{name}_passed": passed for name, passed in source_checks.items()},
        "exact_heldout_model_matrix": set(models) == set(MODEL_ORDER),
        "exactly_three_grpo_seeds": grpo.get("seed_count") == 3
        and len(grpo.get("seeds") or []) == 3,
        "same_grpo_training_contracts": grpo.get("same_training_contracts") is True,
        "all_grpo_seed_gates_passed": len(grpo.get("seeds") or []) == 3
        and all(seed.get("passed") is True for seed in grpo.get("seeds") or []),
        "all_adapter_hashes_present": all(
            bool((models.get(name) or {}).get("adapter_weight_sha256")) for name in MODEL_ORDER
        ),
        "heldout_inputs_frozen_across_models": all(
            heldout.get("integrity_checks", {}).get(f"same_{dataset}_input_hash") is True
            for dataset in DATASETS
        ),
    }

    rows: list[dict[str, Any]] = []
    metric_names = sorted({
        metric
        for model in models.values()
        for dataset in DATASETS
        for metric in (model.get("metrics", {}).get(dataset, {}) or {})
    })
    for name in MODEL_ORDER:
        model = models.get(name) or {}
        row: dict[str, Any] = {
            "model": name,
            "kind": "checkpoint",
            "adapter_weight_sha256": model.get("adapter_weight_sha256", ""),
        }
        for dataset in DATASETS:
            values = model.get("metrics", {}).get(dataset, {}) or {}
            for metric in metric_names:
                row[f"{dataset}.{metric}"] = values.get(metric, "")
        rows.append(row)

    mean_row: dict[str, Any] = {
        "model": "grpo_three_seed_mean",
        "kind": "aggregate",
        "adapter_weight_sha256": "three-seed-distribution",
    }
    std_row: dict[str, Any] = {
        "model": "grpo_three_seed_std",
        "kind": "aggregate",
        "adapter_weight_sha256": "three-seed-distribution",
    }
    for dataset in DATASETS:
        for metric, summary in (heldout.get("grpo_three_seed", {}).get(dataset, {}) or {}).items():
            mean_row[f"{dataset}.{metric}"] = summary.get("mean", "")
            std_row[f"{dataset}.{metric}"] = summary.get("std", "")
    rows.extend((mean_row, std_row))

    gains = heldout.get("grpo_gains") or {}
    reproducibility = {
        "schema_version": "video-skills/l2-paper-reproducibility-manifest-v1",
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "schema_version": payload.get("schema_version"),
                "passed": payload.get("passed"),
            }
            for name, (path, payload) in sources.items()
        },
        "adapter_hashes": {
            name: (models.get(name) or {}).get("adapter_weight_sha256", "")
            for name in MODEL_ORDER
        },
        "grpo_seeds": [int(row["seed"]) for row in grpo.get("seeds") or []],
        "opd_checkpoint": payloads["opd_selection"].get("selected") or {},
        "grpo_runs": [
            {
                "seed": seed.get("seed"),
                "train_report": seed.get("train_report"),
                "terminal_report": seed.get("terminal_report"),
                "trained_adapter": seed.get("trained_adapter"),
                "contracts": seed.get("contracts"),
                "training_dataset_metrics": seed.get("training_dataset_metrics"),
            }
            for seed in grpo.get("seeds") or []
        ],
        "frozen_protocol": {
            "datasets": list(DATASETS),
            "cg_bench_top_k": 2,
            "video_holmes_top_k": 4,
            "cg_boundary_anchor_index0": True,
            "video_holmes_boundary_anchor_index0": False,
            "video_holmes_official_test_video_count": 270,
        },
    }
    report = {
        "schema_version": "video-skills/l2-paper-artifacts-v1",
        "passed": all(checks.values()),
        "integrity_checks": checks,
        "main_table_rows": rows,
        "stage_ablation": {
            "stages": ["sft", "opd_alpha075", "grpo_three_seed_mean"],
            "grpo_gains": gains,
            "performance_is_not_an_integrity_gate": True,
        },
        "three_seed_dev_metrics": grpo.get("metrics") or {},
        "interpretation": {
            "atomic_skill_generalization_evidence": "heldout SFT/OPD/GRPO deltas in grpo_gains",
            "reward_normalization_scope": payloads["reward_normalization"].get(
                "normalization_contract"
            ),
            "terminal_evidence_scope": "frozen independent dev terminal rollouts",
            "heldout_evidence_scope": "frozen official/test-role pointwise evidence retrieval",
        },
    }
    return report, rows, reproducibility


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["model", "kind", "adapter_weight_sha256"] + sorted(
        {key for row in rows for key in row if key not in {"model", "kind", "adapter_weight_sha256"}}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any) -> str:
    return "" if value == "" else f"{100.0 * float(value):.2f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["main_table_rows"]
    chosen = (
        "cg_bench.mean_recall",
        "cg_bench.hit_rate",
        "video_holmes.segment_recall",
        "video_holmes.inference_shot_recall",
        "video_holmes.relationship_support",
    )
    labels = ("Model", "CG Recall@2", "CG Hit", "VH Segment", "VH Inference", "VH Relation")
    lines = [
        "# CG-Bench + Video-Holmes L2 paper results",
        "",
        "All values are percentages. GRPO mean/std use seeds 42, 43, and 44.",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(labels) - 1)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join([str(row["model"])] + [_format_metric(row.get(key, "")) for key in chosen]) + " |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"Overall artifact audit: **{'PASS' if report['passed'] else 'FAIL'}**.",
        "",
        "Performance deltas are reported as measurements and are not used to make the integrity audit pass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        name: (getattr(args, name), _load(getattr(args, name)))
        for name in SOURCE_NAMES
    }
    report, rows, reproducibility = build_paper_artifacts(sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "reproducibility_manifest.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "main_results.csv", rows)
    _write_markdown(args.output_dir / "main_results.md", report)
    print(json.dumps({"passed": report["passed"], "output_dir": str(args.output_dir)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

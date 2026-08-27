#!/usr/bin/env python3
"""Post-SFT quality gates for five independent LoRA warm-ups.

These gates only validate format / action-space warm-up quality. They compare a
LoRA generation report against optional base-9B and majority-action baselines.
Functional metrics (repair conversion, motif utility, verified answer success)
are deferred to OPD.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import read_json, read_jsonl, write_json


DEFAULT_THRESHOLDS = {
    "min_json_valid_rate": 0.95,
    "min_action_match_rate": 0.50,
    "min_schema_or_parse_margin_over_base": 0.0,
    "require_beat_majority_action_match": True,
    "max_hidden_leakage": 0,
}


def majority_action_baseline(dev_jsonl: Path) -> dict[str, Any]:
    """Compute the majority-action baseline on a specialist dev split."""
    rows = read_jsonl(dev_jsonl)
    families: Counter[str] = Counter()
    gold_actions: list[str] = []
    for row in rows:
        content = ""
        for message in row.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = str(message.get("content") or "")
                break
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            gold_actions.append("parse_fail")
            families["parse_fail"] += 1
            continue
        nested = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        family = str(payload.get("tool_name") or nested.get("action_type") or "unknown")
        families[family] += 1
        gold_actions.append(family)
    if not gold_actions:
        return {
            "n_rows": 0,
            "majority_family": None,
            "majority_rate": 0.0,
            "json_valid_rate": 0.0,
            "action_match_rate": 0.0,
        }
    majority_family, majority_count = families.most_common(1)[0]
    return {
        "n_rows": len(gold_actions),
        "majority_family": majority_family,
        "majority_rate": majority_count / len(gold_actions),
        "json_valid_rate": 1.0 - (families.get("parse_fail", 0) / len(gold_actions)),
        # Predicting the majority family on every row.
        "action_match_rate": majority_count / len(gold_actions),
        "family_counts": dict(families),
    }


def evaluate_lora_report(
    *,
    specialist: str,
    lora_generation_report: dict[str, Any],
    base_generation_report: dict[str, Any] | None = None,
    majority_baseline: dict[str, Any] | None = None,
    thresholds: dict[str, float | bool] | None = None,
    hidden_leakage: int = 0,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    json_rate = float(lora_generation_report.get("json_valid_rate") or 0.0)
    action_rate = float(lora_generation_report.get("action_match_rate") or 0.0)
    failures: list[str] = []
    warnings: list[str] = []

    if hidden_leakage > int(thresholds["max_hidden_leakage"]):
        failures.append(f"hidden_leakage={hidden_leakage}")
    if json_rate < float(thresholds["min_json_valid_rate"]):
        failures.append(
            f"json_valid_rate={json_rate:.3f} < {float(thresholds['min_json_valid_rate']):.3f}"
        )
    if action_rate < float(thresholds["min_action_match_rate"]):
        # Warm-up only needs moderate action match; low values are still a hard fail.
        failures.append(
            f"action_match_rate={action_rate:.3f} < {float(thresholds['min_action_match_rate']):.3f}"
        )

    if base_generation_report is not None:
        base_json = float(base_generation_report.get("json_valid_rate") or 0.0)
        margin = json_rate - base_json
        if margin < float(thresholds["min_schema_or_parse_margin_over_base"]):
            failures.append(
                f"json_valid_rate margin over base={margin:.3f} "
                f"< {float(thresholds['min_schema_or_parse_margin_over_base']):.3f}"
            )
        base_action = float(base_generation_report.get("action_match_rate") or 0.0)
        if action_rate + 1e-9 < base_action:
            warnings.append(
                f"action_match_rate={action_rate:.3f} did not exceed base={base_action:.3f}"
            )

    if majority_baseline is not None and thresholds["require_beat_majority_action_match"]:
        majority_rate = float(majority_baseline.get("action_match_rate") or 0.0)
        family_counts = majority_baseline.get("family_counts") or {}
        n_families = len([k for k, v in family_counts.items() if int(v) > 0 and k != "parse_fail"])
        # Single-family packs (e.g. verifier/motif warm-up) already have majority=1.0;
        # "beat majority" is undefined there — only enforce when collapse is diagnosable.
        if majority_rate >= 1.0 - 1e-12 or n_families <= 1:
            warnings.append(
                f"majority baseline skipped (majority_rate={majority_rate:.3f}, n_families={n_families}); "
                "pack is single-family so collapse check is undefined"
            )
        elif action_rate <= majority_rate + 1e-12:
            # Equal to majority counts as collapse; must strictly beat it.
            failures.append(
                f"action_match_rate={action_rate:.3f} did not beat majority baseline={majority_rate:.3f}; "
                "likely majority-class collapse"
            )

    return {
        "specialist": specialist,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "json_valid_rate": json_rate,
            "action_match_rate": action_rate,
            "hidden_leakage": hidden_leakage,
            "n_generation_examples": lora_generation_report.get("examples"),
        },
        "baselines": {
            "base": {
                "json_valid_rate": (base_generation_report or {}).get("json_valid_rate"),
                "action_match_rate": (base_generation_report or {}).get("action_match_rate"),
            },
            "majority": majority_baseline,
        },
        "thresholds": thresholds,
    }


def evaluate_five_lora_sft_gates(
    *,
    reports_root: Path,
    package_root: Path | None = None,
    thresholds: dict[str, float | bool] | None = None,
) -> dict[str, Any]:
    """Evaluate per-specialist folders under reports_root.

    Expected layout:
      reports_root/<specialist>/generation_report.json
      reports_root/<specialist>/base_generation_report.json   (optional)
      reports_root/<specialist>/train_metrics.json           (optional)
    If package_root is provided, majority baselines are computed from
    package_root/<specialist>/dev.jsonl.
    """
    specialist_results = []
    for specialist_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        specialist = specialist_dir.name
        lora_path = specialist_dir / "generation_report.json"
        if not lora_path.exists():
            specialist_results.append(
                {
                    "specialist": specialist,
                    "passed": False,
                    "failures": [f"missing {lora_path}"],
                    "warnings": [],
                }
            )
            continue
        lora_report = read_json(lora_path)
        base_path = specialist_dir / "base_generation_report.json"
        base_report = read_json(base_path) if base_path.exists() else None
        majority = None
        if package_root is not None:
            dev_path = package_root / specialist / "dev.jsonl"
            if dev_path.exists():
                majority = majority_action_baseline(dev_path)
        metrics_path = specialist_dir / "train_metrics.json"
        hidden_leakage = 0
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            hidden_leakage = int(metrics.get("prompt_forbidden_key_hits") or metrics.get("hidden_leakage") or 0)
        specialist_results.append(
            evaluate_lora_report(
                specialist=specialist,
                lora_generation_report=lora_report,
                base_generation_report=base_report,
                majority_baseline=majority,
                thresholds=thresholds,
                hidden_leakage=hidden_leakage,
            )
        )
    failures = [item for item in specialist_results if not item.get("passed")]
    return {
        "schema_version": "video-skills/lora-sft-gates-v1",
        "reports_root": str(reports_root),
        "package_root": str(package_root) if package_root else None,
        "passed": not failures,
        "specialists": specialist_results,
        "failed_specialists": [item["specialist"] for item in failures],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-json-valid-rate", type=float, default=0.95)
    parser.add_argument("--min-action-match-rate", type=float, default=0.50)
    parser.add_argument("--allow-not-beating-majority", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_five_lora_sft_gates(
        reports_root=args.reports_root,
        package_root=args.package_root,
        thresholds={
            "min_json_valid_rate": args.min_json_valid_rate,
            "min_action_match_rate": args.min_action_match_rate,
            "require_beat_majority_action_match": not args.allow_not_beating_majority,
        },
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "failed_specialists": report["failed_specialists"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

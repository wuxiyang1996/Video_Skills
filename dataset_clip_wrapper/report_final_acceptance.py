#!/usr/bin/env python3
"""Merge base L1/L2 quality with repair reports into final acceptance status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repair_status_to_l2(report: dict[str, Any]) -> dict[str, Any]:
    status = report.get("repair_status")
    if status == "resolved_strong":
        acceptance = "accepted_strong"
    elif status == "accepted_bridge":
        acceptance = "accepted_bridge"
    else:
        acceptance = status or "missing"
    support_refs = _repair_support_refs(report)
    return {
        "acceptance_status": acceptance,
        "final_answer": report.get("best_option") or {},
        "support_ref_count": len(support_refs),
        "support_refs": support_refs,
        "repair_status": status,
        "repair_failure_type": report.get("failure_type"),
        "verifier_backend": report.get("verifier_backend"),
        "not_direct_visual_evidence": bool(report.get("not_direct_visual_evidence")),
    }


def _repair_support_refs(report: dict[str, Any]) -> list[str]:
    bridge_refs = ((report.get("background_bridge_verification") or {}).get("visual_anchor_refs") or [])
    refs = [str(ref) for ref in bridge_refs if ref]
    l2_path = ((report.get("artifact_paths") or {}).get("l2_verifier"))
    if not l2_path:
        return list(dict.fromkeys(refs))
    path = Path(str(l2_path))
    if not path.exists():
        return list(dict.fromkeys(refs))
    try:
        l2 = _read_json(path)
    except Exception:
        return list(dict.fromkeys(refs))
    bridge_check = l2.get("bridge_ref_verification") or {}
    refs.extend(str(ref) for ref in bridge_check.get("evidence_refs") or [] if ref)
    best = l2.get("best_option") or {}
    best_label = str(best.get("label")) if best.get("label") is not None else ""
    for row in l2.get("option_verifications") or []:
        if best_label and str(row.get("option_label")) == best_label and row.get("ok"):
            refs.extend(str(ref) for ref in row.get("evidence_refs") or [] if ref)
    return list(dict.fromkeys(refs))


def _merge_report(base_report: dict[str, Any], repair_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    example_id = str(base_report.get("example_id"))
    merged = dict(base_report)
    clip_stats = (merged.get("L1_quality") or {}).get("clip_schema_stats") or {}
    fallback_count = int(clip_stats.get("fine_fallback") or 0) + int(clip_stats.get("coarse_fallback") or 0)
    error_count = int(clip_stats.get("fine_errors") or 0) + int(clip_stats.get("coarse_errors") or 0)
    merged["strict_vlm_perception"] = {
        "qwen_only": fallback_count == 0 and error_count == 0,
        "fallback_clip_schema_count": fallback_count,
        "model_error_clip_schema_count": error_count,
    }
    repair = repair_by_id.get(example_id)
    if not repair:
        merged["final_acceptance_status"] = (base_report.get("L2_status") or {}).get("acceptance_status")
        merged["final_repair_applied"] = False
        merged["final_repair_needed"] = bool(base_report.get("repair_needed"))
        return merged

    merged["pre_repair_L2_status"] = base_report.get("L2_status") or {}
    merged["L2_status"] = _repair_status_to_l2(repair)
    merged["verifier_reason"] = repair.get("verifier_reason")
    merged["repair_needed"] = bool(repair.get("repair_needed_after_round"))
    merged["final_acceptance_status"] = merged["L2_status"]["acceptance_status"]
    merged["final_repair_applied"] = True
    merged["final_repair_needed"] = bool(repair.get("repair_needed_after_round"))
    merged["repair_report"] = {
        "failure_type": repair.get("failure_type"),
        "selected_coarse_indices": repair.get("selected_coarse_indices") or [],
        "selection_mode": repair.get("selection_mode"),
        "patch_counts": repair.get("patch_counts") or {},
        "not_direct_visual_evidence": bool(repair.get("not_direct_visual_evidence")),
        "llm_budget_summary": repair.get("llm_budget_summary") or {},
    }
    return merged


def _sum_usage(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "calls": sum(int(item.get("calls") or 0) for item in items),
        "prompt_chars": sum(int(item.get("prompt_chars") or 0) for item in items),
        "prompt_approx_tokens": sum(int(item.get("prompt_approx_tokens") or 0) for item in items),
        "output_chars": sum(int(item.get("output_chars") or 0) for item in items),
        "malformed_json_count": sum(int(item.get("malformed_json_count") or 0) for item in items),
        "timeout_count": sum(int(item.get("timeout_count") or 0) for item in items),
        "compact_retry_count": sum(int(item.get("compact_retry_count") or 0) for item in items),
        "cache_hits": sum(int(item.get("cache_hits") or 0) for item in items),
        "cache_misses": sum(int(item.get("cache_misses") or 0) for item in items),
    }


def _budget_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    clip_schema = []
    graph_compose = []
    repair_plan = []
    repair_l2 = []
    for row in reports:
        budget = row.get("llm_budget_report") or {}
        if isinstance(budget.get("clip_schema"), dict):
            clip_schema.append(budget["clip_schema"])
        if isinstance(budget.get("graph_compose"), dict):
            graph_compose.append(budget["graph_compose"])
        repair_budget = ((row.get("repair_report") or {}).get("llm_budget_summary") or {})
        if isinstance(repair_budget.get("repair_plan"), dict):
            repair_plan.append(repair_budget["repair_plan"])
        if isinstance(repair_budget.get("l2_verifier"), dict):
            repair_l2.append(repair_budget["l2_verifier"])
    return {
        "clip_schema": _sum_usage(clip_schema),
        "graph_compose": _sum_usage(graph_compose),
        "repair_plan": _sum_usage(repair_plan),
        "repair_l2_verifier": _sum_usage(repair_l2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final 5-dataset L1/L2/repair acceptance report.")
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base = _read_json(args.quality_report)
    repair_by_id: dict[str, dict[str, Any]] = {}
    for path in args.repair_report:
        payload = _read_json(path)
        for report in payload.get("reports") or []:
            if report.get("example_id"):
                repair_by_id[str(report["example_id"])] = report

    reports = [_merge_report(row, repair_by_id) for row in base.get("reports") or []]
    by_l1: dict[str, int] = {}
    by_l2: dict[str, int] = {}
    for row in reports:
        l1 = str(((row.get("L1_quality") or {}).get("grade")) or "missing")
        l2 = str(row.get("final_acceptance_status") or "missing")
        by_l1[l1] = by_l1.get(l1, 0) + 1
        by_l2[l2] = by_l2.get(l2, 0) + 1

    summary = {
        "examples": len(reports),
        "datasets": sorted({str(row.get("dataset")) for row in reports}),
        "l1_quality_counts": by_l1,
        "final_l2_status_counts": by_l2,
        "repair_applied": sum(1 for row in reports if row.get("final_repair_applied")),
        "repair_needed_after_final": sum(1 for row in reports if row.get("final_repair_needed")),
        "high_l1_all": bool(reports) and all(((row.get("L1_quality") or {}).get("grade")) == "high" for row in reports),
        "accepted_all": bool(reports)
        and all(str(row.get("final_acceptance_status")) in {"accepted_strong", "accepted_bridge"} for row in reports),
        "strict_vlm_perception_all": bool(reports)
        and all((row.get("strict_vlm_perception") or {}).get("qwen_only") for row in reports),
        "fallback_clip_schema_total": sum(
            int((row.get("strict_vlm_perception") or {}).get("fallback_clip_schema_count") or 0) for row in reports
        ),
        "model_error_clip_schema_total": sum(
            int((row.get("strict_vlm_perception") or {}).get("model_error_clip_schema_count") or 0) for row in reports
        ),
        "llm_budget_summary": _budget_summary(reports),
    }
    payload = {"summary": summary, "reports": reports}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

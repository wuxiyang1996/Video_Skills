#!/usr/bin/env python3
"""Audit atomic-skill execution health for L2 terminal rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("cg_bench", "video_holmes")


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("dataset") or ""),
        int(row.get("group") or 0),
        int(row.get("sample") or 0),
    )


def audit_atomic_skill_execution(
    samples: Iterable[dict[str, Any]], traces: Iterable[dict[str, Any]], *, min_rollouts: int = 10
) -> dict[str, Any]:
    sample_rows = list(samples)
    trace_rows = list(traces)
    sample_by_key = {_key(row): row for row in sample_rows}
    duplicate_sample_keys = len(sample_by_key) != len(sample_rows)
    trace_keys = [_key(row) for row in trace_rows]
    duplicate_trace_keys = len(set(trace_keys)) != len(trace_keys)
    unmatched_trace_keys = [key for key in trace_keys if key not in sample_by_key]

    dataset_reports: dict[str, Any] = {}
    all_skill_ids: set[str] = set()
    for dataset in DATASETS:
        ds_samples = [row for row in sample_rows if row.get("dataset") == dataset]
        ds_traces = [row for row in trace_rows if row.get("dataset") == dataset]
        cache_hits = sum(bool(row.get("executor_cache_hit")) for row in ds_samples)
        terminal_successes = sum(bool(row.get("terminal_success")) for row in ds_samples)
        process_supported = sum(bool(row.get("process_supported")) for row in ds_samples)
        trace_ok = [int((row.get("rollout_diagnostic") or {}).get("trace_ok") or 0) for row in ds_samples]
        trace_fail = [int((row.get("rollout_diagnostic") or {}).get("trace_fail") or 0) for row in ds_samples]

        skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
        failure_codes: dict[str, Counter[str]] = defaultdict(Counter)
        planned_steps = planned_steps_observed = traced_steps = 0
        unplanned_trace_steps = repeated_planned_trace_steps = 0
        verified_steps = failed_steps = 0
        for row in ds_traces:
            plan = ((row.get("llm_plan") or {}).get("reasoning_plan") or [])
            planned_ids = {str(step.get("step_id") or "") for step in plan if step.get("step_id")}
            planned_steps += len(planned_ids)
            skill_trace = row.get("skill_trace") or []
            traced_steps += len(skill_trace)
            traced_planned_counts: Counter[str] = Counter(
                str(step.get("step_id") or "")
                for step in skill_trace
                if str(step.get("step_id") or "") in planned_ids
            )
            planned_steps_observed += len(traced_planned_counts)
            repeated_planned_trace_steps += sum(
                max(0, count - 1) for count in traced_planned_counts.values()
            )
            unplanned_trace_steps += sum(
                str(step.get("step_id") or "") not in planned_ids for step in skill_trace
            )
            for step in skill_trace:
                skill_id = str(step.get("skill_id") or "")
                if not skill_id:
                    continue
                all_skill_ids.add(skill_id)
                status = str(step.get("status") or "unknown")
                skill_counts[skill_id]["calls"] += 1
                skill_counts[skill_id][status] += 1
                if status == "verified":
                    verified_steps += 1
                else:
                    failed_steps += 1
                    failure_codes[skill_id][str(step.get("failure_code") or "unknown")] += 1

        skills = {}
        for skill_id, counts in sorted(skill_counts.items()):
            calls = int(counts["calls"])
            verified = int(counts["verified"])
            skills[skill_id] = {
                "calls": calls,
                "verified": verified,
                "failed_or_other": calls - verified,
                "verified_rate": verified / max(1, calls),
                "status_counts": dict(sorted(counts.items())),
                "failure_codes": dict(sorted(failure_codes[skill_id].items())),
            }

        outcome_health = {}
        for outcome, predicate in (
            ("terminal_success", lambda row: bool(row.get("terminal_success"))),
            ("terminal_failure", lambda row: not bool(row.get("terminal_success"))),
        ):
            rows = [row for row in ds_samples if predicate(row)]
            oks = [int((row.get("rollout_diagnostic") or {}).get("trace_ok") or 0) for row in rows]
            fails = [int((row.get("rollout_diagnostic") or {}).get("trace_fail") or 0) for row in rows]
            outcome_health[outcome] = {
                "rollouts": len(rows),
                "mean_trace_ok": sum(oks) / max(1, len(oks)),
                "mean_trace_fail": sum(fails) / max(1, len(fails)),
            }

        dataset_reports[dataset] = {
            "rollout_metrics": {
                "rollouts": len(ds_samples),
                "terminal_successes": terminal_successes,
                "terminal_success_rate": terminal_successes / max(1, len(ds_samples)),
                "process_supported": process_supported,
                "process_supported_rate": process_supported / max(1, len(ds_samples)),
                "executor_cache_hits": cache_hits,
                "executor_cache_misses": len(ds_samples) - cache_hits,
                "mean_trace_ok": sum(trace_ok) / max(1, len(trace_ok)),
                "mean_trace_fail": sum(trace_fail) / max(1, len(trace_fail)),
            },
            "fresh_execution_metrics": {
                "executor_traces": len(ds_traces),
                "planned_steps": planned_steps,
                "planned_steps_observed": planned_steps_observed,
                "planned_step_completion_rate": planned_steps_observed / max(1, planned_steps),
                "traced_steps": traced_steps,
                "unplanned_trace_steps": unplanned_trace_steps,
                "repeated_planned_trace_steps": repeated_planned_trace_steps,
                "verified_steps": verified_steps,
                "failed_or_other_steps": failed_steps,
                "verified_step_rate": verified_steps / max(1, traced_steps),
            },
            "outcome_skill_health": outcome_health,
            "skills": skills,
        }

    checks = {
        "unique_sample_keys": not duplicate_sample_keys,
        "unique_trace_keys": not duplicate_trace_keys,
        "all_traces_match_samples": not unmatched_trace_keys,
        "both_datasets_have_min_rollouts": all(
            dataset_reports[dataset]["rollout_metrics"]["rollouts"] >= min_rollouts
            for dataset in DATASETS
        ),
        "both_datasets_have_fresh_executor_traces": all(
            dataset_reports[dataset]["fresh_execution_metrics"]["executor_traces"] > 0
            for dataset in DATASETS
        ),
        "atomic_skill_ids_observed": bool(all_skill_ids),
    }
    return {
        "schema_version": "video-skills/l2-atomic-skill-execution-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "min_rollouts": min_rollouts,
        "sample_rows": len(sample_rows),
        "fresh_executor_trace_rows": len(trace_rows),
        "unmatched_trace_keys": [list(key) for key in unmatched_trace_keys[:20]],
        "atomic_skill_count": len(all_skill_ids),
        "datasets": dataset_reports,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--min-rollouts", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_atomic_skill_execution(
            _read_jsonl(args.samples), _read_jsonl(args.traces), min_rollouts=args.min_rollouts
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        report = {
            "schema_version": "video-skills/l2-atomic-skill-execution-audit-v1",
            "passed": False,
            "checks": {"inputs_valid": False},
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

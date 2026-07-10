#!/usr/bin/env python3
"""Collect the current gated cold-start SFT snapshot across controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .training.l1_builder_sft_adapter import build_l1_builder_exports
from .training.l1_patch_sft_adapter import build_l1_patch_exports
from .training.l2_retrieval_sft_adapter import build_l2_retrieval_exports
from .training.motif_sft_adapter import build_motif_exports
from .training.sft_common import read_jsonl, write_json, write_jsonl
from .training.stepwise_sft_adapter import build_stepwise_exports
from .training.verifier_sft_adapter import build_verifier_exports


def _existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _glob_files(roots: list[Path], pattern: str) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(dict.fromkeys(paths))


def _glob_dirs(roots: list[Path], pattern: str) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.glob(pattern) if path.is_dir())
    return sorted(dict.fromkeys(paths))


def _write_export(
    output_dir: Path,
    name: str,
    transitions: list[dict[str, Any]],
    chats: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    transition_path = output_dir / f"{name}_transitions.jsonl"
    sft_path = output_dir / f"{name}_sft.jsonl"
    report_path = output_dir / f"{name}_report.json"
    write_jsonl(transition_path, transitions)
    write_jsonl(sft_path, chats)
    write_json(report_path, report)
    return {
        "controller": name,
        "transitions": len(transitions),
        "sft_chats": len(chats),
        "transition_path": str(transition_path),
        "sft_path": str(sft_path),
        "report_path": str(report_path),
        "prompt_forbidden_key_hits": report.get("prompt_forbidden_key_hits"),
    }


def _merge_report(reports: list[dict[str, Any]], schema_version: str) -> dict[str, Any]:
    merged: dict[str, Any] = {"schema_version": schema_version, "parts": reports}
    merged["exported_transitions"] = sum(int(report.get("exported_transitions") or 0) for report in reports)
    merged["exported_sft_chats"] = sum(int(report.get("exported_sft_chats") or 0) for report in reports)
    merged["prompt_forbidden_key_hits"] = sum(int(report.get("prompt_forbidden_key_hits") or 0) for report in reports)
    return merged


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_roots = _existing(args.pilot_root)
    # Keep current pilots before historical rollouts so duplicate example_ids
    # prefer corrected/new teacher traces during L1 export de-duplication.
    rollout_paths = _glob_files(pilot_roots, "**/examples.jsonl")
    rollout_paths.extend(_existing(args.extra_rollout_jsonl))
    rollout_paths = list(dict.fromkeys(rollout_paths))

    repair_results = _glob_files(pilot_roots, "**/repair_results.json")
    repair_results.extend(_glob_files(pilot_roots, "**/repair_results.jsonl"))
    repair_results.extend(_existing(args.repair_results))
    repair_results = sorted(dict.fromkeys(repair_results))

    repair_stage_roots = _glob_dirs(pilot_roots, "**/repair_stages")
    repair_stage_roots.extend(_existing(args.repair_stage_root))
    repair_stage_roots = sorted(dict.fromkeys(repair_stage_roots))

    summary: dict[str, Any] = {
        "schema_version": "video-skills/sft-snapshot-report-v0.1",
        "output_dir": str(output_dir),
        "pilot_roots": [str(path) for path in pilot_roots],
        "rollout_jsonl": [str(path) for path in rollout_paths],
        "repair_results": [str(path) for path in repair_results],
        "repair_stage_roots": [str(path) for path in repair_stage_roots],
        "exports": [],
    }

    if rollout_paths:
        transitions, chats, report = build_l1_builder_exports(
            rollout_paths,
            include_datasets=set(args.include_dataset) or None,
            exclude_datasets=set(args.exclude_dataset) or None,
            max_transitions_per_example=args.max_transitions_per_example,
        )
        summary["exports"].append(_write_export(output_dir, "l1_builder", transitions, chats, report))

        transitions, chats, report = build_l2_retrieval_exports(
            rollout_paths,
            repair_results_paths=repair_results,
        )
        summary["exports"].append(_write_export(output_dir, "l2_retrieval", transitions, chats, report))

    l1_patch_transitions: list[dict[str, Any]] = []
    l1_patch_chats: list[dict[str, Any]] = []
    l1_patch_reports: list[dict[str, Any]] = []
    for stage_root in repair_stage_roots:
        transitions, chats, report = build_l1_patch_exports(stage_root)
        l1_patch_transitions.extend(transitions)
        l1_patch_chats.extend(chats)
        l1_patch_reports.append(report)
    if l1_patch_reports:
        summary["exports"].append(
            _write_export(
                output_dir,
                "l1_patch",
                l1_patch_transitions,
                l1_patch_chats,
                _merge_report(l1_patch_reports, "video-skills/l1-patch-sft-merged-report-v0.1"),
            )
        )

    verifier_transitions: list[dict[str, Any]] = []
    verifier_chats: list[dict[str, Any]] = []
    verifier_reports: list[dict[str, Any]] = []
    for stage_root in repair_stage_roots:
        transitions, chats, report = build_verifier_exports(stage_root, None, balance_decisions=False)
        verifier_transitions.extend(transitions)
        verifier_chats.extend(chats)
        verifier_reports.append(report)
    for expert_demo in _existing(args.expert_demos):
        transitions, chats, report = build_verifier_exports(None, expert_demo, balance_decisions=False)
        verifier_transitions.extend(transitions)
        verifier_chats.extend(chats)
        verifier_reports.append(report)
    if args.balance_verifier:
        supported = [row for row in verifier_transitions if row["action_t"]["arguments"].get("decision") == "supported"]
        insufficient = [
            row for row in verifier_transitions
            if row["action_t"]["arguments"].get("decision") == "insufficient"
        ]
        quota = min(len(supported), len(insufficient))
        verifier_transitions = sorted(supported[:quota] + insufficient[:quota], key=lambda row: row["transition_id"])
        wanted_ids = {row["transition_id"] for row in verifier_transitions}
        verifier_chats = [row for row in verifier_chats if row["transition_id"] in wanted_ids]
    if verifier_reports:
        report = _merge_report(verifier_reports, "video-skills/verifier-sft-merged-report-v0.1")
        report["balanced_snapshot"] = bool(args.balance_verifier)
        report["exported_transitions"] = len(verifier_transitions)
        report["exported_sft_chats"] = len(verifier_chats)
        decision_counts: dict[str, int] = {}
        failure_code_counts: dict[str, int] = {}
        for row in verifier_transitions:
            arguments = row["action_t"]["arguments"]
            decision = str(arguments.get("decision") or "unknown")
            failure_code = str(arguments.get("failure_code") or "none")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            failure_code_counts[failure_code] = failure_code_counts.get(failure_code, 0) + 1
        report["decision_counts"] = decision_counts
        report["failure_code_counts"] = failure_code_counts
        summary["exports"].append(_write_export(output_dir, "verifier", verifier_transitions, verifier_chats, report))

    for expert_demo in _existing(args.expert_demos):
        transitions, chats, report = build_stepwise_exports(read_jsonl(expert_demo), source_path=str(expert_demo))
        summary["exports"].append(
            _write_export(
                output_dir,
                f"l2_repair_rounds_{expert_demo.stem}",
                transitions,
                chats,
                report,
            )
        )

    if args.motif_bank and args.motif_bank.exists():
        transitions, chats, report = build_motif_exports(args.motif_bank)
        summary["exports"].append(_write_export(output_dir, "motif_lifecycle", transitions, chats, report))

    summary["total_sft_chats"] = sum(int(row["sft_chats"]) for row in summary["exports"])
    summary["total_transitions"] = sum(int(row["transitions"]) for row in summary["exports"])
    summary["total_prompt_forbidden_key_hits"] = sum(
        int(row["prompt_forbidden_key_hits"] or 0) for row in summary["exports"]
    )
    write_json(output_dir / "snapshot_report.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, action="append", default=[])
    parser.add_argument("--extra-rollout-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--repair-results", type=Path, action="append", default=[])
    parser.add_argument("--repair-stage-root", type=Path, action="append", default=[])
    parser.add_argument("--expert-demos", type=Path, action="append", default=[])
    parser.add_argument("--motif-bank", type=Path)
    parser.add_argument("--include-dataset", action="append", default=["cg_bench", "video_holmes"])
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--max-transitions-per-example", type=int, default=256)
    parser.add_argument("--balance-verifier", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = collect_snapshot(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

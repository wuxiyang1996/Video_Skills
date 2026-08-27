#!/usr/bin/env python3
"""Export repair protocol reports as MDP-style L2/repair controller SFT rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import write_json, write_jsonl
from .stepwise_sft_adapter import (
    _contains_forbidden_prompt_key,
    _observation_for_round,
    _transition_to_sft_chat,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_reports(stage_root: Path) -> list[Path]:
    if stage_root.is_file() and stage_root.name == "repair_05_report.json":
        return [stage_root]
    if stage_root.is_dir():
        return sorted(stage_root.glob("*/repair_05_report.json"))
    return []


def _state_for_repair_report(
    report: dict[str, Any],
    trajectory: dict[str, Any],
    round_row: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": "video-skills/mdp-round-state-v0.2",
        "process_model": "pomdp_compatible_l2_repair_controller",
        "transition_granularity": "l2_or_repair_round",
        "known_limitation": "repair-protocol stage export; state is compact and hides gold labels",
        "demo_id": f"repair::{report.get('example_id')}",
        "dataset": report.get("dataset"),
        "example_id": report.get("example_id"),
        "video_regime": report.get("video_regime"),
        "task_family": "repair_protocol_l2_controller",
        "visible_demo_inputs": {
            "gap_types": report.get("gap_types") or [],
            "selector_status": report.get("selector_status"),
            "selected_coarse_indices": report.get("selected_coarse_indices") or [],
            "negative_coarse_indices": report.get("negative_coarse_indices") or [],
            "patch_counts": report.get("patch_counts") or {},
        },
        "l1_compact": {
            "patch_counts": report.get("patch_counts") or {},
            "negative_target_evidence_nodes": report.get("negative_target_evidence_nodes", 0),
        },
        "l1_round_snapshot": round_row.get("state_snapshot") or {},
        "prior_round_summaries": [],
        "budget_state": {
            "round_index": round_row.get("round_index", round_index),
            "max_repair_rounds": trajectory.get("max_repair_rounds"),
        },
    }


def _transition_for_round(
    *,
    report: dict[str, Any],
    trajectory: dict[str, Any],
    round_row: dict[str, Any],
    report_path: Path,
    round_index: int,
) -> dict[str, Any]:
    round_type = str(round_row.get("round_type") or "repair_l2_reasoning")
    terminal_status = round_row.get("terminal_status") or report.get("repair_status")
    state = _state_for_repair_report(report, trajectory, round_row, round_index)
    action = {
        "schema_version": "video-skills/controller-action-v0.1",
        "round_type": round_type,
        "action": round_row.get("action") if isinstance(round_row.get("action"), dict) else {},
        "target_policy": "choose_next_l2_or_repair_tool_action",
    }
    transition = {
        "schema_version": "video-skills/mdp-round-transition-v0.2",
        "transition_id": f"{report.get('example_id')}::repair_report_round:{round_row.get('round_index', round_index)}",
        "demo_id": f"repair::{report.get('example_id')}",
        "dataset": report.get("dataset"),
        "example_id": report.get("example_id"),
        "split_role": "cold_start_sft",
        "controller": "l2_repair",
        "state_t": state,
        "action_t": action,
        "observation_t": _observation_for_round(round_row),
        "state_t_plus_1_summary": {
            "terminal_status": terminal_status,
            "acceptance_status": report.get("repair_status"),
            "failure_type": report.get("failure_type"),
            "recommended_next_action": report.get("recommended_next_action"),
            "graph_delta": round_row.get("graph_delta") or {},
        },
        "reward_proxy_t": round_row.get("reward_proxy") or {},
        "done": bool(terminal_status),
        "quality_flags": {
            "training_candidate": report.get("repair_status") in {"resolved_strong", "accepted_bridge"},
            "abstain_candidate": bool(report.get("repair_needed_after_round")),
            "repair_needed_after_round": bool(report.get("repair_needed_after_round")),
            "failure_type": report.get("failure_type"),
        },
        "source_stage_dir": str(report_path.parent),
        "source_report_path": str(report_path),
    }
    return transition


def build_repair_report_stepwise_exports(
    stage_roots: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    chats: list[dict[str, Any]] = []
    round_type_counts: Counter[str] = Counter()
    terminal_status_counts: Counter[str] = Counter()
    repair_status_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    prompt_forbidden_key_hits = 0
    report_paths: list[Path] = []

    for stage_root in stage_roots:
        for report_path in _find_reports(stage_root):
            report_paths.append(report_path)
            report = _read_json(report_path)
            trajectory = report.get("l2_trajectory") if isinstance(report.get("l2_trajectory"), dict) else {}
            rounds = trajectory.get("rounds") if isinstance(trajectory.get("rounds"), list) else []
            for index, round_row in enumerate(rounds):
                if not isinstance(round_row, dict):
                    continue
                transition = _transition_for_round(
                    report=report,
                    trajectory=trajectory,
                    round_row=round_row,
                    report_path=report_path,
                    round_index=index,
                )
                user_payload = {"task": "choose_next_controller_action", "state_t": transition["state_t"]}
                if _contains_forbidden_prompt_key(user_payload):
                    prompt_forbidden_key_hits += 1
                round_type = str(round_row.get("round_type") or "repair_l2_reasoning")
                terminal_status = str(round_row.get("terminal_status") or report.get("repair_status") or "unknown")
                transitions.append(transition)
                chats.append(_transition_to_sft_chat(transition, round_type, terminal_status))
                round_type_counts[round_type] += 1
                terminal_status_counts[terminal_status] += 1
                repair_status_counts[str(report.get("repair_status") or "unknown")] += 1
                failure_type_counts[str(report.get("failure_type") or "unknown")] += 1

    report = {
        "schema_version": "video-skills/repair-report-stepwise-sft-report-v0.1",
        "source_stage_roots": [str(path) for path in stage_roots],
        "input_reports": len(report_paths),
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "round_type_counts": dict(round_type_counts),
        "terminal_status_counts": dict(terminal_status_counts),
        "repair_status_counts": dict(repair_status_counts),
        "failure_type_counts": dict(failure_type_counts),
        "prompt_forbidden_key_hits": prompt_forbidden_key_hits,
        "granularity": "repair_protocol_l2_round",
        "intended_use": "cold-start behavior cloning for paid repair L2/repair controller traces",
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, action="append", required=True)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_repair_report_stepwise_exports(args.stage_root)
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

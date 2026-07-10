#!/usr/bin/env python3
"""Export repair clip schemas as clip-level MDP-style L1 patch SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_json, read_jsonl, write_json, write_jsonl


def _clip_node_payload(node: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "node_id", "node_type", "text", "modality", "confidence", "clip_id",
        "time_span", "source_type", "producer", "video_id", "provenance", "visibility",
    }
    return {key: node[key] for key in allowed if key in node}


def _edge_payload(edge: dict[str, Any]) -> dict[str, Any]:
    return compact_visibility(edge)


def _plan_state(plan: dict[str, Any]) -> dict[str, Any]:
    clue = plan.get("clue_need_spec") if isinstance(plan.get("clue_need_spec"), dict) else {}
    allowed_clue = {
        "answer_mode_hint", "bridge_evidence_criteria", "clip_inspection_instruction",
        "forbidden_modalities", "insufficient_evidence_rule", "must_find_visual_evidence",
        "negative_evidence_to_exclude", "positive_evidence_criteria", "temporal_or_action_cues",
        "visual_attributes_to_resolve", "visual_target",
    }
    return {
        "strategy": plan.get("strategy"),
        "repair_mode": plan.get("repair_mode"),
        "gap_types": plan.get("gap_types") or [],
        "clue_need_spec": {key: compact_visibility(clue[key]) for key in allowed_clue if key in clue},
    }


def _build_one_directory(stage_dir: Path) -> list[dict[str, Any]]:
    plan_path = stage_dir / "repair_01_plan.json"
    clips_path = stage_dir / "repair_02_clip_schemas.jsonl"
    patch_path = stage_dir / "repair_03_l1_patch.json"
    if not (plan_path.exists() and clips_path.exists() and patch_path.exists()):
        return []
    plan = read_json(plan_path)
    patch = read_json(patch_path)
    clips = read_jsonl(clips_path)
    nodes = [row for row in patch.get("nodes", []) if isinstance(row, dict)]
    edges = [row for row in patch.get("edges", []) if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    applied_node_ids: list[str] = []
    eligible_clips = [clip for clip in clips if any(str(row.get("clip_id") or "") == str(clip.get("clip_id") or "") for row in nodes)]

    for index, clip in enumerate(eligible_clips):
        clip_id = str(clip.get("clip_id") or "")
        clip_nodes = [row for row in nodes if str(row.get("clip_id") or "") == clip_id]
        if not clip_nodes:
            continue
        clip_node_ids = {str(row.get("node_id")) for row in clip_nodes}
        clip_edges = []
        for edge in edges:
            endpoints = {
                str(edge.get(key)) for key in ("source", "target", "source_id", "target_id", "from", "to")
                if edge.get(key) is not None
            }
            if endpoints & clip_node_ids:
                clip_edges.append(edge)
        state = {
            "schema_version": "video-skills/l1-patch-state-v0.1",
            "process_model": "pomdp_compatible_l1_evidence_controller",
            "dataset": patch.get("dataset"),
            "example_id": patch.get("example_id"),
            "repair_goal": _plan_state(plan),
            "clip_schema": compact_visibility(clip),
            "partial_l1_summary": {
                "applied_clip_count": index,
                "applied_node_ids": applied_node_ids[-64:],
                "remaining_clip_budget": max(0, len(eligible_clips) - index),
            },
        }
        action = {
            "schema_version": "video-skills/l1-patch-action-v0.1",
            "tool_name": "apply_l1_evidence_patch",
            "arguments": {
                "clip_id": clip_id,
                "nodes": [_clip_node_payload(row) for row in clip_nodes],
                "edges": [_edge_payload(row) for row in clip_edges],
            },
        }
        transition_id = f"{patch.get('example_id')}::l1_patch::{index}"
        rows.append({
            "schema_version": "video-skills/l1-patch-transition-v0.1",
            "transition_id": transition_id,
            "controller": "l1_patch",
            "state_t": state,
            "action_t": action,
            "observation_t": {
                "patch_status": "grounded",
                "created_node_count": len(clip_nodes),
                "created_edge_count": len(clip_edges),
                "validation": "source_patch_membership",
            },
            "state_t_plus_1_summary": {
                "total_applied_node_count": len(applied_node_ids) + len(clip_nodes),
                "last_clip_id": clip_id,
            },
            "reward_proxy_t": {"grounded_patch": 1.0, "hidden_supervision": 0.0},
            "done": index == len(eligible_clips) - 1,
            "source_stage_dir": str(stage_dir),
        })
        applied_node_ids.extend(str(row.get("node_id")) for row in clip_nodes if row.get("node_id"))
    return rows


def build_l1_patch_exports(stage_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for patch_path in sorted(stage_root.glob("*/repair_03_l1_patch.json")):
        transitions.extend(_build_one_directory(patch_path.parent))
    chats: list[dict[str, Any]] = []
    forbidden_hits = 0
    dataset_counts: Counter[str] = Counter()
    for row in transitions:
        user_payload = {"task": "choose_next_l1_graph_action", "state_t": row["state_t"]}
        forbidden_hits += int(contains_forbidden_prompt_key(user_payload))
        dataset_counts[str(row["state_t"].get("dataset") or "unknown")] += 1
        chats.append({
            "schema_version": "video-skills/l1-patch-sft-chat-v0.1",
            "transition_id": row["transition_id"],
            "messages": [
                {"role": "system", "content": "You are the Video_Skills L1 evidence controller. Choose one grounded graph patch tool action from the visible clip schema and partial graph state. Return JSON only."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(row["action_t"], ensure_ascii=False, separators=(",", ":"))},
            ],
            "metadata": {"controller": "l1_patch", "dataset": row["state_t"].get("dataset")},
        })
    report = {
        "schema_version": "video-skills/l1-patch-sft-report-v0.1",
        "source_stage_root": str(stage_root),
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "dataset_counts": dict(dataset_counts),
        "prompt_forbidden_key_hits": forbidden_hits,
        "granularity": "one grounded L1 patch action per inspected clip",
        "known_bias": "repair-focused positives; add no-op and rejected patch actions before broad L1 policy training",
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_l1_patch_exports(args.stage_root)
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

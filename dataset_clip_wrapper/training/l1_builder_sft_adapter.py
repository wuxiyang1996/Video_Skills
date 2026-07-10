#!/usr/bin/env python3
"""Recover atomic L1 builder transitions from strict rollout execution traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_jsonl, write_json, write_jsonl


SUPPORTED_SKILLS = {
    "segment_video_or_select_clip",
    "neighbor_vlm_l1_create_node",
    "neighbor_vlm_l1_create_schema_anchor",
    "neighbor_vlm_l1_create_edge",
    "neighbor_vlm_l1_skip_edge",
    "short_video_recurrence_create_clue",
    "short_video_recurrence_link",
}


def _skill_balanced_cap(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        skill_id = str((row.get("action_t") or {}).get("tool_name") or "unknown")
        grouped.setdefault(skill_id, []).append(row)
    selected: list[dict[str, Any]] = []
    offsets = {skill_id: 0 for skill_id in grouped}
    while len(selected) < limit:
        made_progress = False
        for skill_id in sorted(grouped):
            offset = offsets[skill_id]
            if offset >= len(grouped[skill_id]):
                continue
            selected.append(grouped[skill_id][offset])
            offsets[skill_id] += 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def _node_clip_id(node: dict[str, Any]) -> str:
    return str(node.get("clip_id") or node.get("source_clip_id") or "")


def _edge_clip_ids(edge: dict[str, Any]) -> list[str]:
    values = [edge.get("src_clip_id"), edge.get("dst_clip_id")]
    return [str(value) for value in values if value]


def _action_for_trace(
    trace: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    edge_by_id: dict[str, dict[str, Any]],
    graph: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    skill_id = str(trace.get("skill_id") or "")
    clip_ids: list[str] = []
    if skill_id == "segment_video_or_select_clip":
        arguments = {
            "clip_policy": compact_visibility(graph.get("clip_policy") or {}),
            "video_regime": graph.get("video_regime"),
            "observation_end_s": graph.get("observation_end_s"),
        }
    elif "create_node" in skill_id or "create_schema_anchor" in skill_id or "create_clue" in skill_id:
        node = node_by_id.get(str(trace.get("node_id") or ""))
        if node is None:
            return None, []
        arguments = {"node": compact_visibility(node)}
        clip_id = _node_clip_id(node) or str(trace.get("clip_id") or "")
        if clip_id:
            clip_ids.append(clip_id)
    elif "create_edge" in skill_id or skill_id == "short_video_recurrence_link":
        edge = edge_by_id.get(str(trace.get("edge_id") or ""))
        if edge is None:
            return None, []
        arguments = {"edge": compact_visibility(edge)}
        clip_ids.extend(_edge_clip_ids(edge))
    elif skill_id == "neighbor_vlm_l1_skip_edge":
        arguments = compact_visibility({key: value for key, value in trace.items() if key not in {"skill_id", "ok"}})
        clip_ids.extend(str(value) for key, value in trace.items() if key.endswith("clip_id") and value)
    else:
        return None, []
    return {
        "schema_version": "video-skills/l1-builder-action-v0.1",
        "tool_name": skill_id,
        "arguments": arguments,
    }, list(dict.fromkeys(clip_ids))


def _build_rollout_transitions(row: dict[str, Any], source_path: Path, row_index: int) -> tuple[list[dict[str, Any]], int]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    graph = metadata.get("clue_memory_graph") if isinstance(metadata.get("clue_memory_graph"), dict) else {}
    compose = metadata.get("graph_compose") if isinstance(metadata.get("graph_compose"), dict) else {}
    traces = compose.get("execution_trace") if isinstance(compose.get("execution_trace"), list) else []
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    node_by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id")}
    edge_by_id = {str(edge.get("edge_id")): edge for edge in edges if edge.get("edge_id")}
    clip_schemas = metadata.get("clip_schemas") if isinstance(metadata.get("clip_schemas"), list) else []
    clip_by_id = {str(clip.get("clip_id")): compact_visibility(clip) for clip in clip_schemas if isinstance(clip, dict) and clip.get("clip_id")}
    eligible: list[tuple[int, dict[str, Any], dict[str, Any], list[str]]] = []
    unresolved = 0
    recent_failures: list[dict[str, Any]] = []
    failure_state_by_trace: dict[int, list[dict[str, Any]]] = {}

    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue
        skill_id = str(trace.get("skill_id") or "")
        if not bool(trace.get("ok", False)):
            recent_failures.append(compact_visibility(trace))
            recent_failures = recent_failures[-4:]
            continue
        if skill_id not in SUPPORTED_SKILLS:
            continue
        action, clip_ids = _action_for_trace(trace, node_by_id, edge_by_id, graph)
        if action is None:
            unresolved += 1
            continue
        failure_state_by_trace[trace_index] = list(recent_failures)
        eligible.append((trace_index, trace, action, clip_ids))

    transitions: list[dict[str, Any]] = []
    applied_node_ids: list[str] = []
    applied_edge_ids: list[str] = []
    for action_index, (trace_index, trace, action, clip_ids) in enumerate(eligible):
        visible_clips = [clip_by_id[clip_id] for clip_id in clip_ids if clip_id in clip_by_id]
        state = {
            "schema_version": "video-skills/l1-builder-state-v0.1",
            "process_model": "pomdp_compatible_l1_evidence_controller",
            "dataset": row.get("dataset"),
            "example_id": row.get("example_id"),
            "split": row.get("split"),
            "task_family": row.get("task_family"),
            "video_state": {
                "video_id": graph.get("video_id"),
                "video_regime": graph.get("video_regime"),
                "observation_end_s": graph.get("observation_end_s"),
            },
            "current_clip_schemas": visible_clips,
            "partial_l1_summary": {
                "node_count": len(applied_node_ids),
                "edge_count": len(applied_edge_ids),
                "recent_node_ids": applied_node_ids[-32:],
                "recent_edge_ids": applied_edge_ids[-32:],
            },
            "recent_tool_failures": failure_state_by_trace.get(trace_index, []),
            "budget_state": {
                "actions_taken": action_index,
                "future_teacher_trace_length_visible": False,
            },
        }
        transition_id = f"{row.get('example_id')}::l1_builder::{trace_index}"
        transitions.append({
            "schema_version": "video-skills/l1-builder-transition-v0.1",
            "transition_id": transition_id,
            "controller": "l1_builder",
            "state_t": state,
            "action_t": action,
            "observation_t": {"tool_ok": True, "recorded_trace": compact_visibility(trace)},
            "state_t_plus_1_summary": {
                "node_count": len(applied_node_ids) + int("node" in action["arguments"]),
                "edge_count": len(applied_edge_ids) + int("edge" in action["arguments"]),
            },
            "reward_proxy_t": {"recorded_tool_success": 1.0, "video_only_visible": 1.0},
            "done": action_index == len(eligible) - 1,
            "source_rollout_jsonl": str(source_path),
            "source_row_index": row_index,
        })
        if "node" in action["arguments"]:
            node_id = action["arguments"]["node"].get("node_id")
            if node_id:
                applied_node_ids.append(str(node_id))
        if "edge" in action["arguments"]:
            edge_id = action["arguments"]["edge"].get("edge_id")
            if edge_id:
                applied_edge_ids.append(str(edge_id))
    return transitions, unresolved


def build_l1_builder_exports(
    paths: list[Path],
    *,
    include_datasets: set[str] | None = None,
    exclude_datasets: set[str] | None = None,
    max_transitions_per_example: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    input_rows = 0
    skipped_rows = 0
    skipped_duplicate_examples = 0
    uncapped_transitions = 0
    seen_example_ids: set[str] = set()
    unresolved_actions = 0
    for path in paths:
        for row_index, row in enumerate(read_jsonl(path)):
            input_rows += 1
            dataset = str(row.get("dataset") or "")
            if (include_datasets and dataset not in include_datasets) or (
                exclude_datasets and dataset in exclude_datasets
            ):
                skipped_rows += 1
                continue
            example_id = str(row.get("example_id") or "")
            if example_id and example_id in seen_example_ids:
                skipped_duplicate_examples += 1
                continue
            if example_id:
                seen_example_ids.add(example_id)
            built, unresolved = _build_rollout_transitions(row, path, row_index)
            uncapped_transitions += len(built)
            built = _skill_balanced_cap(built, max_transitions_per_example)
            transitions.extend(built)
            unresolved_actions += unresolved
    chats: list[dict[str, Any]] = []
    forbidden_hits = 0
    skill_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    for row in transitions:
        user_payload = {"task": "choose_next_l1_atomic_skill", "state_t": row["state_t"]}
        forbidden_hits += int(contains_forbidden_prompt_key(user_payload))
        skill_counts[str(row["action_t"].get("tool_name") or "unknown")] += 1
        dataset_counts[str(row["state_t"].get("dataset") or "unknown")] += 1
        chats.append({
            "schema_version": "video-skills/l1-builder-sft-chat-v0.1",
            "transition_id": row["transition_id"],
            "messages": [
                {"role": "system", "content": "You are the Video_Skills L1 evidence controller. Given visible clip schemas, partial graph state, failures, and budget, choose the next atomic L1 tool action. Return JSON only."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(row["action_t"], ensure_ascii=False, separators=(",", ":"))},
            ],
            "metadata": {"controller": "l1_builder", "skill_id": row["action_t"].get("tool_name"), "dataset": row["state_t"].get("dataset")},
        })
    report = {
        "schema_version": "video-skills/l1-builder-sft-report-v0.1",
        "source_rollout_jsonl": [str(path) for path in paths],
        "input_rows": input_rows,
        "skipped_rows_by_dataset_filter": skipped_rows,
        "skipped_duplicate_examples": skipped_duplicate_examples,
        "include_datasets": sorted(include_datasets or []),
        "exclude_datasets": sorted(exclude_datasets or []),
        "exported_transitions": len(transitions),
        "uncapped_transitions": uncapped_transitions,
        "max_transitions_per_example": max_transitions_per_example,
        "exported_sft_chats": len(chats),
        "skill_counts": dict(skill_counts),
        "dataset_counts": dict(dataset_counts),
        "unresolved_successful_actions": unresolved_actions,
        "prompt_forbidden_key_hits": forbidden_hits,
        "granularity": "one successful atomic L1 execution-trace action",
        "known_bias": "successful teacher actions dominate; recorded failures are observations, not imitation targets",
        "future_teacher_trace_length_in_prompt": False,
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--max-transitions-per-example", type=int)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_l1_builder_exports(
        args.rollout_jsonl,
        include_datasets=set(args.include_dataset) or None,
        exclude_datasets=set(args.exclude_dataset) or None,
        max_transitions_per_example=args.max_transitions_per_example,
    )
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

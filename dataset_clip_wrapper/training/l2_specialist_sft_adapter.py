#!/usr/bin/env python3
"""Expand existing L2 retrieval traces into specialist-policy SFT rows.

The exporter is deliberately offline and deterministic.  It never calls a
teacher model and never exposes benchmark answers in a prompt.  Strong
retrieval traces yield atomic select/stop and pairwise ranking supervision;
failed traces yield post-attempt recovery actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .l2_retrieval_sft_adapter import _catalog, _quality_gate
from .sft_common import (
    compact_visibility,
    contains_forbidden_prompt_key,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


SYSTEM = (
    "You are the Video_Skills L2 retrieval controller. Choose only from the "
    "visible retrieval state and return the requested tool action as JSON. "
    "Catalog entries are hypotheses, not verified facts."
)


def _chat(transition: dict[str, Any], task: str) -> dict[str, Any]:
    payload = {"task": task, "state_t": transition["state_t"]}
    return {
        "schema_version": "video-skills/l2-specialist-sft-chat-v0.1",
        "transition_id": transition["transition_id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(transition["action_t"], ensure_ascii=False, separators=(",", ":"))},
        ],
        "metadata": {
            "controller": "l2_controller",
            "task": task,
            "dataset": transition["state_t"].get("dataset"),
            "source_example_id": transition.get("source_example_id"),
            "augmentation_family": transition.get("augmentation_family"),
            "source_family_weight": transition.get("source_family_weight", 1.0),
            "is_core": bool(transition.get("is_core")),
        },
    }


def _tokens(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return {token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(token) > 1}


def _hard_negatives(state: dict[str, Any], selected: set[int], count: int) -> list[int]:
    query_tokens = _tokens((state.get("question") or {}).get("question_text") or state.get("question") or {})
    scored: list[tuple[int, str, int]] = []
    for row in state.get("l1_coarse_summary_catalog") or []:
        index = int(row.get("coarse_index", -1))
        if index < 0 or index in selected:
            continue
        overlap = len(query_tokens & _tokens(row))
        digest = hashlib.sha256(f"{state.get('example_id')}:{index}".encode()).hexdigest()
        scored.append((-overlap, digest, index))
    return [index for _, _, index in sorted(scored)[:count]]


def _bounded_catalog_row(catalog_row: dict[str, Any]) -> dict[str, Any]:
    """Use one label-independent candidate representation.

    Positive and negative rows must have identical field/cardinality budgets;
    otherwise summary length becomes an easier target than query relevance.
    """
    time_span = catalog_row.get("time_span") if isinstance(catalog_row.get("time_span"), dict) else {}
    return {
        "coarse_index": int(catalog_row.get("coarse_index", -1)),
        "time_span": {
            key: time_span[key]
            for key in ("start_s", "end_s")
            if key in time_span
        },
        "scene_description": str(catalog_row.get("scene_description") or "")[:80],
        "observable_facts": [str(value)[:60] for value in (catalog_row.get("observable_facts") or [])[:1]],
        "events": [str(value)[:60] for value in (catalog_row.get("events") or [])[:1]],
        "searchable_phrases": [str(value)[:50] for value in (catalog_row.get("searchable_phrases") or [])[:1]],
    }


def _positive_expansions(
    positive_transitions: list[dict[str, Any]], hard_negatives_per_selected: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in positive_transitions:
        source_state = compact_visibility(source.get("state_t") or {})
        selected = [int(value) for value in (source.get("action_t") or {}).get("arguments", {}).get("selected_coarse_indices", [])]
        if not selected:
            continue
        selected_set = set(selected)
        bounded_catalog = [
            _bounded_catalog_row(catalog_row)
            for catalog_row in source_state.get("l1_coarse_summary_catalog") or []
            if isinstance(catalog_row, dict)
        ]
        source_state = dict(source_state)
        source_state["l1_coarse_summary_catalog"] = bounded_catalog
        example_id = str(source_state.get("example_id") or "")
        catalog_by_index = {
            int(row.get("coarse_index", -1)): row
            for row in source_state.get("l1_coarse_summary_catalog") or []
        }
        negative_pool = _hard_negatives(
            source_state, selected_set, max(16, hard_negatives_per_selected)
        )

        def catalog_state(indices: list[int], view: str) -> dict[str, Any]:
            wanted = set(indices)
            state = dict(source_state)
            state["l1_coarse_summary_catalog"] = [
                row for row in source_state.get("l1_coarse_summary_catalog") or []
                if int(row.get("coarse_index", -1)) in wanted
            ]
            state["catalog_view"] = view
            return state

        select_action = {
            "schema_version": "video-skills/l2-specialist-action-v0.1",
            "tool_name": "select_coarse_clips",
            "arguments": {
                "selected_coarse_indices": selected,
                "rationale_short": (source.get("action_t") or {}).get("arguments", {}).get("rationale_short"),
            },
        }
        for view, negative_count in (("full", None), ("hard_4", 4), ("hard_8", 8), ("hard_16", 16)):
            state = source_state if negative_count is None else catalog_state(
                selected + negative_pool[:negative_count], view
            )
            result.append({
                "schema_version": "video-skills/l2-specialist-transition-v0.1",
                "transition_id": f"{example_id}::l2_select_set::{view}",
                "controller": "l2_controller",
                "task": "select_coarse_set",
                "state_t": state,
                "action_t": select_action,
                "source_transition_id": source.get("transition_id"),
                "source_example_id": example_id,
                "augmentation_family": "select_set",
                "is_core": view == "full",
            })

        selected_so_far: list[int] = []
        for step, index in enumerate(selected):
            action = {
                "schema_version": "video-skills/l2-specialist-action-v0.1",
                "tool_name": "select_next_coarse_clip",
                "arguments": {"coarse_index": index},
            }
            temporal = [candidate for candidate in catalog_by_index if abs(candidate - index) <= 4]
            atomic_views = {
                "full": source_state,
                "hard": catalog_state([index] + negative_pool[:7], "atomic_hard"),
                "temporal": catalog_state(temporal + [index], "atomic_temporal"),
            }
            for view, base_state in atomic_views.items():
                state = dict(base_state)
                state["selected_coarse_indices_so_far"] = list(selected_so_far)
                result.append({
                    "schema_version": "video-skills/l2-specialist-transition-v0.1",
                    "transition_id": f"{example_id}::l2_atomic_select::{step}:{view}",
                    "controller": "l2_controller",
                    "task": "select_next_coarse_clip",
                    "state_t": state,
                    "action_t": action,
                    "source_transition_id": source.get("transition_id"),
                    "source_example_id": example_id,
                    "augmentation_family": "atomic_select",
                    "is_core": False,
                })

            for rank, negative in enumerate(negative_pool[:hard_negatives_per_selected]):
                candidates = [catalog_by_index.get(index), catalog_by_index.get(negative)]
                if any(candidate is None for candidate in candidates):
                    continue
                candidates = sorted(candidates, key=lambda row: int(row["coarse_index"]))
                ranking_state = {
                    "schema_version": "video-skills/l2-ranking-state-v0.1",
                    "process_model": "pairwise_coarse_retrieval_ranking",
                    "dataset": source_state.get("dataset"),
                    "example_id": source_state.get("example_id"),
                    "question": source_state.get("question"),
                    "candidate_coarse_summaries": candidates,
                    "selected_coarse_indices_so_far": list(selected_so_far[:-1]),
                }
                result.append({
                    "schema_version": "video-skills/l2-specialist-transition-v0.1",
                    "transition_id": f"{example_id}::l2_rank_pair::{step}:{rank}",
                    "controller": "l2_controller",
                    "task": "rank_coarse_candidates",
                    "state_t": ranking_state,
                    "action_t": {
                        "schema_version": "video-skills/l2-specialist-action-v0.1",
                        "tool_name": "choose_better_coarse_candidate",
                        "arguments": {"coarse_index": index},
                    },
                    "source_transition_id": source.get("transition_id"),
                    "source_example_id": example_id,
                    "augmentation_family": "ranking",
                    "is_core": False,
                })

            listwise_state = {
                "schema_version": "video-skills/l2-ranking-state-v0.1",
                "process_model": "listwise_coarse_retrieval_ranking",
                "dataset": source_state.get("dataset"),
                "example_id": example_id,
                "question": source_state.get("question"),
                "candidate_coarse_summaries": [
                    catalog_by_index[candidate]
                    for candidate in sorted(set([index] + negative_pool[:3]))
                    if candidate in catalog_by_index
                ],
                "selected_coarse_indices_so_far": list(selected_so_far),
            }
            result.append({
                "schema_version": "video-skills/l2-specialist-transition-v0.1",
                "transition_id": f"{example_id}::l2_rank_list::{step}",
                "controller": "l2_controller",
                "task": "rank_coarse_candidates_listwise",
                "state_t": listwise_state,
                "action_t": {
                    "schema_version": "video-skills/l2-specialist-action-v0.1",
                    "tool_name": "choose_best_coarse_candidate",
                    "arguments": {"coarse_index": index},
                },
                "source_transition_id": source.get("transition_id"),
                "source_example_id": example_id,
                "augmentation_family": "ranking",
                "is_core": False,
            })
            selected_so_far.append(index)

        continue_prefixes = [[]] + [selected[:end] for end in range(1, len(selected))]
        for decision_index, prefix in enumerate(continue_prefixes):
            continue_state = dict(source_state)
            continue_state["selected_coarse_indices_so_far"] = prefix
            result.append({
                "schema_version": "video-skills/l2-specialist-transition-v0.1",
                "transition_id": f"{example_id}::l2_continue::{decision_index}",
                "controller": "l2_controller",
                "task": "decide_retrieval_stop",
                "state_t": continue_state,
                "action_t": {
                    "schema_version": "video-skills/l2-specialist-action-v0.1",
                    "tool_name": "continue_coarse_retrieval",
                    "arguments": {"selected_coarse_indices": prefix},
                },
                "source_transition_id": source.get("transition_id"),
                "source_example_id": example_id,
                "augmentation_family": "stop_continue",
                "is_core": False,
            })

        stop_state = dict(source_state)
        stop_state["selected_coarse_indices_so_far"] = selected
        result.append({
            "schema_version": "video-skills/l2-specialist-transition-v0.1",
            "transition_id": f"{example_id}::l2_stop::0",
            "controller": "l2_controller",
            "task": "decide_retrieval_stop",
            "state_t": stop_state,
            "action_t": {
                "schema_version": "video-skills/l2-specialist-action-v0.1",
                "tool_name": "stop_coarse_retrieval",
                "arguments": {"selected_coarse_indices": selected},
            },
            "source_transition_id": source.get("transition_id"),
            "source_example_id": example_id,
            "augmentation_family": "stop_continue",
            "is_core": False,
        })
    return result


RECOVERY_ACTIONS = {
    "no_gpt_oss_selection": "invoke_coarse_selector",
    "selector_failed": "retry_coarse_selection",
    "fine_perception_failed": "retry_fine_perception",
    "final_not_strong_or_resolved": "continue_retrieval",
    "final_answer_incorrect": "reject_commit_and_retrieve_more",
    "qa_not_answerable_without_resolved_repair": "request_evidence_repair",
}

DIAGNOSABLE_RECOVERY_REASONS = set(RECOVERY_ACTIONS) - {"final_answer_incorrect"}


def _repair_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            payload = read_json(path)
            rows = payload.get("reports") if isinstance(payload.get("reports"), list) else [payload]
        except (json.JSONDecodeError, ValueError):
            rows = read_jsonl(path)
        for row in rows:
            if isinstance(row, dict) and row.get("example_id"):
                result[str(row["example_id"])] = row
    return result


def _recovery_expansions(
    rollout_rows: list[dict[str, Any]],
    repairs: dict[str, dict[str, Any]],
    max_per_reason: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    for row in rollout_rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        perception = metadata.get("perception") if isinstance(metadata.get("perception"), dict) else {}
        selection = perception.get("retrieval") if isinstance(perception.get("retrieval"), dict) else {}
        if selection.get("mode") != "gpt_oss_atomic_select_coarse":
            reason = "no_gpt_oss_selection"
        else:
            eligible, reason = _quality_gate(row, selection, repairs)
            if eligible:
                continue
        if reason not in RECOVERY_ACTIONS:
            excluded[f"unsupported_reason:{reason}"] += 1
            continue
        grouped[reason].append(row)

    result: list[dict[str, Any]] = []
    for reason, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: hashlib.sha256(str(row.get("example_id")).encode()).hexdigest())
        for row in rows[:max_per_reason]:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            perception = metadata.get("perception") if isinstance(metadata.get("perception"), dict) else {}
            selection = perception.get("retrieval") if isinstance(perception.get("retrieval"), dict) else {}
            rollout = metadata.get("reasoning_rollout") if isinstance(metadata.get("reasoning_rollout"), dict) else {}
            fine = metadata.get("clip_schemas") if isinstance(metadata.get("clip_schemas"), list) else []
            coarse = metadata.get("coarse_clip_schemas") if isinstance(metadata.get("coarse_clip_schemas"), list) else []
            selected = [int(value) for value in selection.get("selected_coarse_indices", []) if str(value).isdigit()]
            state = {
                "schema_version": "video-skills/l2-recovery-state-v0.1",
                "process_model": "post_attempt_l2_recovery_controller",
                "dataset": row.get("dataset"),
                "example_id": row.get("example_id"),
                "question": compact_visibility(row.get("question") or {}),
                "retrieval_attempt": {
                    "selector_mode": selection.get("mode"),
                    "selector_ok": bool(selection.get("ok")),
                    "selected_coarse_indices": selected,
                    "rationale_short": selection.get("rationale_short"),
                    "coarse_summary_count": len(coarse),
                    "fine_observation_count": len(fine),
                    "fine_model_error_count": sum(bool(item.get("model_error")) for item in fine if isinstance(item, dict)),
                },
                "visible_runtime_diagnostics": compact_visibility({
                    "acceptance_status": rollout.get("acceptance_status"),
                    "failure_reasons": rollout.get("failure_reasons") or [],
                    "verifier_summary": rollout.get("verifier_summary") or {},
                }),
            }
            result.append({
                "schema_version": "video-skills/l2-specialist-transition-v0.1",
                "transition_id": f"{row.get('example_id')}::l2_recovery::{reason}",
                "controller": "l2_controller",
                "task": "choose_l2_recovery_action",
                "state_t": state,
                "action_t": {
                    "schema_version": "video-skills/l2-specialist-action-v0.1",
                    "tool_name": RECOVERY_ACTIONS[reason],
                    "arguments": {"preserve_verified_evidence": True},
                },
                "offline_gate_reason": reason,
                "source_example_id": str(row.get("example_id") or ""),
                "augmentation_family": "recovery_action",
                "is_core": False,
            })
            if reason in DIAGNOSABLE_RECOVERY_REASONS:
                result.append({
                    "schema_version": "video-skills/l2-specialist-transition-v0.1",
                    "transition_id": f"{row.get('example_id')}::l2_recovery_diagnosis::{reason}",
                    "controller": "l2_controller",
                    "task": "diagnose_l2_recovery_failure",
                    "state_t": state,
                    "action_t": {
                        "schema_version": "video-skills/l2-specialist-action-v0.1",
                        "tool_name": "emit_l2_recovery_diagnosis",
                        "arguments": {
                            "failure_code": reason,
                            "recommended_next_tool": RECOVERY_ACTIONS[reason],
                        },
                    },
                    "offline_gate_reason": reason,
                    "source_example_id": str(row.get("example_id") or ""),
                    "augmentation_family": "recovery_diagnosis",
                    "is_core": False,
                })
        excluded[f"capped:{reason}"] += max(0, len(rows) - max_per_reason)
    return result, excluded


def build_l2_specialist_exports(
    rollout_paths: list[Path],
    positive_transitions_path: Path,
    repair_paths: list[Path],
    *,
    hard_negatives_per_selected: int = 6,
    max_recovery_per_reason: int = 64,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rollout_by_example: dict[str, dict[str, Any]] = {}
    for path in rollout_paths:
        for row in read_jsonl(path):
            if row.get("example_id"):
                rollout_by_example[str(row["example_id"])] = row
    positives = read_jsonl(positive_transitions_path)
    transitions = _positive_expansions(positives, hard_negatives_per_selected)
    recovery, excluded = _recovery_expansions(
        list(rollout_by_example.values()), _repair_index(repair_paths), max_recovery_per_reason
    )
    transitions.extend(recovery)
    transitions.sort(key=lambda row: str(row["transition_id"]))
    family_sizes = Counter(
        (str(row.get("source_example_id") or ""), str(row.get("augmentation_family") or "unknown"))
        for row in transitions
    )
    for row in transitions:
        key = (
            str(row.get("source_example_id") or ""),
            str(row.get("augmentation_family") or "unknown"),
        )
        row["source_family_weight"] = 1.0 / family_sizes[key]
    chats = [_chat(row, str(row["task"])) for row in transitions]
    task_counts = Counter(str(row["task"]) for row in transitions)
    recovery_action_counts = Counter(
        str(row.get("offline_gate_reason"))
        for row in recovery
        if row.get("augmentation_family") == "recovery_action"
    )
    recovery_diagnosis_counts = Counter(
        str(row.get("offline_gate_reason"))
        for row in recovery
        if row.get("augmentation_family") == "recovery_diagnosis"
    )
    augmentation_counts = Counter(str(row.get("augmentation_family")) for row in transitions)
    forbidden_hits = sum(
        contains_forbidden_prompt_key({"task": row["task"], "state_t": row["state_t"]})
        for row in transitions
    )
    report = {
        "schema_version": "video-skills/l2-specialist-sft-report-v0.1",
        "source_rollout_rows": len(rollout_by_example),
        "source_positive_retrieval_transitions": len(positives),
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "task_counts": dict(task_counts),
        "augmentation_family_counts": dict(augmentation_counts),
        "core_sft_chats": sum(bool(row.get("is_core")) for row in transitions),
        "derived_sft_chats": sum(not bool(row.get("is_core")) for row in transitions),
        "recovery_action_reason_counts": dict(recovery_action_counts),
        "recovery_diagnosis_reason_counts": dict(recovery_diagnosis_counts),
        "excluded_or_capped": dict(excluded),
        "prompt_forbidden_key_hits": int(forbidden_hits),
        "hard_negatives_per_selected": hard_negatives_per_selected,
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--positive-transitions", type=Path, required=True)
    parser.add_argument("--repair-results", type=Path, action="append", default=[])
    parser.add_argument("--hard-negatives-per-selected", type=int, default=6)
    parser.add_argument("--max-recovery-per-reason", type=int, default=64)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--core-sft-output-jsonl", type=Path)
    parser.add_argument("--derived-sft-output-jsonl", type=Path)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_l2_specialist_exports(
        args.rollout_jsonl,
        args.positive_transitions,
        args.repair_results,
        hard_negatives_per_selected=args.hard_negatives_per_selected,
        max_recovery_per_reason=args.max_recovery_per_reason,
    )
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    if args.core_sft_output_jsonl:
        write_jsonl(
            args.core_sft_output_jsonl,
            [row for row in chats if (row.get("metadata") or {}).get("is_core")],
        )
    if args.derived_sft_output_jsonl:
        write_jsonl(
            args.derived_sft_output_jsonl,
            [row for row in chats if not (row.get("metadata") or {}).get("is_core")],
        )
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export GPT-OSS coarse retrieval actions as conservative L2 MDP SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_json, read_jsonl, write_json, write_jsonl
from ..verification.evaluate_l1_query_memory import evaluate_example


def _catalog(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, schema in enumerate(schemas):
        rows.append({
            "coarse_index": index,
            "time_span": schema.get("time_span"),
            "scene_description": schema.get("scene_description"),
            "observable_facts": compact_visibility(schema.get("observable_facts") or []),
            "events": compact_visibility(schema.get("events") or []),
            "searchable_phrases": schema.get("searchable_phrases") or [],
        })
    return rows


def _quality_gate(
    row: dict[str, Any],
    selection: dict[str, Any],
    repair_by_example_id: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    if not selection.get("ok") or not selection.get("selected_coarse_indices"):
        return False, "selector_failed"
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    schemas = metadata.get("clip_schemas") if isinstance(metadata.get("clip_schemas"), list) else []
    if not schemas or any(isinstance(schema, dict) and schema.get("model_error") for schema in schemas):
        return False, "fine_perception_failed"
    rollout = metadata.get("reasoning_rollout") if isinstance(metadata.get("reasoning_rollout"), dict) else {}
    acceptance_status = str(rollout.get("acceptance_status") or "")
    repair = repair_by_example_id.get(str(row.get("example_id") or "")) or {}
    resolved_by_repair = repair.get("repair_status") == "resolved_strong"
    if acceptance_status != "accepted_strong" and not resolved_by_repair:
        return False, "final_not_strong_or_resolved"
    final = rollout.get("final_answer") if isinstance(rollout.get("final_answer"), dict) else {}
    answer = (row.get("question") or {}).get("answer") or {}
    if not final.get("label") or str(final.get("label")) != str(answer.get("label")):
        return False, "final_answer_incorrect"
    quality = evaluate_example(row, topk=8).get("qa_answerability") or {}
    if acceptance_status != "accepted_strong" or quality.get("grade") != "answerable":
        if not resolved_by_repair:
            return False, "qa_not_answerable_without_resolved_repair"
        return True, "passed_after_resolved_repair"
    return True, "passed_direct"


def build_l2_retrieval_exports(
    paths: list[Path],
    *,
    repair_results_paths: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    repair_by_example_id: dict[str, dict[str, Any]] = {}
    for repair_path in repair_results_paths or []:
        try:
            payload = read_json(repair_path)
            repair_rows = payload.get("reports") if isinstance(payload.get("reports"), list) else [payload]
        except (json.JSONDecodeError, ValueError):
            repair_rows = read_jsonl(repair_path)
        for repair in repair_rows:
            if not isinstance(repair, dict):
                continue
            if repair.get("example_id"):
                repair_by_example_id[str(repair["example_id"])] = repair
    input_rows = 0
    for path in paths:
        for row_index, row in enumerate(read_jsonl(path)):
            input_rows += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            perception = metadata.get("perception") if isinstance(metadata.get("perception"), dict) else {}
            selection = perception.get("retrieval") if isinstance(perception.get("retrieval"), dict) else {}
            if selection.get("mode") != "gpt_oss_atomic_select_coarse":
                excluded["no_gpt_oss_selection"] += 1
                continue
            eligible, reason = _quality_gate(row, selection, repair_by_example_id)
            if not eligible:
                excluded[reason] += 1
                continue
            coarse_schemas = metadata.get("coarse_clip_schemas") if isinstance(metadata.get("coarse_clip_schemas"), list) else []
            selected = [int(value) for value in selection.get("selected_coarse_indices", [])]
            state = {
                "schema_version": "video-skills/l2-retrieval-state-v0.1",
                "process_model": "mdp_style_l2_retrieval_controller",
                "dataset": row.get("dataset"),
                "example_id": row.get("example_id"),
                "question": compact_visibility(row.get("question") or {}),
                "l1_coarse_summary_catalog": _catalog(coarse_schemas),
                "partial_l1_summary": {
                    "coarse_summary_count": len(coarse_schemas),
                    "fine_observation_count": 0,
                },
                "budget_state": {"topk": selection.get("topk"), "retrieval_round": 0},
            }
            action = {
                "schema_version": "video-skills/l2-retrieval-action-v0.1",
                "tool_name": "select_coarse_clips",
                "arguments": {
                    "selected_coarse_indices": selected,
                    "rationale_short": selection.get("rationale_short"),
                },
            }
            fine = metadata.get("clip_schemas") or []
            initial_strong = (metadata.get("reasoning_rollout") or {}).get("acceptance_status") == "accepted_strong"
            transitions.append({
                "schema_version": "video-skills/l2-retrieval-transition-v0.1",
                "transition_id": f"{row.get('example_id')}::l2_retrieval::{row_index}",
                "controller": "l2_retrieval",
                "state_t": state,
                "action_t": action,
                "observation_t": {
                    "fine_clip_count": len(fine),
                    "fine_clip_ids": [schema.get("clip_id") for schema in fine if isinstance(schema, dict)],
                    "tool_ok": True,
                },
                "state_t_plus_1_summary": {"selected_coarse_indices": selected, "fine_clip_count": len(fine)},
                "reward_proxy_t": {
                    "initial_accepted_strong": float(initial_strong),
                    "downstream_resolved_strong": float(reason == "passed_after_resolved_repair"),
                    "final_answer_correct_eval_only": 1.0,
                    "downstream_quality_gate": reason,
                },
                "done": False,
                "source_rollout_jsonl": str(path),
                "source_row_index": row_index,
            })

    chats = []
    forbidden_hits = 0
    for row in transitions:
        user_payload = {"task": "choose_next_l2_retrieval_action", "state_t": row["state_t"]}
        forbidden_hits += int(contains_forbidden_prompt_key(user_payload))
        chats.append({
            "schema_version": "video-skills/l2-retrieval-sft-chat-v0.1",
            "transition_id": row["transition_id"],
            "messages": [
                {"role": "system", "content": "You are the Video_Skills L2 retrieval controller. Given a question and visible L1 coarse summaries, choose the next bounded clip-selection tool action as JSON. Options are hypotheses, not facts."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(row["action_t"], ensure_ascii=False, separators=(",", ":"))},
            ],
            "metadata": {"controller": "l2_retrieval", "quality_gate": "correct_and_strong_or_repair_resolved"},
        })
    report = {
        "schema_version": "video-skills/l2-retrieval-sft-report-v0.1",
        "source_rollout_jsonl": [str(path) for path in paths],
        "source_repair_results": [str(path) for path in repair_results_paths or []],
        "input_rows": input_rows,
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "excluded_counts": dict(excluded),
        "prompt_forbidden_key_hits": forbidden_hits,
        "quality_gate": "gpt_oss selection + valid fine perception + correct initial answer + accepted_strong or resolved_strong repair",
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--repair-results", type=Path, nargs="*", default=[])
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_l2_retrieval_exports(
        args.rollout_jsonl,
        repair_results_paths=args.repair_results,
    )
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

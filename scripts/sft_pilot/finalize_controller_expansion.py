#!/usr/bin/env python3
"""Finalize deduplicated L2 retrieval and L1/L2 motif expansion data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.training.l2_retrieval_sft_adapter import build_l2_retrieval_exports
from dataset_clip_wrapper.training.motif_sft_adapter import build_motif_exports
from dataset_clip_wrapper.training.sft_common import read_json, read_jsonl, write_json, write_jsonl
from motif.miner import mine_paths


def _selection_is_atomic(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    perception = metadata.get("perception") if isinstance(metadata.get("perception"), dict) else {}
    selection = perception.get("retrieval") if isinstance(perception.get("retrieval"), dict) else {}
    return selection.get("mode") == "gpt_oss_atomic_select_coarse" and bool(selection.get("ok"))


def _row_score(row: dict[str, Any], order: int) -> tuple[int, int, int]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    rollout = metadata.get("reasoning_rollout") if isinstance(metadata.get("reasoning_rollout"), dict) else {}
    return (
        int(_selection_is_atomic(row)),
        int(rollout.get("acceptance_status") == "accepted_strong"),
        order,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--new-rollout-root", type=Path, required=True)
    parser.add_argument("--backfilled-rollouts", type=Path, required=True)
    parser.add_argument("--repair-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-motif-support", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_report = read_json(args.base_report)
    paths = [Path(value) for value in base_report.get("source_rollout_jsonl") or []]
    paths.extend(sorted(args.new_rollout_root.glob("**/examples.jsonl")))
    if args.backfilled_rollouts.exists():
        paths.append(args.backfilled_rollouts)
    paths = list(dict.fromkeys(path for path in paths if path.exists()))

    chosen: dict[str, tuple[tuple[int, int, int], dict[str, Any]]] = {}
    source_rows = 0
    for order, path in enumerate(paths):
        for row in read_jsonl(path):
            source_rows += 1
            example_id = str(row.get("example_id") or "")
            if not example_id:
                continue
            scored = (_row_score(row, order), row)
            if example_id not in chosen or scored[0] > chosen[example_id][0]:
                chosen[example_id] = scored
    deduplicated = [value[1] for _, value in sorted(chosen.items())]
    dedup_path = args.output_dir / "deduplicated_rollouts.jsonl"
    write_jsonl(dedup_path, deduplicated)

    transitions, chats, retrieval_report = build_l2_retrieval_exports(
        [dedup_path], repair_results_paths=[args.repair_results]
    )
    # One retrieval decision per example is the semantic unit. Normalize IDs so
    # rerun-local row indices cannot collide or create duplicate supervision.
    transition_by_example: dict[str, dict[str, Any]] = {}
    chat_by_example: dict[str, dict[str, Any]] = {}
    for transition, chat in zip(transitions, chats):
        example_id = str((transition.get("state_t") or {}).get("example_id") or "")
        transition_id = f"{example_id}::l2_retrieval::0"
        transition["transition_id"] = transition_id
        chat["transition_id"] = transition_id
        transition_by_example[example_id] = transition
        chat_by_example[example_id] = chat
    transitions = [transition_by_example[key] for key in sorted(transition_by_example)]
    chats = [chat_by_example[key] for key in sorted(chat_by_example)]
    write_jsonl(args.output_dir / "l2_retrieval_transitions.jsonl", transitions)
    write_jsonl(args.output_dir / "l2_retrieval_sft.jsonl", chats)
    retrieval_report["source_rows_before_example_dedup"] = source_rows
    retrieval_report["unique_rollout_examples"] = len(deduplicated)
    retrieval_report["deduplicated_exported_sft_chats"] = len(chats)
    write_json(args.output_dir / "l2_retrieval_report.json", retrieval_report)

    motif_result = mine_paths([dedup_path, args.repair_results], min_support=args.min_motif_support)
    motif_bank_path = args.output_dir / "motif_bank_l1_l2.jsonl"
    motif_result.bank.save_jsonl(motif_bank_path)
    write_json(args.output_dir / "motif_summary_l1_l2.json", motif_result.to_dict())
    motif_transitions, motif_chats, motif_report = build_motif_exports(motif_bank_path)
    write_jsonl(args.output_dir / "motif_lifecycle_transitions.jsonl", motif_transitions)
    write_jsonl(args.output_dir / "motif_lifecycle_sft.jsonl", motif_chats)
    write_json(args.output_dir / "motif_lifecycle_report.json", motif_report)

    summary = {
        "source_rollout_paths": len(paths),
        "source_rows": source_rows,
        "unique_rollout_examples": len(deduplicated),
        "l2_retrieval_sft": len(chats),
        "motif_lifecycle_sft": len(motif_chats),
        "retrieval_report": str(args.output_dir / "l2_retrieval_report.json"),
        "motif_report": str(args.output_dir / "motif_lifecycle_report.json"),
    }
    write_json(args.output_dir / "finalization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

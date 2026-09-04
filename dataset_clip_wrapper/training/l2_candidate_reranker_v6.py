#!/usr/bin/env python3
"""Build L2 reranker SFT from fixed, label-independent candidate reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluate_l2_candidate_retrieval import boundary_hybrid_candidates
from .l2_oracle_retrieval_v5 import FAMILY_BUDGETS, _compact_transition_catalogs
from .l2_specialist_sft_adapter import _chat, _positive_expansions
from .sft_common import contains_forbidden_prompt_key, read_json, read_jsonl, write_json, write_jsonl


RETRIEVER = "qwen3_vl_embedding_2b_fine8s_stride4s_boundary_hybrid"


def candidate_index(report: dict[str, Any]) -> dict[str, list[int]]:
    result = {}
    for row in report.get("results") or []:
        indices = row.get("top32_boundary_hybrid")
        if not isinstance(indices, list):
            indices = boundary_hybrid_candidates(
                [int(value) for value in row.get("top32") or []],
                int(row.get("catalog_size", 0)),
            )
        result[str(row["example_id"])] = [int(value) for value in indices]
    return result


def core_transitions(chats: list[dict[str, Any]], candidates: dict[str, list[int]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    result = []
    excluded: Counter[str] = Counter()
    for chat in chats:
        metadata = chat.get("metadata") or {}
        if metadata.get("task") != "select_coarse_set" or metadata.get("is_core") is not True:
            continue
        example_id = str(metadata.get("source_example_id") or "")
        candidate_order = candidates.get(example_id)
        if not candidate_order:
            excluded["missing_candidate_report"] += 1
            continue
        user = json.loads(chat["messages"][1]["content"])
        state = dict(user["state_t"])
        action = json.loads(chat["messages"][2]["content"])
        selected = [int(value) for value in action["arguments"]["selected_coarse_indices"]]
        candidate_set = set(candidate_order)
        if not set(selected) <= candidate_set:
            excluded["gold_outside_candidates"] += 1
            continue
        catalog = state.get("l1_coarse_summary_catalog") or []
        # Sort by absolute time/index rather than retrieval rank, so the SFT
        # policy cannot exploit the frozen retriever's ordering as a label cue.
        state["l1_coarse_summary_catalog"] = sorted(
            (row for row in catalog if int(row.get("coarse_index", -1)) in candidate_set),
            key=lambda row: int(row.get("coarse_index", -1)),
        )
        state["schema_version"] = "video-skills/l2-reranker-state-v0.1"
        state["candidate_retrieval"] = {
            "method": RETRIEVER,
            "candidate_count": len(state["l1_coarse_summary_catalog"]),
            "rank_hidden_from_policy": True,
        }
        if contains_forbidden_prompt_key(state):
            excluded["forbidden_prompt_key"] += 1
            continue
        result.append({
            "schema_version": "video-skills/l2-reranker-transition-v0.1",
            "transition_id": f"{example_id}::l2_candidate_reranker::0",
            "controller": "l2_controller",
            "state_t": state,
            "action_t": action,
            "source_example_id": example_id,
            "is_core": True,
        })
    return result, excluded


def build_split(
    chats: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    split_role: str,
    hard_negatives_per_selected: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core, excluded = core_transitions(chats, candidate_index(report))
    expanded = _positive_expansions(core, hard_negatives_per_selected)
    for transition in expanded:
        _compact_transition_catalogs(transition)
    by_source_family = Counter(
        (str(row.get("source_example_id")), str(row.get("augmentation_family"))) for row in expanded
    )
    output = []
    for transition in expanded:
        family = str(transition.get("augmentation_family"))
        key = (str(transition.get("source_example_id")), family)
        transition["source_family_weight"] = FAMILY_BUDGETS[family] / by_source_family[key]
        chat = _chat(transition, str(transition["task"]))
        chat["split_group_id"] = next(
            (row.get("split_group_id") for row in chats if (row.get("metadata") or {}).get("source_example_id") == transition.get("source_example_id")),
            None,
        )
        chat["specialist"] = "l2"
        chat["metadata"].update({
            "split_role": split_role,
            "teacher": "deterministic_cg_bench_clue_interval_mapper",
            "candidate_retriever": RETRIEVER,
            "candidate_rank_hidden": True,
        })
        output.append(chat)
    return output, {
        "core_examples": len(core),
        "derived_rows": len(output),
        "excluded": dict(excluded),
        "family_counts": dict(Counter(str(row.get("augmentation_family")) for row in expanded)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--dev-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-negatives-per-selected", type=int, default=6)
    args = parser.parse_args(argv)

    outputs = {}
    audit = {"schema_version": "video-skills/l2-candidate-reranker-v6-report-v0.1", "retriever": RETRIEVER}
    for split, jsonl_path, report_path, role in (
        ("train", args.train_jsonl, args.train_report, "sft_seed"),
        ("dev", args.dev_jsonl, args.dev_report, "dev_tune"),
    ):
        rows, split_audit = build_split(
            read_jsonl(jsonl_path),
            read_json(report_path),
            split_role=role,
            hard_negatives_per_selected=args.hard_negatives_per_selected,
        )
        outputs[split] = rows
        audit[split] = split_audit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", outputs["train"])
    write_jsonl(args.output_dir / "dev.jsonl", outputs["dev"])
    write_json(args.output_dir / "report.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

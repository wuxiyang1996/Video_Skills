#!/usr/bin/env python3
"""Build L2 v7 reranker SFT without discarding visual-rank or rich L1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluate_l2_candidate_retrieval import boundary_hybrid_candidates
from .l2_specialist_sft_adapter import _chat
from .sft_common import contains_forbidden_prompt_key, read_json, read_jsonl, write_json, write_jsonl


RETRIEVER = "qwen3_vl_embedding_2b_fine8s_stride4s_boundary_hybrid"
FAMILY_BUDGETS = {"full_select": 0.65, "hard_select": 0.10, "ranking": 0.25}


def _text(value: Any, limit: int) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("description") or ""
    return str(value or "")[:limit]


def rich_candidate(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Keep query-useful evidence while remaining comfortably below 16K tokens."""
    return {
        "coarse_index": int(row.get("coarse_index", spec["coarse_index"])),
        "time_span": row.get("time_span") if isinstance(row.get("time_span"), dict) else {},
        "retrieval_rank": int(spec["retrieval_rank"]),
        "semantic_rank": spec.get("semantic_rank"),
        "candidate_sources": list(spec.get("candidate_sources") or []),
        "scene_description": _text(row.get("scene_description"), 180),
        "observable_facts": [_text(value, 120) for value in (row.get("observable_facts") or [])[:3]],
        "events": [_text(value, 100) for value in (row.get("events") or [])[:2]],
        "searchable_phrases": [_text(value, 60) for value in (row.get("searchable_phrases") or [])[:8]],
    }


def candidate_specs(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("results") or []:
        semantic = [int(value) for value in row.get("top32") or []]
        hybrid = row.get("top32_boundary_hybrid")
        if not isinstance(hybrid, list):
            hybrid = boundary_hybrid_candidates(semantic, int(row.get("catalog_size", 0)))
        catalog_size = int(row.get("catalog_size", 0))
        boundaries = {0, max(0, catalog_size - 1)}
        specs = []
        for rank, value in enumerate(hybrid, start=1):
            index = int(value)
            sources = []
            if index in semantic:
                sources.append("visual_semantic")
            if index in boundaries:
                sources.append("boundary_anchor")
            specs.append({
                "coarse_index": index,
                "retrieval_rank": rank,
                "semantic_rank": semantic.index(index) + 1 if index in semantic else None,
                "candidate_sources": sources,
            })
        result[str(row["example_id"])] = specs
    return result


def source_index(rollouts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("example_id")): row for row in rollouts if row.get("example_id")}


def _minimal_select_action(action: dict[str, Any]) -> dict[str, Any]:
    selected = [int(value) for value in (action.get("arguments") or {}).get("selected_coarse_indices") or []]
    return {
        "schema_version": "video-skills/l2-specialist-action-v0.1",
        "tool_name": "select_coarse_clips",
        "arguments": {"selected_coarse_indices": selected},
    }


def core_transitions(
    chats: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    sources: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    result = []
    excluded: Counter[str] = Counter()
    for chat in chats:
        metadata = chat.get("metadata") or {}
        if metadata.get("task") != "select_coarse_set" or metadata.get("is_core") is not True:
            continue
        example_id = str(metadata.get("source_example_id") or "")
        specs = candidates.get(example_id)
        source = sources.get(example_id)
        if not specs:
            excluded["missing_candidate_report"] += 1
            continue
        if not source:
            excluded["missing_rollout_source"] += 1
            continue
        user = json.loads(chat["messages"][1]["content"])
        state = dict(user["state_t"])
        action = _minimal_select_action(json.loads(chat["messages"][2]["content"]))
        selected = action["arguments"]["selected_coarse_indices"]
        candidate_set = {int(spec["coarse_index"]) for spec in specs}
        if not set(selected) <= candidate_set:
            excluded["gold_outside_candidates"] += 1
            continue
        raw_catalog = (source.get("metadata") or {}).get("coarse_clip_schemas") or []
        by_index = {
            int(row.get("coarse_index", position)): row
            for position, row in enumerate(raw_catalog)
            if isinstance(row, dict)
        }
        if any(int(spec["coarse_index"]) not in by_index for spec in specs):
            excluded["missing_rich_catalog_row"] += 1
            continue
        state["l1_coarse_summary_catalog"] = [
            rich_candidate(by_index[int(spec["coarse_index"])], spec) for spec in specs
        ]
        state["schema_version"] = "video-skills/l2-reranker-state-v0.2"
        state["candidate_retrieval"] = {
            "method": RETRIEVER,
            "candidate_count": len(specs),
            "rank_visible_to_policy": True,
            "score_available": False,
        }
        if contains_forbidden_prompt_key(state):
            excluded["forbidden_prompt_key"] += 1
            continue
        result.append({
            "schema_version": "video-skills/l2-reranker-transition-v0.2",
            "transition_id": f"{example_id}::l2_candidate_reranker_v7::0",
            "controller": "l2_controller",
            "task": "select_coarse_set",
            "state_t": state,
            "action_t": action,
            "source_example_id": example_id,
            "is_core": True,
        })
    return result, excluded


def _catalog_subset(state: dict[str, Any], indices: set[int], view: str) -> dict[str, Any]:
    output = dict(state)
    output["l1_coarse_summary_catalog"] = [
        row for row in state["l1_coarse_summary_catalog"] if int(row["coarse_index"]) in indices
    ]
    output["catalog_view"] = view
    return output


def _stable_candidate_order(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    """Deterministically decorrelate candidate array position from the teacher label."""
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{int(row['coarse_index'])}".encode("utf-8")
        ).hexdigest(),
    )


def expand_core(source: dict[str, Any], hard_negatives: int) -> list[dict[str, Any]]:
    state = source["state_t"]
    selected = [int(value) for value in source["action_t"]["arguments"]["selected_coarse_indices"]]
    selected_set = set(selected)
    negatives = [
        int(row["coarse_index"]) for row in state["l1_coarse_summary_catalog"]
        if int(row["coarse_index"]) not in selected_set
    ]
    rows: list[dict[str, Any]] = []

    def add(task: str, suffix: str, family: str, view_state: dict[str, Any], action: dict[str, Any], core: bool = False) -> None:
        rows.append({
            "schema_version": "video-skills/l2-reranker-transition-v0.2",
            "transition_id": f"{source['source_example_id']}::{suffix}",
            "controller": "l2_controller",
            "task": task,
            "state_t": view_state,
            "action_t": action,
            "source_example_id": source["source_example_id"],
            "augmentation_family": family,
            "is_core": core,
        })

    add("select_coarse_set", "l2_v7_select::full", "full_select", state, source["action_t"], True)
    hard_state = _catalog_subset(state, selected_set | set(negatives[:8]), "hard_8")
    add("select_coarse_set", "l2_v7_select::hard_8", "hard_select", hard_state, source["action_t"])

    by_index = {int(row["coarse_index"]): row for row in state["l1_coarse_summary_catalog"]}
    for step, positive in enumerate(selected):
        for rank, negative in enumerate(negatives[:hard_negatives]):
            pair = _stable_candidate_order(
                [by_index[index] for index in (positive, negative)],
                f"{source['source_example_id']}:pair:{step}:{rank}",
            )
            ranking_state = {
                "schema_version": "video-skills/l2-ranking-state-v0.2",
                "process_model": "pairwise_visual_coarse_reranking",
                "dataset": state.get("dataset"),
                "example_id": state.get("example_id"),
                "question": state.get("question"),
                "candidate_coarse_summaries": pair,
            }
            add(
                "rank_coarse_candidates", f"l2_v7_rank_pair::{step}:{rank}", "ranking", ranking_state,
                {"schema_version": "video-skills/l2-specialist-action-v0.1", "tool_name": "choose_better_coarse_candidate", "arguments": {"coarse_index": positive}},
            )
        list_indices = [positive] + negatives[:3]
        list_state = {
            "schema_version": "video-skills/l2-ranking-state-v0.2",
            "process_model": "listwise_visual_coarse_reranking",
            "dataset": state.get("dataset"),
            "example_id": state.get("example_id"),
            "question": state.get("question"),
            "candidate_coarse_summaries": _stable_candidate_order(
                [by_index[index] for index in list_indices],
                f"{source['source_example_id']}:list:{step}",
            ),
        }
        add(
            "rank_coarse_candidates_listwise", f"l2_v7_rank_list::{step}", "ranking", list_state,
            {"schema_version": "video-skills/l2-specialist-action-v0.1", "tool_name": "choose_best_coarse_candidate", "arguments": {"coarse_index": positive}},
        )
    return rows


def build_split(
    chats: list[dict[str, Any]], report: dict[str, Any], sources: dict[str, dict[str, Any]], *,
    split_role: str, hard_negatives_per_selected: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core, excluded = core_transitions(chats, candidate_specs(report), sources)
    expanded = [row for source in core for row in expand_core(source, hard_negatives_per_selected)]
    counts = Counter((str(row["source_example_id"]), str(row["augmentation_family"])) for row in expanded)
    output = []
    split_groups = {
        str((row.get("metadata") or {}).get("source_example_id")): row.get("split_group_id") for row in chats
    }
    for transition in expanded:
        family = str(transition["augmentation_family"])
        key = (str(transition["source_example_id"]), family)
        transition["source_family_weight"] = FAMILY_BUDGETS[family] / counts[key]
        chat = _chat(transition, str(transition["task"]))
        chat["split_group_id"] = split_groups.get(str(transition["source_example_id"]))
        chat["specialist"] = "l2"
        chat["metadata"].update({
            "split_role": split_role,
            "teacher": "deterministic_cg_bench_clue_interval_mapper",
            "candidate_retriever": RETRIEVER,
            "candidate_rank_visible": True,
            "target_has_rationale": False,
        })
        output.append(chat)
    return output, {
        "core_examples": len(core), "derived_rows": len(output), "excluded": dict(excluded),
        "family_counts": dict(Counter(str(row["augmentation_family"]) for row in expanded)),
        "family_loss_budgets": FAMILY_BUDGETS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--dev-report", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-negatives-per-selected", type=int, default=6)
    args = parser.parse_args(argv)
    sources = source_index(read_jsonl(args.rollouts))
    outputs = {}
    audit = {"schema_version": "video-skills/l2-candidate-reranker-v7-report-v0.1", "retriever": RETRIEVER}
    for split, data_path, report_path, role in (
        ("train", args.train_jsonl, args.train_report, "sft_seed"),
        ("dev", args.dev_jsonl, args.dev_report, "dev_tune"),
    ):
        rows, summary = build_split(
            read_jsonl(data_path), read_json(report_path), sources,
            split_role=role, hard_negatives_per_selected=args.hard_negatives_per_selected,
        )
        outputs[split], audit[split] = rows, summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", outputs["train"])
    write_jsonl(args.output_dir / "dev.jsonl", outputs["dev"])
    write_json(args.output_dir / "report.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

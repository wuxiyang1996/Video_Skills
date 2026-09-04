#!/usr/bin/env python3
"""Build balanced L2 pointwise relevance chats from v7 rich candidate states."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .l2_specialist_sft_adapter import SYSTEM
from .sft_common import contains_forbidden_prompt_key, read_json, read_jsonl, write_json, write_jsonl


TASK = "score_coarse_candidate_relevance"


def relevance_action(relevant: bool) -> dict[str, Any]:
    return {
        "schema_version": "video-skills/l2-relevance-action-v0.1",
        "tool_name": "score_coarse_candidate",
        "arguments": {"relevant": relevant},
    }


def pointwise_state(state: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "video-skills/l2-pointwise-state-v0.1",
        "process_model": "independent_visual_candidate_relevance",
        "dataset": state.get("dataset"),
        "example_id": state.get("example_id"),
        "question": state.get("question"),
        "candidate_retrieval": state.get("candidate_retrieval"),
        "candidate_coarse_summary": candidate,
    }


def teacher_feature_index(teacher_report: dict[str, Any] | None) -> dict[str, dict[int, dict[str, Any]]]:
    """Index label-independent visual teacher scores by example/candidate."""
    if teacher_report is None:
        return {}
    if not bool(teacher_report.get("label_independent")):
        raise ValueError("Visual teacher report must be label-independent")
    model = str(teacher_report.get("model") or "")
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for item in teacher_report.get("results") or []:
        example_id = str(item["example_id"])
        ranking = list(item.get("ranking") or [])
        if not ranking:
            continue
        scores = [float(row["score"]) for row in ranking]
        top_score = max(scores)
        bottom_score = min(scores)
        width = top_score - bottom_score
        by_candidate: dict[int, dict[str, Any]] = {}
        for rank, row in enumerate(ranking, start=1):
            index = int(row["candidate_index"])
            score = float(row["score"])
            by_candidate[index] = {
                "model": model,
                "rank": rank,
                "score": score,
                "normalized_score": (score - bottom_score) / width if width > 0 else 0.0,
                "gap_to_top": top_score - score,
            }
        output[example_id] = by_candidate
    return output


def add_teacher_feature(
    candidate: dict[str, Any],
    teacher_features: dict[str, dict[int, dict[str, Any]]],
    example_id: str,
) -> dict[str, Any]:
    """Return a candidate copy augmented with optional visual teacher features."""
    output = dict(candidate)
    feature = teacher_features.get(example_id, {}).get(int(output["coarse_index"]))
    if feature is not None:
        output["visual_teacher_reranker"] = feature
    return output


def _stable_rows(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{int(row['coarse_index'])}".encode("utf-8")
        ).hexdigest(),
    )


def build_split(
    source_rows: list[dict[str, Any]], *, split_role: str, hard_negatives: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    sources = 0
    for source in source_rows:
        metadata = source.get("metadata") or {}
        if metadata.get("task") != "select_coarse_set" or metadata.get("is_core") is not True:
            continue
        sources += 1
        example_id = str(metadata.get("source_example_id") or "")
        user = json.loads(source["messages"][1]["content"])
        state = user.get("state_t") or {}
        gold_action = json.loads(source["messages"][2]["content"])
        selected = {
            int(value)
            for value in (gold_action.get("arguments") or {}).get("selected_coarse_indices") or []
        }
        catalog = list(state.get("l1_coarse_summary_catalog") or [])
        positives = [row for row in catalog if int(row.get("coarse_index", -1)) in selected]
        negatives = [row for row in catalog if int(row.get("coarse_index", -1)) not in selected]
        if len(positives) != len(selected):
            excluded["gold_outside_candidates"] += 1
            continue
        if not positives or not negatives:
            excluded["missing_class"] += 1
            continue
        negatives = negatives[:hard_negatives] if hard_negatives > 0 else negatives
        families = (("positive", positives, True), ("hard_negative", negatives, False))
        for family, candidates, relevant in families:
            family_weight = 0.5 / len(candidates)
            for candidate in _stable_rows(candidates, f"{example_id}:{family}"):
                candidate_state = pointwise_state(state, candidate)
                if contains_forbidden_prompt_key(candidate_state):
                    excluded["forbidden_prompt_key"] += 1
                    continue
                index = int(candidate["coarse_index"])
                payload = {"task": TASK, "state_t": candidate_state}
                action = relevance_action(relevant)
                output.append({
                    "schema_version": "video-skills/l2-pointwise-sft-chat-v0.1",
                    "transition_id": f"{example_id}::l2_v8_pointwise::{index}",
                    "split_group_id": source.get("split_group_id"),
                    "specialist": "l2",
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                        {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))},
                    ],
                    "metadata": {
                        "controller": "l2_controller",
                        "task": TASK,
                        "dataset": metadata.get("dataset"),
                        "source_example_id": example_id,
                        "candidate_index": index,
                        "candidate_relevant": relevant,
                        "augmentation_family": family,
                        "source_family_weight": family_weight,
                        "is_core": True,
                        "split_role": split_role,
                        "teacher": "deterministic_cg_bench_clue_interval_mapper",
                        "candidate_retriever": metadata.get("candidate_retriever"),
                    },
                })
    return output, {
        "source_examples": sources,
        "derived_rows": len(output),
        "class_counts": dict(Counter(str((row.get("metadata") or {}).get("augmentation_family")) for row in output)),
        "source_weight_sum": sum(float((row.get("metadata") or {}).get("source_family_weight", 0.0)) for row in output),
        "excluded": dict(excluded),
    }


def build_label_independent_eval_split(
    source_rows: list[dict[str, Any]],
    candidate_report: dict[str, Any],
    *,
    visual_teacher_report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build scoring prompts from a candidate pool selected without gold labels."""
    if not bool(candidate_report.get("label_independent")):
        raise ValueError("Candidate report must be label-independent")
    pools = {
        str(row["example_id"]): [
            int(value)
            for value in (row.get("top32_boundary_hybrid") or row.get("top32") or [])
        ]
        for row in candidate_report.get("results") or []
    }
    output: list[dict[str, Any]] = []
    missing_reports: list[str] = []
    missing_teacher_features = 0
    teacher_features = teacher_feature_index(visual_teacher_report)
    gold_outside_pool = 0
    source_examples = 0
    for source in source_rows:
        metadata = source.get("metadata") or {}
        if metadata.get("task") != "select_coarse_set" or metadata.get("is_core") is not True:
            continue
        source_examples += 1
        example_id = str(metadata.get("source_example_id") or "")
        requested = pools.get(example_id)
        if not requested:
            missing_reports.append(example_id)
            continue
        user = json.loads(source["messages"][1]["content"])
        state = user.get("state_t") or {}
        catalog_by_index = {
            int(row["coarse_index"]): row
            for row in state.get("l1_coarse_summary_catalog") or []
        }
        missing = set(requested) - set(catalog_by_index)
        if missing:
            raise ValueError(
                f"Candidate report contains unknown indices for {example_id}: {sorted(missing)}"
            )
        gold_action = json.loads(source["messages"][2]["content"])
        gold = sorted(
            int(value)
            for value in (gold_action.get("arguments") or {}).get("selected_coarse_indices") or []
        )
        if set(gold) - set(requested):
            gold_outside_pool += 1
        for retrieval_rank, index in enumerate(requested, start=1):
            candidate = add_teacher_feature(catalog_by_index[index], teacher_features, example_id)
            if visual_teacher_report is not None and "visual_teacher_reranker" not in candidate:
                missing_teacher_features += 1
            candidate["retrieval_rank"] = retrieval_rank
            candidate_state = pointwise_state(state, candidate)
            payload = {"task": TASK, "state_t": candidate_state}
            relevant = index in set(gold)
            output.append({
                "schema_version": "video-skills/l2-pointwise-eval-chat-v0.1",
                "transition_id": f"{example_id}::l2_v8_label_independent_eval::{index}",
                "split_group_id": source.get("split_group_id"),
                "specialist": "l2",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                    {"role": "assistant", "content": json.dumps(relevance_action(relevant), ensure_ascii=False, separators=(",", ":"))},
                ],
                "metadata": {
                    "controller": "l2_controller",
                    "task": TASK,
                    "dataset": metadata.get("dataset"),
                    "source_example_id": example_id,
                    "candidate_index": index,
                    "candidate_relevant": relevant,
                    "gold_indices": gold,
                    "retrieval_rank": retrieval_rank,
                    "visual_teacher_rank": (
                        candidate.get("visual_teacher_reranker") or {}
                    ).get("rank"),
                    "visual_teacher_score": (
                        candidate.get("visual_teacher_reranker") or {}
                    ).get("score"),
                    "candidate_selection_label_independent": True,
                    "is_core": True,
                    "split_role": "dev_tune",
                },
            })
    if missing_reports:
        raise ValueError(f"Missing candidate reports: {sorted(missing_reports)}")
    return output, {
        "source_examples": source_examples,
        "derived_rows": len(output),
        "gold_outside_candidate_pool_examples": gold_outside_pool,
        "candidate_selection_label_independent": True,
        "visual_teacher": str(visual_teacher_report.get("model")) if visual_teacher_report else None,
        "missing_teacher_features": missing_teacher_features,
    }


def build_label_independent_train_split(
    source_rows: list[dict[str, Any]],
    candidate_report: dict[str, Any],
    *,
    visual_teacher_report: dict[str, Any] | None = None,
    teacher_hard_negatives: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build training prompts from a label-independent retrieval pool."""
    if not bool(candidate_report.get("label_independent")):
        raise ValueError("Candidate report must be label-independent")
    pools = {
        str(row["example_id"]): [
            int(value)
            for value in (row.get("top32_boundary_hybrid") or row.get("top32") or [])
        ]
        for row in candidate_report.get("results") or []
    }
    output: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    teacher_features = teacher_feature_index(visual_teacher_report)
    missing_teacher_features = 0
    sources = 0
    for source in source_rows:
        metadata = source.get("metadata") or {}
        if metadata.get("task") != "select_coarse_set" or metadata.get("is_core") is not True:
            continue
        sources += 1
        example_id = str(metadata.get("source_example_id") or "")
        requested = pools.get(example_id)
        if not requested:
            excluded["missing_candidate_report"] += 1
            continue
        user = json.loads(source["messages"][1]["content"])
        state = user.get("state_t") or {}
        catalog_by_index = {
            int(row["coarse_index"]): row
            for row in state.get("l1_coarse_summary_catalog") or []
        }
        missing = set(requested) - set(catalog_by_index)
        if missing:
            raise ValueError(
                f"Candidate report contains unknown indices for {example_id}: {sorted(missing)}"
            )
        gold_action = json.loads(source["messages"][2]["content"])
        selected = {
            int(value)
            for value in (gold_action.get("arguments") or {}).get("selected_coarse_indices") or []
        }
        positives = [index for index in requested if index in selected]
        negatives = [index for index in requested if index not in selected]
        if teacher_hard_negatives > 0 and visual_teacher_report is not None:
            features = teacher_features.get(example_id, {})
            negatives = sorted(
                negatives,
                key=lambda index: (
                    int(features.get(index, {}).get("rank", len(requested) + 1)),
                    requested.index(index),
                ),
            )[:teacher_hard_negatives]
        if not positives:
            excluded["gold_outside_candidate_pool"] += 1
            continue
        if not negatives:
            excluded["missing_negative_class"] += 1
            continue
        for family, indices, relevant in (
            ("positive", positives, True),
            ("hard_negative", negatives, False),
        ):
            family_weight = 0.5 / len(indices)
            for retrieval_rank, index in enumerate(requested, start=1):
                if index not in indices:
                    continue
                candidate = add_teacher_feature(catalog_by_index[index], teacher_features, example_id)
                if visual_teacher_report is not None and "visual_teacher_reranker" not in candidate:
                    missing_teacher_features += 1
                candidate["retrieval_rank"] = retrieval_rank
                candidate_state = pointwise_state(state, candidate)
                if contains_forbidden_prompt_key(candidate_state):
                    excluded["forbidden_prompt_key"] += 1
                    continue
                payload = {"task": TASK, "state_t": candidate_state}
                action = relevance_action(relevant)
                output.append({
                    "schema_version": "video-skills/l2-pointwise-sft-chat-v0.1",
                    "transition_id": f"{example_id}::l2_v10_label_independent_pointwise::{index}",
                    "split_group_id": source.get("split_group_id"),
                    "specialist": "l2",
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                        {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))},
                    ],
                    "metadata": {
                        "controller": "l2_controller",
                        "task": TASK,
                        "dataset": metadata.get("dataset"),
                        "source_example_id": example_id,
                        "candidate_index": index,
                        "candidate_relevant": relevant,
                        "gold_indices": sorted(selected),
                        "retrieval_rank": retrieval_rank,
                        "visual_teacher_rank": (
                            candidate.get("visual_teacher_reranker") or {}
                        ).get("rank"),
                        "visual_teacher_score": (
                            candidate.get("visual_teacher_reranker") or {}
                        ).get("score"),
                        "augmentation_family": family,
                        "source_family_weight": family_weight,
                        "candidate_selection_label_independent": True,
                        "is_core": True,
                        "split_role": "sft_seed",
                        "teacher": (
                            "gold_relevance_with_label_independent_visual_teacher_features"
                            if visual_teacher_report is not None
                            else "gold_relevance_on_label_independent_retrieval_pool"
                        ),
                        "candidate_retriever": candidate_report.get("retriever")
                        or candidate_report.get("model"),
                    },
                })
    return output, {
        "source_examples": sources,
        "derived_rows": len(output),
        "class_counts": dict(Counter(str((row.get("metadata") or {}).get("augmentation_family")) for row in output)),
        "source_weight_sum": sum(float((row.get("metadata") or {}).get("source_family_weight", 0.0)) for row in output),
        "excluded": dict(excluded),
        "candidate_selection_label_independent": True,
        "visual_teacher": str(visual_teacher_report.get("model")) if visual_teacher_report else None,
        "teacher_hard_negatives": teacher_hard_negatives,
        "missing_teacher_features": missing_teacher_features,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-hard-negatives", type=int, default=12)
    parser.add_argument("--dev-hard-negatives", type=int, default=0)
    parser.add_argument("--label-independent-train-candidate-report", type=Path)
    parser.add_argument("--label-independent-dev-candidate-report", type=Path)
    parser.add_argument("--visual-teacher-train-report", type=Path)
    parser.add_argument("--visual-teacher-dev-report", type=Path)
    parser.add_argument("--teacher-hard-negatives", type=int, default=0)
    args = parser.parse_args(argv)
    outputs: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {"schema_version": "video-skills/l2-pointwise-v8-report-v0.1"}
    for split, path, role, hard_negatives in (
        ("train", args.train_jsonl, "sft_seed", args.train_hard_negatives),
        ("dev", args.dev_jsonl, "dev_tune", args.dev_hard_negatives),
    ):
        outputs[split], report[split] = build_split(
            read_jsonl(path), split_role=role, hard_negatives=hard_negatives
        )
    if args.label_independent_train_candidate_report is not None:
        outputs["train"], report["train"] = build_label_independent_train_split(
            read_jsonl(args.train_jsonl),
            read_json(args.label_independent_train_candidate_report),
            visual_teacher_report=(
                read_json(args.visual_teacher_train_report)
                if args.visual_teacher_train_report is not None
                else None
            ),
            teacher_hard_negatives=args.teacher_hard_negatives,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", outputs["train"])
    write_jsonl(args.output_dir / "dev.jsonl", outputs["dev"])
    if args.label_independent_dev_candidate_report is not None:
        label_independent_dev, label_independent_report = build_label_independent_eval_split(
            read_jsonl(args.dev_jsonl),
            read_json(args.label_independent_dev_candidate_report),
            visual_teacher_report=(
                read_json(args.visual_teacher_dev_report)
                if args.visual_teacher_dev_report is not None
                else None
            ),
        )
        write_jsonl(args.output_dir / "dev_label_independent.jsonl", label_independent_dev)
        report["dev_label_independent"] = label_independent_report
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

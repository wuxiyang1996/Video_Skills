#!/usr/bin/env python3
"""Build on-policy L2 OPD rows from train-only pointwise scoring mistakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.training.l2_pointwise_reranker_v8 import relevance_action
from dataset_clip_wrapper.training.sft_common import read_json, read_jsonl, write_json, write_jsonl


def build_opd_rows(
    chats: list[dict[str, Any]], report: dict[str, Any], *, negatives_per_source: int = 3,
    teacher_confidence: float = 0.98,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.5 < teacher_confidence < 1.0:
        raise ValueError("teacher_confidence must be between 0.5 and 1.0")
    by_key = {
        (str((row.get("metadata") or {}).get("source_example_id")), int((row.get("metadata") or {}).get("candidate_index"))): row
        for row in chats
    }
    output = []
    excluded_gold_outside_candidate_pool = 0
    for result in report.get("results") or []:
        example_id = str(result["example_id"])
        gold = {int(value) for value in result.get("gold") or []}
        ranking = list(result.get("ranking") or [])
        ranked_indices = {int(row["candidate_index"]) for row in ranking}
        if not gold or gold - ranked_indices:
            excluded_gold_outside_candidate_pool += 1
            continue
        positives = [row for row in ranking if int(row["candidate_index"]) in gold]
        negatives = [row for row in ranking if int(row["candidate_index"]) not in gold][:negatives_per_source]
        # Lowest-scoring gold is the student's hardest positive; highest-scoring
        # non-golds are its actual on-policy hard negatives.
        selected = sorted(positives, key=lambda row: float(row["score"]))[:1] + negatives
        for ranked in selected:
            index = int(ranked["candidate_index"])
            chat = by_key.get((example_id, index))
            if chat is None:
                raise ValueError(f"Missing pointwise chat for {example_id}:{index}")
            relevant = index in gold
            sample_weight = 0.5 if relevant else 0.5 / max(1, len(negatives))
            correct_id = "relevant_true" if relevant else "relevant_false"
            wrong_id = "relevant_false" if relevant else "relevant_true"
            output.append({
                "schema_version": "video-skills/opd-distill-v1",
                "state": {
                    "state_id": f"{example_id}::candidate::{index}",
                    "source_example_id": example_id,
                    "candidate_index": index,
                    "messages": chat["messages"][:2],
                    "student_pre_opd_score": float(ranked["score"]),
                    "sample_weight": sample_weight,
                },
                "candidates": {
                    "state_id": f"{example_id}::candidate::{index}",
                    "candidates": [
                        {"action_id": "relevant_true", "action": relevance_action(True)},
                        {"action_id": "relevant_false", "action": relevance_action(False)},
                    ],
                },
                "teacher": {
                    "provider": "deterministic_cg_bench_clue_interval_mapper",
                    "action_probs": {correct_id: teacher_confidence, wrong_id: 1.0 - teacher_confidence},
                    "temperature": 1.0,
                },
                "student_checkpoint": report.get("adapter"),
                "precheck": {"passed": True, "train_only": True, "gold_in_candidate_pool": True},
            })
    summary = {
        "schema_version": "video-skills/l2-pointwise-opd-build-v0.1",
        "source_examples": len(report.get("results") or []),
        "rows": len(output),
        "positive_rows": sum(
            (row["teacher"]["action_probs"].get("relevant_true") or 0.0) > 0.5 for row in output
        ),
        "negative_rows": sum(
            (row["teacher"]["action_probs"].get("relevant_false") or 0.0) > 0.5 for row in output
        ),
        "negatives_per_source": negatives_per_source,
        "teacher_confidence": teacher_confidence,
        "excluded_gold_outside_candidate_pool": excluded_gold_outside_candidate_pool,
    }
    return output, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--negatives-per-source", type=int, default=3)
    parser.add_argument("--teacher-confidence", type=float, default=0.98)
    args = parser.parse_args(argv)
    rows, summary = build_opd_rows(
        read_jsonl(args.train_jsonl), read_json(args.train_report),
        negatives_per_source=args.negatives_per_source,
        teacher_confidence=args.teacher_confidence,
    )
    write_jsonl(args.output_jsonl, rows)
    write_json(args.output_report, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build train-only pointwise OPD rows for CG-Bench and Video-Holmes.

The teacher uses hidden annotations evaluator-side, while policy messages
contain only the question and visible candidate schema. CG relevance comes
from clue intervals. Video-Holmes relevance combines inference-shot overlap
with segment/relationship/causal lexical support.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_clip_wrapper.training.l2_oracle_retrieval_v5 import policy_catalog
from dataset_clip_wrapper.training.l2_pointwise_reranker_v8 import (
    TASK,
    pointwise_state,
    relevance_action,
)
from dataset_clip_wrapper.training.l2_specialist_sft_adapter import SYSTEM
from dataset_clip_wrapper.training.sft_common import (
    compact_visibility,
    contains_forbidden_prompt_key,
    write_json,
    write_jsonl,
)
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.l2_dataset_rewards import (
    RELATIONSHIP_SUPPORT_VERSION,
    VH_PLACEHOLDER_FILTER_VERSION,
    lexical_support,
    load_dataset_reward_supervision,
    supervision_key,
    temporal_hit,
)
from trainer.grpo.train_l2_terminal_on_policy import retrieval_catalog
from trainer.split_filter import assert_role_exclusive, filter_examples_by_role, load_split_manifest


VH_OPD_TEACHER_VERSION = (
    "segment-inference-relationship-causal-0.10-0.60-0.30-relv2"
)


def _spans(entry: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> bool:
    span = entry.get("time_span")
    return bool(isinstance(span, Mapping) and any(temporal_hit(span, target) for target in targets))


def candidate_teacher_score(
    example: Mapping[str, Any],
    entry: Mapping[str, Any],
    supervision: Mapping[str, Any],
) -> float:
    dataset = str(example.get("dataset") or "")
    if dataset == "cg_bench":
        return 1.0 if _spans(entry, supervision.get("clue_spans") or []) else 0.0
    if dataset != "video_holmes":
        return 0.0
    inference_hit = 1.0 if _spans(entry, supervision.get("inference_spans") or []) else 0.0
    reference_texts = [
        *(supervision.get("inference_texts") or []),
        *(supervision.get("relationship_texts") or []),
        str((example.get("question") or {}).get("question_text") or ""),
        str(((example.get("question") or {}).get("answer") or {}).get("text") or ""),
        str((example.get("metadata") or {}).get("explanation") or ""),
    ]
    semantic = lexical_support([entry], reference_texts)
    segment_semantic = lexical_support([entry], supervision.get("segment_texts") or [])
    return min(1.0, 0.60 * inference_hit + 0.30 * semantic + 0.10 * segment_semantic)


def _opd_row(
    example: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    candidate_index: int,
    score: float,
    relevant: bool | None,
    sample_weight: float,
) -> dict[str, Any]:
    """Build one OPD row.

    ``relevant=None`` marks a middle-band candidate: one the teacher scores
    between its negative and positive thresholds.  Those are the candidates that
    actually compete for the top-k slots, so rather than forcing them to a side
    the teacher score is used directly as the relevance probability.
    """
    example_id = str(example.get("example_id") or "")
    dataset = str(example.get("dataset") or "")
    state = {
        "dataset": dataset,
        "example_id": example_id,
        "question": compact_visibility(example.get("question") or {}),
        "candidate_retrieval": {"rank": candidate_index + 1},
    }
    visible = pointwise_state(state, dict(candidate))
    if contains_forbidden_prompt_key(visible):
        raise ValueError(f"hidden supervision leaked into OPD prompt for {example_id}:{candidate_index}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"task": TASK, "state_t": visible},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    if relevant is None:
        relevant_prob = max(0.05, min(0.95, float(score)))
    elif relevant:
        relevant_prob = max(0.55, min(0.98, 0.50 + 0.48 * score))
    else:
        relevant_prob = min(0.45, max(0.02, 0.50 * score))
    return {
        "schema_version": "video-skills/opd-distill-v1",
        "state": {
            "state_id": f"{example_id}::candidate::{candidate_index}",
            "source_example_id": example_id,
            "dataset": dataset,
            "candidate_index": candidate_index,
            "messages": messages,
            "sample_weight": sample_weight,
        },
        "candidates": {
            "state_id": f"{example_id}::candidate::{candidate_index}",
            "candidates": [
                {"action_id": "relevant_true", "action": relevance_action(True)},
                {"action_id": "relevant_false", "action": relevance_action(False)},
            ],
        },
        "teacher": {
            "provider": (
                "deterministic_cg_bench_clue_interval_mapper"
                if dataset == "cg_bench"
                else "video_holmes_segment_inference_relationship_teacher"
            ),
            "action_probs": {
                "relevant_true": relevant_prob,
                "relevant_false": 1.0 - relevant_prob,
            },
            "annotation_score": score,
            "temperature": 1.0,
        },
        "precheck": {
            "passed": True,
            "train_only": True,
            "split_role": "opd_pool",
            "hidden_supervision_visible_to_policy": False,
        },
    }


def build_dataset_opd_rows(
    examples: Sequence[Mapping[str, Any]],
    supervision_index: Mapping[str, Mapping[str, Any]],
    *,
    positives_per_example: int = 3,
    negatives_per_example: int = 3,
    middle_band_per_example: int = 0,
    min_video_holmes_score: float = 0.50,
    max_video_holmes_negative_score: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn frozen L1 examples into pointwise OPD rows.

    ``middle_band_per_example`` emits candidates the thresholds otherwise drop.
    On Video-Holmes that band holds 72% of all candidates (median 59 per
    example) against the 3 positives and 3 negatives actually trained on, and it
    is precisely where the top-k decision is made.  CG-Bench's teacher score is
    binary, so it has no middle band and this has no effect there.
    """
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for example in examples:
        dataset = str(example.get("dataset") or "")
        if dataset not in {"cg_bench", "video_holmes"}:
            continue
        supervision = supervision_index.get(supervision_key(example))
        if not supervision:
            excluded[f"{dataset}:missing_supervision"] += 1
            continue
        catalog, _ = retrieval_catalog(example)
        if not catalog:
            excluded[f"{dataset}:empty_catalog"] += 1
            continue
        visible_catalog = policy_catalog(catalog)
        scored = [
            (candidate_teacher_score(example, raw, supervision), index, visible_catalog[index])
            for index, raw in enumerate(catalog)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        if dataset == "cg_bench":
            positives = [row for row in scored if row[0] >= 1.0][:positives_per_example]
        else:
            positives = [row for row in scored if row[0] >= min_video_holmes_score][:positives_per_example]
        positive_indices = {index for _, index, _ in positives}
        max_negative_score = 0.0 if dataset == "cg_bench" else max_video_holmes_negative_score
        negatives = [
            row for row in scored
            if row[1] not in positive_indices and row[0] <= max_negative_score
        ][:negatives_per_example]
        if not positives or not negatives:
            excluded[f"{dataset}:missing_class"] += 1
            continue
        middle: list[tuple[float, int, Any]] = []
        if middle_band_per_example > 0:
            negative_indices = {index for _, index, _ in negatives}
            # Highest-scoring first: those are the candidates contending for the
            # top-k slots, so they carry the most ranking signal.
            middle = [
                row for row in scored
                if row[1] not in positive_indices
                and row[1] not in negative_indices
                and max_negative_score < row[0] < (
                    1.0 if dataset == "cg_bench" else min_video_holmes_score
                )
            ][:middle_band_per_example]
        source_counts[dataset] += 1
        groups: list[tuple[bool | None, list[tuple[float, int, Any]]]] = [
            (True, positives),
            (False, negatives),
        ]
        if middle:
            groups.append((None, middle))
        # Each group carries equal total mass so a large middle band cannot swamp
        # the confidently-labelled ends.
        group_mass = 1.0 / len(groups)
        for relevant, selected in groups:
            weight = group_mass / len(selected)
            for score, index, candidate in selected:
                rows.append(
                    _opd_row(
                        example,
                        candidate,
                        candidate_index=index,
                        score=score,
                        relevant=relevant,
                        sample_weight=weight,
                    )
                )
                label = "middle" if relevant is None else ("positive" if relevant else "negative")
                row_counts[f"{dataset}:{label}"] += 1
    summary = {
        "schema_version": "video-skills/l2-dataset-opd-build-v0.1",
        "source_examples": dict(source_counts),
        "rows": len(rows),
        "row_counts": dict(row_counts),
        "excluded": dict(excluded),
        "positives_per_example": positives_per_example,
        "negatives_per_example": negatives_per_example,
        "middle_band_per_example": middle_band_per_example,
        "min_video_holmes_score": min_video_holmes_score,
        "max_video_holmes_negative_score": max_video_holmes_negative_score,
        "hidden_supervision_visible_to_policy": False,
        "split_role": "opd_pool",
        "video_holmes_supervision_contract": VH_PLACEHOLDER_FILTER_VERSION,
        "relationship_support_contract": RELATIONSHIP_SUPPORT_VERSION,
        "video_holmes_teacher_contract": VH_OPD_TEACHER_VERSION,
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--datasets", default="cg_bench,video_holmes")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        help="Optional cap applied independently to each dataset before mixing.",
    )
    parser.add_argument("--positives-per-example", type=int, default=3)
    parser.add_argument("--negatives-per-example", type=int, default=3)
    parser.add_argument(
        "--middle-band-per-example",
        type=int,
        default=0,
        help=(
            "Candidates to keep from between the negative and positive thresholds, "
            "highest-scoring first, with the teacher score used directly as the "
            "relevance probability.  0 reproduces earlier builds."
        ),
    )
    parser.add_argument("--min-video-holmes-score", type=float, default=0.50)
    parser.add_argument("--max-video-holmes-negative-score", type=float, default=0.05)
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.frozen_l1_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    examples = load_frozen_l1_examples(paths)
    deduped = {str(example.get("example_id") or ""): example for example in examples if example.get("example_id")}
    examples = list(deduped.values())
    manifest = load_split_manifest(args.split_manifest)
    examples = filter_examples_by_role(examples, manifest=manifest, role="opd_pool", strict=False)
    assert_role_exclusive(examples, manifest=manifest, allowed_roles=("opd_pool",))
    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    examples = [example for example in examples if str(example.get("dataset") or "") in datasets]
    examples.sort(key=lambda row: (str(row.get("dataset") or ""), str(row.get("example_id") or "")))
    if args.limit_per_dataset is not None:
        counts: Counter[str] = Counter()
        balanced = []
        for example in examples:
            dataset = str(example.get("dataset") or "")
            if counts[dataset] >= max(0, args.limit_per_dataset):
                continue
            balanced.append(example)
            counts[dataset] += 1
        examples = balanced
    if args.limit is not None:
        examples = examples[: max(0, args.limit)]
    rows, summary = build_dataset_opd_rows(
        examples,
        load_dataset_reward_supervision(args.dataset_root),
        positives_per_example=args.positives_per_example,
        negatives_per_example=args.negatives_per_example,
        middle_band_per_example=args.middle_band_per_example,
        min_video_holmes_score=args.min_video_holmes_score,
        max_video_holmes_negative_score=args.max_video_holmes_negative_score,
    )
    if not rows:
        raise RuntimeError(f"no OPD rows built: {summary}")
    summary.update(
        {
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
            "dataset_root": str(args.dataset_root),
            "frozen_l1_paths": len(paths),
            "frozen_l1_unique_examples": len(deduped),
            "selected_opd_examples": len(examples),
        }
    )
    write_jsonl(args.output_jsonl, rows)
    write_json(args.output_report, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build leakage-safe CG/VH pointwise evaluation rows from frozen L1."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_clip_wrapper.training.l2_oracle_retrieval_v5 import policy_catalog
from dataset_clip_wrapper.training.l2_pointwise_reranker_v8 import TASK, pointwise_state, relevance_action
from dataset_clip_wrapper.training.l2_specialist_sft_adapter import SYSTEM
from dataset_clip_wrapper.training.sft_common import (
    compact_visibility,
    contains_forbidden_prompt_key,
    read_jsonl,
    write_json,
    write_jsonl,
)
from trainer.build_l2_dataset_opd import candidate_teacher_score
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.l2_dataset_rewards import (
    VH_PLACEHOLDER_FILTER_VERSION,
    load_dataset_reward_supervision,
    supervision_key,
)
from trainer.grpo.train_l2_terminal_on_policy import retrieval_catalog
from trainer.split_filter import assert_role_exclusive, filter_examples_by_role, load_split_manifest


def build_dataset_dev_rows(
    examples: Sequence[Mapping[str, Any]],
    supervision_index: Mapping[str, Mapping[str, Any]],
    *,
    max_candidates: int | None = None,
    candidate_indices_by_example: Mapping[str, Sequence[int]] | None = None,
    prompt_reference_by_candidate: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    video_holmes_positive_threshold: float = 0.20,
    split_role: str = "dev_tune",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split_role not in {"dev_tune", "heldout_test"}:
        raise ValueError(f"unsupported evaluation split role: {split_role!r}")
    output: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for example in examples:
        dataset = str(example.get("dataset") or "")
        supervision = supervision_index.get(supervision_key(example))
        catalog, _ = retrieval_catalog(example)
        if dataset not in {"cg_bench", "video_holmes"} or not supervision or not catalog:
            excluded[f"{dataset or 'unknown'}:missing_input"] += 1
            continue
        # Prefix truncation is label independent. Prefer full catalogs for the
        # paper gate; max_candidates is only for bounded smoke evaluation.
        example_id = str(example.get("example_id") or "")
        if candidate_indices_by_example is not None:
            raw_indices = candidate_indices_by_example.get(example_id)
            if raw_indices is None:
                excluded[f"{dataset}:candidate_manifest_missing"] += 1
                continue
            indices = [int(value) for value in raw_indices]
            if len(indices) != len(set(indices)) or any(
                index < 0 or index >= len(catalog) for index in indices
            ):
                raise ValueError(f"invalid candidate manifest indices for {example_id}")
        else:
            indices = list(range(len(catalog)))
        if max_candidates is not None:
            indices = indices[: max(1, int(max_candidates))]
        visible_catalog = policy_catalog(catalog)
        all_scores = {
            index: candidate_teacher_score(example, candidate, supervision)
            for index, candidate in enumerate(catalog)
        }
        gold = {
            index
            for index, score in all_scores.items()
            if (score >= 1.0 if dataset == "cg_bench" else score >= video_holmes_positive_threshold)
        }
        if not gold:
            excluded[f"{dataset}:no_positive_in_full_catalog"] += 1
            continue
        if len(gold) == len(catalog):
            excluded[f"{dataset}:no_negative_in_full_catalog"] += 1
            continue
        sources[dataset] += 1
        for retrieval_rank, index in enumerate(indices, start=1):
            reference = (
                prompt_reference_by_candidate.get((example_id, index))
                if prompt_reference_by_candidate is not None
                else None
            )
            if prompt_reference_by_candidate is not None and reference is None:
                raise ValueError(f"missing prompt reference for {example_id}:{index}")
            if reference is not None:
                reference_role = str((reference.get("metadata") or {}).get("split_role") or "")
                if reference_role and reference_role != split_role:
                    raise ValueError(
                        f"prompt reference split mismatch for {example_id}:{index}: "
                        f"expected={split_role} actual={reference_role}"
                    )
                reference_messages = list(reference.get("messages") or [])
                if len(reference_messages) < 2:
                    raise ValueError(f"invalid prompt reference for {example_id}:{index}")
                system_message = dict(reference_messages[0])
                user_message = dict(reference_messages[1])
                try:
                    user_payload = json.loads(str(user_message.get("content") or ""))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid prompt reference JSON for {example_id}:{index}") from exc
                state_t = user_payload.get("state_t") or {}
                reference_index = (state_t.get("candidate_coarse_summary") or {}).get("coarse_index")
                if user_payload.get("task") != TASK or int(reference_index) != index:
                    raise ValueError(f"prompt reference candidate mismatch for {example_id}:{index}")
                if contains_forbidden_prompt_key(user_payload):
                    raise ValueError(f"hidden supervision leaked into prompt reference: {example_id}:{index}")
            else:
                state = {
                    "dataset": dataset,
                    "example_id": example.get("example_id"),
                    "question": compact_visibility(example.get("question") or {}),
                    "candidate_retrieval": {"rank": retrieval_rank},
                }
                candidate = dict(visible_catalog[index])
                candidate["retrieval_rank"] = retrieval_rank
                visible = pointwise_state(state, candidate)
                if contains_forbidden_prompt_key(visible):
                    raise ValueError(f"hidden supervision leaked into dev prompt: {example.get('example_id')}:{index}")
                system_message = {"role": "system", "content": SYSTEM}
                user_message = {
                    "role": "user",
                    "content": json.dumps(
                        {"task": TASK, "state_t": visible},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            action = relevance_action(index in gold)
            output.append({
                "schema_version": "video-skills/l2-dataset-eval-chat-v0.2",
                "transition_id": f"{example.get('example_id')}::dataset_dev::{index}",
                "specialist": "l2",
                "messages": [
                    system_message,
                    user_message,
                    {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))},
                ],
                "metadata": {
                    "controller": "l2_controller",
                    "task": TASK,
                    "dataset": dataset,
                    "source_example_id": str(example.get("example_id") or ""),
                    "candidate_index": index,
                    "candidate_relevant": index in gold,
                    "gold_indices": sorted(gold),
                    "gold_in_visible_prefix": any(index in gold for index in indices),
                    "retrieval_rank": retrieval_rank,
                    "candidate_entry": catalog[index],
                    "process_supervision": dict(supervision),
                    "candidate_selection_label_independent": True,
                    "hidden_supervision_visible_to_policy": False,
                    "split_role": split_role,
                },
            })
    return output, {
        "schema_version": "video-skills/l2-dataset-eval-build-v0.2",
        "source_examples": dict(sources),
        "rows": len(output),
        "excluded": dict(excluded),
        "max_candidates": max_candidates,
        "video_holmes_positive_threshold": video_holmes_positive_threshold,
        "candidate_selection_label_independent": True,
        "hidden_supervision_visible_to_policy": False,
        "candidate_selection_mode": (
            "fixed_candidate_manifest"
            if candidate_indices_by_example is not None
            else "catalog_prefix" if max_candidates is not None else "full_catalog"
        ),
        "prompt_payload_mode": (
            "frozen_reference" if prompt_reference_by_candidate is not None else "rebuilt_from_catalog"
        ),
        "split_role": split_role,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--datasets", default="cg_bench,video_holmes")
    parser.add_argument(
        "--split-role",
        choices=("dev_tune", "heldout_test"),
        default="dev_tune",
        help="Frozen manifest role to evaluate; heldout_test must only be used after release gates.",
    )
    parser.add_argument(
        "--example-id-allowlist",
        type=Path,
        help="Optional fixed newline-delimited dev core; filtering remains label independent.",
    )
    parser.add_argument(
        "--candidate-index-manifest",
        type=Path,
        help="Fixed label-independent candidate indices per example, ordered by retrieval rank.",
    )
    parser.add_argument(
        "--prompt-reference-jsonl",
        type=Path,
        help=(
            "Optional frozen label-independent dev JSONL whose system/user prompt payload is reused "
            "exactly by (source_example_id, candidate_index)."
        ),
    )
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--video-holmes-positive-threshold", type=float, default=0.20)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_candidates is not None and args.candidate_index_manifest:
        parser.error("--max-candidates and --candidate-index-manifest are mutually exclusive")

    paths: list[Path] = []
    for pattern in args.frozen_l1_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    examples = load_frozen_l1_examples(paths)
    deduped = {str(row.get("example_id") or ""): row for row in examples if row.get("example_id")}
    manifest = load_split_manifest(args.split_manifest)
    examples = filter_examples_by_role(
        deduped.values(), manifest=manifest, role=args.split_role, strict=False
    )
    assert_role_exclusive(examples, manifest=manifest, allowed_roles=(args.split_role,))
    datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    allowed_example_ids = None
    if args.example_id_allowlist:
        allowed_example_ids = {
            line.strip()
            for line in args.example_id_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not allowed_example_ids:
            raise ValueError(f"empty example id allowlist: {args.example_id_allowlist}")
    examples = sorted(
        (
            row for row in examples
            if str(row.get("dataset") or "") in datasets
            and (
                allowed_example_ids is None
                or str(row.get("example_id") or "") in allowed_example_ids
            )
        ),
        key=lambda row: (str(row.get("dataset") or ""), str(row.get("example_id") or "")),
    )
    if args.limit_per_dataset is not None:
        counts: Counter[str] = Counter()
        selected = []
        for row in examples:
            dataset = str(row.get("dataset") or "")
            if counts[dataset] >= max(0, args.limit_per_dataset):
                continue
            counts[dataset] += 1
            selected.append(row)
        examples = selected
    candidate_indices_by_example = None
    if args.candidate_index_manifest:
        candidate_payload = json.loads(args.candidate_index_manifest.read_text(encoding="utf-8"))
        candidate_indices_by_example = candidate_payload.get("examples")
        if not isinstance(candidate_indices_by_example, Mapping):
            raise ValueError(f"invalid candidate index manifest: {args.candidate_index_manifest}")
    prompt_reference_by_candidate = None
    if args.prompt_reference_jsonl:
        prompt_reference_by_candidate = {}
        for row in read_jsonl(args.prompt_reference_jsonl):
            metadata = row.get("metadata") or {}
            key = (str(metadata.get("source_example_id") or ""), int(metadata["candidate_index"]))
            if not key[0] or key in prompt_reference_by_candidate:
                raise ValueError(f"invalid or duplicate prompt reference key: {key}")
            prompt_reference_by_candidate[key] = row
    rows, report = build_dataset_dev_rows(
        examples,
        load_dataset_reward_supervision(args.dataset_root),
        max_candidates=args.max_candidates,
        candidate_indices_by_example=candidate_indices_by_example,
        prompt_reference_by_candidate=prompt_reference_by_candidate,
        video_holmes_positive_threshold=args.video_holmes_positive_threshold,
        split_role=args.split_role,
    )
    report.update({
        "split_role": args.split_role,
        "frozen_l1_globs": list(args.frozen_l1_glob),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
        "dataset_root": str(args.dataset_root),
        "video_holmes_supervision_contract": VH_PLACEHOLDER_FILTER_VERSION,
        "example_id_allowlist": str(args.example_id_allowlist) if args.example_id_allowlist else None,
        "example_id_allowlist_sha256": (
            hashlib.sha256(args.example_id_allowlist.read_bytes()).hexdigest()
            if args.example_id_allowlist else None
        ),
        "candidate_index_manifest": str(args.candidate_index_manifest) if args.candidate_index_manifest else None,
        "candidate_index_manifest_sha256": (
            hashlib.sha256(args.candidate_index_manifest.read_bytes()).hexdigest()
            if args.candidate_index_manifest else None
        ),
        "prompt_reference_jsonl": str(args.prompt_reference_jsonl) if args.prompt_reference_jsonl else None,
        "prompt_reference_jsonl_sha256": (
            hashlib.sha256(args.prompt_reference_jsonl.read_bytes()).hexdigest()
            if args.prompt_reference_jsonl else None
        ),
    })
    if not rows:
        raise RuntimeError(f"no {args.split_role} rows built: {report}")
    write_jsonl(args.output_jsonl, rows)
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate the official Qwen3-VL cross-encoder on L2 visual candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from .evaluate_l2_retrieval_adapter import retrieval_scores
from .evaluate_l2_visual_candidate_retrieval import fine_spans, question_text, reduce_fine_scores, sample_frames
from .l2_candidate_reranker_v7 import source_index
from .sft_common import read_json, read_jsonl, write_json


INSTRUCTION = "Rank coarse video segments by whether their visual or textual evidence is needed to answer the question."


def aggregate_rankings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [retrieval_scores(row["predicted"], row["gold"]) for row in rows]
    return {
        "examples": len(metrics),
        "mean_precision": sum(float(row["precision"]) for row in metrics) / max(1, len(metrics)),
        "mean_recall": sum(float(row["recall"]) for row in metrics) / max(1, len(metrics)),
        "hit_rate": sum(bool(row["hit"]) for row in metrics) / max(1, len(metrics)),
        "exact_rate": sum(bool(row["exact"]) for row in metrics) / max(1, len(metrics)),
    }


def _load_reranker_class(model_path: Path) -> Any:
    script = model_path / "scripts" / "qwen3_vl_reranker.py"
    if not script.is_file():
        raise ValueError(f"Official reranker helper not found: {script}")
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker_local", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reranker helper: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3VLReranker


def _core_rows(path: Path) -> list[dict[str, Any]]:
    return [
        row for row in read_jsonl(path)
        if (row.get("metadata") or {}).get("task") == "select_coarse_set"
        and (row.get("metadata") or {}).get("is_core") is True
    ]


def process_in_batches(reranker: Any, payload: dict[str, Any], batch_size: int) -> list[Any]:
    """Run the official scorer in small batches instead of its one-pair loop."""
    if batch_size <= 1:
        return list(reranker.process(payload))
    instruction = payload.get("instruction", reranker.default_instruction)
    query = payload.get("query") or {}
    pairs = [
        reranker.format_mm_instruction(
            query.get("text"), query.get("image"), query.get("video"),
            document.get("text"), document.get("image"), document.get("video"),
            instruction=instruction,
            fps=payload.get("fps", reranker.fps),
            max_frames=payload.get("max_frames", reranker.max_frames),
        )
        for document in payload.get("documents") or []
    ]
    scores = []
    for start in range(0, len(pairs), batch_size):
        inputs = reranker.tokenize(pairs[start : start + batch_size]).to(reranker.model.device)
        scores.extend(reranker.compute_scores(inputs))
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=None,
        help="Optional label-independent visual retrieval report whose top-32 pool is reranked.",
    )
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--fine-window-sec", type=float, default=0.0)
    parser.add_argument("--fine-stride-sec", type=float, default=4.0)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    args = parser.parse_args(argv)

    import cv2
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sources = source_index(read_jsonl(args.rollouts))
    candidate_pools: dict[str, list[int]] = {}
    if args.candidate_report is not None:
        candidate_report = read_json(args.candidate_report)
        if not bool(candidate_report.get("label_independent")):
            raise ValueError("Candidate report must be label-independent")
        def candidate_pool(item: dict[str, Any]) -> list[int]:
            requested = int(args.pool_size)
            for key in (
                f"top{requested}_boundary_hybrid",
                f"top{requested}",
                "top64_boundary_hybrid" if requested > 32 else "",
                "top64" if requested > 32 else "",
                "top32_boundary_hybrid",
                "top32",
            ):
                if key and item.get(key):
                    return [int(value) for value in item.get(key) or []]
            return []

        candidate_pools = {
            str(item["example_id"]): candidate_pool(item)
            for item in candidate_report.get("results") or []
        }
    Reranker = _load_reranker_class(args.model_path)
    # Frames are already deterministically sampled below. Passing None prevents
    # the upstream helper from trying to sample the supplied PIL list again.
    reranker = Reranker(
        model_name_or_path=str(args.model_path),
        num_frames=None, max_frames=None,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    # The helper bundled with the initial model revision leaves
    # mm_token_type_ids as a Python list after padding input_ids. Transformers
    # 5.13 expects a tensor and otherwise fails while applying attention_mask.
    upstream_tokenize = reranker.tokenize

    def compatible_tokenize(pairs: list[Any], **kwargs: Any) -> Any:
        inputs = upstream_tokenize(pairs, **kwargs)
        token_types = inputs.get("mm_token_type_ids")
        if isinstance(token_types, list):
            width = int(inputs["input_ids"].shape[1])
            padding_side = str(getattr(reranker.processor.tokenizer, "padding_side", "right"))
            padded = []
            for values in token_types:
                values = list(values)
                if len(values) > width:
                    values = values[-width:] if padding_side == "left" else values[:width]
                padding = [0] * (width - len(values))
                padded.append(padding + values if padding_side == "left" else values + padding)
            inputs["mm_token_type_ids"] = torch.tensor(padded, dtype=torch.long)
        return inputs

    reranker.tokenize = compatible_tokenize

    results = []
    rows = _core_rows(args.dev_jsonl)
    for number, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or {}
        example_id = str(metadata["source_example_id"])
        source = sources.get(example_id)
        if source is None:
            raise ValueError(f"Missing rollout source: {example_id}")
        user = json.loads(row["messages"][1]["content"])
        state = user["state_t"]
        full_catalog = list(state["l1_coarse_summary_catalog"])
        if candidate_pools:
            requested = candidate_pools.get(example_id)
            if not requested:
                raise ValueError(f"Missing candidate pool: {example_id}")
            catalog_by_index = {
                int(candidate["coarse_index"]): candidate for candidate in full_catalog
            }
            missing = set(requested[: args.pool_size]) - set(catalog_by_index)
            if missing:
                raise ValueError(
                    f"Candidate report contains unknown indices for {example_id}: {sorted(missing)}"
                )
            catalog = [catalog_by_index[index] for index in requested[: args.pool_size]]
        else:
            catalog = full_catalog[: args.pool_size]
        gold_action = json.loads(row["messages"][2]["content"])
        gold = [int(value) for value in gold_action["arguments"]["selected_coarse_indices"]]
        video_path = str((source.get("video") or {}).get("primary_path") or "")
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        documents = []
        parent_indices = []
        try:
            for parent_index, candidate in enumerate(catalog):
                for span in fine_spans(
                    candidate.get("time_span") or {},
                    window_sec=args.fine_window_sec, stride_sec=args.fine_stride_sec,
                ):
                    frames = sample_frames(
                        capture, span, num_frames=args.num_frames, max_side=args.max_side,
                    )
                    documents.append({
                        "text": "\n".join([
                            str(candidate.get("scene_description") or ""),
                            *[str(value) for value in candidate.get("observable_facts") or []],
                            *[str(value) for value in candidate.get("events") or []],
                        ]),
                        "video": [Image.fromarray(frame) for frame in frames],
                    })
                    parent_indices.append(parent_index)
        finally:
            capture.release()
        scores = process_in_batches(reranker, {
            "instruction": INSTRUCTION,
            "query": {"text": question_text(source.get("question") or {})},
            "documents": documents,
        }, args.reranker_batch_size)
        if len(scores) != len(documents):
            raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(documents)} fine windows")
        scores = reduce_fine_scores(scores, parent_indices, len(catalog))
        ranking = sorted(
            zip((float(value) for value in scores), (int(candidate["coarse_index"]) for candidate in catalog)),
            key=lambda item: (-item[0], item[1]),
        )
        predicted = [index for _, index in ranking[:2]]
        results.append({
            "example_id": example_id, "gold": gold, "predicted": predicted,
            "pool": [int(candidate["coarse_index"]) for candidate in catalog],
            "ranking": [{"candidate_index": index, "score": score} for score, index in ranking],
        })
        print(f"[{number}/{len(rows)}] {example_id} predicted={predicted}", flush=True)

    report = {
        "schema_version": "video-skills/l2-visual-reranker-eval-v0.1",
        "model": str(args.model_path), "instruction": INSTRUCTION,
        "candidate_report": str(args.candidate_report) if args.candidate_report else None,
        "pool_size": args.pool_size, "num_frames": args.num_frames, "max_side": args.max_side,
        "fine_window_sec": args.fine_window_sec, "fine_stride_sec": args.fine_stride_sec,
        "reranker_batch_size": args.reranker_batch_size,
        "label_independent": True, "metrics": aggregate_rankings(results), "results": results,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

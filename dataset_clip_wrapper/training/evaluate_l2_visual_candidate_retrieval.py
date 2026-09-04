#!/usr/bin/env python3
"""Evaluate label-independent visual+text candidate retrieval on L2 dev data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_l2_candidate_retrieval import (
    K_VALUES,
    _aggregate,
    _core_labels,
    boundary_hybrid_candidates,
    document_text,
)
from .l2_retrieval_sft_adapter import _catalog
from .sft_common import read_jsonl, write_json


PROMPT = "Retrieve coarse video segments likely to contain evidence needed to answer the question."


def question_text(question: dict[str, Any]) -> str:
    options = " | ".join(str(row.get("text") or "") for row in question.get("options") or [])
    return f"{question.get('question_text') or ''}\nAnswer options: {options}"


def sample_frames(
    capture: Any,
    span: dict[str, Any],
    *,
    num_frames: int,
    max_side: int,
) -> np.ndarray:
    """Sample RGB frames uniformly inside one coarse time span."""
    import cv2

    start = float(span.get("start_s", 0.0))
    end = max(start, float(span.get("end_s", start)))
    margin = min(1.0, max(0.0, (end - start) / 10.0))
    times = np.linspace(start + margin, max(start + margin, end - margin), num_frames)
    frames = []
    for timestamp in times:
        frame = None
        # Long web videos occasionally have a broken keyframe near the exact
        # seek point.  Retrying a small, label-independent temporal offset is
        # preferable to dropping the example or substituting oracle footage.
        for offset in (0.0, -0.25, 0.25, -1.0, 1.0, -2.0):
            retry_timestamp = min(end, max(start, float(timestamp) + offset))
            capture.set(cv2.CAP_PROP_POS_MSEC, retry_timestamp * 1000.0)
            ok, candidate = capture.read()
            if ok:
                frame = candidate
                break
        if frame is None:
            raise RuntimeError(f"Could not decode frame near {timestamp:.3f}s")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    return np.stack(frames)


def fine_spans(span: dict[str, Any], *, window_sec: float, stride_sec: float) -> list[dict[str, float]]:
    """Create deterministic sub-windows while ensuring the coarse tail is covered."""
    start = float(span.get("start_s", 0.0))
    end = max(start, float(span.get("end_s", start)))
    if window_sec <= 0 or end - start <= window_sec:
        return [{"start_s": start, "end_s": end}]
    starts = list(np.arange(start, end - window_sec + 1e-9, stride_sec))
    tail_start = end - window_sec
    if not starts or tail_start - starts[-1] > 1e-6:
        starts.append(tail_start)
    return [{"start_s": float(value), "end_s": min(end, float(value) + window_sec)} for value in starts]


def reduce_fine_scores(scores: list[float], parent_indices: list[int], catalog_size: int) -> list[float]:
    """Aggregate fine-window scores into one max score per coarse candidate."""
    result = [float("-inf")] * catalog_size
    for score, parent in zip(scores, parent_indices, strict=True):
        result[parent] = max(result[parent], float(score))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--fine-window-sec", type=float, default=0.0)
    parser.add_argument("--fine-stride-sec", type=float, default=4.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)

    import cv2
    import torch
    from sentence_transformers import SentenceTransformer

    labels = _core_labels(args.dev_jsonl)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    labels = dict(list(labels.items())[args.shard_index :: args.num_shards])
    wanted = set(labels)
    sources = {
        str(row["example_id"]): row
        for row in read_jsonl(args.rollouts)
        if row.get("example_id") in wanted
    }
    missing = sorted(wanted - set(sources))
    if missing:
        raise ValueError(f"Missing rollout sources: {missing[:5]}")

    model = SentenceTransformer(
        args.model,
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        device="cuda",
        local_files_only=True,
    )
    results = []
    for example_number, (example_id, gold) in enumerate(labels.items(), start=1):
        source = sources[example_id]
        catalog = _catalog((source.get("metadata") or {}).get("coarse_clip_schemas") or [])
        video_path = str((source.get("video") or {}).get("primary_path") or "")
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        try:
            documents = []
            parent_indices = []
            for parent_index, row in enumerate(catalog):
                spans = fine_spans(
                    row.get("time_span") or {},
                    window_sec=args.fine_window_sec,
                    stride_sec=args.fine_stride_sec,
                )
                for span in spans:
                    frames = sample_frames(
                        capture,
                        span,
                        num_frames=args.num_frames,
                        max_side=args.max_side,
                    )
                    documents.append({"text": document_text(row), "video": frames})
                    parent_indices.append(parent_index)
        finally:
            capture.release()

        query_embedding = model.encode(
            [question_text(source.get("question") or {})],
            prompt=PROMPT,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        document_embeddings = model.encode(
            documents,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        fine_scores = (query_embedding @ document_embeddings.T)[0]
        scores = torch.tensor(
            reduce_fine_scores(fine_scores.tolist(), parent_indices, len(catalog)),
            device=fine_scores.device,
        )
        ranking = torch.argsort(scores, descending=True).tolist()
        order = [int(catalog[index]["coarse_index"]) for index in ranking]
        from .evaluate_l2_candidate_retrieval import candidate_metrics

        hybrid = boundary_hybrid_candidates(order, len(catalog))
        hybrid64 = boundary_hybrid_candidates(order, len(catalog), k=64)
        gold_set = set(gold)
        hybrid_overlap = set(hybrid) & gold_set
        hybrid64_overlap = set(hybrid64) & gold_set
        results.append({
            "example_id": example_id,
            "catalog_size": len(catalog),
            "gold": gold,
            "top32": order[:32],
            "top32_boundary_hybrid": hybrid,
            "top64": order[:64],
            "top64_boundary_hybrid": hybrid64,
            "boundary_hybrid_at_32": {
                "hit": bool(hybrid_overlap),
                "recall": len(hybrid_overlap) / max(1, len(gold_set)),
            },
            "boundary_hybrid_at_64": {
                "hit": bool(hybrid64_overlap),
                "recall": len(hybrid64_overlap) / max(1, len(gold_set)),
            },
            "metrics": candidate_metrics(order, gold),
        })
        print(f"[{example_number}/{len(labels)}] {example_id} done", flush=True)

    output = {
        "schema_version": "video-skills/l2-visual-candidate-eval-v1",
        "model": args.model,
        "label_independent": True,
        "uses_hidden_supervision_in_query_or_document": False,
        "num_frames_per_coarse_window": args.num_frames,
        "max_frame_side": args.max_side,
        "fine_window_sec": args.fine_window_sec,
        "fine_stride_sec": args.fine_stride_sec,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "summary": _aggregate(results),
        "boundary_hybrid_summary": {
            "examples": len(results),
            "hit_at_32": sum(row["boundary_hybrid_at_32"]["hit"] for row in results) / len(results),
            "recall_at_32": sum(row["boundary_hybrid_at_32"]["recall"] for row in results) / len(results),
            "hit_at_64": sum(row["boundary_hybrid_at_64"]["hit"] for row in results) / len(results),
            "recall_at_64": sum(row["boundary_hybrid_at_64"]["recall"] for row in results) / len(results),
        },
        "results": results,
    }
    write_json(args.output, output)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

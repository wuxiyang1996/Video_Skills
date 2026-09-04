#!/usr/bin/env python3
"""Evaluate label-independent dense candidate retrieval for L2 reranking."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .l2_retrieval_sft_adapter import _catalog
from .sft_common import read_jsonl, write_json


K_VALUES = (4, 8, 16, 24, 32, 64)
TASK = "Given a video question, retrieve coarse video summaries likely to contain direct visual or subtitle evidence."


def query_text(question: dict[str, Any], *, include_options: bool) -> str:
    text = str(question.get("question_text") or "")
    if include_options:
        options = " | ".join(str(row.get("text") or "") for row in question.get("options") or [])
        text = f"{text}\nAnswer options: {options}"
    return f"Instruct: {TASK}\nQuery: {text}"


def document_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("scene_description") or "")]
    for key in ("observable_facts", "events", "searchable_phrases"):
        parts.extend(str(value) for value in row.get(key) or [])
    return "\n".join(value for value in parts if value.strip())


def candidate_metrics(order: list[int], gold: list[int]) -> dict[str, dict[str, float | bool]]:
    gold_set = set(gold)
    result = {}
    for k in K_VALUES:
        predicted = set(order[:k])
        overlap = predicted & gold_set
        result[str(k)] = {
            "hit": bool(overlap),
            "recall": len(overlap) / max(1, len(gold_set)),
        }
    return result


def boundary_hybrid_candidates(order: list[int], catalog_size: int, *, k: int = 32) -> list[int]:
    """Reserve fixed exploration slots for the first and last video windows."""
    semantic_slots = max(0, k - 2)
    candidates = list(order[:semantic_slots]) + [0, max(0, catalog_size - 1)] + list(order[semantic_slots:])
    result = []
    for index in candidates:
        if index not in result:
            result.append(index)
        if len(result) >= min(k, catalog_size):
            break
    return result


def _core_labels(path: Path) -> dict[str, list[int]]:
    result = {}
    for row in read_jsonl(path):
        metadata = row.get("metadata") or {}
        if metadata.get("task") != "select_coarse_set" or metadata.get("is_core") is not True:
            continue
        action = json.loads(row["messages"][2]["content"])
        result[str(metadata["source_example_id"])] = [
            int(value) for value in action["arguments"]["selected_coarse_indices"]
        ]
    return result


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    summary: dict[str, Any] = {"examples": count}
    for k in K_VALUES:
        key = str(k)
        summary[f"hit_at_{k}"] = sum(bool(row["metrics"][key]["hit"]) for row in results) / count
        summary[f"recall_at_{k}"] = sum(float(row["metrics"][key]["recall"]) for row in results) / count
    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        size = int(row["catalog_size"])
        label = "<=32" if size <= 32 else "33-64" if size <= 64 else "65-96" if size <= 96 else ">96"
        bins[label].append(row)
    summary["by_catalog_size"] = {key: _aggregate(value) for key, value in bins.items()} if len(bins) > 1 else {}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    import torch
    from sentence_transformers import SentenceTransformer

    labels = {"train": _core_labels(args.train_jsonl), "dev": _core_labels(args.dev_jsonl)}
    wanted = set(labels["train"]) | set(labels["dev"])
    sources = {}
    for row in read_jsonl(args.rollouts):
        if row.get("example_id") in wanted:
            sources[str(row["example_id"])] = row
    missing = sorted(wanted - set(sources))
    if missing:
        raise ValueError(f"Missing rollout sources: {missing[:5]}")

    model = SentenceTransformer(
        args.model,
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda",
        local_files_only=True,
    )
    modes = {"question": False, "question_options": True}
    output: dict[str, Any] = {
        "schema_version": "video-skills/l2-dense-candidate-eval-v1",
        "model": args.model,
        "label_independent": True,
        "uses_hidden_supervision_in_query_or_document": False,
        "splits": {},
    }
    for split, split_labels in labels.items():
        output["splits"][split] = {}
        for mode, include_options in modes.items():
            results = []
            for example_id, gold in split_labels.items():
                source = sources[example_id]
                question = source.get("question") or {}
                schemas = (source.get("metadata") or {}).get("coarse_clip_schemas") or []
                catalog = _catalog(schemas)
                query_embedding = model.encode(
                    [query_text(question, include_options=include_options)],
                    batch_size=1,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                )
                document_embeddings = model.encode(
                    [document_text(row) for row in catalog],
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                )
                scores = (query_embedding @ document_embeddings.T)[0]
                ranking = torch.argsort(scores, descending=True).tolist()
                order = [int(catalog[index]["coarse_index"]) for index in ranking]
                results.append({
                    "example_id": example_id,
                    "catalog_size": len(catalog),
                    "gold": gold,
                    "top32": order[:32],
                    "metrics": candidate_metrics(order, gold),
                })
            output["splits"][split][mode] = {"summary": _aggregate(results), "results": results}
    write_json(args.output, output)
    print(json.dumps({
        "model": args.model,
        "train": {mode: payload["summary"] for mode, payload in output["splits"]["train"].items()},
        "dev": {mode: payload["summary"] for mode, payload in output["splits"]["dev"].items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

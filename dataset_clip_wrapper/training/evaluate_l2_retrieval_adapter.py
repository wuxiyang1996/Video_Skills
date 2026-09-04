#!/usr/bin/env python3
"""Evaluate an L2 adapter on independent oracle-window dev decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .sft_common import apply_chat_template_no_think, read_jsonl, strip_think_tags, write_json
from .train_lora_sft import _extract_json_object, _json_complete_stopping_criteria


def selected_indices(payload: dict[str, Any] | None) -> list[int]:
    if not payload or payload.get("tool_name") != "select_coarse_clips":
        return []
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    result: list[int] = []
    for value in arguments.get("selected_coarse_indices") or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index >= 0 and index not in result:
            result.append(index)
    return result


def retrieval_scores(predicted: list[int], gold: list[int]) -> dict[str, float | bool]:
    pred_set, gold_set = set(predicted), set(gold)
    intersection = len(pred_set & gold_set)
    return {
        "precision": intersection / max(1, len(pred_set)),
        "recall": intersection / max(1, len(gold_set)),
        "hit": bool(intersection),
        "exact": pred_set == gold_set and bool(gold_set),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--adapter", type=Path,
        help="Optional LoRA adapter. Omit it to evaluate the frozen base model on the identical gate.",
    )
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--min-mean-recall", type=float, default=0.60)
    parser.add_argument("--min-hit-rate", type=float, default=0.60)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trainer.grpo.model_runtime import _disable_torchao_peft_probes

    rows = [
        row for row in read_jsonl(args.dev_jsonl)
        if (row.get("metadata") or {}).get("task") == "select_coarse_set"
        and (row.get("metadata") or {}).get("is_core") is True
    ]
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError("No core select_coarse_set dev rows")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _disable_torchao_peft_probes()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
        if args.adapter is not None else base
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    results = []
    with torch.no_grad():
        for row in rows:
            prompt = apply_chat_template_no_think(
                tokenizer, row["messages"][:2], add_generation_prompt=True, tokenize=False
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            prompt_len = int(encoded["input_ids"].shape[1])
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[_json_complete_stopping_criteria(tokenizer, prompt_len)],
            )[0]
            completion = strip_think_tags(tokenizer.decode(generated[prompt_len:], skip_special_tokens=True))
            payload = _extract_json_object(completion)
            gold_payload = json.loads(row["messages"][2]["content"])
            predicted, gold = selected_indices(payload), selected_indices(gold_payload)
            results.append({
                "example_id": (row.get("metadata") or {}).get("source_example_id"),
                "predicted": predicted,
                "gold": gold,
                "json_valid": payload is not None,
                "completion": completion,
                **retrieval_scores(predicted, gold),
            })
    count = len(results)
    report = {
        "schema_version": "video-skills/l2-oracle-dev-eval-v1",
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "model_variant": "lora_adapter" if args.adapter is not None else "frozen_base",
        "examples": count,
        "json_valid_rate": sum(item["json_valid"] for item in results) / count,
        "mean_precision": sum(float(item["precision"]) for item in results) / count,
        "mean_recall": sum(float(item["recall"]) for item in results) / count,
        "hit_rate": sum(bool(item["hit"]) for item in results) / count,
        "exact_rate": sum(bool(item["exact"]) for item in results) / count,
        "thresholds": {"min_mean_recall": args.min_mean_recall, "min_hit_rate": args.min_hit_rate},
        "results": results,
    }
    report["passed"] = (
        report["mean_recall"] >= args.min_mean_recall
        and report["hit_rate"] >= args.min_hit_rate
        and report["json_valid_rate"] == 1.0
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

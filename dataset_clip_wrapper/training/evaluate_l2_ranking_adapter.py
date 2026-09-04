#!/usr/bin/env python3
"""Evaluate an L2 adapter on all independent pairwise/listwise dev decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .sft_common import apply_chat_template_no_think, read_jsonl, strip_think_tags, write_json
from .train_lora_sft import _extract_json_object, _json_complete_stopping_criteria


TASKS = {"rank_coarse_candidates", "rank_coarse_candidates_listwise"}


def chosen_index(payload: dict[str, Any] | None) -> int | None:
    if not payload or payload.get("tool_name") not in {
        "choose_better_coarse_candidate", "choose_best_coarse_candidate"
    }:
        return None
    try:
        return int((payload.get("arguments") or {}).get("coarse_index"))
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trainer.grpo.model_runtime import _disable_torchao_peft_probes

    rows = [row for row in read_jsonl(args.dev_jsonl) if (row.get("metadata") or {}).get("task") in TASKS]
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError("No L2 ranking dev rows")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _disable_torchao_peft_probes()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to("cuda")
    model.eval()
    model.config.use_cache = True
    results = []
    with torch.no_grad():
        for number, row in enumerate(rows, start=1):
            prompt = apply_chat_template_no_think(tokenizer, row["messages"][:2], add_generation_prompt=True, tokenize=False)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            prompt_len = int(encoded["input_ids"].shape[1])
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[_json_complete_stopping_criteria(tokenizer, prompt_len)],
            )[0]
            completion = strip_think_tags(tokenizer.decode(generated[prompt_len:], skip_special_tokens=True))
            payload = _extract_json_object(completion)
            gold = _extract_json_object(row["messages"][2]["content"])
            predicted, expected = chosen_index(payload), chosen_index(gold)
            task = str((row.get("metadata") or {}).get("task"))
            results.append({
                "record_id": row.get("transition_id"), "task": task,
                "predicted": predicted, "gold": expected, "json_valid": payload is not None,
                "correct": predicted == expected and expected is not None,
            })
            if number % 16 == 0 or number == len(rows):
                print(f"[{number}/{len(rows)}]", flush=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["task"])].append(row)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(items),
            "json_valid_rate": sum(bool(row["json_valid"]) for row in items) / max(1, len(items)),
            "accuracy": sum(bool(row["correct"]) for row in items) / max(1, len(items)),
        }

    report = {
        "schema_version": "video-skills/l2-ranking-dev-eval-v1",
        "adapter": str(args.adapter),
        **metrics(results),
        "by_task": {task: metrics(items) for task, items in grouped.items()},
        "results": results,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

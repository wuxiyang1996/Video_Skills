#!/usr/bin/env python3
"""Evaluate a fixed top-16 pairwise tournament using an L2 ranking adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .evaluate_l2_retrieval_adapter import retrieval_scores, selected_indices
from .evaluate_l2_ranking_adapter import chosen_index
from .l2_candidate_reranker_v7 import _stable_candidate_order
from .l2_specialist_sft_adapter import SYSTEM
from .sft_common import apply_chat_template_no_think, read_json, read_jsonl, strip_think_tags, write_json
from .train_lora_sft import _extract_json_object, _json_complete_stopping_criteria


def knockout_winner(candidates: list[int], chooser: Callable[[int, int, str], int], salt: str) -> int:
    if not candidates:
        raise ValueError("Tournament requires candidates")
    active = list(candidates)
    round_number = 0
    while len(active) > 1:
        next_round = []
        for offset in range(0, len(active), 2):
            if offset + 1 >= len(active):
                next_round.append(active[offset])
            else:
                next_round.append(chooser(active[offset], active[offset + 1], f"{salt}:{round_number}:{offset // 2}"))
        active = next_round
        round_number += 1
    return active[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
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
    full_results = {str(row["example_id"]): row for row in read_json(args.full_report)["results"]}
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
    comparison_count = 0
    with torch.no_grad():
        for number, row in enumerate(rows, start=1):
            metadata = row.get("metadata") or {}
            example_id = str(metadata.get("source_example_id"))
            payload = json.loads(row["messages"][1]["content"])
            state = payload["state_t"]
            catalog = state["l1_coarse_summary_catalog"][: args.pool_size]
            by_index = {int(candidate["coarse_index"]): candidate for candidate in catalog}
            order = list(by_index)

            def choose(left: int, right: int, salt: str) -> int:
                nonlocal comparison_count
                pair = _stable_candidate_order([by_index[left], by_index[right]], salt)
                ranking_state = {
                    "schema_version": "video-skills/l2-ranking-state-v0.2",
                    "process_model": "pairwise_visual_coarse_reranking",
                    "dataset": state.get("dataset"), "example_id": state.get("example_id"),
                    "question": state.get("question"), "candidate_coarse_summaries": pair,
                }
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps({"task": "rank_coarse_candidates", "state_t": ranking_state}, ensure_ascii=False, separators=(",", ":"))},
                ]
                prompt = apply_chat_template_no_think(tokenizer, messages, add_generation_prompt=True, tokenize=False)
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                prompt_len = int(encoded["input_ids"].shape[1])
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=[_json_complete_stopping_criteria(tokenizer, prompt_len)],
                )[0]
                completion = strip_think_tags(tokenizer.decode(generated[prompt_len:], skip_special_tokens=True))
                predicted = chosen_index(_extract_json_object(completion))
                comparison_count += 1
                return predicted if predicted in {left, right} else left

            winner = knockout_winner(order, choose, example_id)
            full = selected_indices(_extract_json_object(str(full_results[example_id].get("completion") or "")))
            gold = [int(value) for value in full_results[example_id]["gold"]]
            full_first = []
            for index in full[:1] + [winner]:
                if index not in full_first:
                    full_first.append(index)
            keep_two = list(full[:2]) if len(full) >= 2 else full_first
            results.append({
                "example_id": example_id, "gold": gold, "tournament_winner": winner,
                "full_predicted": full, "full_first_plus_tournament": full_first[:2],
                "keep_full_two_else_tournament": keep_two[:2],
            })
            print(f"[{number}/{len(rows)}] comparisons={comparison_count}", flush=True)

    def aggregate(key: str) -> dict[str, Any]:
        metrics = [retrieval_scores([int(value) for value in row[key]], row["gold"]) for row in results]
        return {
            "examples": len(metrics),
            "mean_precision": sum(float(row["precision"]) for row in metrics) / len(metrics),
            "mean_recall": sum(float(row["recall"]) for row in metrics) / len(metrics),
            "hit_rate": sum(bool(row["hit"]) for row in metrics) / len(metrics),
            "exact_rate": sum(bool(row["exact"]) for row in metrics) / len(metrics),
        }

    for row in results:
        row["tournament_only"] = [row["tournament_winner"]]
    report = {
        "schema_version": "video-skills/l2-tournament-executor-eval-v1",
        "adapter": str(args.adapter), "pool_size": args.pool_size, "comparisons": comparison_count,
        "metrics": {key: aggregate(key) for key in (
            "tournament_only", "full_first_plus_tournament", "keep_full_two_else_tournament"
        )},
        "results": results,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

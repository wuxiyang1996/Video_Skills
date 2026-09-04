#!/usr/bin/env python3
"""Rank L2 candidates by independent true/false relevance log-odds.

Three scoring modes are available (see ``--scoring-mode``):

``sequence_logprob``
    Legacy default.  Sums the log-likelihood of the whole assistant JSON under
    the ``true`` and ``false`` variants and subtracts.  The two completions
    differ in a single token, so this differences two large, nearly equal sums
    whose per-token logits are bf16-quantised -- catastrophic cancellation that
    collapses scores onto multiples of ~0.125 and produces near-ties.
``sequence_logprob_fp32``
    Same estimator, fp32 logits.  Isolates precision from the token restriction.
``decision_logit``
    Log-odds read directly off the single decision token, in fp32.  The shared
    ``logsumexp`` cancels, so ``logit[true] - logit[false]`` is exactly the
    log-odds -- with no cancellation, and one forward pass per row instead of
    two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate_l2_retrieval_adapter import retrieval_scores
from .l2_pointwise_reranker_v8 import relevance_action
from .sft_common import (
    apply_chat_template_no_think,
    decision_position as _decision_position,
    read_jsonl,
    write_json,
)
from trainer.grpo.l2_dataset_rewards import lexical_support, temporal_retrieval_metrics
from trainer.artifact_hash import adapter_weight_sha256


def evaluation_input_provenance(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Return fail-auditable provenance for a frozen pointwise input artifact."""
    return {
        "evaluation_jsonl": str(path),
        "evaluation_jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "input_schema_versions": sorted({str(row.get("schema_version") or "") for row in rows}),
        "input_split_roles": sorted({
            str((row.get("metadata") or {}).get("split_role") or "") for row in rows
        }),
        "input_datasets": sorted({
            str((row.get("metadata") or {}).get("dataset") or "") for row in rows
        }),
        "input_examples": len({
            str((row.get("metadata") or {}).get("source_example_id") or "") for row in rows
        }),
    }


def _topk_temporal_nms(
    ranked_indices: list[int],
    spans: dict[int, dict[str, Any]],
    *,
    top_k: int,
) -> list[int]:
    """Greedy top-k that skips any candidate overlapping an already-chosen pick.

    No free parameter.  A reranker can concentrate its top-k on one region and
    cover a single gold segment several times; on the Video-Holmes heldout this
    selection rule alone moved OPD segment_recall from 59.93 to 64.89 with
    precision unchanged.  Candidates without a span are kept in rank order.
    """
    from trainer.grpo.l2_dataset_rewards import temporal_hit

    chosen: list[int] = []
    for index in ranked_indices:
        span = spans.get(index)
        if span is not None and any(
            temporal_hit(span, spans[other]) for other in chosen if spans.get(other) is not None
        ):
            continue
        chosen.append(index)
        if len(chosen) >= top_k:
            break
    return chosen


def _topk_with_optional_boundary_anchor(
    ranked_indices: list[int], *, top_k: int, boundary_anchor_index0: bool
) -> list[int]:
    if not boundary_anchor_index0 or top_k < 2:
        return ranked_indices[:top_k]
    result = []
    for index in [ranked_indices[0] if ranked_indices else 0, 0, *ranked_indices[1:]]:
        if index not in result:
            result.append(index)
        if len(result) >= top_k:
            break
    return result


def rank_results(
    scored_rows: list[dict[str, Any]],
    top_k: int = 2,
    *,
    boundary_anchor_index0: bool = False,
    temporal_nms: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["example_id"])].append(row)
    results = []
    for example_id, rows in sorted(grouped.items()):
        dataset = str(rows[0].get("dataset") or "unknown")
        ranked = sorted(rows, key=lambda row: (-float(row["score"]), int(row["candidate_index"])))
        ranked_indices = [int(row["candidate_index"]) for row in ranked]
        if temporal_nms:
            span_by_index = {
                int(row["candidate_index"]): (row.get("candidate_entry") or {}).get("time_span")
                for row in rows
                if isinstance((row.get("candidate_entry") or {}).get("time_span"), dict)
            }
            predicted = _topk_temporal_nms(ranked_indices, span_by_index, top_k=top_k)
        else:
            predicted = _topk_with_optional_boundary_anchor(
                ranked_indices, top_k=top_k, boundary_anchor_index0=boundary_anchor_index0
            )
        explicit_gold = {
            int(value)
            for row in rows
            for value in row.get("gold_indices") or []
        }
        gold = sorted(explicit_gold) if explicit_gold else sorted(
            int(row["candidate_index"]) for row in rows if bool(row["gold_relevant"])
        )
        retrieval = sorted(rows, key=lambda row: (int(row["retrieval_rank"]), int(row["candidate_index"])))
        retrieval_predicted = [int(row["candidate_index"]) for row in retrieval[:top_k]]
        by_index = {int(row["candidate_index"]): row for row in rows}
        selected_entries = [
            by_index[index].get("candidate_entry") or {}
            for index in predicted
            if index in by_index
        ]
        supervision = rows[0].get("process_supervision") or {}
        selected_spans = [
            entry["time_span"]
            for entry in selected_entries
            if isinstance(entry, dict) and isinstance(entry.get("time_span"), dict)
        ]
        process_metrics: dict[str, float] = {}
        if dataset == "cg_bench" and supervision.get("clue_spans"):
            clue = temporal_retrieval_metrics(selected_spans, supervision.get("clue_spans") or [])
            process_metrics = {
                "clue_recall": clue["recall"],
                "evidence_precision": clue["precision"],
                "clue_mean_best_iou": clue["mean_best_iou"],
            }
        elif dataset == "video_holmes" and supervision:
            segment = temporal_retrieval_metrics(selected_spans, supervision.get("segment_spans") or [])
            inference = temporal_retrieval_metrics(selected_spans, supervision.get("inference_spans") or [])
            process_metrics = {
                "segment_recall": segment["recall"],
                "segment_precision": segment["precision"],
                "inference_shot_recall": inference["recall"],
                "relationship_support": lexical_support(
                    selected_entries, supervision.get("relationship_texts") or []
                ),
            }
        results.append({
            "example_id": example_id,
            "dataset": dataset,
            "gold": gold,
            "predicted": predicted,
            "retrieval_predicted": retrieval_predicted,
            "metrics": retrieval_scores(predicted, gold),
            "retrieval_metrics": retrieval_scores(retrieval_predicted, gold),
            "process_metrics": process_metrics,
            "ranking": [
                {"candidate_index": int(row["candidate_index"]), "score": float(row["score"]), "retrieval_rank": int(row["retrieval_rank"])}
                for row in ranked
            ],
        })

    def aggregate(key: str) -> dict[str, Any]:
        values = [row[key] for row in results]
        return {
            "examples": len(values),
            "mean_precision": sum(float(row["precision"]) for row in values) / max(1, len(values)),
            "mean_recall": sum(float(row["recall"]) for row in values) / max(1, len(values)),
            "hit_rate": sum(bool(row["hit"]) for row in values) / max(1, len(values)),
            "exact_rate": sum(bool(row["exact"]) for row in values) / max(1, len(values)),
        }

    metric_name = f"top{top_k}"
    dataset_metrics: dict[str, Any] = {}
    for dataset in sorted({str(row.get("dataset") or "unknown") for row in results}):
        subset = [row for row in results if str(row.get("dataset") or "unknown") == dataset]
        keys = sorted({key for row in subset for key in (row.get("process_metrics") or {})})
        dataset_metrics[dataset] = {
            "examples": len(subset),
            "process_metrics": {
                key: sum(float((row.get("process_metrics") or {}).get(key, 0.0)) for row in subset)
                / max(1, len(subset))
                for key in keys
            },
        }
    return results, {
        f"pointwise_{metric_name}": aggregate("metrics"),
        f"visual_retrieval_{metric_name}": aggregate("retrieval_metrics"),
        "dataset_metrics": dataset_metrics,
    }


def _encoded_variant(tokenizer: Any, row: dict[str, Any], relevant: bool) -> tuple[list[int], int]:
    messages = list(row["messages"][:2])
    assistant = json.dumps(relevance_action(relevant), ensure_ascii=False, separators=(",", ":"))
    prompt = apply_chat_template_no_think(tokenizer, messages, add_generation_prompt=True, tokenize=False)
    full = apply_chat_template_no_think(
        tokenizer, messages + [{"role": "assistant", "content": assistant}],
        add_generation_prompt=False, tokenize=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    if common != len(prompt_ids):
        raise ValueError("Chat prompt is not a prefix of pointwise completion")
    return list(full_ids), common


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--boundary-anchor-index0", action="store_true")
    parser.add_argument(
        "--temporal-nms",
        action="store_true",
        help="Greedy top-k skipping candidates that overlap an already-chosen pick. Default off; validate on an unread split before reporting.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=("sequence_logprob", "sequence_logprob_fp32", "decision_logit"),
        default="sequence_logprob",
        help="Score estimator; the default reproduces existing frozen reports.",
    )
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.torchao as peft_torchao
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    peft_import_utils.is_torchao_available = lambda: False
    peft_torchao.is_torchao_available = lambda: False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to("cuda")
    model.eval()
    causal_lm = model.get_base_model()

    rows = read_jsonl(args.dev_jsonl)
    input_provenance = evaluation_input_provenance(rows, args.dev_jsonl)
    variants: list[tuple[int, bool, list[int], int]] = []
    for row_number, row in enumerate(rows):
        for relevant in (True, False):
            ids, prompt_length = _encoded_variant(tokenizer, row, relevant)
            variants.append((row_number, relevant, ids, prompt_length))
    log_likelihoods: dict[tuple[int, bool], float] = {}
    scores: dict[int, float] = {}
    if args.scoring_mode == "decision_logit":
        # One forward per row over the shared prefix; the log-odds is read off the
        # single divergent token in fp32, so nothing is differenced or accumulated.
        decisions: list[tuple[int, list[int], int, int]] = []
        by_row: dict[int, dict[bool, list[int]]] = defaultdict(dict)
        for row_number, relevant, ids, _ in variants:
            by_row[row_number][relevant] = ids
        for row_number in sorted(by_row):
            prefix_length, true_token, false_token = _decision_position(
                by_row[row_number][True], by_row[row_number][False]
            )
            decisions.append(
                (row_number, by_row[row_number][True][:prefix_length], true_token, false_token)
            )
        with torch.no_grad():
            for start in range(0, len(decisions), args.batch_size):
                batch = decisions[start : start + args.batch_size]
                width = max(len(item[1]) for item in batch)
                input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
                attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
                for position, (_, ids, _, _) in enumerate(batch):
                    input_ids[position, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
                    attention_mask[position, : len(ids)] = 1
                hidden = causal_lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
                for position, (row_number, ids, true_token, false_token) in enumerate(batch):
                    # Only two rows of the unembedding are needed, so the fp32
                    # projection costs nothing next to a full-vocabulary logit.
                    pair = causal_lm.lm_head.weight[[true_token, false_token]].float()
                    state = hidden[position, len(ids) - 1].float()
                    logits = pair @ state
                    # log_softmax shares one logsumexp across the vocabulary, so it
                    # cancels: the raw logit gap is exactly the log-odds.
                    scores[row_number] = float((logits[0] - logits[1]).cpu())
                print(f"[{min(start + len(batch), len(decisions))}/{len(decisions)}] decisions", flush=True)
    else:
        head_weight = causal_lm.lm_head.weight
        if args.scoring_mode == "sequence_logprob_fp32":
            head_weight = head_weight.float()
        with torch.no_grad():
            for start in range(0, len(variants), args.batch_size):
                batch = variants[start : start + args.batch_size]
                width = max(len(item[2]) for item in batch)
                input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
                attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
                for position, (_, _, ids, _) in enumerate(batch):
                    input_ids[position, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
                    attention_mask[position, : len(ids)] = 1
                hidden = causal_lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
                for position, (row_number, relevant, ids, prompt_length) in enumerate(batch):
                    labels = input_ids[position, prompt_length : len(ids)]
                    token_hidden = hidden[position, prompt_length - 1 : len(ids) - 1]
                    if args.scoring_mode == "sequence_logprob_fp32":
                        token_hidden = token_hidden.float()
                    score = 0.0
                    for offset in range(0, labels.numel(), 16):
                        logits = F.linear(token_hidden[offset : offset + 16], head_weight)
                        score += float(F.log_softmax(logits.float(), dim=-1).gather(1, labels[offset : offset + 16, None]).sum().cpu())
                    log_likelihoods[(row_number, relevant)] = score
                print(f"[{min(start + len(batch), len(variants))}/{len(variants)}] variants", flush=True)
        for row_number in range(len(rows)):
            scores[row_number] = log_likelihoods[(row_number, True)] - log_likelihoods[(row_number, False)]

    scored_rows = []
    for row_number, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        user = json.loads(row["messages"][1]["content"])
        candidate = user["state_t"]["candidate_coarse_summary"]
        scored_rows.append({
            "dataset": str(metadata.get("dataset") or "unknown"),
            "example_id": str(metadata["source_example_id"]),
            "candidate_index": int(metadata["candidate_index"]),
            "retrieval_rank": int(candidate["retrieval_rank"]),
            "gold_relevant": bool(metadata["candidate_relevant"]),
            "gold_indices": [int(value) for value in metadata.get("gold_indices") or []],
            "candidate_entry": metadata.get("candidate_entry") or candidate,
            "process_supervision": metadata.get("process_supervision") or {},
            "score": scores[row_number],
            "true_log_likelihood": log_likelihoods.get((row_number, True)),
            "false_log_likelihood": log_likelihoods.get((row_number, False)),
        })
    results, metrics = rank_results(
        scored_rows,
        top_k=max(1, int(args.top_k)),
        boundary_anchor_index0=bool(args.boundary_anchor_index0),
        temporal_nms=bool(args.temporal_nms),
    )
    report = {
        "schema_version": "video-skills/l2-pointwise-eval-v0.1",
        "adapter": str(args.adapter),
        "adapter_weight_sha256": adapter_weight_sha256(args.adapter),
        "boundary_anchor_index0": bool(args.boundary_anchor_index0),
        "scoring_mode": str(args.scoring_mode),
        "temporal_nms": bool(args.temporal_nms),
        "top_k": max(1, int(args.top_k)),
        "candidate_rows": len(scored_rows),
        **input_provenance,
        "metrics": metrics,
        "results": results,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

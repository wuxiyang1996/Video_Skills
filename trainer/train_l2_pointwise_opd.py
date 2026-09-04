#!/usr/bin/env python3
"""Train L2 pointwise LoRA with train-only on-policy teacher distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.training.sft_common import (
    apply_chat_template_no_think,
    decision_position,
    write_json,
)
from trainer.artifact_hash import adapter_weight_sha256
from .opd_action_distill_adapter import load_opd_rows


def _encode_action(tokenizer: Any, messages: list[dict[str, Any]], action: dict[str, Any]) -> tuple[list[int], int]:
    assistant = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    prompt = apply_chat_template_no_think(tokenizer, messages, add_generation_prompt=True, tokenize=False)
    full = apply_chat_template_no_think(
        tokenizer, messages + [{"role": "assistant", "content": assistant}],
        add_generation_prompt=False, tokenize=False,
    )
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("OPD prompt is not a prefix of completion")
    return full_ids, len(prompt_ids)


def _sequence_score(causal_lm: Any, hidden: Any, labels: Any, *, chunk_size: int = 16) -> Any:
    import torch
    import torch.nn.functional as F

    score = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, labels.numel(), chunk_size):
        logits = F.linear(hidden[start : start + chunk_size], causal_lm.lm_head.weight)
        score = score + F.log_softmax(logits.float(), dim=-1).gather(
            1, labels[start : start + chunk_size, None]
        ).sum()
    return score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--distill", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-mode",
        choices=("sequence_logprob", "decision_logit"),
        default="sequence_logprob",
        help=(
            "How the true/false student logits are formed.  'sequence_logprob' "
            "(default) reproduces existing runs.  'decision_logit' reads both "
            "logits off the single divergent token in fp32, matching what the "
            "evaluator scores and removing bf16 cancellation from the gradient."
        ),
    )
    parser.add_argument(
        "--dataset-balanced-loss",
        action="store_true",
        help="Normalize total OPD sample weight independently per dataset.",
    )
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.torchao as peft_torchao
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = load_opd_rows(args.distill)
    if not rows:
        raise ValueError("No OPD rows")
    distill_sha256 = hashlib.sha256(args.distill.read_bytes()).hexdigest()
    build_report_path = args.distill.parent / "build_report.json"
    build_report = (
        json.loads(build_report_path.read_text(encoding="utf-8"))
        if build_report_path.is_file() else None
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    peft_import_utils.is_torchao_available = lambda: False
    peft_torchao.is_torchao_available = lambda: False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    causal_lm = model.get_base_model()
    if hasattr(causal_lm.lm_head, "lora_A"):
        raise RuntimeError("lm_head unexpectedly received LoRA")

    encoded = []
    raw_weights = [float((row.get("state") or {}).get("sample_weight", 1.0)) for row in rows]
    row_datasets = [str((row.get("state") or {}).get("dataset") or "unknown") for row in rows]
    dataset_raw_weight: dict[str, float] = defaultdict(float)
    for dataset, weight in zip(row_datasets, raw_weights, strict=True):
        dataset_raw_weight[dataset] += weight
    if args.dataset_balanced_loss and len(dataset_raw_weight) > 1:
        raw_weights = [
            weight / max(1e-12, dataset_raw_weight[dataset])
            for dataset, weight in zip(row_datasets, raw_weights, strict=True)
        ]
    mean_weight = sum(raw_weights) / len(raw_weights)
    for row, raw_weight in zip(rows, raw_weights, strict=True):
        messages = list((row.get("state") or {}).get("messages") or [])
        candidates = list((row.get("candidates") or {}).get("candidates") or [])
        by_id = {str(candidate["action_id"]): candidate["action"] for candidate in candidates}
        if set(by_id) != {"relevant_true", "relevant_false"}:
            raise ValueError("Each OPD row must contain true and false complete actions")
        teacher_map = (row.get("teacher") or {}).get("action_probs") or {}
        variants = [_encode_action(tokenizer, messages, by_id[action_id]) for action_id in ("relevant_true", "relevant_false")]
        decision = decision_position(variants[0][0], variants[1][0])
        encoded.append({
            "state_id": (row.get("state") or {}).get("state_id"),
            "variants": variants,
            "decision": decision,
            "teacher": torch.tensor([float(teacher_map.get("relevant_true", 0.0)), float(teacher_map.get("relevant_false", 0.0))], dtype=torch.float32),
            "weight": raw_weight / mean_weight,
        })

    total_steps = max(1, math.ceil(len(encoded) * args.epochs / args.gradient_accumulation_steps))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    losses: list[float] = []
    correct = 0
    seen = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(encoded)))
        random.Random(args.seed + epoch).shuffle(order)
        for row_index in order:
            row = encoded[row_index]
            variants = row["variants"]
            if args.score_mode == "decision_logit":
                # The two completions share a prefix, so one forward serves both;
                # the 2-way logits are read straight off the divergent token.
                prefix_length, true_token, false_token = row["decision"]
                prefix = variants[0][0][:prefix_length]
                input_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
                attention_mask = torch.ones_like(input_ids)
                hidden = causal_lm.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                ).last_hidden_state
                pair = causal_lm.lm_head.weight[[true_token, false_token]].float()
                student_logits = pair @ hidden[0, -1].float()
            else:
                width = max(len(ids) for ids, _ in variants)
                input_ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
                attention_mask = torch.zeros((2, width), dtype=torch.long, device="cuda")
                for position, (ids, _) in enumerate(variants):
                    input_ids[position, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
                    attention_mask[position, : len(ids)] = 1
                hidden = causal_lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
                scores = []
                for position, (ids, prompt_length) in enumerate(variants):
                    labels = input_ids[position, prompt_length : len(ids)]
                    token_hidden = hidden[position, prompt_length - 1 : len(ids) - 1]
                    scores.append(_sequence_score(causal_lm, token_hidden, labels))
                student_logits = torch.stack(scores)
            teacher = row["teacher"].to("cuda")
            teacher = teacher / teacher.sum().clamp_min(1e-12)
            loss = -(teacher * F.log_softmax(student_logits, dim=0)).sum() * float(row["weight"])
            (loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            correct += int(student_logits.argmax().item() == teacher.argmax().item())
            seen += 1
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps and micro_step < len(encoded) * args.epochs:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            update_step += 1
            metric = {
                "step": update_step, "epoch": epoch,
                "loss": sum(losses[-args.gradient_accumulation_steps:]) / min(args.gradient_accumulation_steps, len(losses)),
                "teacher_argmax_accuracy": correct / seen,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - started,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")
            print(json.dumps({"event": "opd_step", **metric}), flush=True)

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    report = {
        "schema_version": "video-skills/l2-pointwise-opd-train-v0.1",
        "model": args.model,
        "init_adapter": str(args.adapter),
        "init_adapter_weight_sha256": adapter_weight_sha256(args.adapter),
        "adapter_dir": str(adapter_dir),
        "adapter_weight_sha256": adapter_weight_sha256(adapter_dir),
        "distill": str(args.distill),
        "distill_sha256": distill_sha256,
        "distill_build_report": str(build_report_path) if build_report is not None else None,
        "distill_contracts": (
            {
                "split_role": build_report.get("split_role"),
                "split_manifest": build_report.get("split_manifest"),
                "split_manifest_sha256": build_report.get("split_manifest_sha256"),
                "video_holmes_teacher_contract": build_report.get("video_holmes_teacher_contract"),
                "video_holmes_supervision_contract": build_report.get("video_holmes_supervision_contract"),
                "relationship_support_contract": build_report.get("relationship_support_contract"),
                "hidden_supervision_visible_to_policy": build_report.get(
                    "hidden_supervision_visible_to_policy"
                ),
            }
            if build_report is not None else None
        ),
        "teacher_providers": dict(
            Counter(str((row.get("teacher") or {}).get("provider") or "unknown") for row in rows)
        ),
        "rows": len(encoded), "epochs": args.epochs, "total_steps": update_step,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "first_loss": losses[0], "last_loss": losses[-1],
        "teacher_argmax_accuracy": correct / seen,
        "score_mode": str(args.score_mode),
        "dataset_balanced_loss": bool(args.dataset_balanced_loss),
        "dataset_rows": dict(Counter(row_datasets)),
        "dataset_raw_weight": dict(dataset_raw_weight),
        "dataset_effective_weight": {
            dataset: sum(
                weight / mean_weight
                for row_dataset, weight in zip(row_datasets, raw_weights, strict=True)
                if row_dataset == dataset
            )
            for dataset in sorted(set(row_datasets))
        },
        "elapsed_s": time.time() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }
    write_json(args.output_dir / "training_report.json", report)
    print(json.dumps({"event": "opd_complete", **report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

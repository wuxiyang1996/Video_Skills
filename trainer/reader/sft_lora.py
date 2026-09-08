"""LoRA SFT for the reader: (system, user) → JSON completion, loss on the completion only.

Data: JSONL rows {"messages": [system, user], "completion": "<json>"} from
trainer/reader/build_sft_data.py.  Model: an instruct causal LM (default the
local Qwen3.5-9B), bf16, FlashAttention-2 when available, gradient
checkpointing, LoRA on all projection matrices.  `--dry-run` only tokenises
and reports sequence-length statistics (CPU, no weights).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def encode_example(tokenizer: Any, row: dict[str, Any], max_len: int) -> dict[str, Any] | None:
    """Prompt tokens masked with -100; completion (+eos) supervised. Drops rows whose prompt alone exceeds max_len."""
    # Render exactly what the evaluator's server renders: thinking disabled, so the assistant prefix is
    # "<think>\n\n</think>\n\n" rather than an open "<think>\n".  Training on the default (thinking-open)
    # prefix produced an adapter that collapsed to option A at evaluation (fresh 300: 27.3 vs base 38.0).
    prompt_text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) >= max_len - 8:
        return None
    ids = (prompt_ids + completion_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def completion_only_loss(model: Any, batch: dict[str, Any]) -> Any:
    """Cross-entropy on the supervised tail only, materialising logits for just those positions.

    Prompts are ~13k tokens and the vocabulary ~250k, so full-sequence logits alone are ~13 GB
    in fp32 (the OOM of the first SFT job).  Since the completion is the tail of every
    sequence, `logits_to_keep` restricts the LM head to the last k positions (k = supervised
    tokens + 1); the shift-by-one then lines those logits up with the completion labels.
    """
    import torch

    labels = batch["labels"]
    supervised = (labels != -100).sum(dim=1)
    k = int(supervised.max().item()) + 1
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=k)
    logits = out.logits[:, :-1, :].float()               # positions T-k .. T-2 predict tokens T-k+1 .. T-1
    target = labels[:, -k + 1:]                           # the last k-1 labels
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--eval-data", type=Path, default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--save-steps", type=int, default=0, help="Checkpoint the adapter every N optimizer steps (0 = only at epoch end).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    rows = load_rows(args.data, args.limit)
    random.Random(args.seed).shuffle(rows)
    encoded = [e for e in (encode_example(tokenizer, r, args.max_len) for r in rows) if e]
    lens = sorted(len(e["input_ids"]) for e in encoded)
    sup = [sum(1 for t in e["labels"] if t != -100) for e in encoded]
    print(json.dumps({"rows": len(rows), "kept": len(encoded), "len_p50": lens[len(lens) // 2] if lens else 0,
                      "len_p90": lens[int(0.9 * len(lens))] if lens else 0, "len_max": lens[-1] if lens else 0,
                      "supervised_tokens_mean": round(sum(sup) / max(len(sup), 1), 1)}))
    if args.dry_run:
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    class _DS(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    eval_ds = None
    if args.eval_data:
        eval_ds = _DS([e for e in (encode_example(tokenizer, r, args.max_len) for r in load_rows(args.eval_data)) if e])
    targs = TrainingArguments(output_dir=str(args.output_dir), per_device_train_batch_size=1, per_device_eval_batch_size=1,
                              gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, learning_rate=args.lr,
                              lr_scheduler_type="cosine", warmup_ratio=0.05, bf16=True, logging_steps=10,
                              save_strategy=("steps" if args.save_steps else "epoch"), save_steps=(args.save_steps or 500), save_total_limit=2,
                              eval_strategy="epoch" if eval_ds else "no", report_to=[], seed=args.seed,
                              gradient_checkpointing=True, remove_unused_columns=False, dataloader_num_workers=2)
    class CompletionOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = completion_only_loss(model, inputs)
            return (loss, None) if return_outputs else loss

    trainer = CompletionOnlyTrainer(model=model, args=targs, train_dataset=_DS(encoded), eval_dataset=eval_ds,
                                    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    print(json.dumps({"saved": str(args.output_dir / "adapter"), "attn": attn}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

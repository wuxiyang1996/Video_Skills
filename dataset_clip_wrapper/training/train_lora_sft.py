#!/usr/bin/env python3
"""Single-GPU, assistant-only LoRA SFT for the Video_Skills controller chats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .sft_common import apply_chat_template_no_think, strip_think_tags


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row_task(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("task") or metadata.get("controller") or row.get("controller") or "unknown")


def _row_weight(row: dict[str, Any]) -> float:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("source_family_weight", 1.0)
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return weight if math.isfinite(weight) and weight > 0.0 else 1.0


def _representative_subset(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    by_controller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_controller[_row_task(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for controller in sorted(by_controller):
        longest = max(
            by_controller[controller],
            key=lambda row: sum(len(str(message.get("content") or "")) for message in row.get("messages") or []),
        )
        selected.append(longest)
        selected_ids.add(id(longest))
    remaining = [row for row in rows if id(row) not in selected_ids]
    remaining.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('transition_id') or row.get('demo_id')}".encode("utf-8")
        ).hexdigest()
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected[:limit]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = strip_think_tags(text or "")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _encode_chat(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    *,
    weight_scale: float = 1.0,
) -> dict[str, Any]:
    messages = row.get("messages") or []
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Unexpected chat roles for {row.get('transition_id') or row.get('demo_id')}")
    full_text = apply_chat_template_no_think(
        tokenizer, messages, add_generation_prompt=False, tokenize=False
    )
    prompt_text = apply_chat_template_no_think(
        tokenizer, messages[:2], add_generation_prompt=True, tokenize=False
    )
    input_ids = _token_ids(tokenizer, full_text)
    prompt_ids = _token_ids(tokenizer, prompt_text)
    common_prefix = 0
    for left, right in zip(input_ids, prompt_ids):
        if left != right:
            break
        common_prefix += 1
    if common_prefix != len(prompt_ids):
        raise ValueError(
            f"Chat template prompt is not a prefix of the supervised conversation: "
            f"id={row.get('transition_id') or row.get('demo_id')} "
            f"matched={common_prefix} prompt_tokens={len(prompt_ids)}"
        )
    if len(input_ids) > max_length:
        raise ValueError(
            f"Encoded row exceeds max_length: id={row.get('transition_id') or row.get('demo_id')} "
            f"tokens={len(input_ids)} max_length={max_length}"
        )
    labels = [-100] * common_prefix + input_ids[common_prefix:]
    if not any(value != -100 for value in labels):
        raise ValueError(f"No supervised assistant tokens for {row.get('transition_id') or row.get('demo_id')}")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "record_id": str(row.get("transition_id") or row.get("demo_id")),
        "controller": _row_task(row),
        "sample_weight": _row_weight(row) * weight_scale,
    }


class _ChatDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class _Collator:
    def __init__(self, pad_token_id: int, multiple: int = 8):
        self.pad_token_id = pad_token_id
        self.multiple = multiple

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        width = int(math.ceil(width / self.multiple) * self.multiple)
        input_ids, attention_mask, labels, sample_weights = [], [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
            sample_weights.append(float(feature.get("sample_weight", 1.0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        }


def _fused_causal_lm_loss(model: Any, batch: dict[str, Any], loss_module: Any) -> Any:
    """Compute assistant-masked loss without materializing full-vocabulary logits."""
    import torch.nn.functional as torch_functional

    causal_lm = model.get_base_model()
    outputs = causal_lm.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    hidden_states = outputs.last_hidden_state
    labels = torch_functional.pad(batch["labels"], (0, 1), value=-100)[..., 1:].contiguous()
    hidden_size = hidden_states.shape[-1]
    loss = loss_module(
        causal_lm.lm_head.weight,
        hidden_states.reshape(-1, hidden_size),
        labels.reshape(-1),
    )
    return loss * batch["sample_weight"].mean()


def _evaluate(model: Any, loader: Any, device: Any, loss_module: Any) -> float:
    import torch
    import torch.nn.functional as torch_functional

    model.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    causal_lm = model.get_base_model()
    # Liger's fused loss requires autograd even for its forward call. Evaluation
    # instead keeps the backbone under no_grad and projects only supervised
    # assistant tokens, in small chunks, so neither layer activations nor a full
    # sequence-by-vocabulary logits tensor can exhaust GPU memory.
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = causal_lm.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            labels = torch_functional.pad(batch["labels"], (0, 1), value=-100)[..., 1:].reshape(-1)
            hidden = outputs.last_hidden_state.reshape(-1, outputs.last_hidden_state.shape[-1])
            supervised = labels.ne(-100)
            hidden = hidden[supervised]
            labels = labels[supervised]
            loss_sum = 0.0
            for start in range(0, labels.numel(), 64):
                stop = start + 64
                logits = torch_functional.linear(hidden[start:stop], causal_lm.lm_head.weight)
                chunk_loss = torch_functional.cross_entropy(logits.float(), labels[start:stop], reduction="sum")
                loss_sum += float(chunk_loss.cpu())
            weight = float(batch["sample_weight"].mean().cpu())
            weighted_loss_sum += (loss_sum / max(1, labels.numel())) * weight
            weight_sum += weight
            del outputs, hidden, labels, batch
    model.train()
    return weighted_loss_sum / max(1e-12, weight_sum)


def _action_signature(payload: dict[str, Any] | None) -> tuple[str, str]:
    if not payload:
        return "", ""
    nested = payload.get("action") if isinstance(payload.get("action"), dict) else {}
    return str(payload.get("tool_name") or nested.get("action_type") or ""), str(payload.get("round_type") or "")


def _assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _json_complete_stopping_criteria(tokenizer: Any, prompt_len: int):
    """Stop once the *generated suffix* (not the prompt) is a parseable JSON object."""
    from transformers import StoppingCriteria

    class _JsonComplete(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
            gen_ids = input_ids[0, prompt_len:]
            if gen_ids.numel() < 8:
                return False
            text = strip_think_tags(tokenizer.decode(gen_ids, skip_special_tokens=True))
            return _extract_json_object(text) is not None

    return _JsonComplete()


def _generation_check(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    *,
    max_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    # Prefer shortest *assistant* gold JSON per controller (not shortest prompt).
    shortest_by_controller: dict[str, dict[str, Any]] = {}
    for row in rows:
        controller = _row_task(row)
        asst_len = len(_assistant_text(row))
        prev = shortest_by_controller.get(controller)
        if prev is None or asst_len < len(_assistant_text(prev)):
            shortest_by_controller[controller] = row
    selected = sorted(shortest_by_controller.values(), key=lambda row: len(_assistant_text(row)))[:max_examples]
    results = []
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        for row in selected:
            gold_text = _assistant_text(row)
            gold_token_len = max(1, len(_token_ids(tokenizer, gold_text)))
            # Cap per-example budget near gold size (with headroom); still below global max
            # so runaway loops cannot burn thousands of tokens.
            example_max_new = max(256, min(max_new_tokens, int(gold_token_len * 3.0) + 128))
            prompt = apply_chat_template_no_think(
                tokenizer,
                row["messages"][:2],
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_len = int(encoded["input_ids"].shape[1])
            stop = _json_complete_stopping_criteria(tokenizer, prompt_len)
            generated = model.generate(
                **encoded,
                max_new_tokens=example_max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[stop],
            )
            completion = strip_think_tags(
                tokenizer.decode(
                    generated[0, encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
            )
            payload = _extract_json_object(completion)
            gold = _extract_json_object(gold_text)
            action_match = bool(payload is not None and _action_signature(payload) == _action_signature(gold))
            results.append({
                "record_id": row.get("transition_id") or row.get("demo_id"),
                "controller": _row_task(row),
                "json_valid": payload is not None,
                "action_match": action_match,
                "exact_match": payload == gold,
                "max_new_tokens": example_max_new,
                "completion": completion,
            })
    valid = sum(result["json_valid"] for result in results)
    action_valid = sum(result["action_match"] for result in results)
    return {
        "examples": len(results),
        "json_valid": valid,
        "json_valid_rate": valid / max(1, len(results)),
        "action_match": action_valid,
        "action_match_rate": action_valid / max(1, len(results)),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "pilot", "base_baseline"], required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="If >0, evaluate loss on a representative subset of dev (smoke-friendly).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-examples", type=int, default=16)
    parser.add_argument("--generation-max-new-tokens", type=int, default=384)
    parser.add_argument("--min-json-rate", type=float, default=0.5)
    parser.add_argument("--min-action-rate", type=float, default=0.5)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Prefer flash_attention_2; falls back to sdpa only if FA2 unavailable",
    )
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="Fail if flash_attn cannot be imported (recommended on A6000)",
    )
    args = parser.parse_args(argv)

    import torch
    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.torchao as peft_torchao
    from peft import LoraConfig, TaskType, get_peft_model
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    from trainer.grpo.attn_utils import resolve_attn_implementation

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    attn_implementation = resolve_attn_implementation(
        args.attn_implementation,
        allow_sdpa_fallback=not bool(args.require_flash_attn),
    )
    print(json.dumps({"event": "attn_backend", "attn_implementation": attn_implementation}), flush=True)
    # This environment inherits torchao 0.9 from the shared Swift conda env.
    # PEFT 0.19 probes every optional LoRA dispatcher and raises on that old
    # torchao even for ordinary bf16 Linear weights. TorchAO is not used here.
    peft_import_utils.is_torchao_available = lambda: False
    peft_torchao.is_torchao_available = lambda: False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    source_train = _read_jsonl(args.train_jsonl)
    source_dev = _read_jsonl(args.dev_jsonl)
    # Generation always uses the full (or selected) chat rows; loss eval can be capped.
    dev_rows = source_dev

    # Base baseline only needs a few generation prompts — skip full train encode.
    if args.stage == "base_baseline":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for base baseline generation")
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).to(device)
        model.eval()
        data_report = {
            "stage": args.stage,
            "source_train_rows": len(source_train),
            "source_dev_rows": len(source_dev),
            "enable_thinking": False,
            "attn_implementation": attn_implementation,
        }
        (args.output_dir / "data_report.json").write_text(
            json.dumps(data_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "data_ready", **data_report}, ensure_ascii=False), flush=True)
        generation = _generation_check(
            model,
            tokenizer,
            dev_rows,
            device,
            max_examples=args.generation_examples,
            max_new_tokens=args.generation_max_new_tokens,
        )
        generation["model"] = args.model
        generation["stage"] = "base_baseline"
        out_path = args.output_dir / "base_generation_report.json"
        out_path.write_text(json.dumps(generation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "base_baseline_complete", "output": str(out_path), **generation}, ensure_ascii=False), flush=True)
        return 0

    train_rows = _representative_subset(source_train, args.max_train_samples, args.seed)
    eval_rows = _representative_subset(source_dev, args.max_eval_samples, args.seed + 1)
    mean_train_weight = sum(_row_weight(row) for row in train_rows) / max(1, len(train_rows))
    encoded_train = [
        _encode_chat(tokenizer, row, args.max_length, weight_scale=1.0 / mean_train_weight)
        for row in train_rows
    ]
    encoded_dev = [_encode_chat(tokenizer, row, args.max_length) for row in eval_rows]
    data_report = {
        "stage": args.stage,
        "source_train_rows": len(source_train),
        "train_rows": len(encoded_train),
        "source_dev_rows": len(source_dev),
        "dev_rows": len(encoded_dev),
        "max_eval_samples": args.max_eval_samples,
        "train_token_min": min(len(row["input_ids"]) for row in encoded_train),
        "train_token_max": max(len(row["input_ids"]) for row in encoded_train),
        "dev_token_max": max((len(row["input_ids"]) for row in encoded_dev), default=0),
        "assistant_only_loss": True,
        "sample_weighted_loss": True,
        "raw_train_weight_sum": sum(_row_weight(row) for row in train_rows),
        "train_controllers": dict(__import__("collections").Counter(row["controller"] for row in encoded_train)),
        "enable_thinking": False,
    }
    (args.output_dir / "data_report.json").write_text(
        json.dumps(data_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "data_ready", **data_report}, ensure_ascii=False), flush=True)
    if args.data_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LoRA SFT")

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora).to(device)
    causal_lm = model.get_base_model()
    if hasattr(causal_lm.lm_head, "lora_A"):
        raise RuntimeError("lm_head unexpectedly received LoRA; fused loss requires the frozen output projection")
    loss_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
    trainable, total = model.get_nb_trainable_parameters()
    print(json.dumps({"event": "model_ready", "trainable_parameters": trainable, "total_parameters": total}), flush=True)

    collator = _Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        _ChatDataset(encoded_train), batch_size=1, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(_ChatDataset(encoded_dev), batch_size=1, shuffle=False, collate_fn=collator)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, updates_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    epoch = 0
    losses: list[float] = []
    started = time.time()
    model.train()
    while update_step < total_steps:
        epoch += 1
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            raw_loss = _fused_causal_lm_loss(model, batch, loss_module)
            (raw_loss / args.gradient_accumulation_steps).backward()
            losses.append(float(raw_loss.detach().cpu()))
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            metric = {
                "step": update_step,
                "epoch": epoch,
                "loss": sum(losses[-args.gradient_accumulation_steps :]) / args.gradient_accumulation_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - started,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")
            print(json.dumps({"event": "train_step", **metric}), flush=True)
            if args.save_steps > 0 and update_step % args.save_steps == 0:
                checkpoint = args.output_dir / f"checkpoint-{update_step}"
                model.save_pretrained(checkpoint)
            if update_step >= total_steps:
                break

    eval_loss = _evaluate(model, dev_loader, device, loss_module)
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    # Smoke: generate on the overfit train subset (wiring check). Pilot/base: use dev.
    generation_rows = train_rows if args.stage == "smoke" else dev_rows
    generation = _generation_check(
        model,
        tokenizer,
        generation_rows,
        device,
        max_examples=args.generation_examples,
        max_new_tokens=args.generation_max_new_tokens,
    )
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(generation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        **data_report,
        "model": args.model,
        "lora_rank": args.lora_rank,
        "total_steps": total_steps,
        "epochs_completed": epoch,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "eval_loss": eval_loss,
        "json_valid_rate": generation["json_valid_rate"],
        "action_match_rate": generation["action_match_rate"],
        "elapsed_s": time.time() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "adapter_dir": str(adapter_dir),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "training_complete", **report}), flush=True)
    if generation["json_valid_rate"] < args.min_json_rate:
        raise RuntimeError(
            f"JSON generation gate failed: {generation['json_valid_rate']:.3f} < {args.min_json_rate:.3f}"
        )
    if generation["action_match_rate"] < args.min_action_rate:
        raise RuntimeError(
            f"Action generation gate failed: {generation['action_match_rate']:.3f} < {args.min_action_rate:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FSDP-parallel GRPO trainer for multi-GPU LoRA training.

Spawns one process per GPU using ``torch.multiprocessing.spawn``.
Each process loads the base model, applies the target LoRA adapter,
wraps with FSDP (sharding frozen base weights across ranks), and
trains on its slice of the data.  FSDP handles gradient
synchronisation automatically via AllReduce during ``.backward()``.

Typical 8-GPU FSDP memory per rank (14B bf16):
  - Sharded frozen params:  ~3.5 GB  (28 GB / 8)
  - Full LoRA params:       ~0.25 GB
  - LoRA optimizer states:  ~0.5 GB
  - Activations (grad ckpt): ~10-20 GB
  Total: ~14-24 GB of 80 GB  →  plenty of room

Usage::

    from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo

    stats = run_fsdp_grpo(
        gpu_ids=[4, 5, 6, 7],
        model_name="Qwen/Qwen3-14B",
        adapter_dir="runs/lora_adapters/decision/skill_selection",
        adapter_name="skill_selection",
        prompts=prompts,
        completions=completions,
        advantages=advantages,
    )
"""

from __future__ import annotations

import gc
import json
import logging
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _detect_wrap_cls(model):
    """Return the transformer decoder-layer class for FSDP auto-wrap."""
    for module in model.modules():
        cls_name = type(module).__name__
        if "DecoderLayer" in cls_name:
            return {type(module)}
    return set()


def _fsdp_train_worker(rank: int, args: Dict[str, Any]) -> None:
    """Per-GPU FSDP training worker (spawned by :func:`run_fsdp_grpo`).

    Each rank:
    1. Loads the full base model to its GPU
    2. Applies/creates the LoRA adapter
    3. Wraps with FSDP (shards frozen weights across all ranks)
    4. Trains on its slice of the data
    5. Rank 0 gathers and saves the adapter
    """
    import functools

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    world_size = args["world_size"]
    master_port = args["master_port"]
    model_name = args["model_name"]
    adapter_dir = args["adapter_dir"]
    adapter_name = args["adapter_name"]
    prompts = args["prompts"]
    completions = args["completions"]
    advantages = args["advantages"]
    lr = args["lr"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    clip_ratio = args["clip_ratio"]
    kl_coeff = args["kl_coeff"]
    save_dir = args["save_dir"]
    result_file = args["result_file"]
    io_log_dir = args.get("io_log_dir")

    # CUDA_VISIBLE_DEVICES is set by the launcher so rank maps cleanly
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    is_main = rank == 0

    try:
        t0 = time.time()

        # ── 1. Load base model ──────────────────────────────────────
        if is_main:
            logger.info(
                "FSDP rank 0: loading %s (bf16) onto %d GPUs",
                model_name, world_size,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": rank},
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── 2. Apply LoRA adapter ───────────────────────────────────
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        adapter_path = Path(adapter_dir)
        adapter_config_file = adapter_path / "adapter_config.json"

        if adapter_config_file.exists():
            model = PeftModel.from_pretrained(
                model, str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=True,
            )
            if is_main:
                logger.info("Loaded adapter '%s' from %s", adapter_name, adapter_path)
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                inference_mode=False,
            )
            model = get_peft_model(model, lora_cfg, adapter_name=adapter_name)
            if is_main:
                logger.info("Created fresh adapter '%s' (r=16, alpha=32)", adapter_name)

        for n, p in model.named_parameters():
            is_lora = "lora" in n.lower()
            p.requires_grad = is_lora
            if is_lora and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        model.train()

        # ── 3. Wrap with FSDP ──────────────────────────────────────
        wrap_cls = _detect_wrap_cls(model)
        if wrap_cls:
            auto_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=wrap_cls,
            )
        else:
            auto_policy = None

        bf16_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=bf16_policy,
            device_id=rank,
            use_orig_params=True,
        )

        load_time = time.time() - t0
        if is_main:
            n_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(
                "FSDP model ready (%d GPUs, %s trainable params, %.1fs load)",
                world_size, f"{n_trainable:,}", load_time,
            )

        # ── 4. Shard data ──────────────────────────────────────────
        n_total = len(prompts)
        per_rank = n_total // world_size
        start_idx = rank * per_rank
        end_idx = start_idx + per_rank if rank < world_size - 1 else n_total
        my_prompts = prompts[start_idx:end_idx]
        my_completions = completions[start_idx:end_idx]
        my_advantages = advantages[start_idx:end_idx]
        n_my = len(my_prompts)

        if is_main:
            logger.info(
                "FSDP GRPO [%s]: %d samples → %d/rank, %d epochs, bs=%d",
                adapter_name, n_total, n_my, epochs, batch_size,
            )

        # ── 5. Optimizer ───────────────────────────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # ── 5.5. Pre-tokenize & compute reference log-probs ──────
        # Compute log-probs under the *current* policy (before any
        # gradient steps) to serve as the PPO/GRPO reference.
        # Without this, ratio = exp(new - old) is always 1.0 and the
        # clipping / KL penalty has no effect.
        #
        # Pre-tokenize everything first (CPU-only), then run a single
        # batched no-grad forward pass.  This is ~5-10x faster than
        # processing samples one-by-one.
        t_ref = time.time()
        ref_data: list = [None] * n_my
        tokenized: list = [None] * n_my
        for i in range(n_my):
            full_text = my_prompts[i] + my_completions[i]
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True,
            )
            penc = tokenizer(my_prompts[i], return_tensors="pt")
            plen = penc["input_ids"].shape[1]
            if plen >= enc["input_ids"].shape[1]:
                continue
            tokenized[i] = {
                "input_ids": enc["input_ids"],
                "attn_mask": enc["attention_mask"],
                "plen": plen,
            }

        ref_batch_size = batch_size * 2
        model.eval()
        with torch.no_grad():
            for mb_s in range(0, n_my, ref_batch_size):
                mb_e = min(mb_s + ref_batch_size, n_my)
                for i in range(mb_s, mb_e):
                    tk = tokenized[i]
                    if tk is None:
                        continue
                    input_ids = tk["input_ids"].to(device)
                    attn_mask = tk["attn_mask"].to(device)
                    plen = tk["plen"]

                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    logits = out.logits[:, plen - 1:-1, :]
                    target = input_ids[:, plen:]
                    lp = torch.log_softmax(logits, dim=-1)
                    per_tok = lp.gather(
                        -1, target.unsqueeze(-1),
                    ).squeeze(-1).squeeze(0)

                    if per_tok.numel() == 0:
                        continue
                    ref_data[i] = {
                        "input_ids": input_ids.cpu(),
                        "attn_mask": attn_mask.cpu(),
                        "plen": plen,
                        "ref_lp": per_tok.cpu(),
                    }
        model.train()

        if is_main:
            n_valid = sum(1 for r in ref_data if r is not None)
            logger.info(
                "Reference log-probs: %d/%d valid (%.1fs)",
                n_valid, n_my, time.time() - t_ref,
            )

        # ── 6. Training loop ───────────────────────────────────────
        total_loss = 0.0
        total_tokens = 0
        n_mini = max(1, (n_my + batch_size - 1) // batch_size)
        t_train = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_samples = 0
            t_epoch = time.time()

            for mb in range(n_mini):
                mb_start = mb * batch_size
                mb_end = min(mb_start + batch_size, n_my)
                mb_losses: list = []

                for i in range(mb_start, mb_end):
                    rd = ref_data[i]
                    if rd is None:
                        continue

                    input_ids = rd["input_ids"].to(device)
                    attn_mask = rd["attn_mask"].to(device)
                    plen = rd["plen"]
                    old_lp = rd["ref_lp"].to(device)

                    with torch.enable_grad():
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                        )
                        logits = out.logits[:, plen - 1:-1, :]
                        target = input_ids[:, plen:]
                        lp = torch.log_softmax(logits, dim=-1)
                        per_tok = lp.gather(
                            -1, target.unsqueeze(-1),
                        ).squeeze(-1).squeeze(0)

                    ratio = torch.exp(per_tok - old_lp)
                    clipped = torch.clamp(
                        ratio, 1.0 - clip_ratio, 1.0 + clip_ratio,
                    )
                    adv_t = torch.tensor(
                        my_advantages[i], device=device, dtype=per_tok.dtype,
                    )
                    surr = torch.min(ratio * adv_t, clipped * adv_t)
                    loss = -surr.mean() + kl_coeff * (old_lp - per_tok).mean()

                    mb_losses.append(loss)
                    epoch_loss += loss.item()
                    epoch_samples += 1
                    total_tokens += per_tok.numel()

                if mb_losses:
                    torch.stack(mb_losses).mean().backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += epoch_loss / max(epoch_samples, 1)

            if is_main:
                elapsed = time.time() - t_epoch
                rate = n_my / elapsed if elapsed > 0 else 0
                logger.info(
                    "  FSDP [%s] epoch %d/%d: loss=%.4f (%.1f samples/s, %.1fs)",
                    adapter_name, epoch + 1, epochs,
                    epoch_loss / max(epoch_samples, 1), rate, elapsed,
                )

        train_time = time.time() - t_train

        # ── 7. Save adapter (rank 0 gathers full state) ───────────
        save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True,
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, save_policy,
        ):
            state_dict = model.state_dict()

        if is_main:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            lora_state = {
                k: v for k, v in state_dict.items() if "lora_" in k
            }

            if lora_state:
                try:
                    from safetensors.torch import save_file
                    save_file(lora_state, str(save_path / "adapter_model.safetensors"))
                except ImportError:
                    torch.save(lora_state, str(save_path / "adapter_model.bin"))

                logger.info(
                    "Saved %d LoRA tensors → %s", len(lora_state), save_path,
                )
            else:
                logger.warning("No LoRA parameters found to save!")

            mean_loss = total_loss / max(epochs, 1)
            throughput = n_total * epochs / train_time if train_time > 0 else 0
            logger.info(
                "FSDP GRPO [%s] done: %.1fs train (%.1fs load), "
                "%d tokens, loss=%.4f, %.1f samples/s",
                adapter_name, train_time, load_time,
                total_tokens, mean_loss, throughput,
            )

            result = {
                "n_samples": n_total,
                "n_tokens": total_tokens,
                "mean_loss": mean_loss,
                "epochs": epochs,
                "train_time_s": train_time,
                "load_time_s": load_time,
                "n_gpus": world_size,
                "throughput": throughput,
            }
            with open(result_file, "w") as f:
                json.dump(result, f)

        # Debug I/O
        if io_log_dir and is_main:
            _write_debug_io(
                io_log_dir, adapter_name, prompts, completions, advantages,
            )

        del model, optimizer, tokenizer, state_dict
        gc.collect()
        torch.cuda.empty_cache()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _write_debug_io(
    io_log_dir: str,
    adapter_name: str,
    prompts: List[str],
    completions: List[str],
    advantages: List[float],
) -> None:
    try:
        io_dir = Path(io_log_dir) / "grpo"
        io_dir.mkdir(parents=True, exist_ok=True)
        fname = f"fsdp_{adapter_name}.jsonl"
        with open(io_dir / fname, "w") as f:
            for i, (p, c, a) in enumerate(zip(prompts, completions, advantages)):
                record = {
                    "ts": time.time(),
                    "adapter": adapter_name,
                    "sample_idx": i,
                    "prompt": p,
                    "prompt_len_chars": len(p),
                    "completion": c,
                    "completion_len_chars": len(c),
                    "advantage": a,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("FSDP debug I/O write failed: %s", exc)


def run_fsdp_grpo(
    gpu_ids: List[int],
    model_name: str,
    adapter_dir: str,
    adapter_name: str,
    prompts: List[str],
    completions: List[str],
    advantages: List[float],
    *,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    clip_ratio: float = 0.2,
    kl_coeff: float = 0.05,
    save_dir: Optional[str] = None,
    io_log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch FSDP GRPO training for one adapter across multiple GPUs.

    Blocks until training completes.  Returns training stats dict.

    Parameters
    ----------
    gpu_ids : list[int]
        Physical GPU indices (e.g. ``[4, 5, 6, 7]``).
    model_name : str
        HuggingFace model id (e.g. ``"Qwen/Qwen3-14B"``).
    adapter_dir : str
        Directory containing the LoRA adapter (or where a fresh one
        will be created).
    adapter_name : str
        PEFT adapter name (e.g. ``"skill_selection"``).
    prompts, completions, advantages :
        Parallel lists of training data.
    """
    import torch.multiprocessing as mp

    if not prompts:
        logger.warning("No training data for adapter '%s'", adapter_name)
        return {"n_samples": 0, "skipped": True}

    if save_dir is None:
        save_dir = adapter_dir

    world_size = len(gpu_ids)
    master_port = _find_free_port()

    result_fd, result_file = tempfile.mkstemp(
        suffix=".json", prefix="fsdp_result_",
    )
    os.close(result_fd)

    args = {
        "world_size": world_size,
        "master_port": master_port,
        "gpu_ids": gpu_ids,
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "adapter_name": adapter_name,
        "prompts": prompts,
        "completions": completions,
        "advantages": advantages,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "clip_ratio": clip_ratio,
        "kl_coeff": kl_coeff,
        "save_dir": save_dir,
        "result_file": result_file,
        "io_log_dir": io_log_dir,
    }

    logger.info(
        "Launching FSDP GRPO '%s' on %d GPUs %s (%d samples, %d epochs)",
        adapter_name, world_size, gpu_ids, len(prompts), epochs,
    )

    # Remap CUDA_VISIBLE_DEVICES so rank 0..N-1 map to the target GPUs
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    t0 = time.time()
    try:
        mp.spawn(
            _fsdp_train_worker,
            nprocs=world_size,
            args=(args,),
            join=True,
        )
    finally:
        if original_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    elapsed = time.time() - t0

    try:
        with open(result_file) as f:
            result = json.load(f)
        result["wall_time_s"] = elapsed
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to read FSDP result: %s", exc)
        result = {"error": str(exc), "wall_time_s": elapsed}
    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass

    logger.info(
        "FSDP GRPO '%s' complete: %.1fs wall time", adapter_name, elapsed,
    )
    return result

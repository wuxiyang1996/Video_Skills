"""On-policy action-GRPO continuation for one Video Skills SFT specialist LoRA.

The policy samples K JSON actions from the same local adapter that is updated.
Rewards are deterministic against the specialist action target; no mock semantic
judge, remote rollout policy, hidden prompt field, or cross-adapter claim is used.

This trainer is also the algorithm/memory smoke before terminal-reward L2 GRPO.
Do not treat SFT-train exact-match rewards as end-to-end video-task evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

from dataset_clip_wrapper.training.sft_common import apply_chat_template_no_think, strip_think_tags
from trainer.grpo.objective import centered_group_advantages, clipped_grpo_loss, completion_reward


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    raise ValueError("row has no assistant message")


def _prompt_text(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system/user/assistant chat row")
    return apply_chat_template_no_think(
        tokenizer, messages[:2], add_generation_prompt=True, tokenize=False
    )


def _trim_generated(ids: Sequence[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    result: list[int] = []
    for token in ids:
        token = int(token)
        if pad_token_id is not None and token == int(pad_token_id):
            break
        result.append(token)
        if eos_token_id is not None and token == int(eos_token_id):
            break
    return result


def _token_logprobs(model: Any, input_ids: Any, prompt_len: int, *, requires_grad: bool) -> Any:
    import torch
    import torch.nn.functional as F

    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        start = max(int(prompt_len) - 1, 0)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        hidden = base.model(input_ids=input_ids, use_cache=False).last_hidden_state
        token_hidden = hidden[0, start : input_ids.shape[1] - 1]
        targets = input_ids[0, start + 1 :]
        pieces = []
        for offset in range(0, targets.numel(), 16):
            logits = F.linear(
                token_hidden[offset : offset + 16],
                base.lm_head.weight,
            )
            pieces.append(
                F.log_softmax(logits.float(), dim=-1)
                .gather(1, targets[offset : offset + 16, None])
                .squeeze(1)
            )
        selected = torch.cat(pieces, dim=0) if pieces else targets.new_empty((0,), dtype=torch.float32)
    if selected.numel() == 0:
        raise ValueError("no generated tokens to score")
    return selected


def _sample_group(
    policy: Any,
    reference: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    prompt = _prompt_text(tokenizer, row)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"].to(device)
    prompt_len = int(prompt_ids.shape[1])
    policy.eval()
    policy.config.use_cache = True
    with torch.no_grad():
        generated_rows = []
        # Generate one completion at a time.  K-way KV caches beside the frozen
        # reference model exceed a 48GB A6000 on the longest L2 prompts.
        for _ in range(int(k)):
            generated_rows.append(
                policy.generate(
                    input_ids=prompt_ids,
                    attention_mask=encoded.get("attention_mask").to(device),
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]
            )
    policy.config.use_cache = False
    gold_text = _assistant_text(row)
    samples: list[dict[str, Any]] = []
    for sequence in generated_rows:
        completion_ids = _trim_generated(
            sequence[prompt_len:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        if not completion_ids:
            continue
        full_ids = torch.tensor(
            [prompt_ids[0].tolist() + completion_ids], dtype=torch.long, device=device
        )
        completion = strip_think_tags(tokenizer.decode(completion_ids, skip_special_tokens=True))
        reward = completion_reward(completion, gold_text)
        policy.eval()
        old_lp = _token_logprobs(policy, full_ids, prompt_len, requires_grad=False).detach()
        ref_lp = _token_logprobs(reference, full_ids, prompt_len, requires_grad=False).detach()
        samples.append(
            {
                "input_ids": full_ids,
                "prompt_len": prompt_len,
                "completion": completion,
                "old_logprobs": old_lp,
                "ref_logprobs": ref_lp,
                **reward,
            }
        )
    return samples


def _greedy_eval(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: Any, limit: int) -> dict[str, Any]:
    import torch

    selected = rows[: max(0, int(limit))] if limit > 0 else rows
    results = []
    model.eval()
    model.config.use_cache = True
    for row in selected:
        prompt = _prompt_text(tokenizer, row)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        results.append(completion_reward(strip_think_tags(completion), _assistant_text(row)))
    model.config.use_cache = False
    return {
        "n": len(results),
        "json_valid_rate": sum(item["json_valid"] for item in results) / max(1, len(results)),
        "action_match_rate": sum(item["action_match"] for item in results) / max(1, len(results)),
        "exact_match_rate": sum(item["exact_match"] for item in results) / max(1, len(results)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--specialist", choices=["l2", "repair"], required=True)
    parser.add_argument("--max-groups", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--sft-replay-coef", type=float, default=0.05)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-sdpa-fallback", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trainer.grpo.attn_utils import resolve_attn_implementation
    from trainer.grpo.model_runtime import _disable_torchao_peft_probes

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _disable_torchao_peft_probes()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for on-policy GRPO")
    attn = resolve_attn_implementation(
        "flash_attention_2", allow_sdpa_fallback=bool(args.allow_sdpa_fallback)
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def load_base() -> Any:
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=attn,
        )

    policy = PeftModel.from_pretrained(load_base(), args.adapter, is_trainable=True).to(device)
    reference = PeftModel.from_pretrained(load_base(), args.adapter, is_trainable=False).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads()

    train_rows = _read_jsonl(args.train_jsonl)
    dev_rows = _read_jsonl(args.dev_jsonl)
    random.Random(args.seed).shuffle(train_rows)
    if args.max_groups > 0:
        train_rows = train_rows[: args.max_groups]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    before = _greedy_eval(policy, tokenizer, dev_rows, device, args.eval_samples)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=float(args.learning_rate),
    )
    started = time.time()
    trained_groups = 0
    skipped_equal_reward = 0
    all_rewards: list[float] = []
    all_kls: list[float] = []
    for group_index, row in enumerate(train_rows):
        samples = _sample_group(
            policy,
            reference,
            tokenizer,
            row,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        rewards = [float(sample["reward"]) for sample in samples]
        advantages = centered_group_advantages(rewards)
        all_rewards.extend(rewards)
        if not samples or not any(abs(value) > 1e-8 for value in advantages):
            skipped_equal_reward += 1
            continue
        epoch_losses = []
        group_kls = []
        for _ in range(max(1, args.ppo_epochs)):
            optimizer.zero_grad(set_to_none=True)
            policy_loss_values = []
            for sample, advantage in zip(samples, advantages):
                policy.train()
                new_lp = _token_logprobs(
                    policy, sample["input_ids"], sample["prompt_len"], requires_grad=True
                )
                loss, _, kl = clipped_grpo_loss(
                    new_lp,
                    sample["old_logprobs"],
                    sample["ref_logprobs"],
                    advantage,
                    clip_eps=args.clip_eps,
                    kl_coef=args.kl_coef,
                )
                (loss / len(samples)).backward()
                policy_loss_values.append(float(loss.detach().cpu()))
                group_kls.append(float(kl.detach().cpu()))
            # SFT replay on the gold action from this same visible prompt.
            prompt = _prompt_text(tokenizer, row)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            gold_ids = tokenizer(_assistant_text(row), add_special_tokens=False)["input_ids"]
            replay_ids = torch.tensor([prompt_ids + gold_ids], dtype=torch.long, device=device)
            replay_lp = _token_logprobs(policy, replay_ids, len(prompt_ids), requires_grad=True)
            replay_loss = -replay_lp.mean()
            replay_term = float(args.sft_replay_coef) * replay_loss
            replay_term.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(
                sum(policy_loss_values) / len(policy_loss_values)
                + float(replay_term.detach().cpu())
            )
        trained_groups += 1
        all_kls.extend(group_kls)
        metric = {
            "group": group_index,
            "trained_groups": trained_groups,
            "rewards": rewards,
            "advantages": advantages,
            "loss": sum(epoch_losses) / len(epoch_losses),
            "kl": sum(group_kls) / max(1, len(group_kls)),
            "json_valid_rate": sum(sample["json_valid"] for sample in samples) / len(samples),
            "action_match_rate": sum(sample["action_match"] for sample in samples) / len(samples),
            "exact_match_rate": sum(sample["exact_match"] for sample in samples) / len(samples),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        print(json.dumps({"event": "grpo_step", **metric}), flush=True)

    adapter_out = args.output_dir / "adapter"
    policy.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    after = _greedy_eval(policy, tokenizer, dev_rows, device, args.eval_samples)
    summary = {
        "schema_version": "video-skills/specialist-on-policy-grpo-v1",
        "specialist": args.specialist,
        "source_adapter": str(args.adapter),
        "adapter_out": str(adapter_out),
        "on_policy": True,
        "remote_rollout_policy": False,
        "mock_semantic_judge": False,
        "objective": "token_clipped_action_grpo+k3_kl+sft_replay",
        "reward_scope": "sft_action_exact_match_not_video_terminal_reward",
        "groups_seen": len(train_rows),
        "groups_trained": trained_groups,
        "groups_skipped_equal_reward": skipped_equal_reward,
        "samples": len(all_rewards),
        "mean_reward": sum(all_rewards) / max(1, len(all_rewards)),
        "mean_kl": sum(all_kls) / max(1, len(all_kls)),
        "before_dev": before,
        "after_dev": after,
        "elapsed_s": time.time() - started,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }
    (args.output_dir / "grpo_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "grpo_complete", **summary}, ensure_ascii=False), flush=True)
    if summary["mean_kl"] < -1e-8:
        raise RuntimeError("non-negative KL invariant violated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

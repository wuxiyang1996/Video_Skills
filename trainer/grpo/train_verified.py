"""GRPO / RLVR trainer over verified lexicographic rewards (plan §7).

Supports:
  - ``l2_repair`` (default): update L2 + Repair LoRA only; L1 frozen
  - ``joint_l1``: small-LR L1 + L2 + Repair after L2/Repair stability gate

Full GPU LoRA optimization plugs into ``StudentLogprobFn``. Smoke mode validates
group advantages, KL surrogate, and module gating without loading weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trainer.grpo.advantages import grpo_surrogate_loss
from trainer.grpo.collect_rollouts import load_grpo_groups
from trainer.grpo.quality import filter_group_dicts_for_training, summarize_group_quality
from trainer.grpo.types import (
    MODE_JOINT_L1,
    MODE_L2_REPAIR,
    GrpoTrainConfig,
)
from trainer.posttraining_manifest import build_posttraining_manifest, save_posttraining_manifest
from trainer.reward import REWARD_SPEC_VERSION, VerifiedRewardBreakdown, group_rank_advantages

StudentLogprobFn = Callable[[Mapping[str, Any]], float]


def _reward_from_dict(payload: Mapping[str, Any]) -> VerifiedRewardBreakdown:
    progress = payload.get("verified_atomic_progress") or (0, 0, 0, 0, 0)
    progress_t = tuple(int(x) for x in progress)
    rank_key = payload.get("rank_key")
    if rank_key is None:
        rank_key = (
            int(payload.get("hard_feasible", 0)),
            int(payload.get("terminal_success", 0)),
            progress_t,
            int(payload.get("evidence_checks", 0)),
            -int(payload.get("cost_total", 0)),
        )
    else:
        # JSON turns tuples into lists; normalize progress slot.
        rk = list(rank_key)
        if len(rk) >= 3 and isinstance(rk[2], list):
            rk[2] = tuple(int(x) for x in rk[2])
        rank_key = tuple(rk)
    return VerifiedRewardBreakdown(
        spec_version=str(payload.get("spec_version") or REWARD_SPEC_VERSION),
        hard_feasible=bool(payload.get("hard_feasible")),
        terminal_success=bool(payload.get("terminal_success")),
        verified_atomic_progress=progress_t,
        progress_total=int(payload.get("progress_total") or sum(progress_t)),
        evidence_checks=int(payload.get("evidence_checks") or 0),
        cost_total=int(payload.get("cost_total") or 0),
        rank_key=rank_key,  # type: ignore[arg-type]
        hard_failures=tuple(payload.get("hard_failures") or ()),
        blocked_strong_commit=bool(payload.get("blocked_strong_commit")),
    )


def proxy_logprob_from_reward(rollout: Mapping[str, Any]) -> float:
    """Deterministic logprob proxy for smoke: prefer higher rank_key via soft score."""
    reward = rollout.get("reward") or {}
    progress = reward.get("progress_total") or 0
    success = 1.0 if reward.get("terminal_success") else 0.0
    feasible = 1.0 if reward.get("hard_feasible") else 0.0
    cost = float(reward.get("cost_total") or 0.0)
    score = 3.0 * feasible + 2.0 * success + 0.1 * float(progress) - 0.001 * cost
    # Map to a negative logprob-like value.
    return float(score - 5.0)


def train_step_on_group(
    group: Mapping[str, Any],
    *,
    config: GrpoTrainConfig,
    student_logprob_fn: StudentLogprobFn = proxy_logprob_from_reward,
    reference_logprob_fn: StudentLogprobFn | None = None,
) -> dict[str, Any]:
    """One GRPO group step (CPU surrogate)."""
    modules = config.update_modules()
    rollouts = list(group.get("rollouts") or [])
    if len(rollouts) < 2:
        raise ValueError("group must contain >= 2 rollouts")

    rewards = [_reward_from_dict(r.get("reward") or {}) for r in rollouts]
    advantages = group_rank_advantages(rewards)
    logprobs = [float(student_logprob_fn(r)) for r in rollouts]
    ref_fn = reference_logprob_fn or student_logprob_fn
    # Reference is frozen OPD/SFT policy; for smoke use a slightly damped copy.
    ref_logprobs = [float(ref_fn(r)) - 0.01 for r in rollouts]
    loss_stats = grpo_surrogate_loss(
        advantages=advantages,
        logprobs=logprobs,
        ref_logprobs=ref_logprobs,
        kl_coef=config.kl_coef,
    )
    if any(not math.isfinite(v) for v in logprobs + advantages + [loss_stats["loss"]]):
        raise ValueError("non-finite values in GRPO step")

    return {
        "group_id": group.get("group_id"),
        "mode": config.mode,
        "update_modules": list(modules),
        "l1_lr_scale": config.l1_lr_scale if "l1" in modules else 0.0,
        "advantages": advantages,
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
        **loss_stats,
        "n_terminal_success": sum(1 for r in rewards if r.terminal_success),
        "mean_progress": sum(r.progress_total for r in rewards) / len(rewards),
    }


def run_grpo_smoke(
    groups_path: str | Path,
    *,
    config: GrpoTrainConfig,
    output_path: str | Path | None = None,
    drop_dirty: bool = True,
) -> dict[str, Any]:
    groups = load_grpo_groups(groups_path)
    if drop_dirty:
        groups = filter_group_dicts_for_training(groups, drop_dirty=True, min_k=2)
    quality = summarize_group_quality(groups)
    steps = [train_step_on_group(g, config=config) for g in groups]
    summary = {
        "n_groups": len(steps),
        "mode": config.mode,
        "update_modules": list(config.update_modules()),
        "mean_loss": sum(s["loss"] for s in steps) / max(len(steps), 1),
        "mean_kl": sum(s["kl"] for s in steps) / max(len(steps), 1),
        "mean_policy_loss": sum(s["policy_loss"] for s in steps) / max(len(steps), 1),
        "steps": steps,
        "backend": "cpu_proxy",
        "quality": quality,
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def run_grpo_gpu(
    groups_path: str | Path,
    *,
    config: GrpoTrainConfig,
    output_dir: str | Path,
    base_model: str,
    l2_adapter: str,
    repair_adapter: str | None = None,
    l1_adapter: str | None = None,
    learning_rate: float = 5e-6,
    max_groups: int = 0,
    allow_sdpa_fallback: bool = False,
    save_adapter: bool = True,
) -> dict[str, Any]:
    """Single-GPU GRPO with FlashAttention-2 + PEFT LoRA (not verl / ms-swift)."""
    import torch

    from trainer.grpo.model_runtime import load_grpo_runtime

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modules = config.update_modules()
    runtime = load_grpo_runtime(
        base_model=base_model,
        l2_adapter=l2_adapter,
        repair_adapter=repair_adapter,
        l1_adapter=l1_adapter,
        update_modules=modules,
        load_reference=True,
        attn_implementation="flash_attention_2",
        allow_sdpa_fallback=allow_sdpa_fallback,
    )
    lr = float(learning_rate)
    if "l1" in modules:
        lr = lr * float(config.l1_lr_scale)
    optimizer = torch.optim.AdamW(
        (p for p in runtime.policy.parameters() if p.requires_grad),
        lr=lr,
    )

    groups = load_grpo_groups(groups_path)
    groups = filter_group_dicts_for_training(groups, drop_dirty=True, min_k=2)
    quality = summarize_group_quality(groups)
    if max_groups > 0:
        groups = groups[: int(max_groups)]
    steps: list[dict[str, Any]] = []
    runtime.policy.train()

    for group in groups:
        rollouts = list(group.get("rollouts") or [])
        rewards = [_reward_from_dict(r.get("reward") or {}) for r in rollouts]
        advantages = group_rank_advantages(rewards)
        ref_logprobs: list[float] = []
        for rollout in rollouts:
            messages = runtime.rollout_messages_from_policy_view(rollout)
            with torch.no_grad():
                ref_logprobs.append(float(runtime.sequence_logprob(messages, use_reference=True)))

        # Surrogate: mean_i [ -A_i * logπ_i + β (logπ_i - logπ_ref_i) ]
        diff_lps = []
        runtime.policy.train()
        for rollout in rollouts:
            messages = runtime.rollout_messages_from_policy_view(rollout)
            diff_lps.append(_differentiable_sequence_logprob(runtime, messages))

        loss = None
        kl_vals = []
        for adv, lp, rlp in zip(advantages, diff_lps, ref_logprobs):
            kl = lp - float(rlp)
            kl_vals.append(float(kl.detach().cpu()))
            term = -float(adv) * lp + float(config.kl_coef) * kl
            loss = term if loss is None else loss + term
        assert loss is not None
        loss = loss / max(len(diff_lps), 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runtime.policy.parameters(), config.max_grad_norm)
        optimizer.step()
        steps.append(
            {
                "group_id": group.get("group_id"),
                "loss": float(loss.detach().cpu()),
                "kl": sum(kl_vals) / max(len(kl_vals), 1),
                "advantages": advantages,
                "n_terminal_success": sum(1 for r in rewards if r.terminal_success),
            }
        )

    if save_adapter:
        adapter_out = out_dir / "adapter"
        runtime.policy.save_pretrained(adapter_out)

    summary = {
        "n_groups": len(steps),
        "mode": config.mode,
        "update_modules": list(modules),
        "mean_loss": sum(s["loss"] for s in steps) / max(len(steps), 1),
        "mean_kl": sum(s["kl"] for s in steps) / max(len(steps), 1),
        "backend": "gpu_hf_peft",
        "attn_implementation": runtime.attn_implementation,
        "framework": "hf_peft_custom_grpo",
        "verl": False,
        "ms_swift": False,
        "adapter_paths": runtime.adapter_paths,
        "quality": quality,
        "steps": steps,
    }
    (out_dir / "grpo_train_gpu.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _differentiable_sequence_logprob(runtime: Any, messages: Sequence[Mapping[str, str]]):
    import torch
    import torch.nn.functional as F
    from dataset_clip_wrapper.training.sft_common import apply_chat_template_no_think

    full_text = apply_chat_template_no_think(
        runtime.tokenizer, list(messages), add_generation_prompt=False, tokenize=False
    )
    prompt_text = apply_chat_template_no_think(
        runtime.tokenizer, list(messages[:2]), add_generation_prompt=True, tokenize=False
    )
    full_ids = runtime.tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = runtime.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if isinstance(full_ids[0], list):
        full_ids = full_ids[0]
        prompt_ids = prompt_ids[0]
    input_ids = torch.tensor([full_ids], device=runtime.device)
    outputs = runtime.policy(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = max(len(prompt_ids) - 1, 0)
    return token_lp[0, start:].mean()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True, help="grpo_groups.jsonl from collector")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=[MODE_L2_REPAIR, MODE_JOINT_L1], default=MODE_L2_REPAIR)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--l1-lr-scale", type=float, default=0.1)
    parser.add_argument("--l2-stable", action="store_true")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--gpu", action="store_true", help="Run FlashAttention-2 PEFT GRPO on CUDA")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--l2-adapter",
        default=(
            "dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/"
            "pilot/l2/pilot/adapter"
        ),
    )
    parser.add_argument(
        "--repair-adapter",
        default=(
            "dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/"
            "pilot/repair/pilot/adapter"
        ),
    )
    parser.add_argument(
        "--l1-adapter",
        default=(
            "dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/"
            "pilot_l1_full/l1/pilot/adapter"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument(
        "--allow-sdpa-fallback",
        action="store_true",
        help="Debug only; GRPO should use flash_attention_2",
    )
    args = parser.parse_args(argv)

    config = GrpoTrainConfig(
        mode=args.mode,
        kl_coef=float(args.kl_coef),
        l1_lr_scale=float(args.l1_lr_scale),
        l2_stable_flag=bool(args.l2_stable),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu:
        summary = run_grpo_gpu(
            args.groups,
            config=config,
            output_dir=out_dir,
            base_model=args.base_model,
            l2_adapter=args.l2_adapter,
            repair_adapter=args.repair_adapter,
            l1_adapter=args.l1_adapter if args.mode == MODE_JOINT_L1 else None,
            learning_rate=float(args.learning_rate),
            max_groups=int(args.max_groups),
            allow_sdpa_fallback=bool(args.allow_sdpa_fallback),
        )
    else:
        summary = run_grpo_smoke(args.groups, config=config, output_path=out_dir / "grpo_train_smoke.json")

    if args.split_manifest:
        manifest = build_posttraining_manifest(
            stage="grpo_train",
            split_manifest_path=args.split_manifest,
            reward_spec_version=REWARD_SPEC_VERSION,
            grpo_mode=config.mode,
            update_modules=list(config.update_modules()),
            policy_checkpoint=args.policy_checkpoint or args.l2_adapter,
            reference_checkpoint=args.reference_checkpoint,
            extras={
                "smoke": not args.gpu,
                "gpu": bool(args.gpu),
                "mean_loss": summary["mean_loss"],
                "attn_implementation": summary.get("attn_implementation"),
                "framework": summary.get("framework", "hf_peft_custom_grpo"),
                "verl": False,
                "ms_swift": False,
            },
        )
        save_posttraining_manifest(out_dir / "posttraining_run_manifest.json", manifest)

    print(
        json.dumps(
            {
                "n_groups": summary["n_groups"],
                "mode": summary["mode"],
                "update_modules": summary["update_modules"],
                "mean_loss": summary["mean_loss"],
                "mean_kl": summary["mean_kl"],
                "backend": summary.get("backend"),
                "attn_implementation": summary.get("attn_implementation"),
                "framework": summary.get("framework", "hf_peft_custom_grpo"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

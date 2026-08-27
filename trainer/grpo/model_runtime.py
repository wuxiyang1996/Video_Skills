"""Local Qwen3.5-9B + PEFT LoRA runtime for GRPO (FlashAttention-2 required)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_clip_wrapper.training.sft_common import apply_chat_template_no_think
from trainer.grpo.attn_utils import assert_model_uses_flash_attn, resolve_attn_implementation


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"


def _disable_torchao_peft_probes() -> None:
    """Match SFT trainer: inherited Swift env has old torchao that breaks PEFT probes."""
    import peft.import_utils as peft_import_utils
    import peft.tuners.lora.torchao as peft_torchao

    peft_import_utils.is_torchao_available = lambda: False
    peft_torchao.is_torchao_available = lambda: False


@dataclass
class GrpoModelRuntime:
    """Holds tokenizer + trainable LoRA policy and optional frozen reference."""

    tokenizer: Any
    policy: Any
    reference: Any | None
    device: Any
    attn_implementation: str
    update_modules: tuple[str, ...] = ("l2", "repair")
    adapter_paths: dict[str, str] = field(default_factory=dict)

    def sequence_logprob(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        use_reference: bool = False,
    ) -> float:
        """Length-normalized mean logprob of assistant tokens (teacher-forced)."""
        import torch
        import torch.nn.functional as F

        model = self.reference if use_reference and self.reference is not None else self.policy
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("messages must be [system, user, assistant]")
        full_text = apply_chat_template_no_think(
            self.tokenizer, list(messages), add_generation_prompt=False, tokenize=False
        )
        prompt_text = apply_chat_template_no_think(
            self.tokenizer, list(messages[:2]), add_generation_prompt=True, tokenize=False
        )
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if isinstance(full_ids[0], list):
            full_ids = full_ids[0]
            prompt_ids = prompt_ids[0]
        if len(prompt_ids) >= len(full_ids):
            raise ValueError("no assistant tokens to score")
        input_ids = torch.tensor([full_ids], device=self.device)
        with torch.set_grad_enabled(model is self.policy and model.training):
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            # Mask prompt tokens (aligned to targets = input_ids[1:]).
            start = max(len(prompt_ids) - 1, 0)
            assistant_lp = token_lp[0, start:]
            if assistant_lp.numel() == 0:
                raise ValueError("empty assistant logprob span")
            return float(assistant_lp.mean().detach().cpu())

    def rollout_messages_from_policy_view(self, rollout: Mapping[str, Any]) -> list[dict[str, str]]:
        """Build a scorable chat from a collected GRPO policy_view / extras."""
        extras = rollout.get("extras") or {}
        meta = (rollout.get("policy_view") or {}).get("metadata") or {}
        plan = (meta.get("llm_plan") or {}).get("reasoning_plan") or meta.get("reasoning_plan") or []
        motif = rollout.get("motif_online") or meta.get("motif_online") or {}
        question = (
            ((rollout.get("policy_view") or {}).get("question") or {}).get("question_text")
            or extras.get("question_text")
            or ""
        )
        assistant_payload = {
            "schema_version": "video-skills/grpo-controller-action-v0.1",
            "motif_selected_id": motif.get("selected_motif_id"),
            "motif_phase": motif.get("motif_phase"),
            "reasoning_plan": plan,
            "final_answer": (rollout.get("policy_view") or {}).get("final_answer")
            or extras.get("final_answer"),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are the Video Skills L2/Repair controller. "
                    "Emit one JSON object for the executable reasoning plan / commit."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Motif phase: {motif.get('motif_phase')}\n"
                    f"Selected motif: {motif.get('selected_motif_id')}\n"
                    "Produce the controller JSON for this episode."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload, ensure_ascii=False, sort_keys=True),
            },
        ]


def load_grpo_runtime(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    l2_adapter: str | Path | None = None,
    repair_adapter: str | Path | None = None,
    l1_adapter: str | Path | None = None,
    update_modules: Sequence[str] = ("l2", "repair"),
    load_reference: bool = True,
    attn_implementation: str = "flash_attention_2",
    allow_sdpa_fallback: bool = False,
    local_files_only: bool = True,
    dtype: str = "bfloat16",
) -> GrpoModelRuntime:
    """Load policy LoRA stack with FlashAttention-2.

    Adapter merge strategy for v1:
      - load base + L2 adapter as active trainable PEFT model when ``l2`` in update_modules
      - Repair/L1 adapters are recorded in ``adapter_paths`` for run manifest; multi-adapter
        simultaneous train can be extended once PEFT multi-adapter routing is wired.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _disable_torchao_peft_probes()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO model runtime")

    attn = resolve_attn_implementation(
        attn_implementation, allow_sdpa_fallback=allow_sdpa_fallback
    )
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _load_base() -> Any:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            local_files_only=local_files_only,
            dtype=torch_dtype,
            attn_implementation=attn,
        )
        return model

    base = _load_base()
    adapter_paths: dict[str, str] = {}
    active_adapter = None
    if l2_adapter and Path(l2_adapter).exists():
        adapter_paths["l2"] = str(l2_adapter)
        active_adapter = str(l2_adapter)
    if repair_adapter and Path(repair_adapter).exists():
        adapter_paths["repair"] = str(repair_adapter)
        if active_adapter is None:
            active_adapter = str(repair_adapter)
    if l1_adapter and Path(l1_adapter).exists():
        adapter_paths["l1"] = str(l1_adapter)

    if active_adapter is None:
        raise RuntimeError("need at least one existing L2/Repair LoRA adapter path")

    policy = PeftModel.from_pretrained(base, active_adapter, is_trainable=True)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads()
    policy.to(device)
    if attn == "flash_attention_2":
        assert_model_uses_flash_attn(policy)

    reference = None
    if load_reference:
        ref_base = _load_base()
        reference = PeftModel.from_pretrained(ref_base, active_adapter, is_trainable=False)
        reference.to(device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad = False
        if attn == "flash_attention_2":
            assert_model_uses_flash_attn(reference)

    # Freeze modules not in update list (best-effort: whole PEFT is L2/repair active adapter).
    if "l2" not in update_modules and "repair" not in update_modules and "l1" not in update_modules:
        raise ValueError(f"empty update_modules: {update_modules}")

    return GrpoModelRuntime(
        tokenizer=tokenizer,
        policy=policy,
        reference=reference,
        device=device,
        attn_implementation=attn,
        update_modules=tuple(update_modules),
        adapter_paths=adapter_paths,
    )

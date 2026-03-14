"""
GRPO LoRA trainer: reads the buffer, computes advantages, performs
policy-gradient updates on LoRA adapter weights.

Uses HuggingFace native PEFT + PyTorch optimizer (Option A from the
architecture plan). Upgrade to VERL/TRL integration when scaling to
multi-GPU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from skill_agents_grpo.grpo.buffer import GRPOBuffer, GRPOSample
from skill_agents_grpo.grpo.config import GRPOConfig, StageGRPOConfig
from skill_agents_grpo.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


class GRPOLoRATrainer:
    """Train LoRA adapters using GRPO samples from the buffer.

    One trainer instance handles all adapters. Each ``train_step`` call
    processes one adapter's buffer, computes the GRPO loss, and performs
    a gradient step on the adapter's parameters.

    Parameters
    ----------
    llm : MultiLoraSkillBankLLM
        The shared model — provides ``log_probs()`` for gradient computation.
    config : GRPOConfig
        Per-stage hyperparameters.
    """

    def __init__(self, llm: Any, config: Optional[GRPOConfig] = None) -> None:
        self._llm = llm
        self._config = config or GRPOConfig()
        self._optimizers: Dict[str, Any] = {}
        self._step_counts: Dict[str, int] = {}

    def _get_optimizer(self, adapter: SkillFunction, cfg: StageGRPOConfig) -> Any:
        """Lazily create an AdamW optimizer for the adapter's LoRA parameters."""
        import torch

        key = adapter.value
        if key not in self._optimizers:
            self._llm._activate_adapter(adapter)
            if self._llm._is_peft_model:
                params = [
                    p for n, p in self._llm._model.named_parameters()
                    if p.requires_grad and "lora" in n.lower()
                ]
            else:
                params = [p for p in self._llm._model.parameters() if p.requires_grad]

            if not params:
                logger.warning("No trainable parameters found for adapter %s", key)
                return None

            self._optimizers[key] = torch.optim.AdamW(params, lr=cfg.lr)
            self._step_counts[key] = 0
            logger.info(
                "Created optimizer for %s: %d params, lr=%.2e",
                key, len(params), cfg.lr,
            )
        return self._optimizers[key]

    def train_step(self, buffer: GRPOBuffer) -> Dict[str, Any]:
        """Run one GRPO training step per adapter that has data.

        Returns per-adapter training stats.
        """
        import torch

        stats: Dict[str, Any] = {}

        for adapter in buffer.adapters_with_data():
            if not self._config.is_enabled(adapter):
                continue

            samples = buffer.samples_for(adapter)
            if not samples:
                continue

            cfg = self._config.for_stage(adapter)
            adapter_stats = self._train_adapter(adapter, samples, cfg)
            stats[adapter.value] = adapter_stats

        return stats

    def _train_adapter(
        self,
        adapter: SkillFunction,
        samples: List[GRPOSample],
        cfg: StageGRPOConfig,
    ) -> Dict[str, Any]:
        """Train a single adapter on its buffered samples."""
        import torch

        optimizer = self._get_optimizer(adapter, cfg)
        if optimizer is None:
            return {"error": "no_trainable_params"}

        # Flatten samples into (prompt, completion, advantage) tuples
        prompts: List[str] = []
        completions: List[str] = []
        advantages: List[float] = []

        for sample in samples:
            if not sample.prompt or not sample.completions:
                continue
            group_advantages = self._compute_advantages(sample.rewards)
            for comp, adv in zip(sample.completions, group_advantages):
                if comp:
                    prompts.append(sample.prompt)
                    completions.append(comp)
                    advantages.append(adv)

        if not prompts:
            return {"n_samples": 0, "skipped": True}

        total_loss = 0.0
        n_tokens = 0

        for epoch in range(cfg.epochs_per_batch):
            optimizer.zero_grad()
            epoch_loss = torch.tensor(0.0, device=self._llm._model.device)

            for prompt, completion, advantage in zip(prompts, completions, advantages):
                log_p = self._llm.log_probs(adapter, prompt, completion)
                if log_p.numel() == 0:
                    continue

                with torch.no_grad():
                    old_log_p = log_p.detach().clone()

                ratio = torch.exp(log_p - old_log_p)
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                adv_tensor = torch.tensor(advantage, device=log_p.device)

                # PPO-clip objective (maximized → negate for loss)
                surrogate = torch.min(ratio * adv_tensor, clipped * adv_tensor)
                token_loss = -surrogate.mean()

                # KL penalty (per-token mean)
                kl = (old_log_p - log_p).mean()
                token_loss = token_loss + cfg.kl_coeff * kl

                epoch_loss = epoch_loss + token_loss
                n_tokens += log_p.numel()

            if n_tokens > 0:
                epoch_loss = epoch_loss / len(prompts)
                epoch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._llm._model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += epoch_loss.item()

        key = adapter.value
        self._step_counts[key] = self._step_counts.get(key, 0) + 1

        stats = {
            "n_samples": len(prompts),
            "n_tokens": n_tokens,
            "epochs": cfg.epochs_per_batch,
            "mean_loss": total_loss / max(cfg.epochs_per_batch, 1),
            "step": self._step_counts[key],
        }
        logger.info("GRPO train [%s] step %d: %s", key, self._step_counts[key], stats)
        return stats

    @staticmethod
    def _compute_advantages(rewards: List[float]) -> List[float]:
        """Group-normalize rewards to advantages (zero-mean, unit-variance)."""
        if not rewards:
            return []
        n = len(rewards)
        if n == 1:
            return [0.0]
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        std = var ** 0.5 if var > 0 else 1.0
        return [(r - mean) / std for r in rewards]

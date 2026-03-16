"""GRPO training wrappers for the co-evolution loop.

Provides two independent training paths that run **sequentially** on
the same set of GPUs using FSDP data parallelism:

1. **Decision Agent GRPO** — updates ``skill_selection`` and
   ``action_taking`` LoRA adapters using per-step environment rewards.
2. **Skill Bank GRPO** — updates ``segment``, ``contract``, and
   ``curator`` LoRA adapters using stage-specific reward signals.

Both delegate the actual training to
:func:`skill_agents_grpo.grpo.fsdp_trainer.run_fsdp_grpo` which
spawns one process per GPU and uses PyTorch FSDP to shard the frozen
14B base model across all ranks while each rank processes its own
data slice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.coevolution.episode_runner import EpisodeResult, GRPORecord

logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainStats:
    adapter: str
    n_samples: int = 0
    n_tokens: int = 0
    mean_loss: float = 0.0
    epochs: int = 0
    wall_time_s: float = 0.0


@dataclass
class GRPOStepResult:
    decision_stats: Dict[str, GRPOTrainStats] = field(default_factory=dict)
    skillbank_stats: Dict[str, GRPOTrainStats] = field(default_factory=dict)
    wall_time_s: float = 0.0
    records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────


def _compute_advantages(rewards: List[float]) -> List[float]:
    """Group-normalize rewards to zero-mean, unit-variance advantages."""
    if not rewards:
        return []
    n = len(rewards)
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5 if var > 0 else 1.0
    return [(r - mean) / std for r in rewards]


def _collect_grpo_records(results: List[EpisodeResult]) -> Dict[str, List[GRPORecord]]:
    """Group GRPO records by adapter from all episode results."""
    records: Dict[str, List[GRPORecord]] = {
        "action_taking": [],
        "skill_selection": [],
    }
    for r in results:
        for rec in r.grpo_records:
            adapter = rec.adapter
            if adapter in records:
                records[adapter].append(rec)
    return records


def _records_to_training_data(
    records: List[GRPORecord],
) -> tuple:
    """Convert GRPORecords to flat (prompts, completions, advantages) lists.

    Records are grouped by (episode_id, step) and advantages are
    computed within each group.
    """
    groups: Dict[str, List[GRPORecord]] = {}
    for rec in records:
        key = f"{rec.episode_id}_{rec.step}"
        groups.setdefault(key, []).append(rec)

    prompts: List[str] = []
    completions: List[str] = []
    advantages: List[float] = []

    for group in groups.values():
        rewards = [r.reward for r in group]
        advs = _compute_advantages(rewards)
        for rec, adv in zip(group, advs):
            if rec.completion:
                prompts.append(rec.prompt)
                completions.append(rec.completion)
                advantages.append(adv)

    return prompts, completions, advantages


def _samples_to_training_data(
    samples: List[Dict[str, Any]],
) -> tuple:
    """Convert skill-bank sample dicts to flat training lists.

    Each sample has: prompt, completions: [], rewards: [].
    """
    prompts: List[str] = []
    completions: List[str] = []
    advantages: List[float] = []

    for s in samples:
        prompt = s.get("prompt", "")
        comps = s.get("completions", [])
        rewards = s.get("rewards", [])
        if not prompt or not comps:
            continue
        advs = _compute_advantages(rewards)
        for comp, adv in zip(comps, advs):
            if comp:
                prompts.append(prompt)
                completions.append(comp)
                advantages.append(adv)

    return prompts, completions, advantages


# ── Decision Agent Trainer ──────────────────────────────────────────────


class DecisionGRPOTrainer:
    """GRPO trainer for decision agent LoRAs (skill_selection + action_taking).

    Launches FSDP training across all specified GPUs for each adapter
    sequentially.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        adapter_dir: str = "runs/lora_adapters",
        devices: Optional[List[int]] = None,
        group_size: int = 8,
        lr: float = 5e-5,
        temperature: float = 0.7,
        kl_coeff: float = 0.05,
        io_log_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.devices = devices or [4, 5, 6, 7]
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.io_log_dir = io_log_dir

    def train_step(
        self,
        records: Dict[str, List[GRPORecord]],
    ) -> Dict[str, GRPOTrainStats]:
        """Run FSDP GRPO for each decision adapter that has data."""
        from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo

        result: Dict[str, GRPOTrainStats] = {}

        adapter_configs = {
            "skill_selection": {
                "lr": self.lr * 0.6,
                "kl_coeff": min(self.kl_coeff, 0.02),
                "epochs": 3,
            },
            "action_taking": {
                "lr": self.lr,
                "kl_coeff": self.kl_coeff,
                "epochs": 2,
            },
        }

        for adapter_name, cfg in adapter_configs.items():
            recs = records.get(adapter_name, [])
            if not recs:
                logger.info("No GRPO records for '%s', skipping", adapter_name)
                continue

            prompts, completions, advantages = _records_to_training_data(recs)
            if not prompts:
                continue

            adapter_path = str(Path(self.adapter_dir) / adapter_name)
            logger.info(
                "Decision GRPO [%s]: %d samples on %d GPUs",
                adapter_name, len(prompts), len(self.devices),
            )

            stats = run_fsdp_grpo(
                gpu_ids=self.devices,
                model_name=self.model_name,
                adapter_dir=adapter_path,
                adapter_name=adapter_name,
                prompts=prompts,
                completions=completions,
                advantages=advantages,
                lr=cfg["lr"],
                epochs=cfg["epochs"],
                batch_size=8,
                clip_ratio=0.2,
                kl_coeff=cfg["kl_coeff"],
                save_dir=adapter_path,
                io_log_dir=self.io_log_dir,
            )

            result[adapter_name] = GRPOTrainStats(
                adapter=adapter_name,
                n_samples=stats.get("n_samples", 0),
                n_tokens=stats.get("n_tokens", 0),
                mean_loss=stats.get("mean_loss", 0.0),
                epochs=stats.get("epochs", 0),
                wall_time_s=stats.get("wall_time_s", 0.0),
            )

        return result

    def save_adapters(self) -> None:
        pass  # FSDP workers save directly

    def cleanup(self) -> None:
        pass  # FSDP workers clean up their own GPU memory


# ── Skill Bank Trainer ──────────────────────────────────────────────────


class SkillBankGRPOTrainer:
    """GRPO trainer for skill bank LoRAs (segment, contract, curator).

    Launches FSDP training across all specified GPUs for each adapter
    sequentially.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        adapter_dir: str = "runs/lora_adapters",
        devices: Optional[List[int]] = None,
        lr: float = 5e-5,
        temperature: float = 0.7,
        kl_coeff: float = 0.05,
        io_log_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.devices = devices or [4, 5, 6, 7]
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.io_log_dir = io_log_dir

    def train_step(
        self,
        grpo_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, GRPOTrainStats]:
        """Run FSDP GRPO for each skill bank adapter that has data."""
        from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo

        result: Dict[str, GRPOTrainStats] = {}

        adapter_configs = {
            "segment": {
                "lr": self.lr * 0.6,
                "kl_coeff": min(self.kl_coeff, 0.02),
                "epochs": 3,
            },
            "contract": {
                "lr": self.lr,
                "kl_coeff": self.kl_coeff,
                "epochs": 2,
            },
            "curator": {
                "lr": self.lr,
                "kl_coeff": self.kl_coeff,
                "epochs": 2,
            },
        }

        for adapter_name, cfg in adapter_configs.items():
            samples = grpo_data.get(adapter_name, [])
            if not samples:
                logger.info("No GRPO data for '%s', skipping", adapter_name)
                continue

            prompts, completions, advantages = _samples_to_training_data(samples)
            if not prompts:
                continue

            adapter_path = str(Path(self.adapter_dir) / adapter_name)
            logger.info(
                "SkillBank GRPO [%s]: %d samples on %d GPUs",
                adapter_name, len(prompts), len(self.devices),
            )

            stats = run_fsdp_grpo(
                gpu_ids=self.devices,
                model_name=self.model_name,
                adapter_dir=adapter_path,
                adapter_name=adapter_name,
                prompts=prompts,
                completions=completions,
                advantages=advantages,
                lr=cfg["lr"],
                epochs=cfg["epochs"],
                batch_size=8,
                clip_ratio=0.2,
                kl_coeff=cfg["kl_coeff"],
                save_dir=adapter_path,
                io_log_dir=self.io_log_dir,
            )

            result[adapter_name] = GRPOTrainStats(
                adapter=adapter_name,
                n_samples=stats.get("n_samples", 0),
                n_tokens=stats.get("n_tokens", 0),
                mean_loss=stats.get("mean_loss", 0.0),
                epochs=stats.get("epochs", 0),
                wall_time_s=stats.get("wall_time_s", 0.0),
            )

        return result

    def save_adapters(self) -> None:
        pass  # FSDP workers save directly

    def cleanup(self) -> None:
        pass  # FSDP workers clean up their own GPU memory


# ── Top-level entry point ───────────────────────────────────────────────


async def run_grpo_training(
    rollout_results: List[EpisodeResult],
    skillbank_grpo_data: Dict[str, List[Dict[str, Any]]],
    config: Any,
    *,
    step: int = 0,
    executor=None,
) -> GRPOStepResult:
    """Run GRPO training for both decision agent and skill bank.

    Decision and skill bank adapters train **sequentially** on the
    same GPUs using FSDP data parallelism.  Each adapter spawns N
    worker processes (one per GPU) that load the model shard, train
    on their data slice, and exit.

    Parameters
    ----------
    step : int
        Current co-evolution step (used for learning rate /
        temperature schedule).
    executor : ThreadPoolExecutor | None
        Used to run blocking FSDP training off the event loop.
    """
    import asyncio

    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    sched = config.grpo_schedule(step)
    lr = sched["lr"]
    temperature = sched["temperature"]
    kl_coeff = sched["kl_coeff"]
    logger.info(
        "GRPO step %d schedule: lr=%.2e, temp=%.2f, kl=%.3f",
        step, lr, temperature, kl_coeff,
    )

    devices = config.effective_grpo_devices
    io_dir = config.debug_io_dir if getattr(config, "debug_io", False) else None

    train_results = [r for r in rollout_results if not r.eval_only]
    decision_records = _collect_grpo_records(train_results)
    has_decision_data = any(len(v) > 0 for v in decision_records.values())
    has_skillbank_data = any(len(v) > 0 for v in skillbank_grpo_data.values())

    decision_stats: Dict[str, GRPOTrainStats] = {}
    skillbank_stats: Dict[str, GRPOTrainStats] = {}

    # Run decision and skillbank SEQUENTIALLY (both use all GPUs via FSDP)

    if has_decision_data:
        logger.info("Phase C.1: Decision agent GRPO on GPUs %s", devices)
        trainer = DecisionGRPOTrainer(
            model_name=config.model_name,
            adapter_dir=config.decision_adapter_dir,
            devices=devices,
            lr=lr,
            temperature=temperature,
            kl_coeff=kl_coeff,
            io_log_dir=io_dir,
        )
        decision_stats = await loop.run_in_executor(
            executor, trainer.train_step, decision_records,
        )

    if has_skillbank_data:
        logger.info("Phase C.2: Skill bank GRPO on GPUs %s", devices)
        trainer = SkillBankGRPOTrainer(
            model_name=config.model_name,
            adapter_dir=config.skillbank_adapter_dir,
            devices=devices,
            lr=lr,
            temperature=temperature,
            kl_coeff=kl_coeff,
            io_log_dir=io_dir,
        )
        skillbank_stats = await loop.run_in_executor(
            executor, trainer.train_step, skillbank_grpo_data,
        )

    # Collect serializable records for disk export
    all_records: Dict[str, List[Dict[str, Any]]] = {}
    for adapter_name, recs in decision_records.items():
        all_records[adapter_name] = [
            {"prompt": r.prompt, "completion": r.completion,
             "reward": r.reward, "episode_id": r.episode_id,
             "step": r.step, "adapter": r.adapter}
            for r in recs
        ]
    for adapter_name, samples in skillbank_grpo_data.items():
        all_records[adapter_name] = list(samples)

    elapsed = time.monotonic() - t0
    return GRPOStepResult(
        decision_stats=decision_stats,
        skillbank_stats=skillbank_stats,
        wall_time_s=elapsed,
        records=all_records,
    )

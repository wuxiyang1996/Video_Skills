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
import math
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
    finite = [r for r in rewards if math.isfinite(r)]
    if not finite:
        return [0.0] * len(rewards)
    fallback = sum(finite) / len(finite)
    sanitized = [r if math.isfinite(r) else fallback for r in rewards]
    n = len(sanitized)
    if n == 1:
        return [0.0]
    mean = sum(sanitized) / n
    var = sum((r - mean) ** 2 for r in sanitized) / n
    std = var ** 0.5 if var > 0 else 1.0
    return [(r - mean) / std for r in sanitized]


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

    Advantages are computed per-episode (all steps within the same
    episode form a group).  Within an episode, steps with above-average
    rewards get positive advantages, below-average get negative.

    This replaces the earlier per-(episode, step) grouping which always
    produced groups of size 1 and advantages of exactly 0.0.
    """
    by_episode: Dict[str, List[GRPORecord]] = {}
    for rec in records:
        by_episode.setdefault(rec.episode_id, []).append(rec)

    prompts: List[str] = []
    completions: List[str] = []
    advantages: List[float] = []

    for ep_records in by_episode.values():
        rewards = [r.reward for r in ep_records]
        advs = _compute_advantages(rewards)
        for rec, adv in zip(ep_records, advs):
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
        model_name: str = "Qwen/Qwen3-8B",
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

    def build_jobs(
        self,
        records: Dict[str, List[GRPORecord]],
    ) -> tuple:
        """Build FSDP job dicts for all decision adapters.

        Returns ``(jobs, names)`` without launching training.
        """
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

        jobs: List[Dict] = []
        job_names: List[str] = []

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

            jobs.append({
                "adapter_dir": adapter_path,
                "adapter_name": adapter_name,
                "prompts": prompts,
                "completions": completions,
                "advantages": advantages,
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "batch_size": 32,
                "clip_ratio": 0.2,
                "kl_coeff": cfg["kl_coeff"],
                "save_dir": adapter_path,
            })
            job_names.append(adapter_name)

        return jobs, job_names

    def train_step(
        self,
        records: Dict[str, List[GRPORecord]],
    ) -> Dict[str, GRPOTrainStats]:
        """Run FSDP GRPO for all decision adapters in a single spawn."""
        from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo_multi

        jobs, job_names = self.build_jobs(records)
        if not jobs:
            return {}

        all_stats = run_fsdp_grpo_multi(
            gpu_ids=self.devices,
            model_name=self.model_name,
            jobs=jobs,
            io_log_dir=self.io_log_dir,
        )

        result: Dict[str, GRPOTrainStats] = {}
        for name, stats in zip(job_names, all_stats):
            result[name] = GRPOTrainStats(
                adapter=name,
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
        model_name: str = "Qwen/Qwen3-8B",
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

    def build_jobs(
        self,
        grpo_data: Dict[str, List[Dict[str, Any]]],
    ) -> tuple:
        """Build FSDP job dicts for all skill bank adapters.

        Returns ``(jobs, names)`` without launching training.
        """
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

        jobs: List[Dict] = []
        job_names: List[str] = []

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

            jobs.append({
                "adapter_dir": adapter_path,
                "adapter_name": adapter_name,
                "prompts": prompts,
                "completions": completions,
                "advantages": advantages,
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "batch_size": 32,
                "clip_ratio": 0.2,
                "kl_coeff": cfg["kl_coeff"],
                "save_dir": adapter_path,
            })
            job_names.append(adapter_name)

        return jobs, job_names

    def train_step(
        self,
        grpo_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, GRPOTrainStats]:
        """Run FSDP GRPO for all skill bank adapters in a single spawn."""
        from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo_multi

        jobs, job_names = self.build_jobs(grpo_data)
        if not jobs:
            return {}

        all_stats = run_fsdp_grpo_multi(
            gpu_ids=self.devices,
            model_name=self.model_name,
            jobs=jobs,
            io_log_dir=self.io_log_dir,
        )

        result: Dict[str, GRPOTrainStats] = {}
        for name, stats in zip(job_names, all_stats):
            result[name] = GRPOTrainStats(
                adapter=name,
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

    All adapters (decision + skill bank) are trained in a **single**
    FSDP process spawn.  The persistent model is loaded once and LoRA
    weights are swapped between adapters, eliminating redundant model
    loads (~20-30 s saved per step).

    Parameters
    ----------
    step : int
        Current co-evolution step (used for learning rate /
        temperature schedule).
    executor : ThreadPoolExecutor | None
        Used to run blocking FSDP training off the event loop.
    """
    import asyncio
    import functools

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

    # ── Build unified job list from both trainers ──
    all_jobs: List[Dict] = []
    job_categories: List[tuple] = []

    decision_trainer = DecisionGRPOTrainer(
        model_name=config.model_name,
        adapter_dir=config.decision_adapter_dir,
        devices=devices,
        lr=lr,
        temperature=temperature,
        kl_coeff=kl_coeff,
        io_log_dir=io_dir,
    )
    d_jobs, d_names = decision_trainer.build_jobs(decision_records)
    all_jobs.extend(d_jobs)
    job_categories.extend(("decision", n) for n in d_names)

    skillbank_trainer = SkillBankGRPOTrainer(
        model_name=config.model_name,
        adapter_dir=config.skillbank_adapter_dir,
        devices=devices,
        lr=lr,
        temperature=temperature,
        kl_coeff=kl_coeff,
        io_log_dir=io_dir,
    )
    s_jobs, s_names = skillbank_trainer.build_jobs(skillbank_grpo_data)
    all_jobs.extend(s_jobs)
    job_categories.extend(("skillbank", n) for n in s_names)

    decision_stats: Dict[str, GRPOTrainStats] = {}
    skillbank_stats: Dict[str, GRPOTrainStats] = {}

    # ── Single FSDP spawn for all adapters ──
    if all_jobs:
        from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo_multi

        logger.info(
            "Phase C: GRPO training %d adapters on GPUs %s",
            len(all_jobs), devices,
        )

        all_stats = await loop.run_in_executor(
            executor,
            functools.partial(
                run_fsdp_grpo_multi,
                gpu_ids=devices,
                model_name=config.model_name,
                jobs=all_jobs,
                io_log_dir=io_dir,
            ),
        )

        for (cat, name), stats in zip(job_categories, all_stats):
            stat = GRPOTrainStats(
                adapter=name,
                n_samples=stats.get("n_samples", 0),
                n_tokens=stats.get("n_tokens", 0),
                mean_loss=stats.get("mean_loss", 0.0),
                epochs=stats.get("epochs", 0),
                wall_time_s=stats.get("wall_time_s", 0.0),
            )
            if cat == "decision":
                decision_stats[name] = stat
            else:
                skillbank_stats[name] = stat

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

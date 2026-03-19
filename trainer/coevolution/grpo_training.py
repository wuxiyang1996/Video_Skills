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
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from trainer.coevolution.episode_runner import EpisodeResult, GRPORecord

logger = logging.getLogger(__name__)


# ── Experience Replay Buffer ────────────────────────────────────────────

class ReplayBuffer:
    """Ring buffer of GRPORecords with importance-weighted staleness.

    Records from the current step get weight 1.0; older records are
    down-weighted so the gradient is dominated by fresh on-policy data
    while still benefiting from past experience.
    """

    STALENESS_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.5}
    DEFAULT_WEIGHT = 0.3

    def __init__(self, max_size: int = 2000):
        self._buf: Deque[Tuple[GRPORecord, int]] = deque(maxlen=max_size)

    def add(self, records: List[GRPORecord], step: int) -> None:
        for rec in records:
            self._buf.append((rec, step))

    def sample_all(self, current_step: int) -> Tuple[List[GRPORecord], List[float]]:
        """Return all records with their staleness weights."""
        records: List[GRPORecord] = []
        weights: List[float] = []
        for rec, rec_step in self._buf:
            age = current_step - rec_step
            w = self.STALENESS_WEIGHTS.get(age, self.DEFAULT_WEIGHT)
            records.append(rec)
            weights.append(w)
        return records, weights

    def __len__(self) -> int:
        return len(self._buf)


_decision_replay_buffers: Dict[str, ReplayBuffer] = {}


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
    # Per-game sample counts: {adapter_name: {game: n_samples}}
    per_game_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────


def _compute_advantages(rewards: List[float]) -> List[float]:
    """Group-normalize rewards (decision-agent path: no per-group completions)."""
    from skill_agents_grpo.grpo.advantage_utils import compute_grpo_group_advantages

    return compute_grpo_group_advantages(rewards, completions=None)


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
    weights: Optional[List[float]] = None,
) -> tuple:
    """Convert GRPORecords to flat (prompts, completions, advantages) lists.

    Advantages are computed **per-game** so the baseline reflects
    "average performance in this game" rather than "average within
    this single episode".  This fixes two critical failure modes:

    1. **Constant-reward episodes** (e.g. Candy Crush returning 0.5
       every step): per-episode normalization produces advantage=0 for
       every sample, wasting all training data from that game.  Per-game
       normalization uses the cross-episode mean, giving meaningful
       signal when some episodes do better than others.

    2. **Single-step episodes**: per-episode normalization always yields
       advantage=0.  Per-game normalization gives a meaningful value
       relative to other episodes.

    When *weights* are provided (from experience replay), advantages are
    scaled by the importance weight so stale records contribute less.
    """
    if weights is None:
        weights = [1.0] * len(records)

    by_game: Dict[str, List[tuple]] = {}
    for rec, w in zip(records, weights):
        by_game.setdefault(rec.game, []).append((rec, w))

    prompts: List[str] = []
    completions: List[str] = []
    advantages: List[float] = []

    for game, game_entries in sorted(by_game.items()):
        game_records = [e[0] for e in game_entries]
        game_weights = [e[1] for e in game_entries]
        rewards = [r.reward for r in game_records]
        advs = _compute_advantages(rewards)

        n_nonzero = sum(1 for a in advs if abs(a) > 1e-6)
        logger.debug(
            "GRPO advantages [%s]: %d samples, mean_reward=%.4f, "
            "%d/%d non-zero advantages",
            game, len(rewards),
            sum(rewards) / len(rewards) if rewards else 0.0,
            n_nonzero, len(advs),
        )

        for rec, adv, w in zip(game_records, advs, game_weights):
            if rec.completion:
                prompts.append(rec.prompt)
                completions.append(rec.completion)
                advantages.append(adv * w)

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
        from skill_agents_grpo.grpo.advantage_utils import compute_grpo_group_advantages

        advs = compute_grpo_group_advantages(rewards, completions=comps)
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

    _MIN_SAMPLES = 16

    def build_jobs(
        self,
        records: Dict[str, List[GRPORecord]],
        step: int = 0,
    ) -> tuple:
        """Build FSDP job dicts for all decision adapters.

        Fresh records are added to per-adapter replay buffers, then
        training data is drawn from the full buffer with staleness
        weights so past experience is down-weighted.  Adapters with
        fewer than ``_MIN_SAMPLES`` total are skipped.  Epoch count
        scales inversely with sample count so small datasets get more
        passes.

        Returns ``(jobs, names)`` without launching training.
        """
        global _decision_replay_buffers

        adapter_configs = {
            "skill_selection": {
                "lr": self.lr * 0.6,
                "kl_coeff": min(self.kl_coeff, 0.02),
                "base_epochs": 3,
            },
            "action_taking": {
                "lr": self.lr,
                "kl_coeff": self.kl_coeff,
                "base_epochs": 2,
            },
        }

        jobs: List[Dict] = []
        job_names: List[str] = []

        for adapter_name, cfg in adapter_configs.items():
            fresh_recs = records.get(adapter_name, [])

            if adapter_name not in _decision_replay_buffers:
                _decision_replay_buffers[adapter_name] = ReplayBuffer(max_size=2000)
            buf = _decision_replay_buffers[adapter_name]

            if fresh_recs:
                buf.add(fresh_recs, step)

            all_recs, weights = buf.sample_all(step)
            if not all_recs:
                logger.info("No GRPO records for '%s', skipping", adapter_name)
                continue

            if len(all_recs) < self._MIN_SAMPLES:
                logger.warning(
                    "Decision GRPO [%s]: only %d samples (< %d min), skipping",
                    adapter_name, len(all_recs), self._MIN_SAMPLES,
                )
                continue

            prompts, completions, advantages = _records_to_training_data(
                all_recs, weights=weights,
            )
            if not prompts:
                continue

            n_samples = len(prompts)
            epochs = max(2, min(8, 256 // n_samples))

            from collections import Counter
            game_counts = Counter(r.game for r in all_recs)
            game_breakdown = ", ".join(
                f"{g}={n}" for g, n in sorted(game_counts.items())
            )
            n_fresh = len(fresh_recs)
            n_replay = len(all_recs) - n_fresh

            adapter_path = str(Path(self.adapter_dir) / adapter_name)
            logger.info(
                "Decision GRPO [%s]: %d samples (%d fresh + %d replay), "
                "%d epochs, on %d GPUs (%s)",
                adapter_name, n_samples, n_fresh, n_replay,
                epochs, len(self.devices), game_breakdown,
            )

            jobs.append({
                "adapter_dir": adapter_path,
                "adapter_name": adapter_name,
                "prompts": prompts,
                "completions": completions,
                "advantages": advantages,
                "lr": cfg["lr"],
                "epochs": epochs,
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


_skillbank_accum: Dict[str, List[Dict[str, Any]]] = {
    "segment": [],
    "contract": [],
    "curator": [],
}

_SKILLBANK_TRAIN_THRESHOLD = int(
    os.environ.get("SKILLBANK_TRAIN_THRESHOLD", "32")
)
_SKILLBANK_MAX_ACCUM = 512


class SkillBankGRPOTrainer:
    """GRPO trainer for skill bank LoRAs (segment, contract, curator).

    Uses a **cross-step accumulation buffer** (module-level) so that
    sparse skill-bank data pools across co-evolution steps.  Training
    fires only when an adapter has accumulated enough samples
    (>= ``_SKILLBANK_TRAIN_THRESHOLD``), then the buffer is drained.
    This avoids the problem of either gating out adapters that rarely
    produce data (curator, contract) or overfitting tiny batches.
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
        """Build FSDP job dicts for skill bank adapters with enough data.

        New samples from ``grpo_data`` are appended to the persistent
        cross-step accumulation buffer.  An adapter is trained only when
        its buffer reaches ``_SKILLBANK_TRAIN_THRESHOLD``; after
        training, the consumed samples are removed.

        Returns ``(jobs, names)`` without launching training.
        """
        global _skillbank_accum

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

        for adapter_name in adapter_configs:
            new_samples = grpo_data.get(adapter_name, [])
            if new_samples:
                _skillbank_accum[adapter_name].extend(new_samples)
                if len(_skillbank_accum[adapter_name]) > _SKILLBANK_MAX_ACCUM:
                    _skillbank_accum[adapter_name] = _skillbank_accum[adapter_name][-_SKILLBANK_MAX_ACCUM:]

        jobs: List[Dict] = []
        job_names: List[str] = []

        for adapter_name, cfg in adapter_configs.items():
            buf = _skillbank_accum[adapter_name]
            n_new = len(grpo_data.get(adapter_name, []))
            n_total = len(buf)

            if n_total == 0:
                continue

            if n_total < _SKILLBANK_TRAIN_THRESHOLD:
                logger.info(
                    "SkillBank GRPO [%s]: %d accumulated (+%d new), "
                    "waiting for %d — deferring",
                    adapter_name, n_total, n_new,
                    _SKILLBANK_TRAIN_THRESHOLD,
                )
                continue

            prompts, completions, advantages = _samples_to_training_data(buf)
            if not prompts:
                continue

            adapter_path = str(Path(self.adapter_dir) / adapter_name)
            logger.info(
                "SkillBank GRPO [%s]: training on %d accumulated samples "
                "(+%d new this step) on %d GPUs",
                adapter_name, n_total, n_new, len(self.devices),
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

            _skillbank_accum[adapter_name] = []

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
    d_jobs, d_names = decision_trainer.build_jobs(decision_records, step=step)
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
    per_game_counts: Dict[str, Dict[str, int]] = {}

    for adapter_name, recs in decision_records.items():
        all_records[adapter_name] = [
            {"prompt": r.prompt, "completion": r.completion,
             "reward": r.reward, "episode_id": r.episode_id,
             "step": r.step, "adapter": r.adapter, "game": r.game}
            for r in recs
        ]
        from collections import Counter
        per_game_counts[adapter_name] = dict(Counter(r.game for r in recs))

    for adapter_name, samples in skillbank_grpo_data.items():
        all_records[adapter_name] = list(samples)

    elapsed = time.monotonic() - t0
    return GRPOStepResult(
        decision_stats=decision_stats,
        skillbank_stats=skillbank_stats,
        wall_time_s=elapsed,
        records=all_records,
        per_game_counts=per_game_counts,
    )

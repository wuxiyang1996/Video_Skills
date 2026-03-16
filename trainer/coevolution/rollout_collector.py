"""Rollout collector with LPT scheduling and async concurrency control.

Collects episodes across all games concurrently, using
Longest-Processing-Time (LPT) ordering to start the slowest games first
and a semaphore to cap GPU-memory pressure from concurrent LLM requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from trainer.coevolution.config import (
    CoEvolutionConfig,
    GAME_DURATION_ORDER,
    GAME_MAX_STEPS,
)
from trainer.coevolution.episode_runner import EpisodeResult, run_episode_async
from trainer.coevolution.vllm_client import AsyncVLLMClient

logger = logging.getLogger(__name__)


@dataclass
class EpisodeSpec:
    """One episode to run: game name + settings."""
    game: str
    max_steps: int
    episode_idx: int
    estimated_duration_s: float = 0.0
    eval_only: bool = False


def build_lpt_schedule(
    games: List[str],
    episodes_per_game: int,
) -> List[EpisodeSpec]:
    """Build an LPT-ordered list of episode specs.

    Strategy: sort games by descending max_steps (proxy for duration), then
    interleave episodes round-robin so long-game episodes are spread across
    the entire collection window (enabling cross-system overlap with the
    skill bank pipeline).
    """
    sorted_games = sorted(
        games,
        key=lambda g: GAME_MAX_STEPS.get(g, 200),
        reverse=True,
    )

    PER_STEP_S = 1.0
    buckets: Dict[str, List[EpisodeSpec]] = {}
    for g in sorted_games:
        ms = GAME_MAX_STEPS.get(g, 200)
        est = ms * PER_STEP_S
        buckets[g] = [
            EpisodeSpec(game=g, max_steps=ms, episode_idx=i, estimated_duration_s=est)
            for i in range(episodes_per_game)
        ]

    schedule: List[EpisodeSpec] = []
    for ep_idx in range(episodes_per_game):
        for g in sorted_games:
            if ep_idx < len(buckets[g]):
                schedule.append(buckets[g][ep_idx])

    return schedule


async def collect_rollouts(
    config: CoEvolutionConfig,
    vllm_client: AsyncVLLMClient,
    skill_bank: Any = None,
    skill_banks: Optional[Dict[str, Any]] = None,
    *,
    on_episode_done: Optional[Callable[[EpisodeResult], None]] = None,
    thread_executor: Optional[ThreadPoolExecutor] = None,
) -> List[EpisodeResult]:
    """Collect rollouts for all games with LPT scheduling and concurrency cap.

    Parameters
    ----------
    skill_bank : object | None
        Legacy single shared bank (used if *skill_banks* is not provided).
    skill_banks : dict | None
        Per-game banks: ``{game_name: bank_object}``.  Takes priority
        over *skill_bank* when both are provided.
    on_episode_done : callable | None
        Called (synchronously) each time an episode finishes.  Used by the
        orchestrator to feed completed trajectories into the skill bank
        pipeline concurrently (cross-system overlap, Strategy E).
    """
    schedule = build_lpt_schedule(config.games, config.episodes_per_game)

    eval_games = getattr(config, "eval_games", [])
    eval_eps = getattr(config, "eval_episodes_per_game", 3)
    if eval_games:
        eval_schedule = build_lpt_schedule(eval_games, eval_eps)
        for spec in eval_schedule:
            spec.eval_only = True
        schedule.extend(eval_schedule)

    semaphore = asyncio.Semaphore(config.max_concurrent_episodes)

    results: List[EpisodeResult] = []
    results_lock = asyncio.Lock()

    def _bank_for(game: str) -> Any:
        if skill_banks:
            return skill_banks.get(game)
        return skill_bank

    async def _run_one(spec: EpisodeSpec) -> None:
        async with semaphore:
            game_bank = _bank_for(spec.game)
            try:
                result = await run_episode_async(
                    game=spec.game,
                    max_steps=spec.max_steps,
                    vllm_client=vllm_client,
                    skill_bank=game_bank,
                    temperature=config.temperature,
                    executor=thread_executor,
                    stuck_window=config.stuck_window,
                    min_steps_before_stuck=config.min_steps_before_stuck_check,
                )
            except Exception as exc:
                logger.error("Episode %s/%d failed: %s", spec.game, spec.episode_idx, exc)
                result = EpisodeResult(
                    game=spec.game,
                    episode_id=f"{spec.game}_FAILED_{spec.episode_idx}",
                )

            result.eval_only = spec.eval_only
            async with results_lock:
                results.append(result)

            if on_episode_done is not None:
                try:
                    on_episode_done(result)
                except Exception as exc:
                    logger.warning("on_episode_done callback failed: %s", exc)

    t0 = time.monotonic()
    logger.info(
        "Collecting rollouts: %d episodes (%d games × %d eps), max_concurrent=%d",
        len(schedule), len(config.games), config.episodes_per_game,
        config.max_concurrent_episodes,
    )

    tasks = [asyncio.create_task(_run_one(spec)) for spec in schedule]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    n_ok = sum(1 for r in results if r.steps > 0)
    total_steps = sum(r.steps for r in results)
    total_reward = sum(r.total_reward for r in results)

    logger.info(
        "Rollout collection done: %d/%d ok, %d total steps, %.1f total reward, "
        "%.1fs wall time",
        n_ok, len(results), total_steps, total_reward, elapsed,
    )

    return results


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def compute_episode_metrics(results: List[EpisodeResult]) -> Dict[str, Any]:
    """Aggregate episode-level metrics for logging."""
    from collections import defaultdict

    per_game: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "episodes": 0, "total_reward": 0.0, "total_steps": 0,
        "rewards": [], "step_counts": [],
    })

    for r in results:
        if r.steps == 0:
            continue
        g = per_game[r.game]
        g["episodes"] += 1
        g["total_reward"] += r.total_reward
        g["total_steps"] += r.steps
        g["rewards"].append(r.total_reward)
        g["step_counts"].append(r.steps)

    summary: Dict[str, Any] = {"per_game": {}}
    for game, g in per_game.items():
        rewards = g["rewards"]
        summary["per_game"][game] = {
            "n_episodes": g["episodes"],
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "std_reward": _std(rewards),
            "mean_steps": sum(g["step_counts"]) / len(g["step_counts"]) if g["step_counts"] else 0,
            "total_reward": g["total_reward"],
        }

    all_rewards = [r.total_reward for r in results if r.steps > 0]
    summary["aggregate"] = {
        "n_episodes": len(all_rewards),
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "max_reward": max(all_rewards) if all_rewards else 0.0,
        "min_reward": min(all_rewards) if all_rewards else 0.0,
        "std_reward": _std(all_rewards),
        "total_steps": sum(r.steps for r in results),
        "wall_time_s": max(r.wall_time_s for r in results) if results else 0.0,
    }

    return summary

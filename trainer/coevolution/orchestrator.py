"""Co-evolution orchestrator — the main training loop.

Implements the three-phase co-evolution step:
  Phase A — Rollout collection (decision agent plays all games)
  Phase B — Skill bank update (segment, contracts, maintenance)
  Phase C — GRPO training (5 LoRA adapters across two agents)

Phases A and B overlap via cross-system batching: as short-game episodes
complete, their trajectories are immediately fed into the skill bank
pipeline while longer games continue running.

Checkpoints every ``checkpoint_interval`` steps and logs all metrics
to Weights & Biases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.coevolution.checkpoint import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from trainer.coevolution.config import CoEvolutionConfig, prepare_adapters
from trainer.coevolution.episode_runner import EpisodeResult
from trainer.coevolution.grpo_training import GRPOStepResult, run_grpo_training
from trainer.coevolution.rollout_collector import (
    collect_rollouts,
    compute_episode_metrics,
)
from trainer.coevolution.skillbank_pipeline import (
    AsyncSkillBankPipeline,
    PerGameSkillBankManager,
    SkillBankUpdateResult,
)
from trainer.coevolution.vllm_client import AsyncVLLMClient

logger = logging.getLogger(__name__)


async def co_evolution_loop(config: CoEvolutionConfig) -> None:
    """Main co-evolution training loop.

    Step 0:  Rollouts (no bank) → Skill bank extraction → Bank_v1
    Step 1+: Rollouts (with bank) → Skill bank update → Bank_v{k+1}
             + GRPO updates for all 5 LoRAs

    Checkpoints saved every ``checkpoint_interval`` steps.
    All metrics logged to W&B in real time.
    """
    # ── Resolve all paths under run_dir ───────────────────────────
    config.resolve_paths()
    logger.info("Run directory: %s", config.run_dir)

    # ── Ensure LoRA adapters exist ──────────────────────────────────
    adapter_map = prepare_adapters(config)
    logger.info("Adapters ready: %s", list(adapter_map.keys()))

    # ── Initialize W&B ────────────────────────────────────────────
    wandb = None
    if config.wandb_enabled:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.to_dict(),
                resume="allow",
            )
            logger.info("W&B initialized: project=%s", config.wandb_project)
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)
            wandb = None

    # ── vLLM lifecycle manager (persistent on dedicated GPUs) ────
    vllm_manager = None
    if config.manage_vllm:
        from trainer.coevolution.vllm_server import VLLMServerManager

        vllm_log_dir = str(Path(config.run_dir) / "vllm_logs")
        vllm_manager = VLLMServerManager(
            model_name=config.model_name,
            adapter_dir=config.adapter_dir,
            gpu_ids=config.vllm_gpu_ids,
            base_port=config.vllm_base_port,
            gpu_util=config.vllm_gpu_util,
            log_dir=vllm_log_dir,
        )
        logger.info(
            "vLLM managed mode: %d × TP=1 instances on GPUs %s, "
            "ports %d–%d  |  GRPO on GPUs %s",
            len(config.vllm_gpu_ids), config.vllm_gpu_ids,
            config.vllm_base_port,
            config.vllm_base_port + len(config.vllm_gpu_ids) - 1,
            config.grpo_devices,
        )

    # ── vLLM client (supports multiple backends) ──────────────
    vllm_client = AsyncVLLMClient(
        base_urls=config.vllm_base_urls,
        model=config.model_name,
        default_temperature=config.temperature,
        default_max_tokens=config.max_tokens,
    )

    if config.debug_io:
        vllm_client.enable_io_logging(config.debug_io_dir, step=0)
        logger.info("Debug I/O logging enabled: %s", config.debug_io_dir)

    # For externally-managed vLLM, verify reachability upfront
    if not vllm_manager:
        healthy = await vllm_client.health_check()
        if not healthy:
            logger.error("vLLM server not reachable at %s", config.vllm_base_url)
            raise ConnectionError(f"vLLM not reachable: {config.vllm_base_url}")
        logger.info("vLLM server healthy at %s", config.vllm_base_url)

    # ── Executors ─────────────────────────────────────────────────
    thread_executor = ThreadPoolExecutor(
        max_workers=config.thread_workers, thread_name_prefix="coevo",
    )

    # ── Per-game skill bank pipelines ────────────────────────────
    all_games = list(config.games) + list(getattr(config, "eval_games", []))
    sb_manager = PerGameSkillBankManager(
        games=all_games,
        bank_dir=config.bank_dir,
        model_name=config.model_name,
        executor=thread_executor,
    )

    # ── Determine start step ─────────────────────────────────────
    start_step = 0

    if config.start_mode == "from_scratch":
        logger.info("Starting from scratch — ignoring any existing checkpoints")

    elif config.start_mode == "resume":
        if config.resume_from_step is not None:
            start_step = config.resume_from_step
            try:
                metadata = load_checkpoint(
                    config.checkpoint_dir, start_step,
                    adapter_dir=config.adapter_dir,
                    bank_agents=sb_manager.get_agents(),
                )
                logger.info("Resumed from checkpoint step %d", start_step)
                start_step += 1
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"--resume-from-step {start_step}: checkpoint not found "
                    f"in {config.checkpoint_dir}"
                )
        else:
            latest = find_latest_checkpoint(config.checkpoint_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"--resume requested but no checkpoint found "
                    f"in {config.checkpoint_dir}"
                )
            metadata = load_checkpoint(
                config.checkpoint_dir, latest,
                adapter_dir=config.adapter_dir,
                bank_agents=sb_manager.get_agents(),
            )
            start_step = latest + 1
            logger.info("Resumed from latest checkpoint step %d", latest)

    else:  # auto
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            try:
                metadata = load_checkpoint(
                    config.checkpoint_dir, latest,
                    adapter_dir=config.adapter_dir,
                    bank_agents=sb_manager.get_agents(),
                )
                start_step = latest + 1
                logger.info("Auto-resumed from checkpoint step %d", latest)
            except Exception:
                logger.warning("Auto-resume failed, starting from step 0")
                start_step = 0
        else:
            logger.info("No checkpoint found, starting from step 0")

    # ── Ensure output directories ─────────────────────────────────
    dirs_to_create = [
        config.bank_dir, config.adapter_dir,
        config.decision_adapter_dir, config.skillbank_adapter_dir,
        config.checkpoint_dir,
        config.log_dir, config.grpo_data_dir, config.rewards_dir,
        config.tensorboard_dir,
    ]
    if config.debug_io:
        dirs_to_create.append(config.debug_io_dir)
    for d in dirs_to_create:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Initialize TensorBoard ────────────────────────────────────
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=config.tensorboard_dir)
        logger.info("TensorBoard initialized: %s", config.tensorboard_dir)
    except ImportError:
        logger.warning("torch.utils.tensorboard not available, TensorBoard disabled")
    except Exception as exc:
        logger.warning("TensorBoard init failed: %s", exc)

    # ── Persist full config snapshot ──────────────────────────────
    config_path = Path(config.log_dir) / "config.json"
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        logger.info("Config saved: %s", config_path)
    except Exception as exc:
        logger.warning("Config save failed: %s", exc)

    # ── Step history for logging ──────────────────────────────────
    step_log_path = Path(config.log_dir) / "step_log.jsonl"

    logger.info(
        "Starting co-evolution loop: steps %d→%d, %d games × %d eps/game",
        start_step, config.total_steps - 1,
        len(config.games), config.episodes_per_game,
    )

    # ── Start persistent vLLM instances (once) ─────────────────────
    if vllm_manager:
        logger.info(
            "Starting %d persistent vLLM instances...",
            vllm_manager.n_instances,
        )
        vllm_manager.start()
        healthy = await vllm_manager.wait_healthy()
        if not healthy:
            raise RuntimeError(
                "vLLM instances failed to start — check "
                f"{Path(config.run_dir) / 'vllm_logs'} for details"
            )

    # ==================================================================
    # MAIN LOOP
    # ==================================================================
    for step in range(start_step, config.total_steps):
        step_t0 = time.monotonic()
        is_cold_start = (step == 0)

        skill_banks = sb_manager.get_banks()
        skill_counts = sb_manager.skill_counts()
        n_total_skills = sum(skill_counts.values())
        bank_available = n_total_skills > 0
        mode = "cold-start" if is_cold_start or not bank_available else "warm"

        per_game_str = ", ".join(
            f"{g}={c}" for g, c in skill_counts.items() if c > 0
        ) or "empty"
        logger.info(
            "═══ Step %d/%d [%s] ═══ bank=%d skills (%s) ═══",
            step, config.total_steps - 1, mode,
            n_total_skills, per_game_str,
        )

        sb_manager.reset_for_step()
        vllm_client.reset_stats()
        if config.debug_io:
            vllm_client.set_io_step(step)

        # ── Phase A + B: Rollout collection with cross-system overlap ──
        phase_ab_t0 = time.monotonic()

        completed_queue: asyncio.Queue[EpisodeResult] = asyncio.Queue()

        def on_episode_done(result: EpisodeResult) -> None:
            completed_queue.put_nowait(result)

        async def skill_bank_consumer() -> int:
            """Consume completed episodes and feed to skill bank pipeline.

            Greedily drains the queue after each blocking wait so that
            episodes which completed during segmentation are batched together
            instead of being dequeued one-by-one with 5 s gaps.
            """
            n_processed = 0
            batch: List[EpisodeResult] = []
            sentinel_seen = False
            while not sentinel_seen:
                # Block until at least one item arrives (or timeout)
                try:
                    result = await asyncio.wait_for(
                        completed_queue.get(), timeout=5.0,
                    )
                    if result.game == "__SENTINEL__":
                        sentinel_seen = True
                    elif result.steps > 0:
                        batch.append(result)
                except asyncio.TimeoutError:
                    pass

                # Drain all remaining items without waiting
                while not completed_queue.empty():
                    try:
                        result = completed_queue.get_nowait()
                        if result.game == "__SENTINEL__":
                            sentinel_seen = True
                        elif result.steps > 0:
                            batch.append(result)
                    except asyncio.QueueEmpty:
                        break

                # Process when batch is big enough or we're done
                if len(batch) >= config.em_micro_batch_size or (sentinel_seen and batch):
                    logger.info(
                        "Skill bank consumer: processing %d episodes "
                        "(total so far: %d)",
                        len(batch), n_processed + len(batch),
                    )
                    await sb_manager.process_batch_async(batch)
                    n_processed += len(batch)
                    batch = []

            # Flush any remaining partial batch
            if batch:
                logger.info(
                    "Skill bank consumer: flushing final %d episodes "
                    "(total: %d)",
                    len(batch), n_processed + len(batch),
                )
                await sb_manager.process_batch_async(batch)
                n_processed += len(batch)

            logger.info("Skill bank consumer finished: %d episodes processed", n_processed)
            return n_processed

        rollout_task = asyncio.create_task(
            collect_rollouts(
                config, vllm_client,
                skill_banks=skill_banks if bank_available else None,
                on_episode_done=on_episode_done,
                thread_executor=thread_executor,
            )
        )
        consumer_task = asyncio.create_task(skill_bank_consumer())

        rollout_results: List[EpisodeResult] = await rollout_task

        # Signal consumer to drain remaining
        completed_queue.put_nowait(
            EpisodeResult(game="__SENTINEL__", episode_id="__SENTINEL__", steps=0)
        )
        n_consumed = await consumer_task

        phase_ab_time = time.monotonic() - phase_ab_t0
        logger.info(
            "Phase A+B: %.1fs, %d episodes collected, %d consumed by skill bank",
            phase_ab_time, len(rollout_results), n_consumed,
        )

        # ── Export per-episode rewards ────────────────────────────────
        try:
            rewards_path = Path(config.rewards_dir) / f"step_{step:04d}.jsonl"
            with open(rewards_path, "w", encoding="utf-8") as f:
                for ep in rollout_results:
                    if ep.game == "__SENTINEL__":
                        continue
                    record = {
                        "game": ep.game,
                        "episode_id": ep.episode_id,
                        "steps": ep.steps,
                        "total_reward": ep.total_reward,
                        "terminated": ep.terminated,
                        "eval_only": ep.eval_only,
                        "wall_time_s": ep.wall_time_s,
                    }
                    f.write(json.dumps(record, default=str) + "\n")
            logger.debug("Rewards exported: %s", rewards_path)
        except Exception as exc:
            logger.warning("Rewards export failed: %s", exc)

        # ── Phase B: Finalize all per-game skill bank updates ────────
        phase_b_t0 = time.monotonic()
        sb_update_results: Dict[str, SkillBankUpdateResult] = {}
        try:
            sb_update_results = await sb_manager.finalize_all()
        except Exception as exc:
            logger.error("Skill bank finalize failed: %s", exc)

        total_new_skills = sum(
            r.n_new_skills for r in sb_update_results.values()
        )

        phase_b_time = time.monotonic() - phase_b_t0
        logger.info("Phase B finalize: %.1fs (%d game banks)", phase_b_time, len(sb_update_results))

        # ── Phase C: GRPO training (on dedicated GPUs) ───────────────
        grpo_result: Optional[GRPOStepResult] = None
        phase_c_time = 0.0
        if config.grpo_enabled:
            phase_c_t0 = time.monotonic()
            try:
                grpo_result = await run_grpo_training(
                    rollout_results,
                    sb_manager.grpo_data,
                    config,
                    step=step,
                    executor=thread_executor,
                )
            except Exception as exc:
                logger.error("GRPO training failed: %s", exc, exc_info=True)
                try:
                    import torch, gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cache cleared after GRPO failure")
                except Exception:
                    pass
            phase_c_time = time.monotonic() - phase_c_t0
            logger.info("Phase C (GRPO): %.1fs", phase_c_time)

            # ── Hot-reload adapters on persistent vLLM instances ──────
            if vllm_manager and grpo_result:
                try:
                    await vllm_manager.reload_adapters()
                except Exception as exc:
                    logger.warning("Adapter hot-reload failed: %s", exc)

            # ── Export GRPO training data ─────────────────────────────
            if grpo_result:
                try:
                    grpo_step_dir = Path(config.grpo_data_dir) / f"step_{step:04d}"
                    grpo_step_dir.mkdir(parents=True, exist_ok=True)
                    for adapter_name, records in grpo_result.records.items():
                        out_path = grpo_step_dir / f"{adapter_name}.jsonl"
                        with open(out_path, "w", encoding="utf-8") as f:
                            for rec in records:
                                f.write(json.dumps(rec, default=str) + "\n")
                    logger.debug("GRPO data exported: %s", grpo_step_dir)
                except Exception as exc:
                    logger.warning("GRPO data export failed: %s", exc)

        # ── Metrics ──────────────────────────────────────────────────
        step_elapsed = time.monotonic() - step_t0
        episode_metrics = compute_episode_metrics(rollout_results)

        skill_counts = sb_manager.skill_counts()
        n_skills = sum(skill_counts.values())

        vllm_stats = vllm_client.stats()

        step_summary = {
            "step": step,
            "mode": mode,
            "wall_time_s": step_elapsed,
            "phase_ab_time_s": phase_ab_time,
            "phase_b_finalize_time_s": phase_b_time,
            "phase_c_grpo_time_s": phase_c_time,
            "n_episodes": episode_metrics["aggregate"]["n_episodes"],
            "total_steps_played": episode_metrics["aggregate"]["total_steps"],
            "mean_reward": episode_metrics["aggregate"]["mean_reward"],
            "n_skills": n_skills,
            "n_new_skills": total_new_skills,
            "skills_per_game": skill_counts,
            "vllm_calls": vllm_stats["call_count"],
            "vllm_prompt_tokens": vllm_stats["total_prompt_tokens"],
            "vllm_completion_tokens": vllm_stats["total_completion_tokens"],
        }

        logger.info(
            "Step %d complete: %.1fs | %d eps | mean_reward=%.2f | "
            "%d skills (+%d) across %d games | %d vLLM calls",
            step, step_elapsed,
            step_summary["n_episodes"], step_summary["mean_reward"],
            n_skills, total_new_skills,
            sum(1 for c in skill_counts.values() if c > 0),
            step_summary["vllm_calls"],
        )

        # ── W&B logging ──────────────────────────────────────────────
        if wandb is not None:
            log_dict = {
                "step": step,
                "wall_time_s": step_elapsed,
                "phase_ab_time_s": phase_ab_time,
                "phase_b_finalize_time_s": phase_b_time,
                "phase_c_grpo_time_s": phase_c_time,
                "mode": 0 if mode == "cold-start" else 1,
                "n_skills": n_skills,
                "n_new_skills": total_new_skills,
                "vllm/calls": vllm_stats["call_count"],
                "vllm/prompt_tokens": vllm_stats["total_prompt_tokens"],
                "vllm/completion_tokens": vllm_stats["total_completion_tokens"],
            }

            # Per-game rewards
            for game, m in episode_metrics["per_game"].items():
                log_dict[f"reward/{game}/mean"] = m["mean_reward"]
                log_dict[f"reward/{game}/max"] = m["max_reward"]
                log_dict[f"reward/{game}/min"] = m["min_reward"]
                log_dict[f"reward/{game}/std"] = m["std_reward"]
                log_dict[f"reward/{game}/n_episodes"] = m["n_episodes"]
                log_dict[f"reward/{game}/mean_steps"] = m["mean_steps"]

            # Aggregate reward
            log_dict["reward/mean"] = episode_metrics["aggregate"]["mean_reward"]
            log_dict["reward/max"] = episode_metrics["aggregate"]["max_reward"]
            log_dict["reward/min"] = episode_metrics["aggregate"]["min_reward"]
            log_dict["reward/std"] = episode_metrics["aggregate"]["std_reward"]
            log_dict["reward/total_steps"] = episode_metrics["aggregate"]["total_steps"]

            # Per-game skill counts
            for game, cnt in skill_counts.items():
                log_dict[f"skillbank/{game}/n_skills"] = cnt

            # Skill bank stage times (aggregated across games)
            for game, result in sb_update_results.items():
                for stage, t in result.stage_times.items():
                    log_dict[f"skillbank/{game}/{stage}_time_s"] = t

            # GRPO metrics
            if grpo_result:
                for adapter, stats in grpo_result.decision_stats.items():
                    log_dict[f"grpo/decision/{adapter}/loss"] = stats.mean_loss
                    log_dict[f"grpo/decision/{adapter}/n_samples"] = stats.n_samples
                for adapter, stats in grpo_result.skillbank_stats.items():
                    log_dict[f"grpo/skillbank/{adapter}/loss"] = stats.mean_loss
                    log_dict[f"grpo/skillbank/{adapter}/n_samples"] = stats.n_samples
                log_dict["grpo/wall_time_s"] = grpo_result.wall_time_s

            try:
                wandb.log(log_dict, step=step)
            except Exception as exc:
                logger.warning("W&B log failed: %s", exc)

        # ── TensorBoard logging ──────────────────────────────────────
        if tb_writer is not None:
            try:
                tb_writer.add_scalar("timing/wall_time_s", step_elapsed, step)
                tb_writer.add_scalar("timing/phase_ab_s", phase_ab_time, step)
                tb_writer.add_scalar("timing/phase_b_finalize_s", phase_b_time, step)
                tb_writer.add_scalar("timing/phase_c_grpo_s", phase_c_time, step)

                tb_writer.add_scalar("reward/mean", episode_metrics["aggregate"]["mean_reward"], step)
                tb_writer.add_scalar("reward/max", episode_metrics["aggregate"]["max_reward"], step)
                tb_writer.add_scalar("reward/min", episode_metrics["aggregate"]["min_reward"], step)
                tb_writer.add_scalar("reward/std", episode_metrics["aggregate"]["std_reward"], step)
                tb_writer.add_scalar("reward/total_steps", episode_metrics["aggregate"]["total_steps"], step)
                for game, m in episode_metrics["per_game"].items():
                    tb_writer.add_scalar(f"reward/{game}/mean", m["mean_reward"], step)
                    tb_writer.add_scalar(f"reward/{game}/max", m["max_reward"], step)
                    tb_writer.add_scalar(f"reward/{game}/min", m["min_reward"], step)
                    tb_writer.add_scalar(f"reward/{game}/std", m["std_reward"], step)

                tb_writer.add_scalar("skillbank/n_skills", n_skills, step)
                tb_writer.add_scalar("skillbank/n_new_skills", total_new_skills, step)
                for game, cnt in skill_counts.items():
                    tb_writer.add_scalar(f"skillbank/{game}/n_skills", cnt, step)
                for game, result in sb_update_results.items():
                    for stage, t in result.stage_times.items():
                        tb_writer.add_scalar(f"skillbank/{game}/{stage}_time_s", t, step)

                tb_writer.add_scalar("vllm/calls", vllm_stats["call_count"], step)
                tb_writer.add_scalar("vllm/prompt_tokens", vllm_stats["total_prompt_tokens"], step)
                tb_writer.add_scalar("vllm/completion_tokens", vllm_stats["total_completion_tokens"], step)

                if grpo_result:
                    for adapter, stats in grpo_result.decision_stats.items():
                        tb_writer.add_scalar(f"grpo/decision/{adapter}/loss", stats.mean_loss, step)
                        tb_writer.add_scalar(f"grpo/decision/{adapter}/n_samples", stats.n_samples, step)
                    for adapter, stats in grpo_result.skillbank_stats.items():
                        tb_writer.add_scalar(f"grpo/skillbank/{adapter}/loss", stats.mean_loss, step)
                        tb_writer.add_scalar(f"grpo/skillbank/{adapter}/n_samples", stats.n_samples, step)
                    tb_writer.add_scalar("grpo/wall_time_s", grpo_result.wall_time_s, step)

                tb_writer.flush()
            except Exception as exc:
                logger.warning("TensorBoard log failed: %s", exc)

        # ── Step log (JSONL) ─────────────────────────────────────────
        try:
            with open(step_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_summary, default=str) + "\n")
        except Exception:
            pass

        # ── Checkpoint ───────────────────────────────────────────────
        should_checkpoint = (
            (step + 1) % config.checkpoint_interval == 0
            or step == 0
            or step == config.total_steps - 1
        )
        if should_checkpoint:
            try:
                ckpt_metadata = {
                    "n_skills": n_skills,
                    "skills_per_game": skill_counts,
                    "n_new_skills": total_new_skills,
                    "mean_reward": step_summary["mean_reward"],
                    "n_episodes": step_summary["n_episodes"],
                    "mode": mode,
                }
                save_checkpoint(
                    config.checkpoint_dir, step,
                    bank_agents=sb_manager.get_agents(),
                    adapter_dir=config.adapter_dir,
                    metadata=ckpt_metadata,
                )
                cleanup_old_checkpoints(config.checkpoint_dir, keep_last=10)
                logger.info("Checkpoint saved at step %d", step)
            except Exception as exc:
                logger.error("Checkpoint save failed: %s", exc)

    # ── Cleanup ───────────────────────────────────────────────────
    if vllm_manager:
        vllm_manager.stop()

    thread_executor.shutdown(wait=False)

    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception:
            pass

    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    logger.info(
        "Co-evolution loop complete: %d steps",
        config.total_steps - start_step,
    )

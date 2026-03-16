"""Async skill bank pipeline wrapper for the co-evolution loop.

Wraps the synchronous ``SkillBankAgent`` pipeline (Stage 1+2 segmentation,
Stage 3 contract learning, Stage 4 bank maintenance) to run concurrently
with rollout collection.  Uses ``asyncio.Queue`` to receive completed
episodes and processes them in micro-batches through the pipeline stages.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.coevolution.episode_runner import EpisodeResult

logger = logging.getLogger(__name__)


@dataclass
class SkillBankUpdateResult:
    accepted: bool = False
    bank_version: int = 0
    n_skills: int = 0
    n_new_skills: int = 0
    n_episodes_processed: int = 0
    wall_time_s: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    grpo_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class AsyncSkillBankPipeline:
    """Manages the skill bank update lifecycle across a co-evolution step.

    Receives completed episodes (via ``ingest_episode()`` or
    ``process_batch_async()``), processes them through the SkillBankAgent
    pipeline, and produces an updated bank.
    """

    def __init__(
        self,
        bank_dir: str = "runs/skillbank",
        model_name: str = "Qwen/Qwen3-14B",
        executor: Optional[ThreadPoolExecutor] = None,
        report_dir: Optional[str] = None,
    ):
        self.bank_dir = bank_dir
        self.model_name = model_name
        self._executor = executor
        self.report_dir = report_dir or str(Path(bank_dir) / "reports")
        self._agent: Any = None
        self._pending_episodes: List[Any] = []
        self._grpo_data: Dict[str, List[Dict[str, Any]]] = {
            "segment": [],
            "contract": [],
            "curator": [],
        }
        self._update_result: Optional[SkillBankUpdateResult] = None

    def _ensure_agent(self) -> Any:
        """Lazily create the SkillBankAgent."""
        if self._agent is not None:
            return self._agent

        from skill_agents_grpo.pipeline import SkillBankAgent, PipelineConfig

        bank_path = str(Path(self.bank_dir) / "skill_bank.jsonl")
        config = PipelineConfig(
            bank_path=bank_path,
            env_name="llm",
            llm_model=self.model_name,
            extractor_model=self.model_name,
            segmentation_method="dp",
            preference_iterations=1,
            new_skill_penalty=2.0,
            eff_freq=0.5,
            min_instances_per_skill=1,
            start_end_window=3,
            new_pool_min_cluster_size=1,
            new_pool_min_consistency=0.3,
            new_pool_min_distinctiveness=0.15,
            min_new_cluster_size=1,
            report_dir=self.report_dir,
        )
        self._agent = SkillBankAgent(config=config)

        if Path(bank_path).exists():
            try:
                self._agent.load()
                n = len(self._agent.skill_ids)
                logger.info("Loaded existing skill bank: %d skills", n)
            except Exception as exc:
                logger.warning("Failed to load skill bank: %s", exc)

        return self._agent

    def load_bank(self, bank: Any) -> None:
        """Inject a pre-loaded bank into the pipeline agent."""
        agent = self._ensure_agent()
        if hasattr(agent, "bank") and bank is not None:
            agent.bank = bank

    def _convert_episode_result(self, result: EpisodeResult) -> Any:
        """Convert ``EpisodeResult`` to the ``Episode`` format for the pipeline."""
        from data_structure.experience import Experience, Episode

        experiences = []
        for exp_dict in result.experiences:
            exp = Experience(
                state=exp_dict.get("state", ""),
                action=exp_dict.get("action", ""),
                reward=exp_dict.get("reward", 0.0),
                next_state=exp_dict.get("next_state", ""),
                done=exp_dict.get("done", False),
                intentions=exp_dict.get("intention"),
            )
            exp.idx = exp_dict.get("step", 0)
            exp.summary_state = exp_dict.get("summary_state", "")
            exp.action_type = "primitive"
            exp.interface = {"env_name": "gamingagent", "game_name": result.game}
            experiences.append(exp)

        episode = Episode(
            experiences=experiences,
            task=f"Play {result.game}",
            env_name="gamingagent",
            game_name=result.game,
            episode_id=result.episode_id,
            metadata={
                "done": result.terminated or result.truncated,
                "steps": result.steps,
                "total_reward": result.total_reward,
            },
        )
        episode.set_outcome()
        return episode

    async def ingest_episode(self, result: EpisodeResult) -> None:
        """Convert and queue a completed episode for processing."""
        if result.steps == 0:
            return
        episode = self._convert_episode_result(result)
        self._pending_episodes.append(episode)

    async def process_batch_async(
        self,
        results: List[EpisodeResult],
    ) -> None:
        """Process a micro-batch of completed episodes through Stages 1+2.

        Segments episodes concurrently via the thread executor (each
        segmentation involves LLM calls, so parallelism overlaps the
        network I/O).
        """
        episodes = []
        for r in results:
            if r.steps > 0:
                episodes.append(self._convert_episode_result(r))

        if not episodes:
            return

        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()
        executor = self._executor
        t0 = time.monotonic()

        def _segment_one(ep):
            try:
                result, sub_eps = agent.segment_episode(ep, env_name="llm")
                n_segs = len(result.segments) if hasattr(result, "segments") else 0
                logger.debug(
                    "Segmented %s: %d steps → %d segments",
                    ep.episode_id, len(ep.experiences), n_segs,
                )
                return True
            except Exception as exc:
                logger.warning("Segmentation failed for %s: %s", ep.episode_id, exc)
                return False

        futures = [
            loop.run_in_executor(executor, _segment_one, ep)
            for ep in episodes
        ]
        results_ok = await asyncio.gather(*futures, return_exceptions=True)

        n_ok = sum(1 for r in results_ok if r is True)
        elapsed = time.monotonic() - t0
        logger.info(
            "Segmented %d/%d episodes in %.1fs",
            n_ok, len(episodes), elapsed,
        )

        self._pending_episodes.extend(episodes)

    async def finalize_update(self) -> SkillBankUpdateResult:
        """Run contract learning + bank maintenance after all episodes ingested.

        Returns the update result with bank metrics.
        """
        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()
        executor = self._executor
        t0 = time.monotonic()
        stage_times: Dict[str, float] = {}

        n_episodes = len(self._pending_episodes)
        n_skills_before = len(agent.skill_ids)

        # Stage 3: Contract learning
        t_s3 = time.monotonic()

        def _run_contracts():
            if agent._all_segments:
                try:
                    return agent.run_contract_learning()
                except Exception as exc:
                    logger.warning("Contract learning failed: %s", exc)
            return None

        s3_result = await loop.run_in_executor(executor, _run_contracts)
        stage_times["contract_learning"] = time.monotonic() - t_s3

        # Stage 4: Bank maintenance
        t_s4 = time.monotonic()

        def _run_maintenance():
            if agent._all_segments and len(agent.skill_ids) > 0:
                try:
                    return agent.run_bank_maintenance()
                except Exception as exc:
                    logger.warning("Bank maintenance failed: %s", exc)
            return None

        s4_result = await loop.run_in_executor(executor, _run_maintenance)
        stage_times["bank_maintenance"] = time.monotonic() - t_s4

        # Proto-skill materialization
        t_mat = time.monotonic()

        def _materialize():
            try:
                n_formed = agent.form_proto_skills()
                n_verified = agent.verify_proto_skills()
                n_promoted = agent.promote_proto_skills()
                n_materialized = agent.materialize_new_skills()
                return {
                    "formed": n_formed, "verified": n_verified,
                    "promoted": n_promoted, "materialized": n_materialized,
                }
            except Exception as exc:
                logger.warning("Proto-skill processing failed: %s", exc)
                return {}

        mat_result = await loop.run_in_executor(executor, _materialize)
        stage_times["materialization"] = time.monotonic() - t_mat

        # Save bank
        def _save_bank():
            try:
                agent.save()
            except Exception as exc:
                logger.warning("Bank save failed: %s", exc)

        await loop.run_in_executor(executor, _save_bank)

        n_skills_after = len(agent.skill_ids)
        elapsed = time.monotonic() - t0

        self._update_result = SkillBankUpdateResult(
            accepted=True,
            bank_version=getattr(agent, "_iteration_count", 0),
            n_skills=n_skills_after,
            n_new_skills=max(0, n_skills_after - n_skills_before),
            n_episodes_processed=n_episodes,
            wall_time_s=elapsed,
            stage_times=stage_times,
            grpo_data=self._grpo_data,
        )

        logger.info(
            "Skill bank update: %d→%d skills (+%d), %d episodes, %.1fs",
            n_skills_before, n_skills_after,
            self._update_result.n_new_skills, n_episodes, elapsed,
        )

        return self._update_result

    def get_bank(self) -> Any:
        """Return the current skill bank object."""
        if self._agent is not None:
            return self._agent.bank
        return None

    def get_agent(self) -> Any:
        """Return the SkillBankAgent instance."""
        return self._agent

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._grpo_data

    def reset_for_step(self) -> None:
        """Clear per-step state (pending episodes, GRPO data)."""
        self._pending_episodes.clear()
        self._grpo_data = {"segment": [], "contract": [], "curator": []}
        self._update_result = None
        if self._agent is not None:
            self._agent._all_segments = []
            self._agent._new_pool = []


class PerGameSkillBankManager:
    """Maintains a separate ``AsyncSkillBankPipeline`` per game.

    Each game gets its own ``skill_bank.jsonl`` under
    ``<bank_dir>/<game>/skill_bank.jsonl``, so skills learned in Tetris
    stay separate from Diplomacy, etc.

    The manager exposes an interface similar to ``AsyncSkillBankPipeline``
    but routes operations by game name.
    """

    def __init__(
        self,
        games: List[str],
        bank_dir: str = "runs/skillbank",
        model_name: str = "Qwen/Qwen3-14B",
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self._pipelines: Dict[str, AsyncSkillBankPipeline] = {}
        for game in games:
            game_dir = str(Path(bank_dir) / game)
            Path(game_dir).mkdir(parents=True, exist_ok=True)
            self._pipelines[game] = AsyncSkillBankPipeline(
                bank_dir=game_dir,
                model_name=model_name,
                executor=executor,
                report_dir=str(Path(game_dir) / "reports"),
            )
        self._bank_dir = bank_dir
        logger.info(
            "PerGameSkillBankManager: %d game banks under %s",
            len(games), bank_dir,
        )

    def pipeline_for(self, game: str) -> Optional[AsyncSkillBankPipeline]:
        return self._pipelines.get(game)

    def get_bank(self, game: str) -> Any:
        pipe = self._pipelines.get(game)
        return pipe.get_bank() if pipe else None

    def get_banks(self) -> Dict[str, Any]:
        """Return ``{game: bank}`` for all games that have a loaded bank."""
        return {
            game: pipe.get_bank()
            for game, pipe in self._pipelines.items()
            if pipe.get_bank() is not None
        }

    def get_agents(self) -> Dict[str, Any]:
        return {
            game: pipe.get_agent()
            for game, pipe in self._pipelines.items()
        }

    def reset_for_step(self) -> None:
        for pipe in self._pipelines.values():
            pipe.reset_for_step()

    async def process_batch_async(
        self, results: List[EpisodeResult],
    ) -> None:
        """Route episodes to the correct per-game pipeline."""
        by_game: Dict[str, List[EpisodeResult]] = {}
        for r in results:
            by_game.setdefault(r.game, []).append(r)

        tasks = []
        for game, game_results in by_game.items():
            pipe = self._pipelines.get(game)
            if pipe is None:
                logger.warning(
                    "No skill bank pipeline for game '%s', skipping %d episodes",
                    game, len(game_results),
                )
                continue
            tasks.append(pipe.process_batch_async(game_results))

        if tasks:
            await asyncio.gather(*tasks)

    async def finalize_all(self) -> Dict[str, SkillBankUpdateResult]:
        """Finalize all per-game banks and return per-game results."""
        results: Dict[str, SkillBankUpdateResult] = {}

        async def _finalize_one(game: str, pipe: AsyncSkillBankPipeline):
            try:
                results[game] = await pipe.finalize_update()
            except Exception as exc:
                logger.error("Skill bank finalize failed for %s: %s", game, exc)

        tasks = [
            _finalize_one(game, pipe)
            for game, pipe in self._pipelines.items()
        ]
        await asyncio.gather(*tasks)
        return results

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Merge GRPO data from all per-game pipelines."""
        merged: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        for pipe in self._pipelines.values():
            for key in merged:
                merged[key].extend(pipe.grpo_data.get(key, []))
        return merged

    def total_skills(self) -> int:
        total = 0
        for pipe in self._pipelines.values():
            bank = pipe.get_bank()
            if bank and hasattr(bank, "skill_ids"):
                total += len(list(bank.skill_ids))
        return total

    def skill_counts(self) -> Dict[str, int]:
        """Return ``{game: n_skills}``."""
        counts = {}
        for game, pipe in self._pipelines.items():
            bank = pipe.get_bank()
            if bank and hasattr(bank, "skill_ids"):
                counts[game] = len(list(bank.skill_ids))
            else:
                counts[game] = 0
        return counts

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
        model_name: str = "Qwen/Qwen3-8B",
        executor: Optional[ThreadPoolExecutor] = None,
        report_dir: Optional[str] = None,
        game_name: str = "generic",
    ):
        self.bank_dir = bank_dir
        self.model_name = model_name
        self.game_name = game_name
        self._executor = executor
        self.report_dir = report_dir or str(Path(bank_dir) / "reports")
        self._agent: Any = None
        self._query_engine: Any = None
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
            game_name=self.game_name,
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
        """Run the full skill-bank update pipeline.

        Execution order (changed from earlier versions):
          1. Proto-skill materialization — turn __NEW__ clusters into real
             skills so that contract learning and bank maintenance have
             non-empty skill vocabularies from the very first step.
          2. Contract learning (Stage 3) — learn effect summaries; now runs
             on materialized skill labels instead of seeing only __NEW__.
          3. Bank maintenance (Stage 4) — split / merge / refine existing
             skills, with LLM curator filtering.

        Returns the update result with bank metrics.
        """
        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()
        executor = self._executor
        t0 = time.monotonic()
        stage_times: Dict[str, float] = {}

        n_episodes = len(self._pending_episodes)
        n_skills_before = len(agent.skill_ids)

        # ── 1. Proto-skill materialization (FIRST) ───────────────────
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

        n_after_materialize = len(agent.skill_ids)
        if n_after_materialize > n_skills_before:
            logger.info(
                "Materialized %d new skills (%d→%d) — "
                "relabelling __NEW__ segments before contract learning",
                n_after_materialize - n_skills_before,
                n_skills_before, n_after_materialize,
            )

        # ── 2. Contract learning (Stage 3) ───────────────────────────
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

        # ── 3. Bank maintenance (Stage 4) ────────────────────────────
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

        # ── Save bank ────────────────────────────────────────────────
        def _save_bank():
            try:
                agent.save()
            except Exception as exc:
                logger.warning("Bank save failed: %s", exc)

        await loop.run_in_executor(executor, _save_bank)

        self._query_engine = None

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

    def get_raw_bank(self) -> Any:
        """Return the raw ``SkillBankMVP`` (has ``.skill_ids``, etc.)."""
        if self._agent is not None:
            return self._agent.bank
        return None

    def get_bank(self) -> Any:
        """Return a query-engine-wrapped skill bank for decision agents.

        Wrapping in ``SkillQueryEngine`` provides the ``.select()`` method
        needed by ``get_top_k_skill_candidates`` for multi-candidate skill
        selection.  Without this, only a single fallback candidate is
        returned and the skill_selection adapter never fires.
        """
        if self._agent is None:
            return None
        bank = self._agent.bank
        if bank is None or len(bank) == 0:
            return bank
        if self._query_engine is not None:
            return self._query_engine
        try:
            from skill_agents_grpo.query import SkillQueryEngine
            self._query_engine = SkillQueryEngine(bank)
            return self._query_engine
        except Exception:
            return bank

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
        model_name: str = "Qwen/Qwen3-8B",
        executor: Optional[ThreadPoolExecutor] = None,
        grpo_group_size: int = 4,
        seed_bank_dir: Optional[str] = None,
        process_executor: Optional[ProcessPoolExecutor] = None,
    ):
        self._process_executor = process_executor
        self._pipelines: Dict[str, AsyncSkillBankPipeline] = {}
        for game in games:
            game_dir = str(Path(bank_dir) / game)
            Path(game_dir).mkdir(parents=True, exist_ok=True)
            self._pipelines[game] = AsyncSkillBankPipeline(
                bank_dir=game_dir,
                model_name=model_name,
                executor=executor,
                report_dir=str(Path(game_dir) / "reports"),
                game_name=game,
            )
        self._bank_dir = bank_dir
        self._grpo_group_size = grpo_group_size
        self._grpo_buffer: Optional[Any] = None
        self._collected_grpo: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        logger.info(
            "PerGameSkillBankManager: %d game banks under %s (process_pool=%s)",
            len(games), bank_dir, process_executor is not None,
        )

        if seed_bank_dir:
            self._seed_from_coldstart(seed_bank_dir)

    # ── Bank seeding ─────────────────────────────────────────────────

    def _seed_from_coldstart(self, seed_dir: str) -> None:
        """Copy skills from a cold-start bank into empty per-game banks.

        Only seeds a game's bank when it currently contains zero skills,
        so an in-progress run that already has its own skills is never
        overwritten.

        Works at the file level (via ``SkillBankMVP``) rather than
        requiring the lazy ``SkillBankAgent`` to be initialised.
        """
        from skill_agents_grpo.skill_bank.bank import SkillBankMVP

        seed_path = Path(seed_dir)
        if not seed_path.is_dir():
            logger.warning("seed_bank_dir %s does not exist — skipping seed", seed_dir)
            return

        for game, pipe in self._pipelines.items():
            dest_file = Path(pipe.bank_dir) / "skill_bank.jsonl"

            if dest_file.exists() and dest_file.stat().st_size > 0:
                logger.info(
                    "Seed skip %s: bank file already exists at %s", game, dest_file,
                )
                continue

            candidate = seed_path / game / "skill_bank.jsonl"
            if not candidate.exists():
                logger.info("Seed skip %s: no seed file at %s", game, candidate)
                continue

            bank = SkillBankMVP(str(dest_file))
            bank.load(str(candidate))
            n = len(bank)
            if n > 0:
                bank.save()
                logger.info(
                    "Seeded %s bank with %d skills from %s", game, n, candidate,
                )
            else:
                logger.info("Seed file %s was empty — nothing to load", candidate)

    # ── GRPO wrapper management ─────────────────────────────────────

    def _enable_grpo_wrappers(self) -> None:
        """Activate GRPO wrappers on skill-bank LLM calls (module-level)."""
        from skill_agents_grpo.grpo.buffer import GRPOBuffer
        from skill_agents_grpo.stage3_mvp.llm_contract import enable_contract_grpo
        from skill_agents_grpo.bank_maintenance.llm_curator import enable_curator_grpo
        from skill_agents_grpo.infer_segmentation.llm_teacher import enable_segment_grpo
        from skill_agents_grpo.infer_segmentation.episode_adapter import (
            grpo_scorer_factory,
            grpo_decode_fn,
        )

        self._grpo_buffer = GRPOBuffer()
        gs = self._grpo_group_size

        enable_segment_grpo(
            buffer=self._grpo_buffer, group_size=gs, temperature=1.0,
            scorer_factory=grpo_scorer_factory,
            decode_fn=grpo_decode_fn,
        )
        enable_contract_grpo(buffer=self._grpo_buffer, group_size=gs, temperature=0.8)
        enable_curator_grpo(buffer=self._grpo_buffer, group_size=gs, temperature=0.8)
        logger.info("Contract/Curator reward context is dynamic — set before each LLM call")
        logger.info("Skill-bank GRPO wrappers enabled (G=%d)", gs)

    def _disable_grpo_wrappers(self) -> None:
        """Deactivate GRPO wrappers and restore original functions."""
        from skill_agents_grpo.stage3_mvp.llm_contract import disable_contract_grpo
        from skill_agents_grpo.bank_maintenance.llm_curator import disable_curator_grpo
        from skill_agents_grpo.infer_segmentation.llm_teacher import disable_segment_grpo

        disable_segment_grpo()
        disable_contract_grpo()
        disable_curator_grpo()
        logger.info("Skill-bank GRPO wrappers disabled")

    def _collect_grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Drain the shared GRPO buffer into the per-adapter dict format.

        Preserves ``metadata`` (including skill_id, game context) from
        each sample for downstream logging and per-game diagnostics.
        """
        from skill_agents_grpo.lora.skill_function import SkillFunction

        collected: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        if self._grpo_buffer is None:
            return collected

        adapter_map = {
            SkillFunction.SEGMENT: "segment",
            SkillFunction.CONTRACT: "contract",
            SkillFunction.CURATOR: "curator",
        }
        for sf, key in adapter_map.items():
            for sample in self._grpo_buffer.samples_for(sf):
                if sample.prompt and sample.completions:
                    collected[key].append({
                        "prompt": sample.prompt,
                        "completions": sample.completions,
                        "rewards": sample.rewards,
                        "metadata": sample.metadata,
                    })

        n_total = sum(len(v) for v in collected.values())
        if n_total:
            logger.info(
                "Collected %d GRPO samples: segment=%d, contract=%d, curator=%d",
                n_total, len(collected["segment"]),
                len(collected["contract"]), len(collected["curator"]),
            )
        return collected

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
        self._grpo_buffer = None
        self._collected_grpo = {"segment": [], "contract": [], "curator": []}
        try:
            self._disable_grpo_wrappers()
        except Exception:
            pass
        try:
            self._enable_grpo_wrappers()
        except Exception as exc:
            logger.warning("Failed to enable GRPO wrappers: %s", exc)

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
        """Finalize all per-game banks and return per-game results.

        When a ``ProcessPoolExecutor`` was provided, per-game finalization
        runs in separate processes for true parallelism on CPU-bound
        stages.  Otherwise falls back to asyncio tasks (concurrent but
        GIL-bound).
        """
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

        try:
            self._disable_grpo_wrappers()
        except Exception as exc:
            logger.warning("Failed to disable GRPO wrappers: %s", exc)

        self._collected_grpo = self._collect_grpo_data()

        return results

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return GRPO training data collected by the wrappers."""
        return self._collected_grpo

    def total_skills(self) -> int:
        total = 0
        for pipe in self._pipelines.values():
            bank = pipe.get_raw_bank()
            if bank and hasattr(bank, "skill_ids"):
                total += len(list(bank.skill_ids))
        return total

    def skill_counts(self) -> Dict[str, int]:
        """Return ``{game: n_skills}``."""
        counts = {}
        for game, pipe in self._pipelines.items():
            bank = pipe.get_raw_bank()
            if bank and hasattr(bank, "skill_ids"):
                counts[game] = len(list(bank.skill_ids))
            else:
                counts[game] = 0
        return counts

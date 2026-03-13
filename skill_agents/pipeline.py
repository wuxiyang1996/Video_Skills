"""
SkillBank Agent — agentic pipeline that builds, maintains, and serves a Skill Bank.

Orchestrates boundary_proposal (Stage 1), infer_segmentation (Stage 2),
stage3_mvp (Stage 3 contract learn/verify/refine), bank_maintenance (Stage 4
split/merge/refine), and skill_evaluation.  Exposes a query API consumed by
decision_agents.

Typical usage::

    from skill_agents.pipeline import SkillBankAgent, PipelineConfig

    agent = SkillBankAgent(bank_path="skills/bank.jsonl")
    agent.ingest_episodes(episodes, env_name="llm+overcooked")
    agent.run_until_stable(max_iterations=3)

    # Decision-agent queries the bank
    result = agent.query_skill("navigate to pot and place onion")
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.skill_bank.new_pool import NewPoolManager, NewPoolConfig
from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Top-level configuration for the SkillBank Agent pipeline."""

    # Paths
    bank_path: Optional[str] = None
    preference_store_path: Optional[str] = None
    report_dir: Optional[str] = None

    # Stage 1: boundary proposal
    env_name: str = "llm"
    merge_radius: int = 5
    extractor_model: str = "gpt-4o-mini"

    # Stage 2: segmentation
    segmentation_method: str = "dp"
    preference_iterations: int = 3
    margin_threshold: float = 1.0
    max_queries_per_iter: int = 5
    new_skill_penalty: float = 5.0

    # Boundary scoring (tag-change penalty model)
    consistency_penalty: float = 0.3
    min_segment_length: int = 3
    min_skill_length: int = 2
    boundary_score_threshold: float = 0.3

    # Contract feedback: Stage 3 → Stage 2 closed loop
    contract_feedback_mode: str = "off"  # "off" | "weak" | "strong"
    contract_feedback_strength: float = 0.3

    # Stage 3: contract learning
    eff_freq: float = 0.8
    min_instances_per_skill: int = 5
    start_end_window: int = 5

    # NEW pool management
    new_pool_min_cluster_size: int = 5
    new_pool_min_consistency: float = 0.5
    new_pool_min_distinctiveness: float = 0.25

    # Stage 4: bank maintenance
    split_pass_rate_threshold: float = 0.7
    child_pass_rate_threshold: float = 0.8
    merge_jaccard_threshold: float = 0.85
    merge_embedding_threshold: float = 0.90
    min_child_size: int = 3
    min_new_cluster_size: int = 5

    # Convergence
    max_iterations: int = 5
    convergence_margin_std: float = 0.5
    convergence_new_rate: float = 0.05

    # LLM
    llm_model: Optional[str] = None
    max_concurrent_llm_calls: Optional[int] = None  # cap for local GPU (e.g. 1)


# ─────────────────────────────────────────────────────────────────────
# Pipeline state (serialisable snapshot)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IterationSnapshot:
    """Diagnostics captured after one pipeline iteration."""

    iteration: int = 0
    n_skills: int = 0
    n_segments: int = 0
    n_new_segments: int = 0
    new_rate: float = 0.0
    mean_margin: float = 0.0
    mean_pass_rate: float = 0.0
    n_splits: int = 0
    n_merges: int = 0
    n_refines: int = 0
    n_materialized: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────
# SkillBankAgent
# ─────────────────────────────────────────────────────────────────────

class SkillBankAgent:
    """Agentic pipeline that builds, maintains, and serves a Skill Bank.

    Lifecycle::

        agent = SkillBankAgent(...)
        agent.load()                         # load existing bank if any
        agent.ingest_episodes(episodes, ...)  # Stage 1+2+3 per episode
        agent.run_until_stable()              # iterate Stage 2→3→4 to convergence
        result = agent.query_skill("...")     # decision-agent queries

    The agent accumulates ``SegmentRecord`` objects across ingestion calls and
    uses them for bank maintenance and evaluation.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        bank_path: Optional[str] = None,
        bank: Optional[SkillBankMVP] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        if bank_path:
            self.config.bank_path = bank_path

        self.bank = bank or SkillBankMVP(path=self.config.bank_path)
        self._all_segments: List[SegmentRecord] = []
        self._new_pool: List[SegmentRecord] = []  # legacy list (kept for compat)
        self._new_pool_mgr = NewPoolManager(config=NewPoolConfig(
            min_cluster_size=self.config.new_pool_min_cluster_size,
            min_consistency=self.config.new_pool_min_consistency,
            min_distinctiveness=self.config.new_pool_min_distinctiveness,
            min_pass_rate=self.config.split_pass_rate_threshold,
        ))
        self._observations_by_traj: Dict[str, list] = {}
        self._traj_lengths: Dict[str, int] = {}
        self._preference_store: Any = None  # lazily created
        self._iteration: int = 0
        self._history: List[IterationSnapshot] = []
        self._query_engine: Any = None  # lazily created

    # ── Persistence ──────────────────────────────────────────────────

    def load(self) -> None:
        """Load skill bank (and preference store if path set)."""
        if self.config.bank_path:
            self.bank.load(self.config.bank_path)
            logger.info("Loaded bank with %d skills from %s", len(self.bank), self.config.bank_path)

        if self.config.preference_store_path:
            store = self._get_or_create_preference_store()
            store.load(self.config.preference_store_path)

    def save(self) -> None:
        """Persist bank, preference store, and iteration history."""
        if self.config.bank_path:
            self.bank.save(self.config.bank_path)
            logger.info("Saved bank (%d skills) to %s", len(self.bank), self.config.bank_path)

        if self.config.preference_store_path and self._preference_store is not None:
            self._preference_store.save(self.config.preference_store_path)

        if self.config.report_dir and self._history:
            report_dir = Path(self.config.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            history_path = report_dir / "iteration_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in self._history], f, indent=2, default=str)

    # ── Lazy helpers ─────────────────────────────────────────────────

    def _get_or_create_preference_store(self):
        if self._preference_store is None:
            from skill_agents.infer_segmentation.preference import PreferenceStore
            self._preference_store = PreferenceStore()
        return self._preference_store

    def _get_query_engine(self):
        if self._query_engine is None:
            from skill_agents.query import SkillQueryEngine
            self._query_engine = SkillQueryEngine(self.bank)
        return self._query_engine

    def _invalidate_query_engine(self):
        self._query_engine = None

    # ── Stage 1+2: Segment one episode ───────────────────────────────

    def segment_episode(
        self,
        episode,
        env_name: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
        *,
        embedder=None,
        surprisal=None,
        extractor_kwargs: Optional[dict] = None,
    ) -> Tuple[Any, list]:
        """Run Stage 1 (boundary proposal) + Stage 2 (segmentation) on one episode.

        Returns (SegmentationResult, list[SubTask_Experience]).
        Side-effect: accumulates segment records for later stages.
        """
        from skill_agents.infer_segmentation.episode_adapter import (
            infer_and_segment,
            infer_and_segment_offline,
        )
        from skill_agents.infer_segmentation.config import (
            ContractFeedbackConfig,
            SegmentationConfig,
            NewSkillConfig,
            PreferenceLearningConfig,
            LLMTeacherConfig,
        )
        from skill_agents.boundary_proposal.proposal import ProposalConfig

        cfg = self.config
        _env = env_name or cfg.env_name
        _skill_names = skill_names or list(self.bank.skill_ids)

        seg_config = SegmentationConfig(
            method=cfg.segmentation_method,
            new_skill=NewSkillConfig(
                enabled=True,
                penalty=cfg.new_skill_penalty,
            ),
            preference=PreferenceLearningConfig(
                num_iterations=cfg.preference_iterations,
                margin_threshold=cfg.margin_threshold,
                max_queries_per_iter=cfg.max_queries_per_iter,
            ),
            llm_teacher=LLMTeacherConfig(
                model=cfg.llm_model,
                max_concurrent_llm_calls=cfg.max_concurrent_llm_calls,
            ),
            contract_feedback=ContractFeedbackConfig(
                mode=cfg.contract_feedback_mode,
                strength=cfg.contract_feedback_strength,
            ),
        )
        proposal_config = ProposalConfig(merge_radius=cfg.merge_radius)
        _extractor_kwargs = extractor_kwargs or {"model": cfg.extractor_model}

        # Build compat_fn from the bank when contract feedback is enabled
        _compat_fn = None
        if cfg.contract_feedback_mode != "off" and len(self.bank) > 0:
            _compat_fn = self.bank.compat_fn

        store = self._get_or_create_preference_store()

        if _skill_names:
            result, sub_episodes, store = infer_and_segment(
                episode,
                skill_names=_skill_names,
                env_name=_env,
                config=seg_config,
                proposal_config=proposal_config,
                embedder=embedder,
                surprisal=surprisal,
                preference_store=store,
                extractor_kwargs=_extractor_kwargs,
                compat_fn=_compat_fn,
            )
            self._preference_store = store
        else:
            result, sub_episodes = infer_and_segment_offline(
                episode,
                skill_names=[],
                env_name=_env,
                config=seg_config,
                proposal_config=proposal_config,
                embedder=embedder,
                surprisal=surprisal,
                extractor_kwargs=_extractor_kwargs,
                compat_fn=_compat_fn,
            )

        traj_id = getattr(episode, "task", None) or f"traj_{len(self._traj_lengths)}"
        self._cache_trajectory(episode, traj_id, result)

        return result, sub_episodes

    def _cache_trajectory(self, episode, traj_id: str, result) -> None:
        """Store observations and convert segmentation result to SegmentRecord."""
        exps = episode.experiences
        self._observations_by_traj[traj_id] = [
            getattr(e, "state", None) for e in exps
        ]
        self._traj_lengths[traj_id] = len(exps)

        segments = result.segments
        for idx, seg in enumerate(segments):
            seg_id = f"{traj_id}_seg{idx:04d}"
            rec = SegmentRecord(
                seg_id=seg_id,
                traj_id=traj_id,
                t_start=seg.start,
                t_end=seg.end,
                skill_label=seg.assigned_skill,
            )
            if seg.assigned_skill == "__NEW__":
                self._new_pool.append(rec)
                pred_skill = segments[idx - 1].assigned_skill if idx > 0 else None
                succ_skill = segments[idx + 1].assigned_skill if idx < len(segments) - 1 else None
                self._new_pool_mgr.add(rec, predecessor_skill=pred_skill, successor_skill=succ_skill)
            self._all_segments.append(rec)

    # ── Batch ingestion ──────────────────────────────────────────────

    def ingest_episodes(
        self,
        episodes: Sequence,
        env_name: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[Any, list]]:
        """Ingest a batch of episodes through Stage 1+2, then run Stage 3.

        Returns list of (SegmentationResult, SubTask_Experience list) per episode.
        """
        results = []
        for ep in episodes:
            r = self.segment_episode(ep, env_name=env_name, skill_names=skill_names, **kwargs)
            results.append(r)

        if self._all_segments:
            self.run_contract_learning()

        self._invalidate_query_engine()
        return results

    # ── Stage 3: contract learning ───────────────────────────────────

    def run_contract_learning(self) -> Any:
        """Run Stage 3 MVP: learn, verify, and refine contracts for all accumulated segments.

        Returns Stage3MVPSummary.
        """
        from skill_agents.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
            Stage3MVPSummary,
        )
        from skill_agents.stage3_mvp.config import Stage3MVPConfig

        cfg = self.config
        s3_config = Stage3MVPConfig(
            eff_freq=cfg.eff_freq,
            min_instances_per_skill=cfg.min_instances_per_skill,
            start_end_window=cfg.start_end_window,
        )

        specs = [
            SegmentSpec(
                seg_id=rec.seg_id,
                traj_id=rec.traj_id,
                t_start=rec.t_start,
                t_end=rec.t_end,
                skill_label=rec.skill_label,
                ui_events=list(rec.events) if rec.events else [],
            )
            for rec in self._all_segments
            if rec.skill_label.upper() != "NEW" and rec.skill_label != "__NEW__"
        ]

        if not specs:
            logger.info("No non-NEW segments to process in Stage 3.")
            return Stage3MVPSummary()

        summary = run_stage3_mvp(
            segments=specs,
            observations_by_traj=self._observations_by_traj,
            config=s3_config,
            bank=self.bank,
            bank_path=cfg.bank_path,
        )
        logger.info("Stage 3: %s", summary)
        self._invalidate_query_engine()
        return summary

    # ── Protocol update ────────────────────────────────────────────

    def update_protocols(self) -> int:
        """Synthesize or update protocols for skills that need it.

        Called after quality check identifies skills with enough high-quality
        sub-episodes.  Uses LLM to synthesize actionable step-by-step
        protocols from sub-episode evidence.

        Returns the number of protocols updated.
        """
        from skill_agents.stage3_mvp.schemas import Protocol, Skill

        updated = 0
        for sid in self.bank.skill_ids:
            skill = self.bank.get_skill(sid)
            if skill is None or skill.retired:
                continue

            high_quality = [se for se in skill.sub_episodes if se.quality_score >= 0.6]
            if len(high_quality) < 3 and skill.protocol.steps:
                continue

            if len(high_quality) < 3:
                continue

            protocol = self._synthesize_protocol(skill, high_quality)
            if protocol is not None:
                skill.bump_version()
                skill.protocol = protocol
                self.bank.add_or_update_skill(skill)
                updated += 1
                logger.info(
                    "Updated protocol for skill %s (v%d, %d steps)",
                    sid, skill.version, len(protocol.steps),
                )

        if updated:
            self._invalidate_query_engine()
        return updated

    def _synthesize_protocol(
        self,
        skill: Skill,
        high_quality_eps: list,
    ) -> Optional:
        """Synthesize a Protocol from high-quality sub-episode pointers.

        Uses sub-episode summaries and intention_tags (cached on the
        pointers) plus contract effects.  No need to load full rollouts.
        """
        from skill_agents.stage3_mvp.schemas import Protocol

        contract = skill.contract

        steps = []
        preconditions = []
        success_criteria = []

        # Derive steps from the evidence summaries on the pointers
        summaries = [se.summary for se in high_quality_eps if se.summary]
        if summaries:
            seen = set()
            for s in summaries:
                key = s.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    steps.append(s.strip())
                if len(steps) >= 5:
                    break

        # Fall back to contract effects when no summaries are available
        if not steps and contract:
            if contract.eff_add:
                for lit in sorted(contract.eff_add)[:5]:
                    steps.append(f"Achieve: {lit}")
                    success_criteria.append(f"{lit} is true")
            if contract.eff_del:
                for lit in sorted(contract.eff_del)[:3]:
                    steps.append(f"Remove: {lit}")
            if contract.eff_event:
                for lit in sorted(contract.eff_event)[:3]:
                    steps.append(f"Trigger: {lit}")

        if not steps:
            steps = ["Execute skill actions as needed"]

        # Derive expected_tag_pattern from the majority tag set
        from collections import Counter
        tag_counter: Counter = Counter()
        for se in high_quality_eps:
            tag_counter.update(se.intention_tags)
        if tag_counter:
            skill.expected_tag_pattern = [
                tag for tag, _ in tag_counter.most_common(5)
            ]

        avg_len = 0
        if high_quality_eps:
            lengths = [se.length for se in high_quality_eps]
            avg_len = sum(lengths) // len(lengths)

        abort_criteria = []
        if skill.success_rate < 0.5:
            abort_criteria.append("Abort if no progress after expected duration")

        return Protocol(
            preconditions=preconditions,
            steps=steps,
            success_criteria=success_criteria,
            abort_criteria=abort_criteria,
            expected_duration=max(1, avg_len),
        )

    # ── Stage 4.5: sub-episode quality check ────────────────────────

    def run_sub_episode_quality_check(self) -> List[dict]:
        """Run quality scoring, drop low-quality sub-episodes, flag
        skills needing protocol updates, and retire depleted skills.

        Returns list of per-skill quality check results.
        """
        from skill_agents.quality.sub_episode_evaluator import run_quality_check_batch

        skills = [
            self.bank.get_skill(sid)
            for sid in self.bank.skill_ids
            if self.bank.get_skill(sid) is not None
        ]
        results = run_quality_check_batch(skills)

        for r in results:
            if r.get("retired"):
                logger.info("Retiring skill %s after quality check.", r["skill_id"])
            self.bank.recompute_stats(r["skill_id"])

        if results:
            self._invalidate_query_engine()
        return results

    # ── Stage 4: bank maintenance ────────────────────────────────────

    def run_bank_maintenance(
        self,
        embeddings: Optional[Dict[str, List[float]]] = None,
        stage2_diagnostics: Optional[List[dict]] = None,
    ) -> Any:
        """Run Stage 4: split, merge, refine skills and local re-decode.

        Returns BankMaintenanceResult.
        """
        from skill_agents.bank_maintenance.run_bank_maintenance import (
            run_bank_maintenance,
            BankMaintenanceResult,
        )
        from skill_agents.bank_maintenance.config import BankMaintenanceConfig

        cfg = self.config

        maint_config = BankMaintenanceConfig(
            split_pass_rate_threshold=cfg.split_pass_rate_threshold,
            child_pass_rate_threshold=cfg.child_pass_rate_threshold,
            merge_jaccard_threshold=cfg.merge_jaccard_threshold,
            merge_embedding_threshold=cfg.merge_embedding_threshold,
            min_child_size=cfg.min_child_size,
        )

        report_path = None
        if cfg.report_dir:
            report_path = str(Path(cfg.report_dir) / f"bank_diff_iter{self._iteration}.json")

        result = run_bank_maintenance(
            bank=self.bank,
            all_segments=self._all_segments,
            config=maint_config,
            embeddings=embeddings,
            stage2_diagnostics=stage2_diagnostics,
            traj_lengths=self._traj_lengths,
            report_path=report_path,
        )

        if result.alias_map:
            self._apply_alias_map(result.alias_map)

        self._invalidate_query_engine()
        logger.info(
            "Stage 4: %d splits, %d merges, %d refines",
            len(result.split_results),
            len(result.merge_results),
            len(result.refine_results),
        )
        return result

    def _apply_alias_map(self, alias_map: Dict[str, str]) -> None:
        """Relabel segment records after merges."""
        for seg in self._all_segments:
            if seg.skill_label in alias_map:
                seg.skill_label = alias_map[seg.skill_label]
        for seg in self._new_pool:
            if seg.skill_label in alias_map:
                seg.skill_label = alias_map[seg.skill_label]

    # ── Materialize NEW ──────────────────────────────────────────────

    def materialize_new_skills(self) -> int:
        """Promote qualifying ``__NEW__`` clusters to real skills.

        Uses the ``NewPoolManager`` for rich clustering (effect similarity,
        consistency, separability).  Falls back to the legacy signature-based
        approach only when the pool manager is empty but _new_pool has records.

        Returns the number of new skills created.
        """
        from skill_agents.infer_segmentation.config import LLMTeacherConfig

        llm_cfg = LLMTeacherConfig(
            model=self.config.llm_model,
            max_concurrent_llm_calls=self.config.max_concurrent_llm_calls,
        ) if self.config.llm_model else None

        # Try the new pool manager first
        if self._new_pool_mgr.size >= self.config.new_pool_min_cluster_size:
            logger.info(
                "NEW pool: %d candidates, running promotion.",
                self._new_pool_mgr.size,
            )
            created_ids = self._new_pool_mgr.promote(
                bank=self.bank,
                observations_by_traj=self._observations_by_traj,
                llm_config=llm_cfg,
            )
            # Sync legacy pool
            promoted_seg_ids = {c.seg_id for c in self._new_pool_mgr.records}
            self._new_pool = [
                r for r in self._new_pool
                if r.seg_id not in self._new_pool_mgr._promoted_ids
            ]

            for cid in created_ids:
                logger.info("Materialized new skill %s via NewPoolManager.", cid)

            if created_ids and self.config.bank_path:
                self.bank.save(self.config.bank_path)
            self._invalidate_query_engine()
            return len(created_ids)

        if len(self._new_pool) < self.config.min_new_cluster_size:
            logger.info(
                "NEW pool too small (%d < %d), skipping materialization.",
                len(self._new_pool), self.config.min_new_cluster_size,
            )
            return 0

        logger.info(
            "NEW pool: %d legacy records, running legacy promotion.",
            len(self._new_pool),
        )
        return self._materialize_legacy()

    def _materialize_legacy(self) -> int:
        """Legacy promotion: cluster by exact effect_signature string."""
        from skill_agents.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
        )
        from skill_agents.stage3_mvp.config import Stage3MVPConfig

        by_sig: Dict[str, List[SegmentRecord]] = defaultdict(list)
        for rec in self._new_pool:
            sig = rec.effect_signature()
            by_sig[sig].append(rec)

        created = 0
        ts = int(time.time())

        for sig, cluster in by_sig.items():
            if len(cluster) < self.config.min_new_cluster_size:
                continue

            new_id = f"S_new_{ts}_{created}"
            for rec in cluster:
                rec.skill_label = new_id

            specs = [
                SegmentSpec(
                    seg_id=rec.seg_id,
                    traj_id=rec.traj_id,
                    t_start=rec.t_start,
                    t_end=rec.t_end,
                    skill_label=new_id,
                    ui_events=list(rec.events) if getattr(rec, "events", None) else [],
                )
                for rec in cluster
            ]

            s3_config = Stage3MVPConfig(
                eff_freq=self.config.eff_freq,
                min_instances_per_skill=max(1, self.config.min_new_cluster_size),
            )

            summary = run_stage3_mvp(
                segments=specs,
                observations_by_traj=self._observations_by_traj,
                config=s3_config,
                bank=self.bank,
            )

            if new_id in summary.skill_results:
                sr = summary.skill_results[new_id]
                if sr.get("pass_rate", 0) >= self.config.split_pass_rate_threshold:
                    created += 1
                    for rec in cluster:
                        if rec in self._new_pool:
                            self._new_pool.remove(rec)
                else:
                    self.bank.remove(new_id)
                    for rec in cluster:
                        rec.skill_label = "__NEW__"

        if created and self.config.bank_path:
            self.bank.save(self.config.bank_path)
        self._invalidate_query_engine()
        return created

    # ── Skill evaluation ─────────────────────────────────────────────

    def run_evaluation(
        self,
        episode_outcomes: Optional[Dict[str, bool]] = None,
    ) -> Any:
        """Run the skill evaluation pipeline.

        Returns EvaluationSummary.
        """
        from skill_agents.skill_evaluation.run_evaluation import run_skill_evaluation
        from skill_agents.skill_evaluation.config import SkillEvaluationConfig

        eval_config = SkillEvaluationConfig()
        if self.config.llm_model:
            eval_config.llm.model = self.config.llm_model

        report_path = None
        if self.config.report_dir:
            report_path = str(Path(self.config.report_dir) / f"eval_iter{self._iteration}.json")

        summary = run_skill_evaluation(
            bank=self.bank,
            all_segments=self._all_segments,
            config=eval_config,
            episode_outcomes=episode_outcomes,
            report_path=report_path,
        )
        logger.info("Evaluation: %d skills evaluated", len(summary.skill_reports) if hasattr(summary, 'skill_reports') else 0)
        return summary

    # ── Full iteration ───────────────────────────────────────────────

    def run_full_iteration(
        self,
        episodes: Optional[Sequence] = None,
        env_name: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
        **kwargs,
    ) -> IterationSnapshot:
        """Execute one full pipeline iteration: (optional ingest) → Stage 3 → Stage 4 → materialize → snapshot.

        If *episodes* is provided, runs Stage 1+2 first.
        """
        self._iteration += 1
        logger.info("=== Pipeline iteration %d ===", self._iteration)

        if episodes is not None:
            self.ingest_episodes(episodes, env_name=env_name, skill_names=skill_names, **kwargs)
        else:
            if self._all_segments:
                self.run_contract_learning()

        self.run_sub_episode_quality_check()
        self.run_bank_maintenance()
        n_mat = self.materialize_new_skills()

        snap = self._take_snapshot(n_materialized=n_mat)
        self._history.append(snap)
        return snap

    def _take_snapshot(self, n_materialized: int = 0, maint_result=None) -> IterationSnapshot:
        n_new = sum(1 for s in self._all_segments if s.skill_label in ("__NEW__", "NEW"))
        n_total = len(self._all_segments) or 1

        pass_rates = []
        for sid in self.bank.skill_ids:
            r = self.bank.get_report(sid)
            if r is not None:
                pass_rates.append(r.overall_pass_rate)

        return IterationSnapshot(
            iteration=self._iteration,
            n_skills=len(self.bank),
            n_segments=len(self._all_segments),
            n_new_segments=n_new,
            new_rate=n_new / n_total,
            mean_pass_rate=sum(pass_rates) / len(pass_rates) if pass_rates else 0.0,
            n_materialized=n_materialized,
        )

    # ── Convergence loop ─────────────────────────────────────────────

    def run_until_stable(
        self,
        max_iterations: Optional[int] = None,
    ) -> List[IterationSnapshot]:
        """Iterate Stage 3 → Stage 4 until convergence or max iterations.

        Convergence: NEW rate < threshold and merge/split events become rare.
        """
        max_iter = max_iterations or self.config.max_iterations
        snapshots: List[IterationSnapshot] = []

        for _ in range(max_iter):
            snap = self.run_full_iteration()
            snapshots.append(snap)

            if self._is_converged(snap):
                logger.info("Pipeline converged at iteration %d", snap.iteration)
                break

        self.save()
        return snapshots

    def _is_converged(self, snap: IterationSnapshot) -> bool:
        if snap.new_rate > self.config.convergence_new_rate:
            return False
        if snap.n_splits > 0 or snap.n_merges > 0:
            return False
        if snap.mean_pass_rate < self.config.split_pass_rate_threshold:
            return False
        return True

    # ── Skill CRUD ───────────────────────────────────────────────────

    def add_skill(
        self,
        skill_id: str,
        eff_add: Optional[set] = None,
        eff_del: Optional[set] = None,
        eff_event: Optional[set] = None,
    ) -> SkillEffectsContract:
        """Manually add a skill to the bank."""
        contract = SkillEffectsContract(
            skill_id=skill_id,
            eff_add=eff_add or set(),
            eff_del=eff_del or set(),
            eff_event=eff_event or set(),
        )
        self.bank.add_or_update(contract)
        self._invalidate_query_engine()
        logger.info("Added skill %s", skill_id)
        return contract

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill from the bank."""
        if not self.bank.has_skill(skill_id):
            return False
        self.bank.remove(skill_id)
        self._invalidate_query_engine()
        logger.info("Removed skill %s", skill_id)
        return True

    def update_skill(
        self,
        skill_id: str,
        eff_add: Optional[set] = None,
        eff_del: Optional[set] = None,
        eff_event: Optional[set] = None,
    ) -> Optional[SkillEffectsContract]:
        """Update an existing skill's contract.  Returns updated contract or None."""
        contract = self.bank.get_contract(skill_id)
        if contract is None:
            logger.warning("Skill %s not found for update", skill_id)
            return None

        if eff_add is not None:
            contract.eff_add = eff_add
        if eff_del is not None:
            contract.eff_del = eff_del
        if eff_event is not None:
            contract.eff_event = eff_event
        contract.bump_version()
        self.bank.add_or_update(contract)
        self._invalidate_query_engine()
        logger.info("Updated skill %s to version %d", skill_id, contract.version)
        return contract

    # ── Query API (used by decision_agents) ──────────────────────────

    def query_skill(self, key: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query skills by natural-language key (scene/objective/entities).

        Returns a list of dicts with skill_id, contract summary, and match info,
        suitable for decision_agents.
        """
        engine = self._get_query_engine()
        return engine.query(key, top_k=top_k)

    def select_skill(
        self,
        query: str,
        current_state: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rich skill selection combining retrieval relevance with execution
        applicability.

        Preferred API for decision agents.  When ``current_state`` is
        provided, the result includes ``applicability`` and ``confidence``
        in addition to ``relevance``.

        Returns list of ``SkillSelectionResult.to_dict()``.
        """
        engine = self._get_query_engine()
        results = engine.select(query, current_state=current_state, top_k=top_k)
        return [r.to_dict() for r in results]

    def query_by_effects(
        self,
        desired_add: Optional[set] = None,
        desired_del: Optional[set] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find skills whose effects best match desired state changes."""
        engine = self._get_query_engine()
        return engine.query_by_effects(desired_add=desired_add, desired_del=desired_del, top_k=top_k)

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all skills with compact summaries."""
        engine = self._get_query_engine()
        return engine.list_all()

    def get_skill_detail(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get full detail for one skill (contract + report + quality)."""
        engine = self._get_query_engine()
        return engine.get_detail(skill_id)

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def skill_ids(self) -> List[str]:
        return self.bank.skill_ids

    def get_contract(self, skill_id: str) -> Optional[SkillEffectsContract]:
        return self.bank.get_contract(skill_id)

    def get_bank(self) -> SkillBankMVP:
        return self.bank

    @property
    def segments(self) -> List[SegmentRecord]:
        return list(self._all_segments)

    @property
    def iteration_history(self) -> List[IterationSnapshot]:
        return list(self._history)

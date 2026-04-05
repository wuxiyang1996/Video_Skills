"""
SkillBank Agent — agentic pipeline that builds, maintains, and serves a Skill Bank.

Orchestrates boundary_proposal (Stage 1), infer_segmentation (Stage 2),
stage3_mvp (Stage 3 contract learn/verify/refine), bank_maintenance (Stage 4
split/merge/refine), and skill_evaluation.  Exposes a query API consumed by
decision_agents.

Typical usage::

    from skill_agents.pipeline import SkillBankAgent, PipelineConfig

    agent = SkillBankAgent(bank_path="skills/bank.jsonl")
    agent.ingest_episodes(episodes, env_name="llm+avalon")
    agent.run_until_stable(max_iterations=3)

    # Decision-agent queries the bank
    result = agent.query_skill("propose team and vote on quest")
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
from skill_agents.skill_bank.new_pool import (
    NewPoolManager,
    NewPoolConfig,
    ProtoSkillManager,
    ProtoSkillConfig,
)
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
    extractor_model: Optional[str] = None

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
        self._proto_mgr = ProtoSkillManager(config=ProtoSkillConfig())
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

    @staticmethod
    def _seed_skills_from_intentions(episode) -> List[str]:
        """Extract unique canonical intention tags from an episode.

        Used as seed skill names for the first pass when the bank is empty,
        so the DP decoder has real labels to assign instead of only __NEW__.
        """
        from skill_agents.boundary_proposal.signal_extractors import (
            parse_intention_tag,
        )
        tags = set()
        for exp in episode.experiences:
            intent = getattr(exp, "intentions", None)
            if intent:
                tag = parse_intention_tag(intent)
                if tag != "UNKNOWN":
                    tags.add(tag)
        return sorted(tags)

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

        # When the bank is empty, seed skill names from the episode's
        # intention tags so the DP decoder has real labels to assign
        # (otherwise only __NEW__ is available and the penalty makes it
        # always prefer 1 segment).
        if not _skill_names:
            _skill_names = self._seed_skills_from_intentions(episode)

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

    def refine_low_pass_protocols(
        self,
        pass_rate_threshold: float = 0.4,
        min_episodes: int = 3,
    ) -> int:
        """Re-synthesize protocols for skills with a low success rate.

        Called periodically to give struggling skills a chance to improve
        their protocols using the latest sub-episode evidence.

        Returns the number of protocols re-synthesized.
        """
        from skill_agents.stage3_mvp.schemas import Skill

        refined = 0
        for sid in self.bank.skill_ids:
            skill = self.bank.get_skill(sid)
            if skill is None:
                continue
            if not skill.protocol or not skill.protocol.steps:
                continue
            if skill.success_rate >= pass_rate_threshold:
                continue
            if len(skill.sub_episodes) < min_episodes:
                continue

            all_eps = sorted(
                skill.sub_episodes, key=lambda se: se.quality_score, reverse=True,
            )
            best_eps = all_eps[:max(3, len(all_eps) // 2)]

            new_protocol = self._synthesize_protocol(skill, best_eps)
            if new_protocol is not None:
                skill.bump_version()
                skill.protocol = new_protocol
                self.bank.add_or_update_skill(skill)
                refined += 1
                logger.info(
                    "Refined protocol for low-pass skill %s (rate=%.2f, v%d)",
                    sid, skill.success_rate, skill.version,
                )

        if refined:
            self._invalidate_query_engine()
        return refined

    def _synthesize_protocol(
        self,
        skill: Skill,
        high_quality_eps: list,
    ) -> Optional:
        """Synthesize a Protocol from high-quality sub-episode pointers.

        Uses sub-episode summaries and intention_tags (cached on the
        pointers) plus contract effects.  When an LLM model is configured
        (via ``config.llm_model`` or ``config.extractor_model``), generates
        richer protocols via the same ``ask_model`` routing used by both
        GPT and Qwen backends.
        """
        from skill_agents.stage3_mvp.schemas import Protocol

        contract = skill.contract

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
            from decision_agents.protocol_utils import compute_expected_duration
            avg_len = compute_expected_duration(lengths)

        # Collect evidence for LLM synthesis
        summaries = [se.summary for se in high_quality_eps if se.summary]
        effects_desc = ""
        if contract:
            parts = []
            if contract.eff_add:
                parts.append("achieves: " + ", ".join(sorted(contract.eff_add)[:5]))
            if contract.eff_del:
                parts.append("removes: " + ", ".join(sorted(contract.eff_del)[:3]))
            if contract.eff_event:
                parts.append("triggers: " + ", ".join(sorted(contract.eff_event)[:3]))
            effects_desc = "; ".join(parts)

        # Try LLM-based synthesis (model-agnostic: routes to GPT or Qwen)
        llm_model = self.config.llm_model or self.config.extractor_model
        action_vocab = getattr(self.config, "action_vocab", None) or []
        protocol = self._llm_synthesize_protocol(
            skill, summaries, effects_desc, llm_model,
            action_vocab=action_vocab,
        )
        if protocol is not None:
            protocol.expected_duration = max(1, avg_len)
            if skill.success_rate < 0.5:
                protocol.abort_criteria.append(
                    "Abort if no progress after expected duration"
                )
            return protocol

        # Deterministic fallback (no LLM available)
        steps = []
        preconditions = []
        success_criteria = []

        if summaries:
            seen = set()
            for s in summaries:
                key = s.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    steps.append(s.strip())
                if len(steps) >= 5:
                    break

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

    def _llm_synthesize_protocol(
        self,
        skill,
        summaries: List[str],
        effects_desc: str,
        model: Optional[str],
        action_vocab: Optional[List[str]] = None,
    ) -> Optional:
        """Use ask_model (model-agnostic) to generate a structured protocol.

        Works identically for GPT and Qwen because ask_model routes to the
        correct backend based on the model name.  Reasoning models (Qwen3)
        are handled transparently via ``wrap_ask_for_reasoning_models``.
        """
        try:
            from API_func import ask_model as _raw_ask
        except ImportError:
            return None
        if _raw_ask is None or model is None:
            return None

        from skill_agents._llm_compat import wrap_ask_for_reasoning_models
        from skill_agents.stage3_mvp.schemas import Protocol
        import json as _json

        _ask = wrap_ask_for_reasoning_models(_raw_ask, model_hint=model)

        evidence = "\n".join(f"  - {s}" for s in summaries[:5]) if summaries else "(none)"
        skill_name = skill.name or skill.skill_id

        action_block = ""
        if action_vocab:
            action_block = (
                f"\nAvailable game actions: {', '.join(action_vocab[:20])}\n"
                f"IMPORTANT: Reference these exact action names in your steps.\n"
            )

        prompt = (
            f"You are a game-AI protocol designer. Generate a concrete execution "
            f"protocol for the skill below.\n\n"
            f"Skill: {skill_name}\n"
            f"Description: {skill.strategic_description or '(none)'}\n"
            f"Effects: {effects_desc or '(none)'}\n"
            f"Evidence from successful executions:\n{evidence}\n"
            f"{action_block}\n"
            f"Generate a structured protocol as JSON with these exact keys:\n"
            f'{{"preconditions": ["..."], "steps": ["..."], '
            f'"step_checks": ["..."], '
            f'"success_criteria": ["..."], "abort_criteria": ["..."], '
            f'"predicate_success": ["key=value", ...], '
            f'"predicate_abort": ["key>N", ...]}}\n'
            f"Rules:\n"
            f"- preconditions: 1-3 specific conditions (game situation, state) "
            f"that must hold before starting\n"
            f"- steps: 2-7 concrete action steps (imperative, game-specific). "
            f"Do NOT write generic steps like 'Achieve: X' or 'Execute skill'. "
            f"Reference actual game actions when possible.\n"
            f"- step_checks: one entry per step — a key=value condition from "
            f"the game state that indicates the step is complete "
            f"(e.g. 'stack_h<5', 'quest=2'). Use empty string '' if no "
            f"specific check applies.\n"
            f"- success_criteria: 1-3 human-readable descriptions of success\n"
            f"- abort_criteria: 1-2 human-readable conditions to stop early\n"
            f"- predicate_success: 1-3 machine-checkable key=value or key<N "
            f"conditions from the game state (e.g. 'phase=endgame', "
            f"'holes<5')\n"
            f"- predicate_abort: 1-2 machine-checkable conditions "
            f"(e.g. 'stack_h>18', 'moves<3')\n"
            f"Reply with ONLY the JSON object."
        )
        try:
            reply = _ask(prompt, model=model, temperature=0.2, max_tokens=800)
            if not reply or reply.startswith("Error"):
                return None
            import re
            json_m = re.search(r"\{[\s\S]*\}", reply)
            if not json_m:
                return None
            data = _json.loads(json_m.group(0))
            steps = data.get("steps", [])[:7]
            step_checks = data.get("step_checks", [])[:7]
            if step_checks and len(step_checks) < len(steps):
                step_checks.extend([""] * (len(steps) - len(step_checks)))
            return Protocol(
                preconditions=data.get("preconditions", [])[:5],
                steps=steps,
                success_criteria=data.get("success_criteria", [])[:5],
                abort_criteria=data.get("abort_criteria", [])[:3],
                step_checks=step_checks,
                predicate_success=data.get("predicate_success", [])[:5],
                predicate_abort=data.get("predicate_abort", [])[:3],
                action_vocab=action_vocab or [],
            )
        except Exception as exc:
            logger.debug("LLM protocol synthesis failed: %s", exc)
            return None

    # ── Phase 5: Distill execution hints ────────────────────────────

    def distill_execution_hints(self, min_successful: int = 3) -> int:
        """Derive lightweight execution hints for skills with enough evidence.

        For each skill, aggregates successful sub-episodes to produce:
        - common preconditions
        - common target objects
        - state-transition pattern
        - termination cues
        - common failure modes
        - natural-language execution description

        Returns the number of skills updated with hints.
        """
        from skill_agents.stage3_mvp.schemas import ExecutionHint
        from collections import Counter

        updated = 0
        for sid in self.bank.skill_ids:
            skill = self.bank.get_skill(sid)
            if skill is None or skill.retired:
                continue

            successful = [
                se for se in skill.sub_episodes
                if se.outcome == "success" or se.quality_score >= 0.6
            ]
            if len(successful) < min_successful:
                continue

            # Skip if hint is recent and based on enough data
            if (skill.execution_hint is not None
                    and skill.execution_hint.n_source_segments >= len(successful)):
                continue

            contract = skill.contract

            # --- common preconditions ---
            preconditions: List[str] = []
            if skill.protocol.preconditions:
                preconditions = list(skill.protocol.preconditions[:5])

            # --- common target objects ---
            obj_counter: Counter = Counter()
            for se in successful:
                for tag in se.intention_tags:
                    parts = tag.strip("[]").split("_")
                    for p in parts:
                        if p and len(p) > 2 and p.upper() != p:
                            obj_counter[p.lower()] += 1
            target_objects = [obj for obj, _ in obj_counter.most_common(5)]

            # --- state-transition pattern ---
            transition_pattern = ""
            if contract is not None:
                add_str = ", ".join(sorted(contract.eff_add)[:3]) if contract.eff_add else ""
                del_str = ", ".join(sorted(contract.eff_del)[:3]) if contract.eff_del else ""
                parts = []
                if del_str:
                    parts.append(f"removes [{del_str}]")
                if add_str:
                    parts.append(f"achieves [{add_str}]")
                transition_pattern = " → ".join(parts)

            # --- termination cues ---
            termination_cues: List[str] = []
            if skill.protocol.success_criteria:
                termination_cues = list(skill.protocol.success_criteria[:3])
            elif contract is not None and contract.eff_add:
                termination_cues = [
                    f"{lit} becomes true" for lit in sorted(contract.eff_add)[:3]
                ]

            # --- failure modes ---
            failure_modes: List[str] = []
            report = self.bank.get_report(sid)
            if report is not None and report.failure_signatures:
                failure_modes = [
                    f"{sig} ({cnt}x)"
                    for sig, cnt in sorted(
                        report.failure_signatures.items(), key=lambda x: -x[1]
                    )[:3]
                ]
            if skill.protocol.abort_criteria:
                failure_modes.extend(skill.protocol.abort_criteria[:2])

            if not failure_modes and skill.tags:
                primary_tag = skill.tags[0].upper() if skill.tags else ""
                _TAG_FAILURE_DEFAULTS = {
                    "SURVIVE": "Board state deteriorates despite defensive moves",
                    "DEFEND": "Board state deteriorates despite defensive moves",
                    "MERGE": "No merge opportunities available on any legal move",
                    "POSITION": "Structure broken — anchor tile dislodged or ordering disrupted",
                    "SETUP": "Structure broken — anchor tile dislodged or ordering disrupted",
                    "CLEAR": "Clearing move creates worse congestion than before",
                    "ATTACK": "Attack creates vulnerability without achieving objective",
                    "NAVIGATE": "Path blocked or position unchanged after several moves",
                    "COLLECT": "Target resource depleted or unreachable",
                    "BUILD": "Required materials unavailable or placement blocked",
                }
                if primary_tag in _TAG_FAILURE_DEFAULTS:
                    failure_modes.append(_TAG_FAILURE_DEFAULTS[primary_tag])
                else:
                    failure_modes.append(
                        "No progress toward skill objective after several moves"
                    )

            # --- execution description ---
            exec_desc = ""
            if skill.strategic_description:
                exec_desc = skill.strategic_description
            elif skill.protocol.steps:
                exec_desc = " → ".join(skill.protocol.steps[:4])
            elif successful:
                summaries = [se.summary for se in successful if se.summary][:3]
                if summaries:
                    exec_desc = "; ".join(summaries)

            # Compute typical durations
            lengths = [se.length for se in successful]
            avg_len = sum(lengths) / max(len(lengths), 1)

            hint = ExecutionHint(
                common_preconditions=preconditions,
                common_target_objects=target_objects,
                state_transition_pattern=transition_pattern,
                termination_cues=termination_cues,
                common_failure_modes=failure_modes[:5],
                execution_description=exec_desc,
                n_source_segments=len(successful),
            )

            skill.execution_hint = hint
            self.bank.add_or_update_skill(skill)
            updated += 1

        if updated:
            logger.info("Distilled execution hints for %d skills.", updated)
            self._invalidate_query_engine()

        return updated

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

    # ── Proto-Skill Management ────────────────────────────────────────

    def form_proto_skills(self) -> int:
        """Scan NEW pool clusters and form proto-skills.

        Proto-skills are lightweight intermediates that can participate
        in Stage 2 decoding before full promotion.

        Returns the number of proto-skills created.
        """
        existing = set(self.bank.skill_ids)
        created = self._proto_mgr.form_from_pool(
            self._new_pool_mgr, existing_bank_skills=existing,
        )
        if created:
            logger.info(
                "Formed %d proto-skills: %s",
                len(created),
                [p.proto_id for p in created[:5]],
            )
        return len(created)

    def verify_proto_skills(self) -> int:
        """Run light verification on all unverified proto-skills.

        Returns the number of proto-skills verified.
        """
        verified = 0
        for pid in list(self._proto_mgr.proto_ids):
            proto = self._proto_mgr.get(pid)
            if proto is None or proto.verified:
                continue
            pass_rate = self._proto_mgr.verify(
                pid, self.bank, self._observations_by_traj,
            )
            if pass_rate is not None:
                verified += 1
                logger.info(
                    "Proto-skill %s verified: pass_rate=%.3f",
                    pid, pass_rate,
                )
        return verified

    def promote_proto_skills(self) -> int:
        """Promote qualifying proto-skills to real skills.

        Returns the number of skills promoted.
        """
        promoted = self._proto_mgr.promote_ready(self.bank)
        if promoted:
            logger.info("Promoted %d proto-skills to real skills: %s", len(promoted), promoted[:5])
            self._invalidate_query_engine()
            if self.config.bank_path:
                self.bank.save(self.config.bank_path)
        return len(promoted)

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
        """Execute one full pipeline iteration.

        Pipeline: (optional ingest) → Stage 3 → quality check
        → Stage 4 → proto-skill formation → verification → promotion
        → materialize remaining NEW → execution hints → snapshot.

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

        # Phase 4: proto-skill layer (NEW → cluster → proto → verify → promote)
        n_proto = self.form_proto_skills()
        if n_proto > 0:
            self.verify_proto_skills()
        n_promoted = self.promote_proto_skills()

        n_mat = self.materialize_new_skills()

        # Phase 5: distill execution hints for skills with enough evidence
        self.distill_execution_hints()

        snap = self._take_snapshot(n_materialized=n_mat + n_promoted)
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

    def query_skill(
        self,
        key: str,
        top_k: int = 3,
        current_state: Optional[Dict[str, float]] = None,
        current_predicates: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Query skills by natural-language key (scene/objective/entities).

        When *current_state* or *current_predicates* is provided, defaults
        to the rich state-aware ``select()`` path which uses applicability,
        pass rate, matched/missing effects, and state-conditioned ranking.

        Falls back to the simpler retrieval-only ``query()`` path only when
        no state information is available.

        Returns a list of structured guidance dicts suitable for
        decision_agents.
        """
        engine = self._get_query_engine()
        state = current_state or current_predicates
        if state is not None:
            results = engine.select(
                key, current_state=state, top_k=top_k,
            )
            return [r.to_dict() for r in results]
        return engine.query(key, top_k=top_k)

    def select_skill(
        self,
        query: str,
        current_state: Optional[Dict[str, float]] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rich skill selection combining retrieval relevance with execution
        applicability and structured guidance.

        Preferred API for decision agents.  When ``current_state`` or
        ``current_predicates`` is provided, the result includes full
        guidance: applicability, confidence, why_selected, expected_effects,
        preconditions, termination_hint, failure_modes, execution_hint.

        Returns list of ``SkillSelectionResult.to_dict()``.
        """
        engine = self._get_query_engine()
        state = current_state or current_predicates
        results = engine.select(
            query, current_state=state, top_k=top_k,
        )
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

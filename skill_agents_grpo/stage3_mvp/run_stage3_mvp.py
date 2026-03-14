"""
Step 7 — Orchestrator: run the full Stage 3 MVP pipeline.

Pipeline:
  1. Summarize predicates for each segment.
  2. Compute per-instance effects.
  3. Group by skill label (skip "NEW").
  4. For each skill with enough instances:
     a. Learn initial effects contract.
     b. Verify contract.
     c. Refine contract.
     d. Persist into skill bank.
  5. Emit a debug report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig
from skill_agents_grpo.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents_grpo.stage3_mvp.predicate_vocab import PredicateVocab
from skill_agents_grpo.stage3_mvp.extract_predicates import default_extract_predicates
from skill_agents_grpo.stage3_mvp.segment_summarize import summarize_segment
from skill_agents_grpo.stage3_mvp.effects_compute import compute_effects
from skill_agents_grpo.stage3_mvp.contract_learn import learn_effects_contract
from skill_agents_grpo.stage3_mvp.contract_verify import verify_effects_contract
from skill_agents_grpo.stage3_mvp.contract_refine import refine_effects_contract


# ── Segment specification (light input schema) ──────────────────────

@dataclass
class SegmentSpec:
    """Minimal descriptor for a segment from Stage 2 output."""

    seg_id: str
    traj_id: str
    t_start: int
    t_end: int
    skill_label: str
    ui_events: List[str] = field(default_factory=list)


# ── Per-skill result ─────────────────────────────────────────────────

@dataclass
class SkillResult:
    """Contract + report for one skill after learn-verify-refine."""

    skill_id: str
    contract: SkillEffectsContract
    report_initial: VerificationReport
    contract_refined: SkillEffectsContract
    report_refined: Optional[VerificationReport] = None

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "contract": self.contract_refined.to_dict(),
            "initial_contract": self.contract.to_dict(),
            "report_initial": self.report_initial.to_dict(),
            "report_refined": (
                self.report_refined.to_dict() if self.report_refined else None
            ),
        }


# ── Debug report ─────────────────────────────────────────────────────

@dataclass
class Stage3MVPSummary:
    """Compact summary of a full Stage 3 MVP run."""

    n_segments: int = 0
    n_skills_processed: int = 0
    n_skills_skipped: int = 0
    skill_results: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_segments": self.n_segments,
            "n_skills_processed": self.n_skills_processed,
            "n_skills_skipped": self.n_skills_skipped,
            "skill_results": self.skill_results,
        }

    def __str__(self) -> str:
        lines = [
            f"Stage 3 MVP Summary: {self.n_skills_processed} skills processed, "
            f"{self.n_skills_skipped} skipped, {self.n_segments} segments total",
        ]
        for sid, info in self.skill_results.items():
            lines.append(
                f"  {sid}: {info['n_literals_kept']} literals, "
                f"pass_rate={info['pass_rate']:.2f}, "
                f"top_fails={info.get('top_failing_literals', [])}"
            )
        return "\n".join(lines)


# ── Main pipeline ────────────────────────────────────────────────────

def run_stage3_mvp(
    segments: List[SegmentSpec],
    observations_by_traj: Dict[str, Sequence[Any]],
    config: Optional[Stage3MVPConfig] = None,
    vocab: Optional[PredicateVocab] = None,
    extract_fn: Optional[Callable] = None,
    bank: Optional[SkillBankMVP] = None,
    bank_path: Optional[str] = None,
) -> Stage3MVPSummary:
    """Run the full Stage 3 MVP pipeline.

    Parameters
    ----------
    segments : list[SegmentSpec]
        Segment descriptors from Stage 2 output.
    observations_by_traj : dict[str, Sequence]
        Mapping ``traj_id -> observations`` (indexed by timestep).
    config : Stage3MVPConfig, optional
        Pipeline configuration. Uses defaults if not provided.
    vocab : PredicateVocab, optional
        Predicate registry. A fresh one is created if not provided.
    extract_fn : callable, optional
        ``(obs) -> {predicate: prob}``.  Falls back to no-op extractor.
    bank : SkillBankMVP, optional
        Existing bank to update.  A new one is created if not provided.
    bank_path : str, optional
        Path to save the skill bank JSONL after the run.

    Returns
    -------
    Stage3MVPSummary
    """
    if config is None:
        config = Stage3MVPConfig()
    if vocab is None:
        vocab = PredicateVocab()
    if extract_fn is None:
        extract_fn = default_extract_predicates
    if bank is None:
        from skill_agents_grpo.skill_bank.bank import SkillBankMVP
        bank = SkillBankMVP(path=bank_path)

    # Step 1+2: summarize and compute effects for every segment
    records: List[SegmentRecord] = []
    for spec in segments:
        obs = observations_by_traj.get(spec.traj_id, [])
        rec = summarize_segment(
            seg_id=spec.seg_id,
            traj_id=spec.traj_id,
            t_start=spec.t_start,
            t_end=spec.t_end,
            skill_label=spec.skill_label,
            observations=obs,
            extract_predicates=extract_fn,
            config=config,
            vocab=vocab,
            ui_events=spec.ui_events,
        )
        compute_effects(rec, config, vocab)
        records.append(rec)

    # Group by skill label, exclude "NEW"
    by_skill: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for rec in records:
        if rec.skill_label.upper() == "NEW":
            continue
        by_skill[rec.skill_label].append(rec)

    summary = Stage3MVPSummary(n_segments=len(records))
    skill_results: Dict[str, SkillResult] = {}

    for skill_id, instances in sorted(by_skill.items()):
        if len(instances) < config.min_instances_per_skill:
            summary.n_skills_skipped += 1
            continue

        # Step 3: learn
        prev_ver = 0
        existing = bank.get_contract(skill_id)
        if existing is not None:
            prev_ver = existing.version

        contract = learn_effects_contract(skill_id, instances, config, prev_ver)

        # Step 4: verify
        report = verify_effects_contract(contract, instances, config)

        # Step 5: refine
        refined = refine_effects_contract(contract, report, config)

        # Re-verify the refined contract
        report_refined = verify_effects_contract(refined, instances, config)

        sr = SkillResult(
            skill_id=skill_id,
            contract=contract,
            report_initial=report,
            contract_refined=refined,
            report_refined=report_refined,
        )
        skill_results[skill_id] = sr

        # Step 6: persist
        bank.add_or_update(refined, report_refined)

        # Build summary entry
        top_fails = _top_failing_literals(report, n=5)
        summary.skill_results[skill_id] = {
            "n_instances": len(instances),
            "n_literals_initial": contract.total_literals,
            "n_literals_kept": refined.total_literals,
            "pass_rate": report_refined.overall_pass_rate,
            "pass_rate_initial": report.overall_pass_rate,
            "top_failing_literals": top_fails,
            "worst_segments": report_refined.worst_segments[:3],
        }
        summary.n_skills_processed += 1

    # Save bank
    if bank_path or bank._path:
        bank.save(bank_path)

    return summary


def _top_failing_literals(report: VerificationReport, n: int = 5) -> List[str]:
    """Return up to *n* literals with the lowest success rates."""
    all_rates: List[tuple] = []
    for p, r in report.eff_add_success_rate.items():
        all_rates.append((f"add:{p}", r))
    for p, r in report.eff_del_success_rate.items():
        all_rates.append((f"del:{p}", r))
    for e, r in report.eff_event_rate.items():
        all_rates.append((f"evt:{e}", r))
    all_rates.sort(key=lambda x: x[1])
    return [lit for lit, _ in all_rates[:n]]


# ── Convenience: build SegmentSpec from Stage 2 SegmentationResult ───

def specs_from_segmentation_result(
    result: Any,
    traj_id: str,
    ui_event_log: Optional[List[dict]] = None,
    timestamp_key: str = "t",
    event_key: str = "event",
) -> List[SegmentSpec]:
    """Convert a Stage 2 ``SegmentationResult`` into ``SegmentSpec`` list.

    Parameters
    ----------
    result
        Stage 2 output with a ``.segments`` attribute (list of
        ``SegmentDiagnostic`` with ``.start``, ``.end``, ``.assigned_skill``).
    traj_id : str
        Trajectory identifier.
    ui_event_log : list[dict], optional
        UI event log entries with timestamp and event fields.
    """
    from skill_agents_grpo.stage3_mvp.extract_predicates import extract_ui_events_from_log

    specs: List[SegmentSpec] = []
    for idx, seg in enumerate(result.segments):
        seg_id = f"{traj_id}_seg{idx:04d}"
        events: List[str] = []
        if ui_event_log is not None:
            events = extract_ui_events_from_log(
                ui_event_log, seg.start, seg.end,
                timestamp_key=timestamp_key, event_key=event_key,
            )
        specs.append(SegmentSpec(
            seg_id=seg_id,
            traj_id=traj_id,
            t_start=seg.start,
            t_end=seg.end,
            skill_label=seg.assigned_skill,
            ui_events=events,
        ))
    return specs

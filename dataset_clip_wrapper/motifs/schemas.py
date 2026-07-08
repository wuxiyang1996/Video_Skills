"""Schemas for reusable L1/L2 graph motifs.

The design mirrors the useful parts of COS-PLAY's skill bank while keeping a
video-QA boundary: motifs are graph priors, not executable actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "video-skills/motif-v0.1"


class MotifLifecycleStatus(str, Enum):
    """Lifecycle states for banked motifs."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    VERIFIED = "verified"
    ACTIVE = "active"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class MotifEvidenceRef:
    """Pointer to one rollout or graph slice that supports a motif."""

    dataset: str
    example_id: str
    task_family: str = ""
    source_path: str = ""
    l1_node_ids: tuple[str, ...] = ()
    l2_node_ids: tuple[str, ...] = ()
    final_answer_correct: bool | None = None
    verifier_passed: bool | None = None
    evidence_valid: bool | None = None
    no_hidden_leakage: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "task_family": self.task_family,
            "source_path": self.source_path,
            "l1_node_ids": list(self.l1_node_ids),
            "l2_node_ids": list(self.l2_node_ids),
            "final_answer_correct": self.final_answer_correct,
            "verifier_passed": self.verifier_passed,
            "evidence_valid": self.evidence_valid,
            "no_hidden_leakage": self.no_hidden_leakage,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotifEvidenceRef":
        return cls(
            dataset=str(payload.get("dataset") or ""),
            example_id=str(payload.get("example_id") or ""),
            task_family=str(payload.get("task_family") or ""),
            source_path=str(payload.get("source_path") or ""),
            l1_node_ids=tuple(str(x) for x in payload.get("l1_node_ids") or ()),
            l2_node_ids=tuple(str(x) for x in payload.get("l2_node_ids") or ()),
            final_answer_correct=payload.get("final_answer_correct"),
            verifier_passed=payload.get("verifier_passed"),
            evidence_valid=payload.get("evidence_valid"),
            no_hidden_leakage=payload.get("no_hidden_leakage"),
        )


@dataclass
class MotifTransferReport:
    """Empirical transfer evidence for a motif on heldout examples."""

    target_dataset: str
    target_task_family: str
    n_total: int = 0
    n_success: int = 0
    baseline_success_rate: float | None = None
    motif_success_rate: float | None = None
    verifier_pass_rate: float = 0.0
    evidence_valid_rate: float = 0.0
    no_leakage_rate: float = 0.0
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def pass_rate(self) -> float:
        if self.n_total <= 0:
            return 0.0
        return self.n_success / self.n_total

    @property
    def delta_over_baseline(self) -> float | None:
        if self.baseline_success_rate is None or self.motif_success_rate is None:
            return None
        return self.motif_success_rate - self.baseline_success_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dataset": self.target_dataset,
            "target_task_family": self.target_task_family,
            "n_total": self.n_total,
            "n_success": self.n_success,
            "pass_rate": round(self.pass_rate, 6),
            "baseline_success_rate": self.baseline_success_rate,
            "motif_success_rate": self.motif_success_rate,
            "delta_over_baseline": self.delta_over_baseline,
            "verifier_pass_rate": self.verifier_pass_rate,
            "evidence_valid_rate": self.evidence_valid_rate,
            "no_leakage_rate": self.no_leakage_rate,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotifTransferReport":
        return cls(
            target_dataset=str(payload.get("target_dataset") or ""),
            target_task_family=str(payload.get("target_task_family") or ""),
            n_total=int(payload.get("n_total") or 0),
            n_success=int(payload.get("n_success") or 0),
            baseline_success_rate=payload.get("baseline_success_rate"),
            motif_success_rate=payload.get("motif_success_rate"),
            verifier_pass_rate=float(payload.get("verifier_pass_rate") or 0.0),
            evidence_valid_rate=float(payload.get("evidence_valid_rate") or 0.0),
            no_leakage_rate=float(payload.get("no_leakage_rate") or 0.0),
            notes=[str(x) for x in payload.get("notes") or []],
            created_at=float(payload.get("created_at") or 0.0),
        )


@dataclass
class MotifRecord:
    """Reusable L1/L2 graph motif stored in a motif bank."""

    motif_id: str
    name: str
    description: str
    motif_type: str = "l1_l2_graph_template"
    status: MotifLifecycleStatus = MotifLifecycleStatus.DRAFT
    trigger_signature: dict[str, Any] = field(default_factory=dict)
    l1_template: dict[str, Any] = field(default_factory=dict)
    l2_template: dict[str, Any] = field(default_factory=dict)
    expansion_constraints: list[str] = field(default_factory=lambda: [
        "expand_before_execution",
        "cite_current_video_evidence_only",
        "do_not_answer_directly",
        "run_l1_l2_verifiers",
        "do_not_mine_heldout_test_for_thresholds",
    ])
    evidence_refs: list[MotifEvidenceRef] = field(default_factory=list)
    transfer_reports: list[MotifTransferReport] = field(default_factory=list)
    proposal_source: str = ""
    proposal_model: str = ""
    proposal_confidence: float | None = None
    curator_model: str = ""
    curator_verdict: str = ""
    curator_reason: str = ""
    false_binding_patterns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def support_count(self) -> int:
        return len(self.evidence_refs)

    @property
    def verified_task_families(self) -> list[str]:
        families = {
            report.target_task_family
            for report in self.transfer_reports
            if report.n_total > 0 and report.pass_rate > 0.0
        }
        return sorted(family for family in families if family)

    @property
    def empirical_confidence(self) -> float:
        """Evidence-backed confidence, separate from LLM proposal confidence."""

        if not self.transfer_reports:
            return 0.0
        scores = []
        for report in self.transfer_reports:
            gate_score = min(
                report.pass_rate,
                report.verifier_pass_rate,
                report.evidence_valid_rate,
                report.no_leakage_rate,
            )
            scores.append(gate_score)
        return sum(scores) / len(scores)

    def add_evidence(self, ref: MotifEvidenceRef) -> None:
        key = (ref.dataset, ref.example_id, ref.l1_node_ids, ref.l2_node_ids)
        existing = {
            (item.dataset, item.example_id, item.l1_node_ids, item.l2_node_ids)
            for item in self.evidence_refs
        }
        if key not in existing:
            self.evidence_refs.append(ref)
            self.updated_at = time.time()

    def add_transfer_report(self, report: MotifTransferReport) -> None:
        self.transfer_reports.append(report)
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "motif_id": self.motif_id,
            "name": self.name,
            "description": self.description,
            "motif_type": self.motif_type,
            "status": self.status.value,
            "trigger_signature": self.trigger_signature,
            "l1_template": self.l1_template,
            "l2_template": self.l2_template,
            "expansion_constraints": self.expansion_constraints,
            "support": {
                "support_count": self.support_count,
                "verified_task_families": self.verified_task_families,
                "empirical_confidence": round(self.empirical_confidence, 6),
            },
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "transfer_reports": [report.to_dict() for report in self.transfer_reports],
            "agent": {
                "proposal_source": self.proposal_source,
                "proposal_model": self.proposal_model,
                "proposal_confidence": self.proposal_confidence,
                "curator_model": self.curator_model,
                "curator_verdict": self.curator_verdict,
                "curator_reason": self.curator_reason,
            },
            "false_binding_patterns": self.false_binding_patterns,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotifRecord":
        agent = payload.get("agent") or {}
        return cls(
            motif_id=str(payload["motif_id"]),
            name=str(payload.get("name") or payload["motif_id"]),
            description=str(payload.get("description") or ""),
            motif_type=str(payload.get("motif_type") or "l1_l2_graph_template"),
            status=MotifLifecycleStatus(payload.get("status") or "draft"),
            trigger_signature=payload.get("trigger_signature") or {},
            l1_template=payload.get("l1_template") or {},
            l2_template=payload.get("l2_template") or {},
            expansion_constraints=[
                str(x) for x in payload.get("expansion_constraints") or []
            ],
            evidence_refs=[
                MotifEvidenceRef.from_dict(x)
                for x in payload.get("evidence_refs") or []
            ],
            transfer_reports=[
                MotifTransferReport.from_dict(x)
                for x in payload.get("transfer_reports") or []
            ],
            proposal_source=str(agent.get("proposal_source") or ""),
            proposal_model=str(agent.get("proposal_model") or ""),
            proposal_confidence=agent.get("proposal_confidence"),
            curator_model=str(agent.get("curator_model") or ""),
            curator_verdict=str(agent.get("curator_verdict") or ""),
            curator_reason=str(agent.get("curator_reason") or ""),
            false_binding_patterns=[
                str(x) for x in payload.get("false_binding_patterns") or []
            ],
            notes=[str(x) for x in payload.get("notes") or []],
            created_at=float(payload.get("created_at") or 0.0),
            updated_at=float(payload.get("updated_at") or 0.0),
        )

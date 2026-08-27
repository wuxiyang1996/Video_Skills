"""Persistent motif bank for mined L1/L2 graph motifs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MotifInstance:
    motif_type: str
    signature: str
    dataset: str
    example_id: str
    task_family: str
    video_regime: str
    final_status: str
    verifier_passed: bool
    graph_template: dict[str, Any]
    trigger_signature: dict[str, Any]
    expansion_template: dict[str, Any]
    source_path: str = ""
    proposal_source: str = "deterministic_seed"
    agent_backend: str = ""
    curator_verdict: str = ""
    curator_reason: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_type": self.motif_type,
            "signature": self.signature,
            "dataset": self.dataset,
            "example_id": self.example_id,
            "task_family": self.task_family,
            "video_regime": self.video_regime,
            "final_status": self.final_status,
            "verifier_passed": self.verifier_passed,
            "graph_template": self.graph_template,
            "trigger_signature": self.trigger_signature,
            "expansion_template": self.expansion_template,
            "source_path": self.source_path,
            "proposal_source": self.proposal_source,
            "agent_backend": self.agent_backend,
            "curator_verdict": self.curator_verdict,
            "curator_reason": self.curator_reason,
            "confidence": self.confidence,
        }


@dataclass
class MotifRecord:
    motif_id: str
    motif_type: str
    signature: str
    status: str = "candidate"
    support_count: int = 0
    success_count: int = 0
    datasets_seen: set[str] = field(default_factory=set)
    task_families_seen: set[str] = field(default_factory=set)
    video_regimes_seen: set[str] = field(default_factory=set)
    example_ids: list[str] = field(default_factory=list)
    graph_template: dict[str, Any] = field(default_factory=dict)
    trigger_signature: dict[str, Any] = field(default_factory=dict)
    expansion_template: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    proposal_sources: set[str] = field(default_factory=set)
    agent_backends: set[str] = field(default_factory=set)
    curator_verdicts: dict[str, int] = field(default_factory=dict)
    curator_reasons: list[str] = field(default_factory=list)

    @property
    def verifier_pass_rate(self) -> float:
        if not self.support_count:
            return 0.0
        return self.success_count / self.support_count

    def add_instance(self, instance: MotifInstance) -> None:
        self.support_count += 1
        if instance.verifier_passed:
            self.success_count += 1
        if instance.dataset:
            self.datasets_seen.add(instance.dataset)
        if instance.task_family:
            self.task_families_seen.add(instance.task_family)
        if instance.video_regime:
            self.video_regimes_seen.add(instance.video_regime)
        if instance.example_id and instance.example_id not in self.example_ids:
            self.example_ids.append(instance.example_id)
        if not self.graph_template:
            self.graph_template = instance.graph_template
        if not self.trigger_signature:
            self.trigger_signature = instance.trigger_signature
        if not self.expansion_template:
            self.expansion_template = instance.expansion_template
        if instance.proposal_source:
            self.proposal_sources.add(instance.proposal_source)
        if instance.agent_backend:
            self.agent_backends.add(instance.agent_backend)
        if instance.curator_verdict:
            self.curator_verdicts[instance.curator_verdict] = self.curator_verdicts.get(instance.curator_verdict, 0) + 1
        if instance.curator_reason and instance.curator_reason not in self.curator_reasons:
            self.curator_reasons.append(instance.curator_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "video-skills-relaunch/composite-motif-v0.1",
            "motif_id": self.motif_id,
            "motif_type": self.motif_type,
            "signature": self.signature,
            "status": self.status,
            "support": {
                "support_count": self.support_count,
                "success_count": self.success_count,
                "verifier_pass_rate": round(self.verifier_pass_rate, 6),
                "datasets_seen": sorted(self.datasets_seen),
                "task_families_seen": sorted(self.task_families_seen),
                "video_regimes_seen": sorted(self.video_regimes_seen),
                "example_ids": self.example_ids,
            },
            "trigger_signature": self.trigger_signature,
            "graph_template": self.graph_template,
            "expansion_template": self.expansion_template,
            "constraints": [
                "expand_before_execution",
                "cite_current_video_evidence_only",
                "do_not_create_atomic_skill",
                "run_node_level_verifiers",
                "do_not_mine_heldout_test_for_thresholds",
            ],
            "agent": {
                "proposal_sources": sorted(self.proposal_sources),
                "agent_backends": sorted(self.agent_backends),
                "curator_verdicts": dict(sorted(self.curator_verdicts.items())),
                "curator_reasons": self.curator_reasons[:5],
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotifRecord":
        support = payload.get("support") or {}
        record = cls(
            motif_id=str(payload["motif_id"]),
            motif_type=str(payload.get("motif_type") or "unknown"),
            signature=str(payload.get("signature") or ""),
            status=str(payload.get("status") or "candidate"),
            support_count=int(support.get("support_count") or 0),
            success_count=int(support.get("success_count") or 0),
            datasets_seen=set(support.get("datasets_seen") or []),
            task_families_seen=set(support.get("task_families_seen") or []),
            video_regimes_seen=set(support.get("video_regimes_seen") or []),
            example_ids=[str(item) for item in support.get("example_ids") or []],
            graph_template=payload.get("graph_template") or {},
            trigger_signature=payload.get("trigger_signature") or {},
            expansion_template=payload.get("expansion_template") or {},
            notes=[str(item) for item in payload.get("notes") or []],
        )
        agent = payload.get("agent") or {}
        record.proposal_sources = set(str(item) for item in agent.get("proposal_sources") or [])
        record.agent_backends = set(str(item) for item in agent.get("agent_backends") or [])
        record.curator_verdicts = {
            str(key): int(value) for key, value in (agent.get("curator_verdicts") or {}).items()
        }
        record.curator_reasons = [str(item) for item in agent.get("curator_reasons") or []]
        return record


class MotifBank:
    """JSONL-backed motif registry inspired by the old skill bank pattern."""

    def __init__(self) -> None:
        self._records: dict[str, MotifRecord] = {}

    @property
    def records(self) -> list[MotifRecord]:
        return list(self._records.values())

    def add_instance(self, motif_id: str, instance: MotifInstance) -> MotifRecord:
        record = self._records.get(motif_id)
        if record is None:
            record = MotifRecord(
                motif_id=motif_id,
                motif_type=instance.motif_type,
                signature=instance.signature,
            )
            self._records[motif_id] = record
        record.add_instance(instance)
        return record

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(self.records, key=lambda item: (item.status, item.motif_id)):
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> "MotifBank":
        bank = cls()
        if not path.exists():
            return bank
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = MotifRecord.from_dict(json.loads(line))
                bank._records[record.motif_id] = record
        return bank

    def summary(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for record in self.records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
            type_counts[record.motif_type] = type_counts.get(record.motif_type, 0) + 1
        return {
            "motif_count": len(self.records),
            "status_counts": status_counts,
            "motif_type_counts": type_counts,
        }

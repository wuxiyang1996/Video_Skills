"""Lifecycle gates for motif promotion and rollback."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bank import MotifBank
from .schemas import MotifLifecycleStatus, MotifRecord


class MotifDiffOp(str, Enum):
    ADD = "add"
    TRANSITION = "transition"
    RECORD_FALSE_BINDING = "record_false_binding"


@dataclass
class MotifDiffEntry:
    op: MotifDiffOp
    motif_id: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "motif_id": self.motif_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class MotifGateConfig:
    min_support_count: int = 2
    min_transfer_pass_rate: float = 0.5
    min_verifier_pass_rate: float = 0.8
    min_evidence_valid_rate: float = 0.8
    require_no_leakage: bool = True
    min_delta_over_baseline: float = 0.0


class MotifLifecycleManager:
    """Single write path for motif status transitions."""

    def __init__(
        self,
        bank: MotifBank,
        gate_config: MotifGateConfig | None = None,
    ) -> None:
        self.bank = bank
        self.gate_config = gate_config or MotifGateConfig()
        self.diff_entries: list[MotifDiffEntry] = []

    def add_candidate(self, record: MotifRecord) -> MotifRecord:
        if record.status == MotifLifecycleStatus.DRAFT:
            record.status = MotifLifecycleStatus.CANDIDATE
        self.bank.add(record)
        self.diff_entries.append(MotifDiffEntry(
            op=MotifDiffOp.ADD,
            motif_id=record.motif_id,
            details={"status": record.status.value},
        ))
        return record

    def transition(
        self,
        motif_id: str,
        target_status: MotifLifecycleStatus,
        reason: str,
    ) -> MotifRecord:
        record = self.bank.require(motif_id)
        source_status = record.status
        if not self._is_allowed_transition(source_status, target_status):
            raise ValueError(
                f"Illegal motif transition: {source_status.value} -> {target_status.value}"
            )
        record.status = target_status
        record.updated_at = time.time()
        record.notes.append(f"transition:{source_status.value}->{target_status.value}:{reason}")
        self.diff_entries.append(MotifDiffEntry(
            op=MotifDiffOp.TRANSITION,
            motif_id=motif_id,
            details={
                "from": source_status.value,
                "to": target_status.value,
                "reason": reason,
            },
        ))
        return record

    def apply_transfer_gates(self, motif_id: str) -> MotifRecord:
        record = self.bank.require(motif_id)
        if self._passes_transfer_gates(record):
            if record.status in {
                MotifLifecycleStatus.CANDIDATE,
                MotifLifecycleStatus.SHADOW,
            }:
                return self.transition(
                    motif_id,
                    MotifLifecycleStatus.VERIFIED,
                    "passed_transfer_gates",
                )
            return record
        if record.status in {
            MotifLifecycleStatus.CANDIDATE,
            MotifLifecycleStatus.SHADOW,
            MotifLifecycleStatus.VERIFIED,
        }:
            return self.transition(
                motif_id,
                MotifLifecycleStatus.REJECTED,
                "failed_transfer_gates",
            )
        return record

    def record_false_binding(self, motif_id: str, pattern: str) -> MotifRecord:
        record = self.bank.require(motif_id)
        if pattern not in record.false_binding_patterns:
            record.false_binding_patterns.append(pattern)
        record.updated_at = time.time()
        self.diff_entries.append(MotifDiffEntry(
            op=MotifDiffOp.RECORD_FALSE_BINDING,
            motif_id=motif_id,
            details={"pattern": pattern},
        ))
        return record

    def diff_report(self) -> dict[str, Any]:
        return {
            "n_entries": len(self.diff_entries),
            "entries": [entry.to_dict() for entry in self.diff_entries],
        }

    def _passes_transfer_gates(self, record: MotifRecord) -> bool:
        cfg = self.gate_config
        if record.support_count < cfg.min_support_count:
            return False
        if not record.transfer_reports:
            return False
        best = max(record.transfer_reports, key=lambda item: item.pass_rate)
        if best.pass_rate < cfg.min_transfer_pass_rate:
            return False
        if best.verifier_pass_rate < cfg.min_verifier_pass_rate:
            return False
        if best.evidence_valid_rate < cfg.min_evidence_valid_rate:
            return False
        if cfg.require_no_leakage and best.no_leakage_rate < 1.0:
            return False
        delta = best.delta_over_baseline
        if delta is not None and delta < cfg.min_delta_over_baseline:
            return False
        return True

    @staticmethod
    def _is_allowed_transition(
        source: MotifLifecycleStatus,
        target: MotifLifecycleStatus,
    ) -> bool:
        allowed = {
            MotifLifecycleStatus.DRAFT: {
                MotifLifecycleStatus.CANDIDATE,
                MotifLifecycleStatus.REJECTED,
            },
            MotifLifecycleStatus.CANDIDATE: {
                MotifLifecycleStatus.SHADOW,
                MotifLifecycleStatus.VERIFIED,
                MotifLifecycleStatus.REJECTED,
            },
            MotifLifecycleStatus.SHADOW: {
                MotifLifecycleStatus.VERIFIED,
                MotifLifecycleStatus.REJECTED,
            },
            MotifLifecycleStatus.VERIFIED: {
                MotifLifecycleStatus.ACTIVE,
                MotifLifecycleStatus.REJECTED,
                MotifLifecycleStatus.DEPRECATED,
            },
            MotifLifecycleStatus.ACTIVE: {
                MotifLifecycleStatus.DEPRECATED,
                MotifLifecycleStatus.ROLLED_BACK,
            },
            MotifLifecycleStatus.REJECTED: set(),
            MotifLifecycleStatus.DEPRECATED: {
                MotifLifecycleStatus.ROLLED_BACK,
            },
            MotifLifecycleStatus.ROLLED_BACK: set(),
        }
        return target in allowed[source] or source == target

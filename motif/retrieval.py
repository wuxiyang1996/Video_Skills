"""Motif retrieval and selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .bank import MotifBank
from .schemas import MotifLifecycleStatus, MotifRecord


def _tokenize(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(part) >= 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass
class MotifSelectionResult:
    motif_id: str
    name: str
    status: str
    relevance: float
    applicability: float
    empirical_confidence: float
    score: float
    why_selected: str = ""
    expansion_constraints: list[str] = field(default_factory=list)
    trigger_signature: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_id": self.motif_id,
            "name": self.name,
            "status": self.status,
            "relevance": round(self.relevance, 6),
            "applicability": round(self.applicability, 6),
            "empirical_confidence": round(self.empirical_confidence, 6),
            "score": round(self.score, 6),
            "why_selected": self.why_selected,
            "expansion_constraints": self.expansion_constraints,
            "trigger_signature": self.trigger_signature,
        }


class MotifQueryEngine:
    """Select motifs as graph-construction priors for L1/L2 agents."""

    def __init__(self, bank: MotifBank) -> None:
        self.bank = bank

    def select(
        self,
        query: str,
        task_family: str = "",
        dataset: str = "",
        include_shadow: bool = False,
        include_candidate: bool = False,
        top_k: int = 3,
        exclude_motif_ids: set[str] | frozenset[str] | None = None,
        phase: str = "accelerate",
    ) -> list[MotifSelectionResult]:
        """Select motifs.

        ``phase=accelerate`` (default): VERIFIED/ACTIVE only — used to skip planner.
        ``phase=repair``: also allows SHADOW/CANDIDATE as repair priors.
        """
        if phase == "repair":
            include_shadow = True
            include_candidate = True
        query_tokens = _tokenize(query)
        excluded = {str(x) for x in (exclude_motif_ids or set())}
        candidates = []
        for record in self.bank.records:
            if record.motif_id in excluded:
                continue
            if not self._is_visible(
                record,
                include_shadow=include_shadow,
                include_candidate=include_candidate,
            ):
                continue
            relevance = self._relevance(record, query_tokens)
            applicability = self._applicability(record, task_family=task_family, dataset=dataset)
            confidence = record.empirical_confidence
            score = 0.45 * relevance + 0.35 * applicability + 0.20 * confidence
            candidates.append(MotifSelectionResult(
                motif_id=record.motif_id,
                name=record.name,
                status=record.status.value,
                relevance=relevance,
                applicability=applicability,
                empirical_confidence=confidence,
                score=score,
                why_selected=self._why(record, relevance, applicability, confidence),
                expansion_constraints=list(record.expansion_constraints),
                trigger_signature=dict(record.trigger_signature),
            ))
        # Prefer positive-score matches; if none, still return visible motifs so a
        # non-empty ACTIVE/VERIFIED bank can satisfy mandatory retrieve attempts.
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        positive = [item for item in ranked if item.score > 0]
        chosen = positive if positive else ranked
        return chosen[:top_k]

    @staticmethod
    def _is_visible(
        record: MotifRecord,
        include_shadow: bool,
        include_candidate: bool = False,
    ) -> bool:
        visible = {
            MotifLifecycleStatus.VERIFIED,
            MotifLifecycleStatus.ACTIVE,
        }
        if include_shadow:
            visible.add(MotifLifecycleStatus.SHADOW)
        if include_candidate:
            visible.add(MotifLifecycleStatus.CANDIDATE)
        return record.status in visible

    @staticmethod
    def _relevance(record: MotifRecord, query_tokens: set[str]) -> float:
        text = " ".join([
            record.motif_id,
            record.name,
            record.description,
            str(record.trigger_signature),
            str(record.l1_template),
            str(record.l2_template),
        ])
        return _jaccard(query_tokens, _tokenize(text))

    @staticmethod
    def _applicability(record: MotifRecord, task_family: str, dataset: str) -> float:
        score = 0.0
        if task_family and task_family in record.verified_task_families:
            score += 0.6
        if dataset:
            datasets = {ref.dataset for ref in record.evidence_refs}
            if dataset in datasets:
                score += 0.3
        if record.support_count > 0:
            score += min(record.support_count, 5) / 50.0
        return min(score, 1.0)

    @staticmethod
    def _why(
        record: MotifRecord,
        relevance: float,
        applicability: float,
        confidence: float,
    ) -> str:
        return (
            f"relevance={relevance:.3f}; applicability={applicability:.3f}; "
            f"empirical_confidence={confidence:.3f}; support={record.support_count}"
        )

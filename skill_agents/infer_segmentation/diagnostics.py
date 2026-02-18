"""
Diagnostic output from InferSegmentation.

For each decoded segment the diagnostics record:
  - top-K skill candidates with scores
  - margin (score gap between rank-1 and rank-2)
  - boundary confidence
  - per-term score breakdown

Small margin → uncertain → good candidate for preference labeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SkillCandidate:
    """One candidate skill label for a segment, with full breakdown."""

    skill: str
    total_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentDiagnostic:
    """Diagnostics for a single decoded segment."""

    start: int
    end: int
    assigned_skill: str
    candidates: List[SkillCandidate] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """Score gap between rank-1 and rank-2.  0 if < 2 candidates."""
        if len(self.candidates) < 2:
            return float("inf")
        return self.candidates[0].total_score - self.candidates[1].total_score

    @property
    def is_uncertain(self) -> bool:
        """Heuristic: margin below 1.0 → worth querying."""
        return self.margin < 1.0


@dataclass
class BoundaryDiagnostic:
    """Diagnostics for a boundary decision (cut vs. no-cut)."""

    time: int
    score_with_cut: float
    score_without_cut: float

    @property
    def confidence(self) -> float:
        return self.score_with_cut - self.score_without_cut


@dataclass
class SegmentationResult:
    """Full output of InferSegmentation, including the best path and diagnostics."""

    segments: List[SegmentDiagnostic] = field(default_factory=list)
    boundaries: List[BoundaryDiagnostic] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def skill_sequence(self) -> List[str]:
        return [s.assigned_skill for s in self.segments]

    @property
    def cut_points(self) -> List[int]:
        return [s.start for s in self.segments]

    def uncertain_segments(self, margin_threshold: float = 1.0) -> List[SegmentDiagnostic]:
        """Return segments where the margin is below threshold (best for preference labeling)."""
        return [s for s in self.segments if s.margin < margin_threshold]

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "assigned_skill": s.assigned_skill,
                    "margin": s.margin,
                    "candidates": [
                        {"skill": c.skill, "score": c.total_score, "breakdown": c.breakdown}
                        for c in s.candidates
                    ],
                }
                for s in self.segments
            ],
            "boundaries": [
                {
                    "time": b.time,
                    "score_with_cut": b.score_with_cut,
                    "score_without_cut": b.score_without_cut,
                    "confidence": b.confidence,
                }
                for b in self.boundaries
            ],
        }

"""
Data schemas for Stage 3 MVP: effects-only contracts.

Three core types:
  - ``SegmentRecord``: enriched segment with booleanized predicates and effects.
  - ``SkillEffectsContract``: eff_add / eff_del / eff_event contract per skill.
  - ``VerificationReport``: per-skill pass rates and failure diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class SegmentRecord:
    """One labelled segment with predicate summaries and per-instance effects."""

    seg_id: str
    traj_id: str
    t_start: int
    t_end: int
    skill_label: str  # existing skill id or "NEW"

    # Smoothed predicate probabilities at segment boundaries
    P_start: Dict[str, float] = field(default_factory=dict)
    P_end: Dict[str, float] = field(default_factory=dict)

    # UI events observed within the segment (optional)
    events: List[str] = field(default_factory=list)

    # Booleanized predicate sets
    B_start: Set[str] = field(default_factory=set)
    B_end: Set[str] = field(default_factory=set)

    # Per-instance effects (computed in Step 2)
    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    eff_event: Set[str] = field(default_factory=set)

    def effect_signature(self) -> str:
        """Canonical string for debugging: ``A:p1,p2|D:q1,q2|E:e1``."""
        a = ",".join(sorted(self.eff_add)) if self.eff_add else ""
        d = ",".join(sorted(self.eff_del)) if self.eff_del else ""
        e = ",".join(sorted(self.eff_event)) if self.eff_event else ""
        return f"A:{a}|D:{d}|E:{e}"

    def to_dict(self) -> dict:
        return {
            "seg_id": self.seg_id,
            "traj_id": self.traj_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "skill_label": self.skill_label,
            "P_start": self.P_start,
            "P_end": self.P_end,
            "events": self.events,
            "B_start": sorted(self.B_start),
            "B_end": sorted(self.B_end),
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "eff_event": sorted(self.eff_event),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SegmentRecord:
        return cls(
            seg_id=d["seg_id"],
            traj_id=d["traj_id"],
            t_start=d["t_start"],
            t_end=d["t_end"],
            skill_label=d["skill_label"],
            P_start=d.get("P_start", {}),
            P_end=d.get("P_end", {}),
            events=d.get("events", []),
            B_start=set(d.get("B_start", [])),
            B_end=set(d.get("B_end", [])),
            eff_add=set(d.get("eff_add", [])),
            eff_del=set(d.get("eff_del", [])),
            eff_event=set(d.get("eff_event", [])),
        )


@dataclass
class SkillEffectsContract:
    """Effects-only contract for one skill (MVP: no pre/inv)."""

    skill_id: str
    version: int = 1

    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    eff_event: Set[str] = field(default_factory=set)

    # Per-literal instance support count
    support: Dict[str, int] = field(default_factory=dict)
    n_instances: int = 0

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = time.time()

    @property
    def total_literals(self) -> int:
        return len(self.eff_add) + len(self.eff_del) + len(self.eff_event)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "eff_event": sorted(self.eff_event),
            "support": self.support,
            "n_instances": self.n_instances,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillEffectsContract:
        return cls(
            skill_id=d["skill_id"],
            version=d.get("version", 1),
            eff_add=set(d.get("eff_add", [])),
            eff_del=set(d.get("eff_del", [])),
            eff_event=set(d.get("eff_event", [])),
            support=d.get("support", {}),
            n_instances=d.get("n_instances", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


@dataclass
class VerificationReport:
    """Per-skill verification metrics for an effects-only contract."""

    skill_id: str
    n_instances: int = 0

    eff_add_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_del_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_event_rate: Dict[str, float] = field(default_factory=dict)

    overall_pass_rate: float = 0.0
    worst_segments: List[str] = field(default_factory=list)
    failure_signatures: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "n_instances": self.n_instances,
            "eff_add_success_rate": self.eff_add_success_rate,
            "eff_del_success_rate": self.eff_del_success_rate,
            "eff_event_rate": self.eff_event_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "worst_segments": self.worst_segments,
            "failure_signatures": self.failure_signatures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerificationReport:
        return cls(**d)

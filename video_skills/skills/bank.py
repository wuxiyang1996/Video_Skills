"""Skill bank schema + registry.

``SkillRecord`` is the canonical, serializable, versioned record for every
entry in the bank (skill_extraction_bank.md §6). Each ``SkillRecord`` is
paired with an executable ``AtomicSkill`` callable that the harness invokes.

V1 policy: the bank is a closed, hand-curated inventory. Promotion / patch /
retire operations are defined here but gated behind explicit synthesizer
invocation (deferred to later phases).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from ..contracts import (
    AtomicStepResult,
    EvidenceBundle,
    HopGoal,
    RetrievalQuery,
    TriggerSpec,
    VerificationCheckSpec,
    new_id,
    now_ts,
)


@dataclass
class SkillUsage:
    """Per-skill running counters (skill_extraction_bank.md §6)."""

    n_invocations: int = 0
    n_success: int = 0
    n_failure_by_mode: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    transfer_rate: float = 0.0  # success rate on tasks outside training family
    last_updated: Optional[float] = None


@dataclass
class SkillVersion:
    """Bank versioning metadata."""

    version_id: str
    parent_version_id: Optional[str] = None
    created_at: float = field(default_factory=now_ts)
    created_by: Literal[
        "crafted", "promoted", "merged", "split", "patched"
    ] = "crafted"
    status: Literal["active", "shadow", "retired"] = "active"
    notes: Optional[str] = None


@dataclass
class SkillRecord:
    """Canonical, serializable bank entry."""

    skill_id: str
    name: str
    type: Literal["atomic", "composite"]
    family: str  # one of §4.x families, e.g. "temporal", "social_belief"
    trigger_conditions: List[TriggerSpec] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    output_type: str = "claim"
    verification_rule: List[VerificationCheckSpec] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    required_memory_fields: List[str] = field(default_factory=list)
    retrieval_hints: List[RetrievalQuery] = field(default_factory=list)
    required_primitives: List[str] = field(default_factory=list)
    protocol_steps: List[str] = field(default_factory=list)
    child_links: List[str] = field(default_factory=list)
    parent_links: List[str] = field(default_factory=list)
    usage_stats: SkillUsage = field(default_factory=SkillUsage)
    version: SkillVersion = field(
        default_factory=lambda: SkillVersion(version_id=new_id("skv"))
    )
    examples: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# A skill executable receives a context dict (from the harness) and returns
# the raw output dict that the harness wraps into an ``AtomicStepResult``.
SkillExecutable = Callable[["SkillContext"], "SkillOutput"]


@dataclass
class SkillContext:
    """The bundle of information the harness passes to an atomic skill.

    The skill is a pure function over this context; it must not mutate
    memory directly (use the ``memory_procedures`` handle to request a
    procedure call instead).
    """

    hop_goal: HopGoal
    inputs: Dict[str, Any]
    evidence: Optional[EvidenceBundle]
    memory: Any  # video_skills.memory.Memory  (typed loosely to avoid cycle)
    memory_procedures: Any  # video_skills.memory.MemoryProcedureRegistry
    retriever: Any  # video_skills.retriever.Retriever
    trace_so_far: List[AtomicStepResult] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutput:
    """Return type of an atomic skill executable."""

    output: Dict[str, Any]
    output_type: str
    confidence: float
    inferred: bool = False
    failure_mode: Optional[str] = None
    used_evidence: Optional[EvidenceBundle] = None
    requested_memory_writes: List[Dict[str, Any]] = field(default_factory=list)
    # ^ each item: {"procedure": "<name>", "args": {...}}
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicSkill:
    """A bound pair of (record, executable). Loaded into the bank at startup."""

    record: SkillRecord
    executable: SkillExecutable

    @property
    def skill_id(self) -> str:
        return self.record.skill_id

    @property
    def name(self) -> str:
        return self.record.name


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------


class ReasoningSkillBank:
    """In-memory, by-name registry of atomic + composite skills.

    Lookup helpers mirror the controller's selection needs: by family, by
    output_type, by trigger evaluation. The bank is intentionally not
    coupled to persistence in v1 — :func:`build_starter_bank` rebuilds it
    deterministically each process startup.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, AtomicSkill] = {}
        self._by_name: Dict[str, str] = {}
        self._by_family: Dict[str, List[str]] = {}
        self._by_output_type: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: AtomicSkill) -> None:
        sid = skill.skill_id
        if sid in self._skills:
            raise ValueError(f"skill_id {sid!r} already registered")
        if skill.name in self._by_name:
            raise ValueError(f"skill name {skill.name!r} already registered")
        self._skills[sid] = skill
        self._by_name[skill.name] = sid
        self._by_family.setdefault(skill.record.family, []).append(sid)
        self._by_output_type.setdefault(skill.record.output_type, []).append(sid)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, skill_id: str) -> AtomicSkill:
        return self._skills[skill_id]

    def get_by_name(self, name: str) -> AtomicSkill:
        return self._skills[self._by_name[name]]

    def has(self, skill_id_or_name: str) -> bool:
        return skill_id_or_name in self._skills or skill_id_or_name in self._by_name

    def all(self) -> List[AtomicSkill]:
        return list(self._skills.values())

    def by_family(self, family: str) -> List[AtomicSkill]:
        return [self._skills[i] for i in self._by_family.get(family, [])]

    def by_output_type(self, output_type: str) -> List[AtomicSkill]:
        return [self._skills[i] for i in self._by_output_type.get(output_type, [])]

    def stats(self) -> Dict[str, Any]:
        return {
            "n_skills": len(self._skills),
            "families": {f: len(ids) for f, ids in self._by_family.items()},
        }

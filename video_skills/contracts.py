"""Canonical runtime data contracts for Video_Skills.

Normative source: ``infra_plans/03_controller/actors_reasoning_model.md`` §2A
and ``infra_plans/00_overview/runtime_contracts.md``.

These dataclasses are the **only** allowed wire format between major runtime
components (controller, harness, memory, retriever, verifier, skill bank).
Per the Contract Rules in §2A.8:

1. Canonical objects only — no ad hoc dict passing.
2. Inferred tagging is mandatory — values not directly read from grounded
   perception or stored memory must set ``inferred=True``.
3. Evidence-bearing objects must carry refs and confidence. Every claim
   either has a non-empty ``EvidenceBundle`` or a verifier action of
   ``"abstain"``.
4. Versioning — every object carries an implicit ``schema_version``.
5. Idempotent reads — same ``RetrievalQuery`` returns same shape within a
   session.
6. No silent enrichment — optional fields are documented; everything else
   goes into ``meta``.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Identifiers / time helpers
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    """Generate a short, unique identifier with a domain prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ts() -> float:
    """Return wall-clock seconds (used only for trace timestamps)."""
    return time.time()


# ---------------------------------------------------------------------------
# Supporting reference objects
# ---------------------------------------------------------------------------


@dataclass
class EvidenceRef:
    """A pointer to a piece of grounded perception or stored memory.

    Mirrors ``visual_grounding.schemas.EvidenceRef`` field-for-field so the
    grounding pipeline's outputs can flow into ``EvidenceBundle`` without
    rewrapping.
    """

    ref_id: str
    modality: str  # "frame" | "clip" | "subtitle" | "voice" | "memory_node" | "state"
    source_id: Optional[str] = None  # backing memory record / node id
    time_span: Optional[Tuple[float, float]] = None
    entities: List[str] = field(default_factory=list)
    provenance: Literal["observed", "inferred", "retrieved", "distilled"] = "observed"
    confidence: float = 1.0
    text: Optional[str] = None
    locator: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityRef:
    """An entity active inside a window (face/voice/character handle)."""

    entity_id: str
    canonical_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    role: Optional[str] = None  # "person" | "object" | "group" | "speaker"
    confidence: float = 1.0


@dataclass
class EventRef:
    """A detected action / utterance / state-change inside a window."""

    event_id: str
    event_type: str  # "action" | "utterance" | "state_change" | ...
    description: str
    participants: List[str] = field(default_factory=list)
    time_span: Optional[Tuple[float, float]] = None
    confidence: float = 1.0


@dataclass
class DialogueSpan:
    """A subtitle / ASR span aligned to the parent window's time_span."""

    span_id: str
    text: str
    speaker: Optional[str] = None
    time_span: Optional[Tuple[float, float]] = None
    confidence: float = 1.0


@dataclass
class FrameRef:
    """A single keyframe pulled by the perception layer."""

    frame_id: str
    timestamp: float
    locator: Optional[Dict[str, Any]] = None  # path / byte offset / video.frame_num


# ---------------------------------------------------------------------------
# §2A.1 GroundedWindow
# ---------------------------------------------------------------------------


@dataclass
class GroundedWindow:
    """A grounded slice of video / dialogue / state from the perception layer.

    Primary consumable for memory writers and atomic grounding skills.
    """

    window_id: str
    clip_id: str
    time_span: Tuple[float, float]
    entities: List[EntityRef] = field(default_factory=list)
    events: List[EventRef] = field(default_factory=list)
    dialogue: List[DialogueSpan] = field(default_factory=list)
    spatial_state: Dict[str, Any] = field(default_factory=dict)
    keyframes: List[FrameRef] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    inferred: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# §2A.2 EvidenceBundle (and RetrievalQuery)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalQuery:
    """A query against memory. Cf. §2B.2 Retriever interface."""

    query_id: str
    text: str
    entity_filter: List[str] = field(default_factory=list)
    time_filter: Optional[Tuple[float, float]] = None
    perspective: Optional[str] = None  # character_id whose viewpoint to retrieve from
    store_filter: Literal["episodic", "semantic", "state", "any"] = "any"
    k: int = 8
    mode: Literal["lexical", "dense", "hybrid"] = "lexical"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """The unit returned by Retriever; the **only** way evidence flows.

    Per §2A.8 rule 3: no claim may be emitted without an attached bundle (or an
    explicit empty bundle marked ``sufficiency_hint=0``).
    """

    bundle_id: str
    refs: List[EvidenceRef]
    query: RetrievalQuery
    coverage: Dict[str, Any] = field(default_factory=dict)
    contradictions: List[EvidenceRef] = field(default_factory=list)
    sufficiency_hint: float = 0.0  # retriever's prior on sufficiency, 0..1
    confidence: float = 0.0
    inferred: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def is_empty(self) -> bool:
        return len(self.refs) == 0


# ---------------------------------------------------------------------------
# §2A.3 HopGoal
# ---------------------------------------------------------------------------


@dataclass
class HopGoal:
    """A single hop's contract: what the hop must establish."""

    hop_id: str
    parent_question_id: str
    goal_text: str
    target_claim_type: str  # "ordering" | "belief" | "causal" | "presence" | "state" | ...
    required_entities: List[str] = field(default_factory=list)
    required_time_scope: Optional[Tuple[float, float]] = None
    perspective_anchor: Optional[str] = None
    retrieval_hints: List[RetrievalQuery] = field(default_factory=list)
    success_predicate: str = ""
    max_atomic_steps: int = 6
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# §2A.5 VerificationResult (defined before AtomicStepResult, which references it)
# ---------------------------------------------------------------------------


@dataclass
class VerificationCheck:
    """One named local check applied to an AtomicStepResult."""

    name: str  # one of §2C.1 catalog or skill-specific
    passed: bool
    evidence_refs: List[str] = field(default_factory=list)
    score: float = 1.0
    notes: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# Severity ordering for next_action aggregation (§2A.5 / §HarnessLocalVerification)
_NEXT_ACTION_SEVERITY: Dict[str, int] = {
    "continue": 0,
    "retry": 1,
    "broaden": 2,
    "switch_skill": 3,
    "abstain": 4,
}


def most_severe_next_action(actions: List[str]) -> str:
    """Aggregate `next_action` from multiple checks per the harness rule
    (most-severe wins)."""
    if not actions:
        return "continue"
    return max(actions, key=lambda a: _NEXT_ACTION_SEVERITY.get(a, 0))


@dataclass
class VerificationResult:
    """Local, per-step verification record."""

    passed: bool
    checks: List[VerificationCheck] = field(default_factory=list)
    score: float = 0.0
    counterevidence: List[EvidenceRef] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    next_action: Literal[
        "continue", "retry", "broaden", "switch_skill", "abstain"
    ] = "continue"
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# §2A.4 AtomicStepResult
# ---------------------------------------------------------------------------


# Standard failure mode codes recognized by the harness (§HarnessFailureModes)
STANDARD_FAILURE_MODES = {
    "schema_violation",  # output did not match output_schema
    "missing_input",  # required input was unresolved at call time
    "empty_evidence",  # required EvidenceBundle was empty after retrieval+broaden
    "verification_failed",  # verifier returned passed=False
    "timeout",  # skill exceeded its per-call latency budget
    "exception",  # skill raised; output is None
}

# Output type tokens (§2A.4 + skill bank §4.9)
OUTPUT_TYPES = {
    "claim",
    "span",
    "ordering",
    "belief",
    "presence",
    "abstain",
    "evidence_set",
    "set[claim]",
    "entity_ref",
    "decision",
    "meta",
}


@dataclass
class AtomicStepResult:
    """Output of every atomic skill invocation."""

    step_id: str
    hop_id: str
    skill_id: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]
    output_type: str
    evidence: Optional[EvidenceBundle]
    verification: VerificationResult
    confidence: float
    inferred: bool = False
    failure_mode: Optional[str] = None
    latency_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def is_claim(self) -> bool:
        return self.output_type in {"claim", "ordering", "belief", "presence", "decision"}


def validate_atomic_step(step: AtomicStepResult) -> List[str]:
    """Apply Contract Rules §2A.8 to an AtomicStepResult.

    Returns a list of rule violations (empty list = valid).
    """
    violations: List[str] = []

    # Rule 3: claims need either evidence with refs OR a verifier abstain.
    if step.is_claim():
        has_evidence = (
            step.evidence is not None and not step.evidence.is_empty()
        )
        is_abstaining = step.verification.next_action == "abstain"
        if not (has_evidence or is_abstaining):
            violations.append(
                f"claim step {step.step_id} has no evidence and is not abstaining"
            )

    # output_type sanity
    if step.output_type not in OUTPUT_TYPES:
        violations.append(
            f"step {step.step_id} has unknown output_type={step.output_type!r}"
        )

    # failure_mode sanity (allow skill-specific codes too, but flag obvious typos)
    if step.failure_mode is not None and not isinstance(step.failure_mode, str):
        violations.append(f"step {step.step_id} failure_mode must be a string")

    return violations


# ---------------------------------------------------------------------------
# §2A.6 AbstainDecision
# ---------------------------------------------------------------------------


@dataclass
class AbstainDecision:
    """Emitted when the controller declines to answer."""

    abstain: bool
    reason: str  # "insufficient_evidence" | "contradictions" | "perspective_unresolved" | ...
    blocking_checks: List[str] = field(default_factory=list)
    last_evidence: Optional[EvidenceBundle] = None
    confidence_ceiling: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Question analysis + HopRecord
# ---------------------------------------------------------------------------


@dataclass
class QuestionAnalysis:
    """Result of ``controller.analyze_question(q)``.

    The controller's first artifact for a new question. Drives planning and
    routing decisions across the trace.
    """

    question_id: str
    question_text: str
    question_type: str  # "ordering" | "belief" | "causal" | "presence" | "state" | "free"
    target_entities: List[str] = field(default_factory=list)
    time_anchor: Optional[Tuple[float, float]] = None
    perspective_anchor: Optional[str] = None
    expected_answer_type: str = "free_text"  # "yes_no" | "free_text" | "entity" | ...
    decomposition: List[str] = field(default_factory=list)  # sub-goal texts
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class HopRecord:
    """One hop's record inside a ReasoningTrace."""

    hop_goal: HopGoal
    steps: List[AtomicStepResult] = field(default_factory=list)
    hop_verification: VerificationResult = field(
        default_factory=lambda: VerificationResult(passed=False)
    )
    outcome: Literal["resolved", "blocked", "abstain", "in_progress"] = "in_progress"
    cost: Dict[str, Any] = field(default_factory=lambda: {
        "atomic_steps": 0,
        "retrieval_calls": 0,
        "broaden_levels": 0,
        "latency_ms": 0,
    })
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# §2A.7 ReasoningTrace
# ---------------------------------------------------------------------------


@dataclass
class ReasoningTrace:
    """End-to-end record of a question's execution."""

    trace_id: str
    question_id: str
    question_analysis: QuestionAnalysis
    hops: List[HopRecord] = field(default_factory=list)
    final_claim: Optional[Dict[str, Any]] = None
    final_evidence: Optional[EvidenceBundle] = None
    final_verification: VerificationResult = field(
        default_factory=lambda: VerificationResult(passed=False)
    )
    abstain: Optional[AbstainDecision] = None
    answer: Optional[str] = None
    bank_skill_ids_used: List[str] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=lambda: {
        "hops": 0,
        "atomic_steps": 0,
        "retrieval_calls": 0,
        "tokens": 0,
        "latency_ms": 0,
        "large_model_calls": 0,
    })
    started_at: float = field(default_factory=now_ts)
    finished_at: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def append_hop(self, hop: HopRecord) -> None:
        self.hops.append(hop)
        self.cost["hops"] = len(self.hops)
        self.cost["atomic_steps"] += hop.cost.get("atomic_steps", 0)
        self.cost["retrieval_calls"] += hop.cost.get("retrieval_calls", 0)
        self.cost["latency_ms"] += hop.cost.get("latency_ms", 0)
        for s in hop.steps:
            if s.skill_id not in self.bank_skill_ids_used:
                self.bank_skill_ids_used.append(s.skill_id)


# ---------------------------------------------------------------------------
# Skill bank — TriggerSpec / VerificationCheckSpec
# (Defined here because controller / harness reference these directly.)
# ---------------------------------------------------------------------------


@dataclass
class TriggerSpec:
    """Trigger condition for skill selection (cf. skill bank §6)."""

    when: str  # symbolic: "question_type=='ordering' && len(target_entities)>=2"
    priority: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationCheckSpec:
    """Local, deterministic check declared by a skill (cf. harness §LocalVerificationFormat)."""

    name: str
    inputs: List[str] = field(default_factory=list)
    predicate: str = ""
    threshold: Optional[float] = None
    on_fail: Literal["retry", "broaden", "switch_skill", "abstain", "continue"] = "continue"
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "SCHEMA_VERSION",
    "STANDARD_FAILURE_MODES",
    "OUTPUT_TYPES",
    "new_id",
    "now_ts",
    "EvidenceRef",
    "EntityRef",
    "EventRef",
    "DialogueSpan",
    "FrameRef",
    "GroundedWindow",
    "RetrievalQuery",
    "EvidenceBundle",
    "HopGoal",
    "VerificationCheck",
    "VerificationResult",
    "most_severe_next_action",
    "AtomicStepResult",
    "validate_atomic_step",
    "AbstainDecision",
    "QuestionAnalysis",
    "HopRecord",
    "ReasoningTrace",
    "TriggerSpec",
    "VerificationCheckSpec",
]

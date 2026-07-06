"""Typed grounding schemas (layered on top of ``schemas.py``).

This module implements the *typed* schema family described in
``infra_plans/01_grounding/video_benchmarks_grounding.md`` and the visual-grounding
plan that motivated it (``EntityState``, ``EventSpan``,
``InteractionEdge``, ``TemporalRelation``, ``VisibilityState``,
``BeliefCandidate``, ``GroundedClip``, plus the upstream observation /
segment / subtitle / memory record types). Every typed object carries:

* ``evidence_refs``    — pointers back into the raw video/subtitle/audio
                         substrate via :class:`EvidenceRef`.
* ``confidence``       — float in ``[0.0, 1.0]``.
* ``source_type``      — ``observed`` | ``inferred`` | ``retrieved`` |
                         ``distilled`` (the four provenance buckets the
                         plan mandates).

The legacy schemas in ``visual_grounding/schemas.py`` (``Entity``,
``Event``, ``Interaction``, ``GroundedWindow``, ``GroundingNode``, …) are
**not** removed. They remain the on-the-wire format used by
``local_grounder.py`` and ``social_video_graph.py``. The typed schemas
here sit one layer above; converters live in
``grounding_normalizer.py`` (legacy → typed) and
``memory_projection.py`` (typed → memory records).

Design rules enforced here (per the visual-grounding plan §13–§14):

* No grounded object may exist without ``evidence_refs`` and ``confidence``.
* Inferred / hypothesis objects (``BeliefCandidate``, hypothesised
  ``EventSpan``, hypothesised ``VisibilityState``, …) must set
  ``source_type="inferred"``.
* Ambiguity is preserved via ``candidates`` lists / ``ambiguity``
  metadata where natural — never collapsed into false certainty.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from visual_grounding.schemas import EvidenceRef


SourceType = Literal["observed", "inferred", "retrieved", "distilled"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_grounding_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _evref_to_dict(refs: List[EvidenceRef]) -> List[Dict[str, Any]]:
    return [r.to_dict() for r in refs]


# ---------------------------------------------------------------------------
# Upstream "observation" objects (Layer 1)
# ---------------------------------------------------------------------------


@dataclass
class VideoSegment:
    """A typed segment descriptor (Layer-1 input to grounding).

    Mirrors :class:`visual_grounding.segmenter.Window` but is the public
    typed contract the rest of the typed pipeline consumes.
    """

    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    subtitle_span_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_range(self) -> Tuple[float, float]:
        return (self.start_time, self.end_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubtitleSpan:
    """One subtitle/ASR line with an attached :class:`EvidenceRef`."""

    span_id: str
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    evidence_ref: Optional[EvidenceRef] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "span_id": self.span_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
        }
        if self.evidence_ref is not None:
            d["evidence_ref"] = self.evidence_ref.to_dict()
        return d


@dataclass
class RawObservation:
    """Low-commitment observation produced by the observation extractor.

    The plan (§5) explicitly says: do not collapse everything into one
    summary string. Each ``RawObservation`` is one decomposed unit
    (caption / participant / action / event-proposal / …) with its own
    evidence refs and confidence so the normalizer can promote or drop
    them independently.
    """

    obs_id: str
    segment_id: str
    observation_type: str  # caption | action | participant | speaker_turn |
                           # entity_mention | event_proposal | interaction_proposal |
                           # subtitle_echo | object_mention | …
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_model: Optional[str] = None
    source_type: SourceType = "observed"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        return d


# ---------------------------------------------------------------------------
# Typed grounded objects (Layer 2)
# ---------------------------------------------------------------------------


@dataclass
class EntityState:
    """A stable typed entity (cross-segment).

    ``candidates`` may hold alternative identity hypotheses for the same
    handle when alias resolution is ambiguous (plan §14).
    """

    entity_id: str
    canonical_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "person"  # person | object | group | place | speaker
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_type: SourceType = "observed"
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        return d


@dataclass
class EventSpan:
    """A grounded event with participants and time bounds."""

    event_id: str
    event_type: str
    description: str = ""
    participants: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    location: Optional[str] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_type: SourceType = "observed"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        return d


@dataclass
class InteractionEdge:
    """Typed (entity, relation, entity) edge anchored to evidence."""

    edge_id: str
    src_entity: str
    dst_entity: str
    interaction_type: str
    event_id: Optional[str] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_type: SourceType = "observed"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        return d


@dataclass
class TemporalRelation:
    """Typed temporal/causal link between two events."""

    relation_id: str
    lhs_event_id: str
    rhs_event_id: str
    relation_type: str  # before | after | overlap | causes | enables
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_type: SourceType = "inferred"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        return d


@dataclass
class VisibilityState:
    """Typed visibility/access state (saw / could_see / heard / …).

    Drives Theory-of-Mind style queries: ``what could X have seen at
    time T?`` for SIV-Bench, Video-Holmes, M3-Bench.
    """

    state_id: str
    holder_entity: str
    target_event_or_object: str
    relation_type: str  # saw | could_see | could_not_see | heard | did_not_hear
    time_range: Optional[Tuple[float, float]] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_type: SourceType = "inferred"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        if self.time_range is not None:
            d["time_range"] = list(self.time_range)
        return d


@dataclass
class BeliefCandidate:
    """Typed belief / motive / suspicion hypothesis (always inferred)."""

    belief_id: str
    holder_entity: str
    proposition: str
    polarity: str = "uncertain"  # believes_true | believes_false | uncertain
    trigger_event_id: Optional[str] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.5
    source_type: SourceType = "inferred"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        return d


@dataclass
class GroundedClip:
    """The typed unit consumed by the runtime / memory layers.

    A ``GroundedClip`` is the typed counterpart of
    :class:`visual_grounding.schemas.GroundedWindow`. Same time range,
    same evidence anchoring, but fully typed objects with provenance.
    """

    clip_id: str
    video_id: str
    start_time: float
    end_time: float
    summary: Optional[str] = None
    entities: List[EntityState] = field(default_factory=list)
    events: List[EventSpan] = field(default_factory=list)
    interactions: List[InteractionEdge] = field(default_factory=list)
    temporal_relations: List[TemporalRelation] = field(default_factory=list)
    visibility_states: List[VisibilityState] = field(default_factory=list)
    belief_candidates: List[BeliefCandidate] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_range(self) -> Tuple[float, float]:
        return (self.start_time, self.end_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "entities": [e.to_dict() for e in self.entities],
            "events": [e.to_dict() for e in self.events],
            "interactions": [i.to_dict() for i in self.interactions],
            "temporal_relations": [r.to_dict() for r in self.temporal_relations],
            "visibility_states": [v.to_dict() for v in self.visibility_states],
            "belief_candidates": [b.to_dict() for b in self.belief_candidates],
            "evidence_refs": _evref_to_dict(self.evidence_refs),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Memory record (Layer 3 — long-video projection)
# ---------------------------------------------------------------------------


MemoryKind = Literal[
    "episodic_event",
    "entity_thread",
    "relation",
    "social",
    "semantic_summary",
]


@dataclass
class MemoryRecord:
    """Memory-friendly projection of a typed grounded object.

    Long-video runtimes index ``MemoryRecord`` rows in a hierarchical
    store; short-video runtimes can skip persistence entirely.
    """

    record_id: str
    kind: MemoryKind
    video_id: str
    text: str
    time_range: Optional[Tuple[float, float]] = None
    entity_ids: List[str] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    source_type: SourceType = "observed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = _evref_to_dict(self.evidence_refs)
        if self.time_range is not None:
            d["time_range"] = list(self.time_range)
        return d


__all__ = [
    "SourceType",
    "MemoryKind",
    "VideoSegment",
    "SubtitleSpan",
    "RawObservation",
    "EntityState",
    "EventSpan",
    "InteractionEdge",
    "TemporalRelation",
    "VisibilityState",
    "BeliefCandidate",
    "GroundedClip",
    "MemoryRecord",
    "new_grounding_id",
]

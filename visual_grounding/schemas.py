r"""Canonical data contracts for visual grounding.

These objects implement the schemas defined in:

- ``infra_plans/01_grounding/video_benchmarks_grounding.md`` §2 (``GroundingNode``), §3.2
  (per-window grounding JSON), §4 (hierarchical memory levels).
- ``infra_plans/03_controller/actors_reasoning_model.md`` §4.2 (``MemoryNode``) and the
  "Canonical Runtime Data Contracts" gap list in
  ``plan_docs_implementation_checklist.md`` §1.
- ``infra_plans/02_memory/agentic_memory_design.md`` — three stores (episodic /
  semantic / state) + evidence as an attachment layer (not a separate
  top-level store).

The grounding layer produces ``GroundedWindow`` objects; the retrieval
layer turns them into typed ``GroundingNode`` rows indexed by a
``SocialVideoGraph``. Short videos keep ``GroundedWindow``\ s in a
``DirectContext`` in-context buffer; long videos persist both windows
and the derived graph.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class GroundingMode(str, Enum):
    """Execution regime for :func:`build_grounded_context`.

    - ``direct``: keep grounded windows in an in-context buffer. Used for
      short, context-rich clips (Video-Holmes, SIV-Bench).
    - ``retrieval``: persist windows + distilled summaries in a
      ``SocialVideoGraph`` for later ``search(...)`` calls. Used for long
      videos (VRBench, LongVideoBench, CG-Bench, M3-Bench).
    - ``auto``: pick based on video duration vs. a configurable threshold.
    """

    direct = "direct"
    retrieval = "retrieval"
    auto = "auto"


# Accept strings too for the public API.
ModeLike = Union[GroundingMode, Literal["direct", "retrieval", "auto"]]


def _coerce_mode(mode: ModeLike) -> GroundingMode:
    if isinstance(mode, GroundingMode):
        return mode
    return GroundingMode(str(mode).lower())


# ---------------------------------------------------------------------------
# Evidence layer (attachment, not a top-level store)
# ---------------------------------------------------------------------------


@dataclass
class EvidenceRef:
    """A pointer into the evidence substrate (visual / subtitle / audio).

    Mirrors ``agentic_memory_design.md``'s stance that visual material is
    not a first-class memory store — it is an attachment on episodic or
    state records.

    The optional ``video_id``, ``clip_id``, ``source_type``, ``confidence``
    fields support the typed grounding layer
    (``grounding_schemas.py`` / ``grounding_runtime.py``); they default to
    ``None`` / ``1.0`` / ``"observed"`` so existing callers stay
    untouched.
    """

    ref_id: str
    modality: str  # frame | clip | subtitle | audio | region | memory
    # Timestamps are always seconds from the start of the video.
    timestamp: Optional[Tuple[float, float]] = None
    # Back-pointer into the raw source — frame index, subtitle span, etc.
    locator: Optional[Dict[str, Any]] = None
    # Optional cached text (subtitle line, ASR span, caption, ...).
    text: Optional[str] = None
    # --- typed-grounding extensions (optional, backwards-compatible) ---
    video_id: Optional[str] = None
    clip_id: Optional[str] = None
    confidence: float = 1.0
    source_type: str = "observed"  # observed | inferred | retrieved | distilled

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def time_range(self) -> Optional[Tuple[float, float]]:
        """Alias for :attr:`timestamp` matching the typed-grounding schema."""
        return self.timestamp


# ---------------------------------------------------------------------------
# Per-window grounded output  (matches §3.2 JSON example)
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """Entity observed in a grounded window.

    ``id`` is a window-local handle (e.g. ``p1``). After entity resolution
    in the consolidator it is mapped to a globally stable ``entity_id``
    (see :class:`EntityProfile`).
    """

    id: str
    type: Literal["person", "object", "group", "speaker"]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Interaction:
    src: str
    rel: str  # talking_to, looking_at, helping, blocking, ...
    dst: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Event:
    type: str  # enters_room, joins_group, confrontation_start, ...
    agents: List[str] = field(default_factory=list)
    confidence: float = 1.0
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SocialHypothesis:
    """Latent interpretation kept as a hypothesis, not hard fact.

    Hypotheses index into the state-memory social subfield at query time
    (see ``agentic_memory_design.md``).
    """

    type: Literal[
        "intention", "belief", "emotion", "trust", "suspicion",
        "alliance", "conflict", "deception_risk", "goal", "commitment",
    ]
    # target may be a single entity, a pair, or a group handle.
    target: Union[str, List[str]]
    value: str
    confidence: float = 0.5
    provenance: Literal[
        "directly_observed", "inferred_from_dialogue",
        "inferred_from_behavior", "inferred_from_absence",
    ] = "inferred_from_behavior"
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundedWindow:
    """One locally-grounded window of video.

    This is the canonical unit consumed by both direct-mode reasoning and
    retrieval-mode memory construction.
    """

    window_id: str
    time_span: Tuple[float, float]
    scene: Optional[str] = None
    subtitle_mode: Literal["origin", "added", "removed", "none"] = "origin"
    entities: List[Entity] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    social_hypotheses: List[SocialHypothesis] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    # frame indices sampled in this window (optional cache for reasoner prompts)
    frame_indices: List[int] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["time_span"] = list(self.time_span)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroundedWindow":
        return cls(
            window_id=d["window_id"],
            time_span=tuple(d["time_span"]),
            scene=d.get("scene"),
            subtitle_mode=d.get("subtitle_mode", "origin"),
            entities=[Entity(**e) for e in d.get("entities", [])],
            interactions=[Interaction(**x) for x in d.get("interactions", [])],
            events=[Event(**ev) for ev in d.get("events", [])],
            social_hypotheses=[
                SocialHypothesis(**h) for h in d.get("social_hypotheses", [])
            ],
            evidence=[EvidenceRef(**e) for e in d.get("evidence", [])],
            frame_indices=list(d.get("frame_indices", [])),
            confidence=float(d.get("confidence", 1.0)),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Graph node schema (retrieval mode)
# ---------------------------------------------------------------------------


NodeType = Literal[
    "entity",
    "interaction",
    "event",
    "social_hypothesis",
    "episodic",
    "semantic",
]


@dataclass
class GroundingNode:
    """Structural node in the :class:`SocialVideoGraph`.

    Matches the ``GroundingNode`` dataclass in
    ``infra_plans/01_grounding/video_benchmarks_grounding.md`` §2.1. Visual/audio/subtitle
    material is attached via ``evidence_refs`` and through the graph's
    evidence index — not as a fifth top-level memory type.
    """

    node_id: str
    node_type: NodeType
    text: str
    timestamp: Tuple[float, float]
    clip_id: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = list(self.timestamp)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GroundingNode":
        return cls(
            node_id=d["node_id"],
            node_type=d["node_type"],
            text=d["text"],
            timestamp=tuple(d["timestamp"]),
            clip_id=d.get("clip_id"),
            entity_ids=list(d.get("entity_ids", [])),
            confidence=float(d.get("confidence", 1.0)),
            evidence_refs=list(d.get("evidence_refs", [])),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class EntityProfile:
    """Cross-window entity record produced by entity resolution.

    ``plan_docs_implementation_checklist.md`` §3 calls this out as a
    high-risk gap (face/voice/person tracking for M3-Bench etc.). Kept
    here as a first-class profile with alias handling.
    """

    entity_id: str  # stable global id, e.g. "person_07"
    canonical_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    appearance_embedding_ids: List[str] = field(default_factory=list)
    voice_embedding_ids: List[str] = field(default_factory=list)
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    window_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Direct-mode in-context buffer (short videos)
# ---------------------------------------------------------------------------


@dataclass
class DirectContext:
    r"""In-context buffer for short, context-rich videos.

    Direct mode does **not** build a persistent retrieval index. Instead,
    the reasoner ingests the raw frames / subtitles and the list of
    ``GroundedWindow``\ s directly. This object is what
    :func:`build_grounded_context` returns when ``mode="direct"``.
    """

    video_path: str
    duration: float
    windows: List[GroundedWindow] = field(default_factory=list)
    subtitle_mode: Literal["origin", "added", "removed", "none"] = "origin"
    subtitles: List[EvidenceRef] = field(default_factory=list)
    frame_index_path: Optional[str] = None  # optional cache file with frames
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- summary views --------------------------------------------------
    @property
    def mode(self) -> GroundingMode:
        return GroundingMode.direct

    def as_reasoner_text(self, include_hypotheses: bool = True) -> str:
        """Pretty-print the grounded state for a reasoner prompt.

        Order: scene -> entities -> interactions -> events -> hypotheses,
        one window block at a time. Keeps the ``[Think]`` loop in
        ``actors_reasoning_model.md`` §3 usable without retrieval.
        """
        lines: List[str] = []
        for w in self.windows:
            s, e = w.time_span
            header = f"[{s:7.2f}s – {e:7.2f}s] {w.scene or ''}".rstrip()
            lines.append(header)
            if w.entities:
                ent_str = "; ".join(
                    f"{ent.id}({ent.type}){'/' + _fmt_attrs(ent.attributes) if ent.attributes else ''}"
                    for ent in w.entities
                )
                lines.append(f"  entities: {ent_str}")
            if w.interactions:
                lines.append("  interactions: " + "; ".join(
                    f"{i.src} -{i.rel}-> {i.dst} (c={i.confidence:.2f})"
                    for i in w.interactions
                ))
            if w.events:
                lines.append("  events: " + "; ".join(
                    f"{ev.type}[{','.join(ev.agents)}] (c={ev.confidence:.2f})"
                    for ev in w.events
                ))
            if include_hypotheses and w.social_hypotheses:
                lines.append("  hypotheses: " + "; ".join(
                    f"{h.type}({h.target})={h.value} (c={h.confidence:.2f})"
                    for h in w.social_hypotheses
                ))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "mode": self.mode.value,
            "subtitle_mode": self.subtitle_mode,
            "windows": [w.to_dict() for w in self.windows],
            "subtitles": [s.to_dict() for s in self.subtitles],
            "frame_index_path": self.frame_index_path,
            "metadata": self.metadata,
        }


def _fmt_attrs(attrs: Dict[str, Any]) -> str:
    if not attrs:
        return ""
    parts = []
    for k, v in attrs.items():
        parts.append(f"{k}={v}")
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_ts() -> float:
    return time.time()


__all__ = [
    "GroundingMode",
    "ModeLike",
    "EvidenceRef",
    "Entity",
    "Interaction",
    "Event",
    "SocialHypothesis",
    "GroundedWindow",
    "GroundingNode",
    "NodeType",
    "EntityProfile",
    "DirectContext",
    "new_id",
    "now_ts",
]

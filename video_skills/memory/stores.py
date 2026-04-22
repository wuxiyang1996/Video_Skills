"""In-memory backing stores for the three-store agentic memory.

Layout:

- :class:`EpisodicStore` — ``what happened``: events keyed by clip / window,
  with ``contradicts`` edges.
- :class:`SemanticStore` — ``what stays true over time``: slowly-changing
  summaries with versioning.
- :class:`StateStore` — ``what is true now for reasoning at query time``:
  social-state (beliefs, knowledge boundaries) + spatial-state subfields,
  each with ``supersedes`` linkage.
- :class:`EvidenceStore` — registry of :class:`EvidenceRef` objects;
  episodic / state writes attach refs by id.
- :class:`EntityProfileRegistry` — union-find style entity aliasing.

All collections are pure-Python dicts so the v1 substrate runs without any
backing service. Persistence is provided by ``Memory.to_dict`` /
``Memory.from_dict``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import EvidenceRef, new_id, now_ts


# ---------------------------------------------------------------------------
# Episodic store — "what happened"
# ---------------------------------------------------------------------------


@dataclass
class EpisodicEvent:
    """A grounded event appended to episodic memory.

    Sourced from a ``GroundedWindow`` event by ``append_grounded_event``.
    """

    event_id: str
    thread_id: str
    clip_id: str
    window_id: str
    event_type: str
    description: str
    participants: List[str] = field(default_factory=list)
    time_span: Optional[Tuple[float, float]] = None
    evidence_ref_ids: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    confidence: float = 1.0
    inferred: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicThread:
    """An episodic thread groups events for one clip / window."""

    thread_id: str
    clip_id: str
    time_span: Optional[Tuple[float, float]] = None
    event_ids: List[str] = field(default_factory=list)
    summary_id: Optional[str] = None  # set when compressed
    meta: Dict[str, Any] = field(default_factory=dict)


class EpisodicStore:
    def __init__(self) -> None:
        self.threads: Dict[str, EpisodicThread] = {}
        self.events: Dict[str, EpisodicEvent] = {}
        # Secondary indices
        self._by_clip: Dict[str, List[str]] = {}  # clip_id -> [event_ids]
        self._by_entity: Dict[str, List[str]] = {}  # entity_id -> [event_ids]

    def add_thread(self, thread: EpisodicThread) -> None:
        self.threads[thread.thread_id] = thread

    def add_event(self, event: EpisodicEvent) -> None:
        self.events[event.event_id] = event
        self._by_clip.setdefault(event.clip_id, []).append(event.event_id)
        for ent in event.participants:
            self._by_entity.setdefault(ent, []).append(event.event_id)
        thread = self.threads.get(event.thread_id)
        if thread is not None and event.event_id not in thread.event_ids:
            thread.event_ids.append(event.event_id)

    def events_for_entity(self, entity_id: str) -> List[EpisodicEvent]:
        return [self.events[eid] for eid in self._by_entity.get(entity_id, [])]

    def events_in_time(
        self,
        time_span: Tuple[float, float],
    ) -> List[EpisodicEvent]:
        lo, hi = time_span
        out: List[EpisodicEvent] = []
        for e in self.events.values():
            if e.time_span is None:
                continue
            es, ee = e.time_span
            if es <= hi and ee >= lo:
                out.append(e)
        return out

    def stats(self) -> Dict[str, int]:
        return {
            "n_threads": len(self.threads),
            "n_events": len(self.events),
        }


# ---------------------------------------------------------------------------
# Semantic store — "what stays true over time"
# ---------------------------------------------------------------------------


@dataclass
class SemanticSummary:
    """A versioned slowly-changing abstraction over episodic content."""

    summary_id: str
    subject: str  # entity_id, pair "ent1+ent2", or "global"
    text: str
    source_episode_ids: List[str] = field(default_factory=list)
    version: int = 1
    parent_version_id: Optional[str] = None
    confidence: float = 1.0
    inferred: bool = True  # semantic items are distilled, hence inferred
    meta: Dict[str, Any] = field(default_factory=dict)


class SemanticStore:
    def __init__(self) -> None:
        self.summaries: Dict[str, SemanticSummary] = {}
        self._by_subject: Dict[str, List[str]] = {}  # subject -> [summary_ids]
        self._archived: Dict[str, SemanticSummary] = {}

    def add(self, summary: SemanticSummary) -> None:
        self.summaries[summary.summary_id] = summary
        self._by_subject.setdefault(summary.subject, []).append(summary.summary_id)

    def archive(self, summary_id: str) -> None:
        s = self.summaries.pop(summary_id, None)
        if s is None:
            return
        self._archived[summary_id] = s
        ids = self._by_subject.get(s.subject, [])
        if summary_id in ids:
            ids.remove(summary_id)

    def for_subject(self, subject: str) -> List[SemanticSummary]:
        return [self.summaries[i] for i in self._by_subject.get(subject, [])]

    def stats(self) -> Dict[str, int]:
        return {"n_active": len(self.summaries), "n_archived": len(self._archived)}


# ---------------------------------------------------------------------------
# State store — "what is true now"
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """A row in the social subfield: who knows / believes what, when."""

    state_id: str
    holder_entity: str
    proposition: str
    polarity: str = "true"  # "true" | "false" | "unknown"
    time_anchor: Optional[float] = None
    evidence_ref_ids: List[str] = field(default_factory=list)
    source_step_id: Optional[str] = None
    supersedes: Optional[str] = None
    is_active: bool = True
    confidence: float = 1.0
    inferred: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialState:
    """A row in the spatial subfield: where an entity / object is."""

    state_id: str
    entity_id: str
    location: str
    visibility: Optional[str] = None  # "visible" | "occluded" | "off_screen"
    time_anchor: Optional[float] = None
    evidence_ref_ids: List[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    confidence: float = 1.0
    inferred: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


class StateStore:
    def __init__(self) -> None:
        self.beliefs: Dict[str, BeliefState] = {}
        self.spatial: Dict[str, SpatialState] = {}
        self._belief_by_holder: Dict[str, List[str]] = {}
        self._spatial_by_entity: Dict[str, List[str]] = {}

    # Beliefs ---------------------------------------------------------------

    def add_belief(self, belief: BeliefState) -> None:
        self.beliefs[belief.state_id] = belief
        self._belief_by_holder.setdefault(belief.holder_entity, []).append(belief.state_id)
        if belief.supersedes is not None:
            prior = self.beliefs.get(belief.supersedes)
            if prior is not None:
                prior.is_active = False

    def beliefs_for(self, holder: str, *, active_only: bool = True) -> List[BeliefState]:
        ids = self._belief_by_holder.get(holder, [])
        out = [self.beliefs[i] for i in ids]
        if active_only:
            out = [b for b in out if b.is_active]
        return out

    def beliefs_about(self, proposition_substr: str) -> List[BeliefState]:
        sub = proposition_substr.lower()
        return [b for b in self.beliefs.values() if sub in b.proposition.lower()]

    # Spatial ---------------------------------------------------------------

    def add_spatial(self, state: SpatialState) -> None:
        self.spatial[state.state_id] = state
        self._spatial_by_entity.setdefault(state.entity_id, []).append(state.state_id)

    def spatial_for(self, entity_id: str) -> List[SpatialState]:
        return [self.spatial[i] for i in self._spatial_by_entity.get(entity_id, [])]

    # Stats -----------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "n_beliefs": len(self.beliefs),
            "n_active_beliefs": sum(1 for b in self.beliefs.values() if b.is_active),
            "n_spatial": len(self.spatial),
        }


# ---------------------------------------------------------------------------
# Evidence store — registry of EvidenceRef
# ---------------------------------------------------------------------------


class EvidenceStore:
    def __init__(self) -> None:
        self._refs: Dict[str, EvidenceRef] = {}
        self._by_entity: Dict[str, List[str]] = {}

    def add(self, ref: EvidenceRef) -> EvidenceRef:
        self._refs[ref.ref_id] = ref
        for ent in ref.entities:
            self._by_entity.setdefault(ent, []).append(ref.ref_id)
        return ref

    def get(self, ref_id: str) -> Optional[EvidenceRef]:
        return self._refs.get(ref_id)

    def for_entity(self, entity_id: str) -> List[EvidenceRef]:
        return [self._refs[i] for i in self._by_entity.get(entity_id, [])]

    def all(self) -> List[EvidenceRef]:
        return list(self._refs.values())

    def stats(self) -> Dict[str, int]:
        return {"n_refs": len(self._refs)}


# ---------------------------------------------------------------------------
# Entity profile registry — union-find aliasing
# ---------------------------------------------------------------------------


@dataclass
class EntityProfile:
    """Stable identity for an entity across detections."""

    entity_id: str
    canonical_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    aliases_pending: List[str] = field(default_factory=list)
    appearance_evidence_ids: List[str] = field(default_factory=list)
    voice_evidence_ids: List[str] = field(default_factory=list)
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class EntityProfileRegistry:
    """Union-find over entity ids, merging never deletes the source id."""

    def __init__(self) -> None:
        self.profiles: Dict[str, EntityProfile] = {}
        self._parent: Dict[str, str] = {}  # entity_id -> canonical entity_id

    def register(self, profile: EntityProfile) -> EntityProfile:
        self.profiles[profile.entity_id] = profile
        self._parent.setdefault(profile.entity_id, profile.entity_id)
        return profile

    def find(self, entity_id: str) -> str:
        cur = entity_id
        while self._parent.get(cur, cur) != cur:
            cur = self._parent[cur]
        # Path compression
        node = entity_id
        while self._parent.get(node, node) != cur:
            nxt = self._parent[node]
            self._parent[node] = cur
            node = nxt
        return cur

    def merge(self, a: str, b: str) -> str:
        """Merge b into a's tree. Returns the canonical id (a's root)."""
        if a not in self._parent or b not in self._parent:
            raise KeyError(f"unknown entity in merge: {a!r} / {b!r}")
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        self._parent[rb] = ra
        # Union-find merge: aliases append-only
        target = self.profiles[ra]
        source = self.profiles[rb]
        for alias in [source.canonical_name, *source.aliases]:
            if alias and alias not in target.aliases:
                target.aliases.append(alias)
        for ev in source.appearance_evidence_ids:
            if ev not in target.appearance_evidence_ids:
                target.appearance_evidence_ids.append(ev)
        for ev in source.voice_evidence_ids:
            if ev not in target.voice_evidence_ids:
                target.voice_evidence_ids.append(ev)
        if source.first_seen is not None:
            target.first_seen = (
                source.first_seen if target.first_seen is None
                else min(target.first_seen, source.first_seen)
            )
        if source.last_seen is not None:
            target.last_seen = (
                source.last_seen if target.last_seen is None
                else max(target.last_seen, source.last_seen)
            )
        return ra

    def resolve(self, alias_or_id: str) -> Optional[str]:
        """Map an alias string OR an entity_id to a canonical entity_id."""
        if alias_or_id in self._parent:
            return self.find(alias_or_id)
        # Search by alias
        target = alias_or_id.lower()
        for p in self.profiles.values():
            if p.canonical_name and p.canonical_name.lower() == target:
                return self.find(p.entity_id)
            if any(a.lower() == target for a in p.aliases):
                return self.find(p.entity_id)
        return None

    def stats(self) -> Dict[str, int]:
        roots = {self.find(eid) for eid in self._parent}
        return {"n_profiles": len(self.profiles), "n_canonical": len(roots)}


# ---------------------------------------------------------------------------
# Memory facade — bundles all four stores + evidence
# ---------------------------------------------------------------------------


@dataclass
class Memory:
    """Top-level facade owning all four stores.

    The harness, retriever, and procedures all hold a reference to this object;
    no module reaches into individual stores by hard-coded name from outside.
    """

    episodic: EpisodicStore = field(default_factory=EpisodicStore)
    semantic: SemanticStore = field(default_factory=SemanticStore)
    state: StateStore = field(default_factory=StateStore)
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    entities: EntityProfileRegistry = field(default_factory=EntityProfileRegistry)
    contradicts: List[Tuple[str, str, str]] = field(default_factory=list)
    # ^ list of (record_id_a, record_id_b, reason)

    def stats(self) -> Dict[str, Any]:
        return {
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "state": self.state.stats(),
            "evidence": self.evidence.stats(),
            "entities": self.entities.stats(),
            "n_contradicts": len(self.contradicts),
        }

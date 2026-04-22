"""Grounding runtime — what the harness / skills should call.

This is the *single* read interface skill code uses to access typed
grounding state. Per the visual-grounding plan §10 + §12:

* Harness must not read raw captions or raw model output directly.
* Direct mode (short videos) and retrieval mode (long videos) share the
  same interface — only the storage backend differs.

The runtime stores:

* A list of :class:`GroundedClip` rows (always populated).
* A list of :class:`MemoryRecord` rows (populated for long videos via
  :class:`MemoryProjection`).
* An optional :class:`SocialVideoGraph` that powers semantic search
  when an embedder is wired in.

Required capabilities (plan §10):

    get_local_grounded_context(video_id, time_range)
    retrieve_by_entity(entity_id)
    retrieve_by_event(event_id)
    retrieve_supporting_evidence(query)
    retrieve_counterevidence(query)

Bonus capabilities the plan calls out under §12 ("Skill-consumable
retrieval API"):

    retrieve_events_for_entity(entity_id)
    retrieve_visibility(holder_entity, target)
    retrieve_event_chain(timestamp)
    retrieve_subtitles_for_event(event_id)
    list_subtitle_spans()
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from visual_grounding.grounding_schemas import (
    BeliefCandidate,
    EntityState,
    EventSpan,
    GroundedClip,
    InteractionEdge,
    MemoryRecord,
    SubtitleSpan,
    TemporalRelation,
    VisibilityState,
)
from visual_grounding.schemas import EvidenceRef


def _kw_score(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    q = set(str(query).lower().split())
    t = set(str(text).lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / (len(q) + 1e-6)


def _negate(query: str) -> List[str]:
    """Naive negation-aware token list: drop common negation tokens.

    Used for ``retrieve_counterevidence`` so e.g. "did NOT leave" finds
    evidence for "leave" via the same scoring path.
    """
    neg_tokens = {"not", "no", "never", "didn't", "doesn't", "isn't",
                  "wasn't", "won't", "can't", "couldn't"}
    out = [t for t in str(query).split() if t.lower() not in neg_tokens]
    return out


class GroundingRuntime:
    """Read-only typed-grounding view consumed by harness and skills."""

    def __init__(
        self,
        *,
        video_id: Optional[str] = None,
        clips: Optional[Sequence[GroundedClip]] = None,
        memory_records: Optional[Sequence[MemoryRecord]] = None,
        subtitle_spans: Optional[Sequence[SubtitleSpan]] = None,
        graph: Optional[Any] = None,
        mode: str = "direct",
    ) -> None:
        self.video_id = video_id
        self.mode = mode
        self._clips: List[GroundedClip] = list(clips or [])
        self._memory: List[MemoryRecord] = list(memory_records or [])
        self._subtitles: List[SubtitleSpan] = list(subtitle_spans or [])
        self._graph = graph

        # Indexes (rebuilt on every add to keep the runtime snappy).
        self._index_clips()

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_clip(self, clip: GroundedClip) -> None:
        self._clips.append(clip)
        self._index_clips()

    def add_memory(self, records: Iterable[MemoryRecord]) -> None:
        self._memory.extend(records)

    def add_subtitle_spans(self, spans: Iterable[SubtitleSpan]) -> None:
        self._subtitles.extend(spans)

    # ------------------------------------------------------------------
    # Required capabilities (plan §10)
    # ------------------------------------------------------------------

    def get_local_grounded_context(
        self,
        video_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[GroundedClip]:
        """Return clips overlapping ``time_range`` (None means whole video)."""
        vid = video_id or self.video_id
        out: List[GroundedClip] = []
        for clip in self._clips:
            if vid is not None and clip.video_id != vid:
                continue
            if time_range is not None:
                s, e = time_range
                if clip.end_time < s or clip.start_time > e:
                    continue
            out.append(clip)
        return out

    def retrieve_by_entity(self, entity_id: str) -> List[MemoryRecord]:
        eid = self._normalize_entity(entity_id)
        if eid is None:
            return []
        return [r for r in self._memory if eid in r.entity_ids]

    def retrieve_by_event(self, event_id: str) -> List[MemoryRecord]:
        if not event_id:
            return []
        return [r for r in self._memory if event_id in r.event_ids]

    def retrieve_supporting_evidence(
        self,
        query: str,
        *,
        top_k: int = 5,
        polarity: str = "support",
    ) -> List[EvidenceRef]:
        results = self._rank_records(query, top_k=top_k, polarity=polarity)
        evidence: List[EvidenceRef] = []
        seen: set = set()
        for rec, _ in results:
            for ref in rec.evidence_refs:
                if ref.ref_id in seen:
                    continue
                evidence.append(ref)
                seen.add(ref.ref_id)
        return evidence

    def retrieve_counterevidence(
        self, query: str, *, top_k: int = 5,
    ) -> List[EvidenceRef]:
        return self.retrieve_supporting_evidence(
            " ".join(_negate(query)), top_k=top_k, polarity="counter",
        )

    # ------------------------------------------------------------------
    # Bonus skill-consumable queries (plan §12)
    # ------------------------------------------------------------------

    def retrieve_events_for_entity(self, entity_id: str) -> List[EventSpan]:
        eid = self._normalize_entity(entity_id)
        if eid is None:
            return []
        out: List[EventSpan] = []
        seen: set = set()
        for clip in self._clips:
            for ev in clip.events:
                if eid in ev.participants and ev.event_id not in seen:
                    out.append(ev)
                    seen.add(ev.event_id)
        out.sort(key=lambda e: (e.start_time or 0.0))
        return out

    def retrieve_visibility(
        self,
        holder_entity: Optional[str] = None,
        target: Optional[str] = None,
        *,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[VisibilityState]:
        hid = self._normalize_entity(holder_entity) if holder_entity else None
        out: List[VisibilityState] = []
        for clip in self._clips:
            for v in clip.visibility_states:
                if hid is not None and v.holder_entity != hid:
                    continue
                if target is not None and v.target_event_or_object != target:
                    continue
                if time_range is not None and v.time_range is not None:
                    s, e = time_range
                    vs, ve = v.time_range
                    if ve < s or vs > e:
                        continue
                out.append(v)
        return out

    def retrieve_event_chain(
        self, timestamp: float, *, window: float = 30.0,
    ) -> List[EventSpan]:
        s, e = timestamp - window, timestamp + window
        out: List[EventSpan] = []
        seen: set = set()
        for clip in self._clips:
            for ev in clip.events:
                est = ev.start_time if ev.start_time is not None else 0.0
                eet = ev.end_time if ev.end_time is not None else est
                if eet < s or est > e:
                    continue
                if ev.event_id not in seen:
                    out.append(ev)
                    seen.add(ev.event_id)
        out.sort(key=lambda e: (e.start_time or 0.0))
        return out

    def retrieve_subtitles_for_event(self, event_id: str) -> List[SubtitleSpan]:
        for clip in self._clips:
            for ev in clip.events:
                if ev.event_id != event_id:
                    continue
                if ev.start_time is None or ev.end_time is None:
                    continue
                return [
                    s for s in self._subtitles
                    if not (s.end_time < ev.start_time
                            or s.start_time > ev.end_time)
                ]
        return []

    def list_subtitle_spans(
        self, time_range: Optional[Tuple[float, float]] = None,
    ) -> List[SubtitleSpan]:
        if time_range is None:
            return list(self._subtitles)
        s, e = time_range
        return [
            sp for sp in self._subtitles
            if not (sp.end_time < s or sp.start_time > e)
        ]

    # ------------------------------------------------------------------
    # Search routing (uses graph when available)
    # ------------------------------------------------------------------

    def search(
        self, query: str, *, top_k: int = 5,
    ) -> List[Tuple[MemoryRecord, float]]:
        return self._rank_records(query, top_k=top_k, polarity="support")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        per_kind: Dict[str, int] = {}
        for r in self._memory:
            per_kind[r.kind] = per_kind.get(r.kind, 0) + 1
        return {
            "clips": len(self._clips),
            "memory_records": len(self._memory),
            "subtitles": len(self._subtitles),
            "entities": len(self._entity_index),
            "events": len(self._event_index),
            "interactions": len(self._interaction_index),
            **{f"memory:{k}": v for k, v in per_kind.items()},
        }

    @property
    def clips(self) -> List[GroundedClip]:
        return list(self._clips)

    @property
    def memory_records(self) -> List[MemoryRecord]:
        return list(self._memory)

    @property
    def subtitle_spans(self) -> List[SubtitleSpan]:
        return list(self._subtitles)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _index_clips(self) -> None:
        self._entity_index: Dict[str, EntityState] = {}
        self._event_index: Dict[str, EventSpan] = {}
        self._interaction_index: Dict[str, InteractionEdge] = {}
        self._alias_to_id: Dict[str, str] = {}
        for clip in self._clips:
            for e in clip.entities:
                self._entity_index.setdefault(e.entity_id, e)
                if e.canonical_name:
                    self._alias_to_id[e.canonical_name.lower()] = e.entity_id
                for a in e.aliases:
                    if a:
                        self._alias_to_id[str(a).lower()] = e.entity_id
                self._alias_to_id[e.entity_id.lower()] = e.entity_id
            for ev in clip.events:
                self._event_index.setdefault(ev.event_id, ev)
            for edge in clip.interactions:
                self._interaction_index.setdefault(edge.edge_id, edge)

    def _normalize_entity(self, entity_id: Optional[str]) -> Optional[str]:
        if not entity_id:
            return None
        return self._alias_to_id.get(str(entity_id).lower(), str(entity_id))

    def _rank_records(
        self,
        query: str,
        *,
        top_k: int,
        polarity: str = "support",
    ) -> List[Tuple[MemoryRecord, float]]:
        if not query or not self._memory:
            return []

        # 1) Try the graph backend first when available.
        if self._graph is not None and hasattr(self._graph, "search"):
            try:
                hits = self._graph.search(query, top_k=top_k * 2)
                # Map graph nodes to memory records via clip_id metadata.
                if hits:
                    out: List[Tuple[MemoryRecord, float]] = []
                    for node, score in hits:
                        for rec in self._memory:
                            if rec.metadata.get("clip_id") == getattr(
                                node, "clip_id", None,
                            ) or rec.event_ids and getattr(
                                node, "node_id", "",
                            ) in rec.event_ids:
                                out.append((rec, float(score)))
                                if len(out) >= top_k:
                                    break
                        if len(out) >= top_k:
                            break
                    if out:
                        return out
            except Exception:
                pass

        # 2) Keyword fallback over memory records.
        qry = query if polarity == "support" else " ".join(_negate(query))
        scored = [(rec, _kw_score(qry, rec.text)) for rec in self._memory]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair in scored[:top_k] if pair[1] > 0.0] or scored[:top_k]


__all__ = ["GroundingRuntime"]

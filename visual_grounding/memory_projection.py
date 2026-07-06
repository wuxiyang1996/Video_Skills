r"""Memory projection (Layer 3 — long-video persistence).

Projects :class:`GroundedClip` objects into a flat list of typed
:class:`MemoryRecord`\ s suitable for indexing in a hierarchical
retrieval store. Short videos may skip this step entirely.

Per the visual-grounding plan §9, four memory kinds are emitted:

* ``episodic_event``    — one record per :class:`EventSpan`
                          (with participants + timestamp).
* ``entity_thread``     — one record per :class:`EntityState` rolled up
                          across clips (the entity's appearance thread).
* ``relation``          — one record per :class:`InteractionEdge`.
* ``social``            — one record per :class:`VisibilityState` and
                          per :class:`BeliefCandidate`.
* ``semantic_summary``  — clip-level summary node (one per
                          :class:`GroundedClip`) for cross-scene
                          compression.

The merge step deduplicates ``entity_thread`` records by ``entity_id``
across clips (a core requirement of M3-Bench style person tracking).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from visual_grounding.grounding_schemas import (
    BeliefCandidate,
    EntityState,
    EventSpan,
    GroundedClip,
    InteractionEdge,
    MemoryRecord,
    VisibilityState,
    new_grounding_id,
)
from visual_grounding.schemas import EvidenceRef


def _dedupe_refs(refs: Iterable[EvidenceRef]) -> List[EvidenceRef]:
    seen: Dict[str, EvidenceRef] = {}
    for r in refs:
        if r.ref_id not in seen:
            seen[r.ref_id] = r
    return list(seen.values())


class MemoryProjection:
    r"""Project :class:`GroundedClip`\ s into :class:`MemoryRecord`\ s."""

    def __init__(
        self,
        *,
        emit_semantic_summary: bool = True,
        emit_visibility_records: bool = True,
        emit_belief_records: bool = True,
    ) -> None:
        self.emit_semantic_summary = emit_semantic_summary
        self.emit_visibility_records = emit_visibility_records
        self.emit_belief_records = emit_belief_records

    # ------------------------------------------------------------------
    # Per-clip projection
    # ------------------------------------------------------------------

    def project_clip(self, grounded_clip: GroundedClip) -> List[MemoryRecord]:
        out: List[MemoryRecord] = []
        vid = grounded_clip.video_id
        time_range = (grounded_clip.start_time, grounded_clip.end_time)

        for ev in grounded_clip.events:
            out.append(MemoryRecord(
                record_id=new_grounding_id("memep"),
                kind="episodic_event",
                video_id=vid,
                text=self._event_text(ev),
                time_range=(
                    ev.start_time if ev.start_time is not None
                    else grounded_clip.start_time,
                    ev.end_time if ev.end_time is not None
                    else grounded_clip.end_time,
                ),
                entity_ids=list(ev.participants),
                event_ids=[ev.event_id],
                evidence_refs=list(ev.evidence_refs),
                confidence=ev.confidence,
                source_type=ev.source_type,
                metadata={
                    "event_type": ev.event_type,
                    "clip_id": grounded_clip.clip_id,
                },
            ))

        for ent in grounded_clip.entities:
            out.append(MemoryRecord(
                record_id=new_grounding_id("memet"),
                kind="entity_thread",
                video_id=vid,
                text=self._entity_text(ent),
                time_range=time_range,
                entity_ids=[ent.entity_id],
                evidence_refs=list(ent.evidence_refs),
                confidence=ent.confidence,
                source_type=ent.source_type,
                metadata={
                    "canonical_name": ent.canonical_name,
                    "aliases": list(ent.aliases),
                    "entity_type": ent.entity_type,
                    "attributes": dict(ent.attributes),
                    "clip_id": grounded_clip.clip_id,
                },
            ))

        for edge in grounded_clip.interactions:
            out.append(MemoryRecord(
                record_id=new_grounding_id("memrel"),
                kind="relation",
                video_id=vid,
                text=f"{edge.src_entity} {edge.interaction_type} {edge.dst_entity}",
                time_range=time_range,
                entity_ids=[edge.src_entity, edge.dst_entity],
                event_ids=[edge.event_id] if edge.event_id else [],
                evidence_refs=list(edge.evidence_refs),
                confidence=edge.confidence,
                source_type=edge.source_type,
                metadata={
                    "interaction_type": edge.interaction_type,
                    "clip_id": grounded_clip.clip_id,
                },
            ))

        if self.emit_visibility_records:
            for vs in grounded_clip.visibility_states:
                out.append(MemoryRecord(
                    record_id=new_grounding_id("memsoc"),
                    kind="social",
                    video_id=vid,
                    text=(
                        f"{vs.holder_entity} {vs.relation_type} "
                        f"{vs.target_event_or_object}"
                    ),
                    time_range=vs.time_range or time_range,
                    entity_ids=[vs.holder_entity],
                    event_ids=(
                        [vs.target_event_or_object]
                        if vs.target_event_or_object.startswith("evt_")
                        else []
                    ),
                    evidence_refs=list(vs.evidence_refs),
                    confidence=vs.confidence,
                    source_type=vs.source_type,
                    metadata={
                        "relation": vs.relation_type,
                        "clip_id": grounded_clip.clip_id,
                        "subkind": "visibility",
                    },
                ))

        if self.emit_belief_records:
            for b in grounded_clip.belief_candidates:
                out.append(MemoryRecord(
                    record_id=new_grounding_id("memsoc"),
                    kind="social",
                    video_id=vid,
                    text=(
                        f"{b.holder_entity} ({b.polarity}) {b.proposition}"
                    ),
                    time_range=time_range,
                    entity_ids=[b.holder_entity],
                    event_ids=[b.trigger_event_id] if b.trigger_event_id else [],
                    evidence_refs=list(b.evidence_refs),
                    confidence=b.confidence,
                    source_type=b.source_type,
                    metadata={
                        "polarity": b.polarity,
                        "clip_id": grounded_clip.clip_id,
                        "subkind": "belief",
                    },
                ))

        if self.emit_semantic_summary:
            out.append(MemoryRecord(
                record_id=new_grounding_id("memsem"),
                kind="semantic_summary",
                video_id=vid,
                text=grounded_clip.summary or self._fallback_summary(grounded_clip),
                time_range=time_range,
                entity_ids=[e.entity_id for e in grounded_clip.entities],
                event_ids=[ev.event_id for ev in grounded_clip.events],
                evidence_refs=list(grounded_clip.evidence_refs),
                confidence=1.0,
                source_type="distilled",
                metadata={"clip_id": grounded_clip.clip_id},
            ))

        return out

    # ------------------------------------------------------------------
    # Cross-clip merge (called once per video for long-video runs)
    # ------------------------------------------------------------------

    def merge_into_long_video_memory(
        self,
        clip_memories: Sequence[MemoryRecord],
    ) -> List[MemoryRecord]:
        out: List[MemoryRecord] = []
        # Group entity_thread records by entity_id so a long video has
        # one stable memory row per entity (with merged evidence + time
        # range).
        threads: Dict[str, MemoryRecord] = {}
        for rec in clip_memories:
            if rec.kind != "entity_thread":
                out.append(rec)
                continue
            eid = rec.entity_ids[0] if rec.entity_ids else rec.record_id
            existing = threads.get(eid)
            if existing is None:
                threads[eid] = rec
            else:
                merged_refs = _dedupe_refs(
                    list(existing.evidence_refs) + list(rec.evidence_refs)
                )
                existing.evidence_refs = merged_refs
                if rec.time_range is not None and existing.time_range is not None:
                    existing.time_range = (
                        min(existing.time_range[0], rec.time_range[0]),
                        max(existing.time_range[1], rec.time_range[1]),
                    )
                existing.confidence = max(existing.confidence, rec.confidence)
                # Aggregate aliases / attributes from clip-level metadata.
                meta = dict(existing.metadata or {})
                rec_meta = dict(rec.metadata or {})
                aliases = list({
                    *meta.get("aliases", []),
                    *rec_meta.get("aliases", []),
                })
                attrs = {**rec_meta.get("attributes", {}),
                         **meta.get("attributes", {})}
                meta["aliases"] = aliases
                meta["attributes"] = attrs
                if not meta.get("canonical_name"):
                    meta["canonical_name"] = rec_meta.get("canonical_name")
                existing.metadata = meta
        out.extend(threads.values())
        return out

    # ------------------------------------------------------------------
    # Convenience top-level driver
    # ------------------------------------------------------------------

    def project_clips(
        self, clips: Sequence[GroundedClip],
    ) -> List[MemoryRecord]:
        per_clip: List[MemoryRecord] = []
        for clip in clips:
            per_clip.extend(self.project_clip(clip))
        return self.merge_into_long_video_memory(per_clip)

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _event_text(self, ev: EventSpan) -> str:
        parts = [ev.event_type]
        if ev.participants:
            parts.append(f"({', '.join(ev.participants)})")
        if ev.description:
            parts.append(f"- {ev.description}")
        return " ".join(parts).strip()

    def _entity_text(self, ent: EntityState) -> str:
        name = ent.canonical_name or ent.entity_id
        attrs = ", ".join(f"{k}={v}" for k, v in (ent.attributes or {}).items())
        if attrs:
            return f"{name} ({ent.entity_type}; {attrs})"
        return f"{name} ({ent.entity_type})"

    def _fallback_summary(self, clip: GroundedClip) -> str:
        bits = []
        if clip.events:
            bits.append("events: " + ", ".join(e.event_type for e in clip.events))
        if clip.entities:
            names = [e.canonical_name or e.entity_id for e in clip.entities]
            bits.append("entities: " + ", ".join(names))
        return " | ".join(bits) or f"clip {clip.clip_id}"


__all__ = ["MemoryProjection"]

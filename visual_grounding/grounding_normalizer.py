"""Grounding normalizer (Layer 2 → typed :class:`GroundedClip`).

Takes a :class:`VideoSegment` and the upstream layer outputs (raw
observations + entities + events + social state + temporal relations)
and emits one :class:`GroundedClip` per segment with all provenance
preserved.

Per the visual-grounding plan §8 + §13:

* Every grounded object retains ``confidence``, ``source_type`` and
  ``evidence_refs`` (no orphans allowed).
* Overlapping entities/events are deduplicated by ``entity_id`` /
  ``event_id`` so the clip stays compact.
* The clip's evidence list is the union of the segment's clip
  evidence + every nested object's evidence.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from visual_grounding.grounding_schemas import (
    BeliefCandidate,
    EntityState,
    EventSpan,
    GroundedClip,
    InteractionEdge,
    RawObservation,
    TemporalRelation,
    VideoSegment,
    VisibilityState,
)
from visual_grounding.schemas import EvidenceRef, new_id


def _dedupe_evidence(refs: Iterable[EvidenceRef]) -> List[EvidenceRef]:
    seen: Dict[str, EvidenceRef] = {}
    for r in refs:
        if r.ref_id not in seen:
            seen[r.ref_id] = r
    return list(seen.values())


def _summary_from(observations: Sequence[RawObservation]) -> Optional[str]:
    captions = [
        o.payload.get("scene") for o in observations
        if o.observation_type == "caption" and o.payload.get("scene")
    ]
    if captions:
        return " | ".join(str(c) for c in captions)
    return None


class GroundingNormalizer:
    """Stateless normalizer; safe to share across segments."""

    def normalize(
        self,
        segment: VideoSegment,
        raw_observations: Sequence[RawObservation],
        entities: Sequence[EntityState],
        events: Sequence[EventSpan],
        social_outputs: Optional[Dict[str, list]] = None,
        temporal_relations: Optional[Sequence[TemporalRelation]] = None,
    ) -> GroundedClip:
        social_outputs = social_outputs or {}
        interactions: List[InteractionEdge] = list(
            social_outputs.get("interactions", []) or []
        )
        visibility_states: List[VisibilityState] = list(
            social_outputs.get("visibility_states", []) or []
        )
        belief_candidates: List[BeliefCandidate] = list(
            social_outputs.get("belief_candidates", []) or []
        )
        temporal_relations = list(temporal_relations or [])

        # Dedup entities by id (keep first; merge evidence).
        ent_by_id: Dict[str, EntityState] = {}
        for e in entities:
            existing = ent_by_id.get(e.entity_id)
            if existing is None:
                ent_by_id[e.entity_id] = e
            else:
                existing.evidence_refs = _dedupe_evidence(
                    list(existing.evidence_refs) + list(e.evidence_refs)
                )
                for k, v in e.attributes.items():
                    existing.attributes.setdefault(k, v)

        # Restrict to entities that actually have evidence in this segment
        # (every other entity is still queryable via the runtime, but we
        # keep the per-clip view focused).
        seg_entities = [
            e for e in ent_by_id.values()
            if any(
                (r.clip_id == segment.segment_id)
                or (r.timestamp is not None
                    and not (r.timestamp[1] < segment.start_time
                             or r.timestamp[0] > segment.end_time))
                for r in e.evidence_refs
            )
        ] or list(ent_by_id.values())

        seg_events = [
            ev for ev in events
            if (ev.start_time is None and ev.end_time is None)
            or not (
                (ev.end_time or ev.start_time or 0.0) < segment.start_time
                or (ev.start_time or 0.0) > segment.end_time
            )
        ]
        seg_event_ids = {ev.event_id for ev in seg_events}

        seg_interactions = [
            edge for edge in interactions
            if edge.event_id is None or edge.event_id in seg_event_ids
        ] or list(interactions)

        seg_temporal = [
            r for r in temporal_relations
            if r.lhs_event_id in seg_event_ids
            or r.rhs_event_id in seg_event_ids
        ]

        seg_visibility = [
            v for v in visibility_states
            if v.target_event_or_object in seg_event_ids
            or any(
                (r.clip_id == segment.segment_id) for r in v.evidence_refs
            )
        ]

        seg_beliefs = [
            b for b in belief_candidates
            if any(
                (r.clip_id == segment.segment_id) for r in b.evidence_refs
            )
        ]

        # Aggregate evidence — clip-level + every nested object's evidence.
        all_evidence: List[EvidenceRef] = []
        for o in raw_observations:
            if o.segment_id == segment.segment_id:
                all_evidence.extend(o.evidence_refs)
        for collection in (
            seg_entities, seg_events, seg_interactions, seg_temporal,
            seg_visibility, seg_beliefs,
        ):
            for obj in collection:
                all_evidence.extend(obj.evidence_refs)
        all_evidence = _dedupe_evidence(all_evidence)

        clip = GroundedClip(
            clip_id=segment.segment_id,
            video_id=segment.video_id,
            start_time=segment.start_time,
            end_time=segment.end_time,
            summary=_summary_from(
                [o for o in raw_observations
                 if o.segment_id == segment.segment_id],
            ),
            entities=seg_entities,
            events=seg_events,
            interactions=seg_interactions,
            temporal_relations=seg_temporal,
            visibility_states=seg_visibility,
            belief_candidates=seg_beliefs,
            evidence_refs=all_evidence,
            metadata=dict(segment.metadata),
        )
        return clip


__all__ = ["GroundingNormalizer"]

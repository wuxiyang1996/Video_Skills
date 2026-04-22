r"""Event grounding (typed).

Promotes ``event_proposal`` :class:`RawObservation`\ s into typed
:class:`EventSpan` rows with deduplication, participant remapping
through the :class:`EntityTracker`, and evidence aggregation.

Per the visual-grounding plan §7:

* Merge raw observations into event spans.
* Assign participants (via the entity tracker mapping).
* Attach evidence refs.
* Avoid overcommitting: if the proposal had a ``description`` flag of
  inferred-only, source_type is ``inferred``; otherwise ``observed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from visual_grounding.entity_tracker import EntityTracker
from visual_grounding.grounding_schemas import (
    EntityState,
    EventSpan,
    RawObservation,
    new_grounding_id,
)
from visual_grounding.schemas import EvidenceRef


def _key(event_type: str, agents: Sequence[str]) -> Tuple[str, Tuple[str, ...]]:
    return (str(event_type or "").lower(), tuple(sorted(str(a) for a in agents)))


class EventGrounder:
    """Build :class:`EventSpan` rows from event-proposal observations."""

    def __init__(
        self,
        *,
        merge_within_seconds: float = 2.0,
        observed_event_types: Optional[Sequence[str]] = None,
    ) -> None:
        self.merge_within_seconds = merge_within_seconds
        # Event types we treat as ``observed`` by default; everything else
        # becomes ``inferred`` so downstream skills know to vet it.
        default_observed = {
            "enters_room", "leaves_room", "stands_up", "sits_down",
            "picks_up", "puts_down", "opens", "closes",
            "speaks", "shows", "hands_over", "walks_to",
        }
        self.observed_event_types = set(
            observed_event_types or default_observed,
        )

    def build_events(
        self,
        observations: Sequence[RawObservation],
        entities: Sequence[EntityState],
        *,
        tracker: Optional[EntityTracker] = None,
        segment_index: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[EventSpan]:
        """Return one :class:`EventSpan` per (event_type, participants).

        ``segment_index`` maps ``segment_id -> (start_time, end_time)``
        so the event grounder can attach time bounds without re-probing
        the video. When omitted, time bounds are taken from the first
        evidence ref's timestamp.
        """
        events_by_key: Dict[Tuple[str, Tuple[str, ...]], EventSpan] = {}
        for obs in observations:
            if obs.observation_type != "event_proposal":
                continue
            event_type = str(obs.payload.get("type") or "event").lower()
            agents_raw = list(obs.payload.get("agents", []) or [])
            agents = [
                self._remap(tracker, obs.segment_id, a)
                for a in agents_raw
            ]
            agents = [a for a in agents if a]

            t0, t1 = self._span_for(obs, segment_index)
            key = _key(event_type, agents)
            existing = events_by_key.get(key)
            if existing is not None and abs(
                (existing.end_time or 0.0) - t0,
            ) <= self.merge_within_seconds:
                existing.end_time = max(existing.end_time or t1, t1)
                existing.start_time = min(existing.start_time or t0, t0)
                self._merge_evidence(existing.evidence_refs, obs.evidence_refs)
                existing.confidence = max(existing.confidence, obs.confidence)
                if not existing.description and obs.payload.get("description"):
                    existing.description = str(obs.payload["description"])
                continue

            span = EventSpan(
                event_id=new_grounding_id("evt"),
                event_type=event_type,
                description=str(obs.payload.get("description", "")),
                participants=agents,
                start_time=t0,
                end_time=t1,
                location=obs.payload.get("location"),
                evidence_refs=list(obs.evidence_refs),
                confidence=float(obs.confidence),
                source_type=(
                    "observed"
                    if event_type in self.observed_event_types
                    else "inferred"
                ),
            )
            events_by_key[key] = span

        # Stable order by start_time then event_type.
        ordered = sorted(
            events_by_key.values(),
            key=lambda e: (e.start_time or 0.0, e.event_type),
        )
        return ordered

    # -- helpers -------------------------------------------------------

    def _remap(
        self,
        tracker: Optional[EntityTracker],
        segment_id: Optional[str],
        local_id: str,
    ) -> str:
        if tracker is None:
            return str(local_id)
        gid = tracker.map_local_id(segment_id, str(local_id))
        return gid or str(local_id)

    def _span_for(
        self,
        obs: RawObservation,
        segment_index: Optional[Dict[str, Tuple[float, float]]],
    ) -> Tuple[float, float]:
        if segment_index and obs.segment_id in segment_index:
            return segment_index[obs.segment_id]
        for ref in obs.evidence_refs:
            if ref.timestamp is not None:
                return ref.timestamp
        return (0.0, 0.0)

    def _merge_evidence(
        self,
        target: List[EvidenceRef],
        new_refs: Sequence[EvidenceRef],
    ) -> None:
        seen = {r.ref_id for r in target}
        for r in new_refs:
            if r.ref_id not in seen:
                target.append(r)
                seen.add(r.ref_id)


__all__ = ["EventGrounder"]

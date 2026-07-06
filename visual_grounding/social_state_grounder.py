"""Social-state grounding (typed).

Builds:

* :class:`InteractionEdge`  — typed (src, rel, dst) interaction.
* :class:`VisibilityState`  — saw / could_see / heard / ... social
                              access state, derived as hypotheses from
                              co-presence + interaction signals.
* :class:`BeliefCandidate`  — typed motive / belief / suspicion
                              hypothesis (always ``inferred``).

Interactions come from ``interaction_proposal`` observations;
visibility states come from co-presence in events; belief candidates
come from ``social_hypothesis_proposal`` observations.

All inferred objects are explicitly tagged ``source_type="inferred"``
so downstream skill execution and failure analysis can treat them as
hypotheses, not facts (plan §13).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from visual_grounding.entity_tracker import EntityTracker
from visual_grounding.grounding_schemas import (
    BeliefCandidate,
    EntityState,
    EventSpan,
    InteractionEdge,
    RawObservation,
    VisibilityState,
    new_grounding_id,
)
from visual_grounding.schemas import EvidenceRef


_VISIBLE_RELS = {
    "looking_at", "talking_to", "facing", "watching", "showing",
    "demonstrating_to", "handing_to", "pointing_at",
}
_AUDIBLE_RELS = {"talking_to", "shouting_at", "calling", "speaking_to"}


class SocialStateGrounder:
    """Promote social observations into typed interaction / state objects."""

    def __init__(
        self,
        *,
        infer_visibility_from_events: bool = True,
        observed_relations: Optional[Sequence[str]] = None,
    ) -> None:
        self.infer_visibility_from_events = infer_visibility_from_events
        # Relations we treat as observed; everything else is inferred.
        default_observed = {
            "talking_to", "looking_at", "facing", "next_to",
            "handing_to", "showing", "calling", "pointing_at",
        }
        self.observed_relations = set(observed_relations or default_observed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_social_states(
        self,
        observations: Sequence[RawObservation],
        entities: Sequence[EntityState],
        events: Sequence[EventSpan],
        *,
        tracker: Optional[EntityTracker] = None,
    ) -> Dict[str, list]:
        interactions = self._build_interactions(observations, tracker, events)
        visibility = self._build_visibility(
            observations, events, interactions,
        )
        beliefs = self._build_beliefs(observations, tracker)
        return {
            "interactions": interactions,
            "visibility_states": visibility,
            "belief_candidates": beliefs,
        }

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def _build_interactions(
        self,
        observations: Sequence[RawObservation],
        tracker: Optional[EntityTracker],
        events: Sequence[EventSpan],
    ) -> List[InteractionEdge]:
        out: List[InteractionEdge] = []
        # Build a quick (start_time -> event_id) lookup so interactions
        # can be linked back to a containing event when present.
        event_by_time: List[Tuple[float, float, str]] = sorted(
            [(e.start_time or 0.0, e.end_time or (e.start_time or 0.0),
              e.event_id) for e in events],
        )

        def find_event(t: float) -> Optional[str]:
            for s, e, eid in event_by_time:
                if s - 0.01 <= t <= e + 0.01:
                    return eid
            return None

        for obs in observations:
            if obs.observation_type != "interaction_proposal":
                continue
            src_local = obs.payload.get("src")
            dst_local = obs.payload.get("dst")
            if src_local is None or dst_local is None:
                continue
            src = self._remap(tracker, obs.segment_id, src_local)
            dst = self._remap(tracker, obs.segment_id, dst_local)
            rel = str(obs.payload.get("rel") or "interacts_with").lower()
            t0 = (
                obs.evidence_refs[0].timestamp[0]
                if obs.evidence_refs and obs.evidence_refs[0].timestamp
                else 0.0
            )
            event_id = find_event(t0)
            out.append(InteractionEdge(
                edge_id=new_grounding_id("int"),
                src_entity=src,
                dst_entity=dst,
                interaction_type=rel,
                event_id=event_id,
                evidence_refs=list(obs.evidence_refs),
                confidence=float(obs.confidence),
                source_type=(
                    "observed" if rel in self.observed_relations else "inferred"
                ),
            ))
        return out

    # ------------------------------------------------------------------
    # Visibility / access states (always inferred)
    # ------------------------------------------------------------------

    def _build_visibility(
        self,
        observations: Sequence[RawObservation],
        events: Sequence[EventSpan],
        interactions: Sequence[InteractionEdge],
    ) -> List[VisibilityState]:
        out: List[VisibilityState] = []
        # A) From explicit interactions (looking_at etc.).
        for edge in interactions:
            if edge.interaction_type in _VISIBLE_RELS:
                t = self._first_ts(edge.evidence_refs)
                out.append(VisibilityState(
                    state_id=new_grounding_id("vis"),
                    holder_entity=edge.src_entity,
                    target_event_or_object=edge.dst_entity,
                    relation_type="saw",
                    time_range=t,
                    evidence_refs=list(edge.evidence_refs),
                    confidence=min(0.95, edge.confidence),
                    source_type="inferred",
                ))
            if edge.interaction_type in _AUDIBLE_RELS:
                t = self._first_ts(edge.evidence_refs)
                out.append(VisibilityState(
                    state_id=new_grounding_id("vis"),
                    holder_entity=edge.dst_entity,
                    target_event_or_object=edge.src_entity,
                    relation_type="heard",
                    time_range=t,
                    evidence_refs=list(edge.evidence_refs),
                    confidence=min(0.9, edge.confidence),
                    source_type="inferred",
                ))

        # B) From co-presence in events: every participant could_see the event.
        if self.infer_visibility_from_events:
            for event in events:
                if not event.participants:
                    continue
                window = (
                    (event.start_time or 0.0, event.end_time or 0.0)
                    if event.start_time is not None else None
                )
                for participant in event.participants:
                    out.append(VisibilityState(
                        state_id=new_grounding_id("vis"),
                        holder_entity=participant,
                        target_event_or_object=event.event_id,
                        relation_type="could_see",
                        time_range=window,
                        evidence_refs=list(event.evidence_refs),
                        confidence=min(0.7, event.confidence),
                        source_type="inferred",
                    ))
        return out

    # ------------------------------------------------------------------
    # Belief candidates (always inferred)
    # ------------------------------------------------------------------

    def _build_beliefs(
        self,
        observations: Sequence[RawObservation],
        tracker: Optional[EntityTracker],
    ) -> List[BeliefCandidate]:
        out: List[BeliefCandidate] = []
        for obs in observations:
            if obs.observation_type != "social_hypothesis_proposal":
                continue
            holder = obs.payload.get("target")
            if isinstance(holder, list):
                holder_id = self._remap(tracker, obs.segment_id, holder[0])
            else:
                holder_id = self._remap(tracker, obs.segment_id, holder)
            value = str(obs.payload.get("value", ""))
            polarity = obs.payload.get("polarity") or "uncertain"
            out.append(BeliefCandidate(
                belief_id=new_grounding_id("blf"),
                holder_entity=holder_id or "",
                proposition=f"{obs.payload.get('type', 'belief')}: {value}".strip(),
                polarity=str(polarity),
                trigger_event_id=None,
                evidence_refs=list(obs.evidence_refs),
                confidence=float(obs.confidence),
                source_type="inferred",
            ))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remap(
        self,
        tracker: Optional[EntityTracker],
        segment_id: Optional[str],
        local_id,
    ) -> str:
        if local_id is None:
            return ""
        if tracker is None:
            return str(local_id)
        gid = tracker.map_local_id(segment_id, str(local_id))
        return gid or str(local_id)

    def _first_ts(
        self, refs: Sequence[EvidenceRef],
    ) -> Optional[Tuple[float, float]]:
        for r in refs:
            if r.timestamp is not None:
                return r.timestamp
        return None


__all__ = ["SocialStateGrounder"]

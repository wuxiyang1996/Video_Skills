"""Entity tracking and identity consolidation (typed grounding).

Maps repeated participant / object / speaker mentions in
:class:`RawObservation` streams to stable :class:`EntityState`
records.

Per the visual-grounding plan §6 / §14:

* Stable entity ids are critical for person-heavy long videos
  (M3-Bench).
* Alias resolution must be exposed; unresolved identity ambiguity must
  be reported via ``candidates`` lists, not silently collapsed.

This module is purposely lightweight: it merges mentions by
``(local_id, attributes, speaker)`` signature, with an optional caller-
supplied ``embedding_fn`` for embedding-based face/voice matching.
Heavier matchers (e.g. the embedding resolver in
:mod:`visual_grounding.consolidator`) can be plugged in via the
``custom_matcher`` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from visual_grounding.grounding_schemas import (
    EntityState,
    RawObservation,
    new_grounding_id,
)
from visual_grounding.schemas import EvidenceRef


# A custom matcher: (existing_state, candidate_attrs, segment_id) -> similarity in [0, 1].
CustomMatcher = Callable[[EntityState, Dict[str, Any], Optional[str]], float]


def _signature(local_id: Optional[str], attrs: Dict[str, Any]) -> str:
    parts = []
    if local_id:
        parts.append(f"id={local_id}")
    for k in ("name", "role", "clothing", "appearance", "speaker"):
        v = attrs.get(k)
        if v:
            parts.append(f"{k}={v}")
    return "|".join(parts).lower()


def _jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta, tb = set(a.split("|")), set(b.split("|"))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class EntityTracker:
    """Stateful entity identity tracker.

    Usage:
        tracker = EntityTracker(match_threshold=0.6)
        states = tracker.update(observations)
        states = tracker.resolve_aliases(states)

    The tracker keeps an internal table of :class:`EntityState`
    keyed by stable global id; ``update(...)`` may be called repeatedly
    (segment-by-segment streaming) and returns *all* entities tracked so
    far. ``snapshot()`` returns a defensive copy.
    """

    def __init__(
        self,
        *,
        match_threshold: float = 0.5,
        custom_matcher: Optional[CustomMatcher] = None,
    ) -> None:
        self.match_threshold = match_threshold
        self.custom_matcher = custom_matcher
        self._states: Dict[str, EntityState] = {}
        self._signatures: Dict[str, str] = {}
        # local-id-per-segment -> stable id, useful for downstream remapping
        self._segment_local_to_global: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def map_local_id(
        self, segment_id: Optional[str], local_id: Optional[str],
    ) -> Optional[str]:
        if segment_id is None or local_id is None:
            return None
        return self._segment_local_to_global.get((segment_id, local_id))

    def snapshot(self) -> List[EntityState]:
        return list(self._states.values())

    # ------------------------------------------------------------------
    # Streaming update
    # ------------------------------------------------------------------

    def update(self, observations: Sequence[RawObservation]) -> List[EntityState]:
        """Consume a batch of observations and return the global state list."""
        for obs in observations:
            if obs.observation_type not in (
                "participant_mention",
                "entity_mention",
                "speaker_turn",
            ):
                continue
            attrs: Dict[str, Any] = dict(obs.payload.get("attributes", {}) or {})
            local_id = obs.payload.get("id")
            entity_type = (
                obs.payload.get("type")
                or ("speaker" if obs.observation_type == "speaker_turn" else "person")
            )
            if obs.observation_type == "speaker_turn":
                attrs.setdefault("speaker", obs.payload.get("speaker"))
            if "name" not in attrs and obs.payload.get("speaker"):
                attrs["name"] = obs.payload.get("speaker")

            chosen_id = self._match_or_alloc(
                segment_id=obs.segment_id,
                local_id=local_id,
                entity_type=entity_type,
                attrs=attrs,
                evidence_refs=obs.evidence_refs,
                confidence=obs.confidence,
            )
            if local_id is not None and obs.segment_id is not None:
                self._segment_local_to_global[(obs.segment_id, str(local_id))] = chosen_id

        return self.snapshot()

    # ------------------------------------------------------------------
    # Alias resolution
    # ------------------------------------------------------------------

    def resolve_aliases(
        self, entities: Optional[Sequence[EntityState]] = None,
    ) -> List[EntityState]:
        """Merge entities whose canonical_name / aliases overlap.

        Conservative: only merges when canonical names match
        case-insensitively or one entity's name appears in another's
        aliases. Anything more aggressive is a job for an embedding
        resolver (see :class:`visual_grounding.consolidator.EmbeddingEntityResolver`).
        """
        states = list(entities) if entities is not None else self.snapshot()
        # Build alias -> [entity_id] map.
        bucket: Dict[str, List[str]] = {}
        for st in states:
            keys = set()
            if st.canonical_name:
                keys.add(st.canonical_name.lower())
            for a in st.aliases:
                if a:
                    keys.add(str(a).lower())
            for k in keys:
                bucket.setdefault(k, []).append(st.entity_id)

        # Union-find over states sharing a key.
        parent: Dict[str, str] = {st.entity_id: st.entity_id for st in states}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ids in bucket.values():
            head = ids[0]
            for other in ids[1:]:
                union(head, other)

        # Group and merge.
        groups: Dict[str, List[EntityState]] = {}
        for st in states:
            groups.setdefault(find(st.entity_id), []).append(st)

        merged: List[EntityState] = []
        for root, members in groups.items():
            if len(members) == 1:
                merged.append(members[0])
                continue
            anchor = members[0]
            seen_aliases = set(anchor.aliases)
            evidence_seen = {r.ref_id for r in anchor.evidence_refs}
            for m in members[1:]:
                for a in [m.canonical_name, *m.aliases, m.entity_id]:
                    if a and a not in seen_aliases:
                        anchor.aliases.append(str(a))
                        seen_aliases.add(a)
                for r in m.evidence_refs:
                    if r.ref_id not in evidence_seen:
                        anchor.evidence_refs.append(r)
                        evidence_seen.add(r.ref_id)
                # remap stored maps that point at merged ids
                for k, v in list(self._segment_local_to_global.items()):
                    if v == m.entity_id:
                        self._segment_local_to_global[k] = anchor.entity_id
                self._states.pop(m.entity_id, None)
                self._signatures.pop(m.entity_id, None)
            merged.append(anchor)
            self._states[anchor.entity_id] = anchor

        return merged

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _match_or_alloc(
        self,
        *,
        segment_id: Optional[str],
        local_id: Optional[str],
        entity_type: str,
        attrs: Dict[str, Any],
        evidence_refs: Sequence[EvidenceRef],
        confidence: float,
    ) -> str:
        # Fast path: same (segment, local_id) we already saw.
        if segment_id is not None and local_id is not None:
            existing = self._segment_local_to_global.get(
                (segment_id, str(local_id)),
            )
            if existing is not None:
                self._extend_state(
                    existing, attrs, evidence_refs, confidence,
                )
                return existing

        sig = _signature(str(local_id) if local_id else None, attrs)
        best_id: Optional[str] = None
        best_score = 0.0

        for eid, st in self._states.items():
            if st.entity_type != entity_type and st.entity_type not in (
                "person", "speaker",
            ):
                continue
            score = 0.0
            if self.custom_matcher is not None:
                try:
                    score = float(self.custom_matcher(st, attrs, segment_id))
                except Exception:
                    score = 0.0
            else:
                score = _jaccard(sig, self._signatures.get(eid, ""))
            if score > best_score:
                best_score = score
                best_id = eid

        if best_id is not None and best_score >= self.match_threshold:
            self._extend_state(best_id, attrs, evidence_refs, confidence)
            return best_id

        # Allocate new entity.
        gid = new_grounding_id("ent")
        canonical = attrs.get("name")
        aliases: List[str] = []
        if local_id:
            aliases.append(str(local_id))
        speaker = attrs.get("speaker")
        if speaker and speaker != canonical:
            aliases.append(str(speaker))
        st = EntityState(
            entity_id=gid,
            canonical_name=canonical,
            aliases=aliases,
            entity_type=entity_type or "person",
            attributes={k: v for k, v in attrs.items() if k != "name"},
            evidence_refs=list(evidence_refs),
            confidence=float(confidence),
            source_type="observed",
            candidates=[],
        )
        self._states[gid] = st
        self._signatures[gid] = sig
        return gid

    def _extend_state(
        self,
        entity_id: str,
        attrs: Dict[str, Any],
        evidence_refs: Sequence[EvidenceRef],
        confidence: float,
    ) -> None:
        st = self._states[entity_id]
        for k, v in attrs.items():
            if k == "name" and not st.canonical_name:
                st.canonical_name = v
                continue
            if k not in st.attributes and v is not None:
                st.attributes[k] = v
        seen = {r.ref_id for r in st.evidence_refs}
        for r in evidence_refs:
            if r.ref_id not in seen:
                st.evidence_refs.append(r)
                seen.add(r.ref_id)
        # Confidence accumulates as a max — repeated observations boost it.
        st.confidence = max(st.confidence, float(confidence))


__all__ = ["EntityTracker", "CustomMatcher"]

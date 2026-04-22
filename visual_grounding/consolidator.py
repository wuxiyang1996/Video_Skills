r"""Temporal consolidation and entity resolution.

Step 4 (and the 8B-builder variant in §3.6) of the unified pipeline.
Responsibilities:

- Merge adjacent grounded windows when the predicate delta is small
  (used mainly in retrieval mode to build event-level nodes).
- Resolve window-local entity handles (``p1`` in window A, ``p1`` in
  window B) into stable cross-window entity IDs via a pluggable
  face/voice embedder.
- Distill clusters into semantic-summary nodes (cross-scene
  compression).
- Emit lists of :class:`GroundingNode`\ s that can be added to a
  :class:`SocialVideoGraph`.

Entity resolution here addresses the explicit
"Entity resolution / re-identification" gap called out in
``infra_plans/99_meta/plan_docs_implementation_checklist.md`` §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from visual_grounding.schemas import (
    Entity,
    EntityProfile,
    Event,
    GroundedWindow,
    GroundingNode,
    Interaction,
    SocialHypothesis,
    new_id,
)


# (window_id, local_id) -> stable entity_id
EntityMap = Dict[Tuple[str, str], str]

# A resolver that maps window-local entities to global IDs. Signature:
#   resolver(window, entity) -> Optional[str]
#   returning None means "no match, allocate a new global id".
EntityResolver = Callable[[GroundedWindow, Entity], Optional[str]]


# ---------------------------------------------------------------------------
# Default entity resolver: attribute-based matcher
# ---------------------------------------------------------------------------


class AttributeEntityResolver:
    """Light-weight cross-window entity matcher.

    Strategy: compare normalized attribute strings (clothing, role,
    position) across the rolling profile table. Works as a reasonable
    default when no face/voice embeddings are available. For M3-Bench we
    expect callers to inject an embedding-based resolver instead — see
    :class:`EmbeddingEntityResolver` below.
    """

    def __init__(self, match_threshold: float = 0.5):
        self.match_threshold = match_threshold
        self.profiles: Dict[str, EntityProfile] = {}

    def _signature(self, e: Entity) -> str:
        parts = [e.type]
        for k in ("role", "clothing", "appearance", "name"):
            v = e.attributes.get(k)
            if v:
                parts.append(f"{k}={v}")
        return "|".join(parts).lower()

    def _sim(self, sig_a: str, sig_b: str) -> float:
        if not sig_a or not sig_b:
            return 0.0
        tokens_a = set(sig_a.split("|"))
        tokens_b = set(sig_b.split("|"))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def __call__(self, window: GroundedWindow, entity: Entity) -> Optional[str]:
        sig = self._signature(entity)
        best_id: Optional[str] = None
        best_score = 0.0
        for eid, prof in self.profiles.items():
            prof_sig = prof.attributes.get("_signature", "")
            score = self._sim(sig, prof_sig)
            if score > best_score:
                best_score = score
                best_id = eid
        if best_id is not None and best_score >= self.match_threshold:
            return best_id
        return None

    def register(self, entity_id: str, profile: EntityProfile) -> None:
        self.profiles[entity_id] = profile


class EmbeddingEntityResolver:
    """Embedding-based entity resolver (cosine similarity, greedy match).

    ``embedding_fn(window, entity) -> np.ndarray`` should return an L2-
    normalized appearance/voice vector. Useful as a drop-in for
    face/voice re-identification required by M3-Bench.
    """

    def __init__(
        self,
        embedding_fn: Callable[[GroundedWindow, Entity], np.ndarray],
        threshold: float = 0.75,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.table: Dict[str, np.ndarray] = {}

    def __call__(self, window: GroundedWindow, entity: Entity) -> Optional[str]:
        try:
            vec = self.embedding_fn(window, entity)
        except Exception:
            return None
        if vec is None or not isinstance(vec, np.ndarray) or vec.size == 0:
            return None
        best_id: Optional[str] = None
        best_sim = -1.0
        for eid, ref in self.table.items():
            sim = float(np.dot(vec, ref))
            if sim > best_sim:
                best_sim = sim
                best_id = eid
        if best_id is not None and best_sim >= self.threshold:
            return best_id
        return None

    def register(self, entity_id: str, embedding: np.ndarray) -> None:
        self.table[entity_id] = embedding


# ---------------------------------------------------------------------------
# Entity resolution pass
# ---------------------------------------------------------------------------


def resolve_entities(
    windows: Sequence[GroundedWindow],
    resolver: Optional[EntityResolver] = None,
) -> Tuple[EntityMap, Dict[str, EntityProfile]]:
    """Walk all windows and return an (entity_map, profiles) pair.

    ``entity_map`` maps ``(window_id, local_id)`` to a stable global id.
    The same ``GroundedWindow`` objects are mutated so their
    entities/interactions/events/hypotheses reference global IDs after
    this call.
    """
    resolver = resolver or AttributeEntityResolver()
    entity_map: EntityMap = {}
    profiles: Dict[str, EntityProfile] = {}

    def _alloc_profile(e: Entity, t0: float, t1: float) -> str:
        gid = new_id(e.type if e.type in ("person", "group", "speaker", "object")
                     else "ent")
        prof = EntityProfile(
            entity_id=gid,
            canonical_name=e.attributes.get("name"),
            aliases=[e.id] if e.id else [],
            first_seen=t0,
            last_seen=t1,
            window_ids=[],
            attributes={
                **{k: v for k, v in e.attributes.items() if k != "name"},
                "_signature": (
                    getattr(resolver, "_signature", lambda x: "")(e)
                    if isinstance(resolver, AttributeEntityResolver) else ""
                ),
            },
        )
        profiles[gid] = prof
        if isinstance(resolver, AttributeEntityResolver):
            resolver.register(gid, prof)
        return gid

    for w in windows:
        t0, t1 = w.time_span
        # First pass: resolve ids.
        for e in w.entities:
            gid = resolver(w, e)
            if gid is None or gid not in profiles:
                gid = _alloc_profile(e, t0, t1)
            else:
                prof = profiles[gid]
                prof.last_seen = max(prof.last_seen or t1, t1)
                if e.id and e.id not in prof.aliases:
                    prof.aliases.append(e.id)
            entity_map[(w.window_id, e.id)] = gid
            if w.window_id not in profiles[gid].window_ids:
                profiles[gid].window_ids.append(w.window_id)

        # Second pass: rewrite in-window references.
        remap = lambda lid: entity_map.get((w.window_id, lid), lid)  # noqa
        for e in w.entities:
            e.id = remap(e.id)
        for x in w.interactions:
            x.src = remap(x.src)
            x.dst = remap(x.dst)
        for ev in w.events:
            ev.agents = [remap(a) for a in ev.agents]
        for h in w.social_hypotheses:
            if isinstance(h.target, list):
                h.target = [remap(a) for a in h.target]
            else:
                h.target = remap(h.target)

    return entity_map, profiles


# ---------------------------------------------------------------------------
# Window merging
# ---------------------------------------------------------------------------


def _predicate_set(w: GroundedWindow) -> set:
    preds = set()
    for x in w.interactions:
        preds.add(("int", x.src, x.rel, x.dst))
    for ev in w.events:
        preds.add(("evt", ev.type, tuple(sorted(ev.agents))))
    return preds


def merge_adjacent_windows(
    windows: Sequence[GroundedWindow],
    *,
    max_gap: float = 1.0,
    min_jaccard: float = 0.7,
) -> List[GroundedWindow]:
    """Merge temporally adjacent windows when their predicates are similar.

    Mirrors the "temporal aggregation" step in
    ``video_benchmarks_grounding.md`` §3.6. Predicate stability is
    measured with Jaccard over ``(interaction / event)`` tuples.
    """
    if not windows:
        return []
    ordered = sorted(windows, key=lambda x: x.time_span[0])
    merged: List[GroundedWindow] = [ordered[0]]
    for w in ordered[1:]:
        prev = merged[-1]
        gap = w.time_span[0] - prev.time_span[1]
        a = _predicate_set(prev)
        b = _predicate_set(w)
        denom = len(a | b) or 1
        jacc = len(a & b) / denom
        if gap <= max_gap and (jacc >= min_jaccard or (not a and not b)):
            prev.time_span = (prev.time_span[0], w.time_span[1])
            prev.entities = _dedupe_entities(prev.entities + w.entities)
            prev.interactions.extend(w.interactions)
            prev.events.extend(w.events)
            prev.social_hypotheses.extend(w.social_hypotheses)
            prev.evidence.extend(w.evidence)
            prev.frame_indices.extend(w.frame_indices)
            prev.confidence = min(prev.confidence, w.confidence)
        else:
            merged.append(w)
    return merged


def _dedupe_entities(entities: Iterable[Entity]) -> List[Entity]:
    seen: Dict[str, Entity] = {}
    for e in entities:
        if e.id in seen:
            # Prefer the richer attribute dict.
            if len(e.attributes) > len(seen[e.id].attributes):
                seen[e.id] = e
        else:
            seen[e.id] = e
    return list(seen.values())


# ---------------------------------------------------------------------------
# Node emission
# ---------------------------------------------------------------------------


def windows_to_nodes(
    windows: Sequence[GroundedWindow],
    profiles: Dict[str, EntityProfile],
) -> List[GroundingNode]:
    """Convert grounded windows + profiles into :class:`GroundingNode` rows.

    Emits: entity, episodic (per window), interaction, event, and
    social_hypothesis nodes. Semantic summaries are produced separately
    by :func:`distill_semantic_summaries`.
    """
    nodes: List[GroundingNode] = []

    # Entity nodes from the profile table.
    for prof in profiles.values():
        nodes.append(
            GroundingNode(
                node_id=prof.entity_id,
                node_type="entity",
                text=prof.canonical_name or prof.entity_id,
                timestamp=(prof.first_seen or 0.0, prof.last_seen or 0.0),
                entity_ids=[prof.entity_id],
                confidence=1.0,
                evidence_refs=[],
                metadata={
                    "aliases": prof.aliases,
                    "attributes": {
                        k: v for k, v in prof.attributes.items()
                        if k != "_signature"
                    },
                    "window_ids": prof.window_ids,
                },
            )
        )

    for w in windows:
        ev_refs = [r.ref_id for r in w.evidence]
        # Episodic backbone node (one per window).
        ep_text_parts = [w.scene or ""]
        if w.interactions:
            ep_text_parts.append(
                "; ".join(f"{i.src} {i.rel} {i.dst}" for i in w.interactions)
            )
        if w.events:
            ep_text_parts.append("; ".join(
                f"{ev.type}({','.join(ev.agents)})" for ev in w.events
            ))
        ep_text = " | ".join(p for p in ep_text_parts if p)
        nodes.append(
            GroundingNode(
                node_id=w.window_id,
                node_type="episodic",
                text=ep_text or "episodic window",
                timestamp=w.time_span,
                entity_ids=[e.id for e in w.entities],
                confidence=w.confidence,
                evidence_refs=ev_refs,
                metadata={"subtitle_mode": w.subtitle_mode},
            )
        )
        for x in w.interactions:
            nodes.append(
                GroundingNode(
                    node_id=new_id("int"),
                    node_type="interaction",
                    text=f"{x.src} {x.rel} {x.dst}",
                    timestamp=w.time_span,
                    clip_id=w.window_id,
                    entity_ids=[x.src, x.dst],
                    confidence=x.confidence,
                    evidence_refs=ev_refs,
                    metadata={"rel": x.rel, **x.metadata},
                )
            )
        for ev in w.events:
            nodes.append(
                GroundingNode(
                    node_id=new_id("evt"),
                    node_type="event",
                    text=f"{ev.type}: {ev.description or ''}".strip(": "),
                    timestamp=w.time_span,
                    clip_id=w.window_id,
                    entity_ids=list(ev.agents),
                    confidence=ev.confidence,
                    evidence_refs=ev_refs,
                    metadata={"type": ev.type, **ev.metadata},
                )
            )
        for h in w.social_hypotheses:
            targets = h.target if isinstance(h.target, list) else [h.target]
            nodes.append(
                GroundingNode(
                    node_id=new_id("hyp"),
                    node_type="social_hypothesis",
                    text=f"{h.type}({','.join(targets)})={h.value}",
                    timestamp=w.time_span,
                    clip_id=w.window_id,
                    entity_ids=list(targets),
                    confidence=h.confidence,
                    evidence_refs=list(h.supporting_evidence) + ev_refs,
                    metadata={
                        "type": h.type,
                        "provenance": h.provenance,
                        "contradicting_evidence": h.contradicting_evidence,
                    },
                )
            )
    return nodes


# ---------------------------------------------------------------------------
# Semantic distillation
# ---------------------------------------------------------------------------


def distill_semantic_summaries(
    windows: Sequence[GroundedWindow],
    *,
    cluster_size: int = 4,
    summarizer: Optional[Callable[[List[GroundedWindow]], str]] = None,
) -> List[GroundingNode]:
    """Produce long-range compression nodes (§4 — Semantic summaries).

    Default summarizer just concatenates scene phrases; callers will
    typically inject a VLM-backed summarizer that emits a compact
    paragraph. Clusters are fixed-size runs of temporally ordered
    windows — a cheap-but-useful default.
    """
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.time_span[0])
    nodes: List[GroundingNode] = []

    def default_summary(ws: List[GroundedWindow]) -> str:
        parts = []
        for w in ws:
            if w.scene:
                parts.append(f"[{w.time_span[0]:.0f}s] {w.scene}")
            elif w.events:
                parts.append(f"[{w.time_span[0]:.0f}s] " + ", ".join(
                    ev.type for ev in w.events
                ))
        return " → ".join(parts)

    fn = summarizer or default_summary
    for i in range(0, len(ordered), cluster_size):
        chunk = list(ordered[i : i + cluster_size])
        if not chunk:
            continue
        t0 = chunk[0].time_span[0]
        t1 = chunk[-1].time_span[1]
        entity_ids = sorted({
            e.id for w in chunk for e in w.entities
        })
        evidence_refs = [r.ref_id for w in chunk for r in w.evidence]
        try:
            text = fn(chunk) or ""
        except Exception:
            text = default_summary(chunk)
        nodes.append(
            GroundingNode(
                node_id=new_id("sem"),
                node_type="semantic",
                text=text,
                timestamp=(t0, t1),
                entity_ids=entity_ids,
                confidence=sum(w.confidence for w in chunk) / len(chunk),
                evidence_refs=evidence_refs,
                metadata={"source_window_ids": [w.window_id for w in chunk]},
            )
        )
    return nodes


__all__ = [
    "EntityMap",
    "EntityResolver",
    "AttributeEntityResolver",
    "EmbeddingEntityResolver",
    "resolve_entities",
    "merge_adjacent_windows",
    "windows_to_nodes",
    "distill_semantic_summaries",
]

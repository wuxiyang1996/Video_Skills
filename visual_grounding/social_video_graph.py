"""SocialVideoGraph — unified grounding index for long videos.

Implements the graph API specified in
``infra_plans/01_grounding/video_benchmarks_grounding.md`` §2.3:

- ``add_entity`` / ``add_interaction`` / ``add_event`` /
  ``add_social_hypothesis`` / ``add_episodic`` / ``add_semantic``.
- ``search``, ``get_timeline``, ``get_relations``, ``get_evidence``.
- ``translate`` / ``back_translate`` hooks for name/ID mapping
  (critical for M3-Bench, see §5.2).
- ``save`` / ``load`` for persisted artifacts.

Retrieval is backed by :class:`rag.retrieval.MemoryStore` so the same
embedding / ranking code powers both the skill bank and grounding
retrieval. When no embedder is supplied, the graph still works with a
lightweight keyword / substring fallback — useful for tests and for
direct-mode smoke runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from visual_grounding.schemas import (
    EntityProfile,
    EvidenceRef,
    GroundingNode,
    NodeType,
    new_id,
)


try:  # MemoryStore is optional — the graph runs without it.
    from rag.retrieval import MemoryStore  # type: ignore
    from rag.embedding.base import TextEmbedderBase, MultimodalEmbedderBase  # type: ignore
    _HAS_RAG = True
except Exception:  # pragma: no cover — keep the module import-safe
    _HAS_RAG = False
    MemoryStore = None  # type: ignore
    TextEmbedderBase = None  # type: ignore
    MultimodalEmbedderBase = None  # type: ignore


SearchResult = Tuple[GroundingNode, float]


# ---------------------------------------------------------------------------
# Simple keyword fallback (no embedder required)
# ---------------------------------------------------------------------------


def _keyword_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / (len(q) + 1e-6)


# ---------------------------------------------------------------------------
# SocialVideoGraph
# ---------------------------------------------------------------------------


class SocialVideoGraph:
    """Unified visual-grounding graph / index.

    Node kinds: ``entity``, ``interaction``, ``event``,
    ``social_hypothesis``, ``episodic``, ``semantic``.

    Design notes:
    - Structural nodes (entity / interaction / event) support traversal.
    - Backbone nodes (episodic / semantic) align with the episodic and
      semantic stores described in
      ``infra_plans/02_memory/agentic_memory_design.md``.
    - Hypothesis nodes index state-memory social entries; the state store
      itself is not duplicated here.
    - Evidence is held as a map ``ref_id -> EvidenceRef`` so
      ``get_evidence(node_id)`` can resolve attachments without recomputing.
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        top_k: int = 5,
        video_path: Optional[str] = None,
    ) -> None:
        self.video_path = video_path
        self.top_k = top_k
        self._nodes: Dict[str, GroundingNode] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._entities: Dict[str, EntityProfile] = {}
        self._evidence: Dict[str, EvidenceRef] = {}
        # Entity name/alias <-> id maps for translate/back-translate.
        self._alias_to_id: Dict[str, str] = {}
        self._id_to_display: Dict[str, str] = {}

        self._store: Optional[MemoryStore] = None  # type: ignore[assignment]
        self._use_embedder = False
        if embedder is not None and _HAS_RAG:
            self._store = MemoryStore(embedder=embedder, top_k=top_k)  # type: ignore[misc]
            self._use_embedder = True

    # ------------------------------------------------------------------
    # Low-level add
    # ------------------------------------------------------------------

    def add_node(self, node: GroundingNode) -> GroundingNode:
        if node.node_id in self._nodes:
            return self._nodes[node.node_id]
        self._nodes[node.node_id] = node
        self._by_type.setdefault(node.node_type, []).append(node.node_id)
        if self._use_embedder and self._store is not None:
            text = node.text or node.node_id
            try:
                self._store.add_texts([text], payloads=[node.node_id])
            except Exception:
                pass
        return node

    def add_nodes(self, nodes: Iterable[GroundingNode]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_evidence(self, refs: Iterable[EvidenceRef]) -> None:
        for r in refs:
            self._evidence[r.ref_id] = r

    def register_entity_profile(self, profile: EntityProfile) -> None:
        self._entities[profile.entity_id] = profile
        self._id_to_display[profile.entity_id] = (
            profile.canonical_name or profile.entity_id
        )
        for alias in [profile.canonical_name, *profile.aliases, profile.entity_id]:
            if alias:
                self._alias_to_id[str(alias).lower()] = profile.entity_id

    # ------------------------------------------------------------------
    # High-level adders (names match §2.3 API)
    # ------------------------------------------------------------------

    def add_entity(self, profile: EntityProfile) -> GroundingNode:
        self.register_entity_profile(profile)
        return self.add_node(
            GroundingNode(
                node_id=profile.entity_id,
                node_type="entity",
                text=profile.canonical_name or profile.entity_id,
                timestamp=(profile.first_seen or 0.0, profile.last_seen or 0.0),
                entity_ids=[profile.entity_id],
                confidence=1.0,
                metadata={
                    "aliases": profile.aliases,
                    "attributes": profile.attributes,
                    "window_ids": profile.window_ids,
                },
            )
        )

    def add_interaction(
        self,
        src: str,
        rel: str,
        dst: str,
        timestamp: Tuple[float, float],
        *,
        confidence: float = 1.0,
        clip_id: Optional[str] = None,
        evidence_refs: Optional[Sequence[str]] = None,
    ) -> GroundingNode:
        return self.add_node(
            GroundingNode(
                node_id=new_id("int"),
                node_type="interaction",
                text=f"{src} {rel} {dst}",
                timestamp=timestamp,
                clip_id=clip_id,
                entity_ids=[src, dst],
                confidence=confidence,
                evidence_refs=list(evidence_refs or []),
                metadata={"rel": rel},
            )
        )

    def add_event(
        self,
        event_type: str,
        agents: Sequence[str],
        timestamp: Tuple[float, float],
        *,
        description: Optional[str] = None,
        confidence: float = 1.0,
        clip_id: Optional[str] = None,
        evidence_refs: Optional[Sequence[str]] = None,
    ) -> GroundingNode:
        return self.add_node(
            GroundingNode(
                node_id=new_id("evt"),
                node_type="event",
                text=f"{event_type}: {description or ''}".strip(": "),
                timestamp=timestamp,
                clip_id=clip_id,
                entity_ids=list(agents),
                confidence=confidence,
                evidence_refs=list(evidence_refs or []),
                metadata={"type": event_type},
            )
        )

    def add_social_hypothesis(
        self,
        hyp_type: str,
        target: Union[str, Sequence[str]],
        value: str,
        timestamp: Tuple[float, float],
        *,
        confidence: float = 0.5,
        provenance: str = "inferred_from_behavior",
        supporting_evidence: Optional[Sequence[str]] = None,
        clip_id: Optional[str] = None,
    ) -> GroundingNode:
        targets = [target] if isinstance(target, str) else list(target)
        return self.add_node(
            GroundingNode(
                node_id=new_id("hyp"),
                node_type="social_hypothesis",
                text=f"{hyp_type}({','.join(targets)})={value}",
                timestamp=timestamp,
                clip_id=clip_id,
                entity_ids=targets,
                confidence=confidence,
                evidence_refs=list(supporting_evidence or []),
                metadata={"type": hyp_type, "provenance": provenance},
            )
        )

    def add_episodic(
        self,
        text: str,
        timestamp: Tuple[float, float],
        *,
        entity_ids: Sequence[str] = (),
        confidence: float = 1.0,
        evidence_refs: Sequence[str] = (),
        clip_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GroundingNode:
        return self.add_node(
            GroundingNode(
                node_id=clip_id or new_id("ep"),
                node_type="episodic",
                text=text,
                timestamp=timestamp,
                clip_id=clip_id,
                entity_ids=list(entity_ids),
                confidence=confidence,
                evidence_refs=list(evidence_refs),
                metadata=dict(metadata or {}),
            )
        )

    def add_semantic(
        self,
        text: str,
        timestamp: Tuple[float, float],
        *,
        entity_ids: Sequence[str] = (),
        confidence: float = 1.0,
        evidence_refs: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GroundingNode:
        return self.add_node(
            GroundingNode(
                node_id=new_id("sem"),
                node_type="semantic",
                text=text,
                timestamp=timestamp,
                entity_ids=list(entity_ids),
                confidence=confidence,
                evidence_refs=list(evidence_refs),
                metadata=dict(metadata or {}),
            )
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _filter(
        self,
        nodes: Iterable[GroundingNode],
        *,
        clip_filter: Optional[str] = None,
        entity_filter: Optional[Iterable[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        node_types: Optional[Iterable[NodeType]] = None,
    ) -> List[GroundingNode]:
        ent_set = set(entity_filter) if entity_filter else None
        type_set = set(node_types) if node_types else None
        out: List[GroundingNode] = []
        for n in nodes:
            if clip_filter is not None and n.clip_id != clip_filter:
                continue
            if ent_set is not None and not (set(n.entity_ids) & ent_set):
                continue
            if time_range is not None:
                s, e = time_range
                ns, ne = n.timestamp
                if ne < s or ns > e:
                    continue
            if type_set is not None and n.node_type not in type_set:
                continue
            out.append(n)
        return out

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        clip_filter: Optional[str] = None,
        entity_filter: Optional[Iterable[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        node_types: Optional[Iterable[NodeType]] = None,
    ) -> List[SearchResult]:
        """Search the graph. Returns ``(node, score)`` pairs.

        Uses the embedder when available, otherwise falls back to a
        keyword overlap score.
        """
        k = top_k or self.top_k
        if not self._nodes:
            return []

        candidates = list(self._nodes.values())
        if any((clip_filter, entity_filter, time_range, node_types)):
            candidates = self._filter(
                candidates,
                clip_filter=clip_filter,
                entity_filter=entity_filter,
                time_range=time_range,
                node_types=node_types,
            )
        if not candidates:
            return []

        # Translate alias-style names to entity IDs before scoring.
        query_effective = self.back_translate(query)

        if self._use_embedder and self._store is not None:
            try:
                ranked = self._store.rank(query_effective, k=max(k * 3, k))
                out: List[SearchResult] = []
                candidate_ids = {n.node_id for n in candidates}
                for _, score, payload in ranked:
                    if payload in candidate_ids and payload in self._nodes:
                        out.append((self._nodes[payload], float(score)))
                    if len(out) >= k:
                        break
                if out:
                    return out
            except Exception:
                pass

        scored = [
            (n, _keyword_score(query_effective, n.text))
            for n in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get_timeline(self, entity_id: str) -> List[GroundingNode]:
        eid = self.back_translate(entity_id)
        hits = [
            n for n in self._nodes.values()
            if eid in n.entity_ids and n.node_type != "entity"
        ]
        hits.sort(key=lambda n: n.timestamp[0])
        return hits

    def get_relations(self, entity_id: str) -> List[GroundingNode]:
        eid = self.back_translate(entity_id)
        return [
            n for n in self._nodes.values()
            if n.node_type in ("interaction", "social_hypothesis")
            and eid in n.entity_ids
        ]

    def get_evidence(self, node_id: str) -> List[EvidenceRef]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        out: List[EvidenceRef] = []
        for ref_id in node.evidence_refs:
            r = self._evidence.get(ref_id)
            if r is not None:
                out.append(r)
        return out

    # ------------------------------------------------------------------
    # Name / ID translation (M3-Bench style)
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Rewrite entity IDs (``person_07``) into display names.

        Useful for returning final answers in the benchmark's native
        vocabulary.
        """
        if not text:
            return text
        out = text
        for eid, display in self._id_to_display.items():
            if display and display != eid:
                out = out.replace(eid, display)
        return out

    def back_translate(self, text: str) -> str:
        """Rewrite display names / aliases into entity IDs.

        Applied to queries before retrieval so the embedder sees the
        canonical handles used in the stored node text.
        """
        if not text or not self._alias_to_id:
            return text
        tokens = text.split()
        out = []
        for tok in tokens:
            key = tok.lower().strip(".,!?:;\"'")
            if key in self._alias_to_id:
                out.append(self._alias_to_id[key])
            else:
                out.append(tok)
        return " ".join(out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "video_path": self.video_path,
            "top_k": self.top_k,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "entities": {k: v.to_dict() for k, v in self._entities.items()},
            "evidence": {k: v.to_dict() for k, v in self._evidence.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        embedder: Optional[Any] = None,
    ) -> "SocialVideoGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        g = cls(
            embedder=embedder,
            top_k=int(data.get("top_k", 5)),
            video_path=data.get("video_path"),
        )
        for e in (data.get("evidence", {}) or {}).values():
            g._evidence[e["ref_id"]] = EvidenceRef(**e)
        for prof in (data.get("entities", {}) or {}).values():
            g.register_entity_profile(EntityProfile(**prof))
        for n in data.get("nodes", []) or []:
            g.add_node(GroundingNode.from_dict(n))
        return g

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "retrieval"

    def __len__(self) -> int:
        return len(self._nodes)

    def stats(self) -> Dict[str, int]:
        out = {t: len(ids) for t, ids in self._by_type.items()}
        out["total"] = len(self._nodes)
        out["evidence"] = len(self._evidence)
        out["entities"] = len(self._entities)
        return out


__all__ = ["SocialVideoGraph"]

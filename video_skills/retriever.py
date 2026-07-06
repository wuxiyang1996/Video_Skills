"""Retriever subsystem.

Implements the §2B contract from
``infra_plans/03_controller/actors_reasoning_model.md``:

- query rewriting (``HopGoal`` → list of :class:`RetrievalQuery`),
- entity- / time- / perspective-conditioned retrieval,
- counterevidence retrieval,
- top-k fusion across episodic + semantic + state stores,
- contradiction-aware retrieval,
- retrieval deduplication (collapse near-duplicate refs),
- the §2B.3 broaden ladder.

V1 backend is **deterministic lexical match over in-memory stores**. Every
hit is wrapped in an :class:`EvidenceRef` keyed by the originating record
id so the verifier and audit trail can resolve back. The interface is
designed so the v2 dense / hybrid backend is a drop-in replacement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import (
    EvidenceBundle,
    EvidenceRef,
    HopGoal,
    RetrievalQuery,
    new_id,
)


_BROADEN_LADDER = (
    "relax_entity_filter",
    "widen_time_filter",
    "expand_store_filter",
    "add_counterevidence_pass",
    "fall_back_dense_only",
)


@dataclass
class RetrieverConfig:
    """Knob settings that the controller / loop can tweak per task family."""

    default_k: int = 8
    max_k: int = 32
    time_widen_factor: float = 2.0
    dedup_overlap_seconds: float = 1.0
    semantic_weight: float = 0.6  # weight applied to semantic-store hits
    state_weight: float = 0.7  # weight applied to state-store hits


class Retriever:
    """The single retrieval subsystem all hops route through."""

    def __init__(self, memory: Any, config: Optional[RetrieverConfig] = None) -> None:
        self.memory = memory
        self.config = config or RetrieverConfig()
        # Per-hop broaden state: maps hop_id -> ladder index
        self._broaden_state: Dict[str, int] = {}
        # Audit log of every retrieve call
        self.audit: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API (matches §2B.2)
    # ------------------------------------------------------------------

    def rewrite(self, hop: HopGoal, ctx: Any = None) -> List[RetrievalQuery]:
        """Expand a ``HopGoal`` into one or more ``RetrievalQuery``s."""
        base = RetrievalQuery(
            query_id=new_id("rq"),
            text=hop.goal_text,
            entity_filter=list(hop.required_entities),
            time_filter=hop.required_time_scope,
            perspective=hop.perspective_anchor,
            store_filter="any",
            k=self.config.default_k,
            mode="lexical",
        )
        out: List[RetrievalQuery] = [base]
        # If hop already carries retrieval hints, honor them too
        for hint in hop.retrieval_hints:
            out.append(hint)
        return out

    def retrieve(self, query: RetrievalQuery) -> EvidenceBundle:
        refs: List[EvidenceRef] = []
        if query.store_filter in ("episodic", "any"):
            refs.extend(self._retrieve_episodic(query))
        if query.store_filter in ("semantic", "any"):
            refs.extend(self._retrieve_semantic(query))
        if query.store_filter in ("state", "any"):
            refs.extend(self._retrieve_state(query))
        # Score-fuse, dedup, take top-k
        refs = self._dedup(refs)
        refs.sort(key=lambda r: r.confidence, reverse=True)
        refs = refs[: query.k or self.config.default_k]
        contradictions = self._collect_contradictions(refs)
        coverage = self._compute_coverage(refs, query)
        bundle = EvidenceBundle(
            bundle_id=new_id("eb"),
            refs=refs,
            query=query,
            coverage=coverage,
            contradictions=contradictions,
            sufficiency_hint=self._sufficiency_hint(refs, query),
            confidence=(sum(r.confidence for r in refs) / len(refs)) if refs else 0.0,
            inferred=False,
            meta={"backend": "lexical-v1"},
        )
        self.audit.append({
            "query_id": query.query_id,
            "n_refs": len(refs),
            "store_filter": query.store_filter,
            "broaden_level": 0,
        })
        return bundle

    def retrieve_counter(self, claim: Dict[str, Any], ctx: Any = None) -> EvidenceBundle:
        """Run a paired retrieval for refuting evidence."""
        text = " ".join(str(v) for v in claim.values() if isinstance(v, (str, int, float)))
        # Heuristic negation rewrite
        for word in ("not", "didn't", "no", "missing", "absent"):
            if word in text.lower():
                text = text.replace(word, "")
                break
        else:
            text = "NOT " + text
        query = RetrievalQuery(
            query_id=new_id("rq"),
            text=text,
            store_filter="any",
            k=self.config.default_k,
            mode="lexical",
            meta={"counter": True},
        )
        bundle = self.retrieve(query)
        bundle.meta["counter"] = True
        return bundle

    def fuse(self, bundles: Sequence[EvidenceBundle]) -> EvidenceBundle:
        """Merge multiple bundles into one, deduplicated and re-sorted."""
        if not bundles:
            return EvidenceBundle(
                bundle_id=new_id("eb"),
                refs=[],
                query=RetrievalQuery(query_id=new_id("rq"), text="<fuse-empty>"),
                sufficiency_hint=0.0,
                confidence=0.0,
            )
        refs: List[EvidenceRef] = []
        for b in bundles:
            refs.extend(b.refs)
        refs = self._dedup(refs)
        refs.sort(key=lambda r: r.confidence, reverse=True)
        merged = EvidenceBundle(
            bundle_id=new_id("eb"),
            refs=refs,
            query=bundles[0].query,
            coverage={
                "fused_query_ids": [b.query.query_id for b in bundles],
            },
            contradictions=self._collect_contradictions(refs),
            sufficiency_hint=max(b.sufficiency_hint for b in bundles),
            confidence=(sum(r.confidence for r in refs) / len(refs)) if refs else 0.0,
        )
        return merged

    # ------------------------------------------------------------------
    # Broaden ladder (§2B.3)
    # ------------------------------------------------------------------

    def broaden(self, hop_id: str, base_query: RetrievalQuery) -> EvidenceBundle:
        """Apply the next step of the broaden ladder for a given hop."""
        idx = self._broaden_state.get(hop_id, 0)
        widened = self._apply_broaden(base_query, idx)
        bundle = self.retrieve(widened)
        bundle.meta["broaden_level"] = idx + 1
        bundle.meta["broaden_step"] = _BROADEN_LADDER[idx] if idx < len(_BROADEN_LADDER) else "exhausted"
        self._broaden_state[hop_id] = min(idx + 1, len(_BROADEN_LADDER))
        return bundle

    def reset_broaden(self, hop_id: str) -> None:
        self._broaden_state.pop(hop_id, None)

    def _apply_broaden(self, query: RetrievalQuery, ladder_idx: int) -> RetrievalQuery:
        if ladder_idx >= len(_BROADEN_LADDER):
            return query
        step = _BROADEN_LADDER[ladder_idx]
        widened = RetrievalQuery(
            query_id=new_id("rq"),
            text=query.text,
            entity_filter=list(query.entity_filter),
            time_filter=query.time_filter,
            perspective=query.perspective,
            store_filter=query.store_filter,
            k=query.k,
            mode=query.mode,
            meta={**query.meta, "broaden_from": query.query_id, "broaden_step": step},
        )
        if step == "relax_entity_filter" and widened.entity_filter:
            widened.entity_filter = widened.entity_filter[:1]  # keep only the primary entity
        elif step == "widen_time_filter" and widened.time_filter is not None:
            lo, hi = widened.time_filter
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * self.config.time_widen_factor
            widened.time_filter = (max(0.0, mid - half), mid + half)
        elif step == "expand_store_filter":
            widened.store_filter = "any"
        elif step == "add_counterevidence_pass":
            widened.meta["include_counter"] = True
        elif step == "fall_back_dense_only":
            widened.mode = "dense"
        return widened

    # ------------------------------------------------------------------
    # Per-store retrieval primitives (lexical baseline)
    # ------------------------------------------------------------------

    def _retrieve_episodic(self, query: RetrievalQuery) -> List[EvidenceRef]:
        refs: List[EvidenceRef] = []
        text_terms = _tokenize(query.text)
        ent_filter = set(query.entity_filter)
        for ev in self.memory.episodic.events.values():
            if ent_filter and not (set(ev.participants) & ent_filter):
                continue
            if query.time_filter is not None and ev.time_span is not None:
                lo, hi = query.time_filter
                es, ee = ev.time_span
                if ee < lo or es > hi:
                    continue
            score = _lexical_score(text_terms, ev.description) + _lexical_score(
                text_terms, " ".join(ev.participants)
            )
            if score <= 0:
                continue
            for ref_id in ev.evidence_ref_ids:
                stored = self.memory.evidence.get(ref_id)
                if stored is not None:
                    new_ref = _clone_ref_with_score(stored, score, ev.confidence, query)
                    new_ref.entities = list(set(new_ref.entities) | set(ev.participants))
                    refs.append(new_ref)
            # Always emit a virtual ref pointing back to the episodic record itself.
            refs.append(EvidenceRef(
                ref_id=f"vepi_{ev.event_id}",
                modality="memory_node",
                source_id=ev.event_id,
                time_span=ev.time_span,
                entities=list(ev.participants),
                provenance="observed" if not ev.inferred else "inferred",
                confidence=min(1.0, score / 4.0) * ev.confidence,
                text=ev.description,
                meta={"store": "episodic", "score": score},
            ))
        return refs

    def _retrieve_semantic(self, query: RetrievalQuery) -> List[EvidenceRef]:
        refs: List[EvidenceRef] = []
        text_terms = _tokenize(query.text)
        for s in self.memory.semantic.summaries.values():
            score = _lexical_score(text_terms, s.text)
            if score <= 0:
                continue
            refs.append(EvidenceRef(
                ref_id=f"vsem_{s.summary_id}",
                modality="memory_node",
                source_id=s.summary_id,
                provenance="distilled",
                confidence=min(1.0, score / 4.0) * s.confidence * self.config.semantic_weight,
                text=s.text,
                entities=[s.subject] if s.subject != "global" else [],
                meta={"store": "semantic", "score": score, "version": s.version},
            ))
        return refs

    def _retrieve_state(self, query: RetrievalQuery) -> List[EvidenceRef]:
        refs: List[EvidenceRef] = []
        text_terms = _tokenize(query.text)
        ent_filter = set(query.entity_filter)
        # Beliefs
        for b in self.memory.state.beliefs.values():
            if not b.is_active:
                continue
            if ent_filter and b.holder_entity not in ent_filter:
                continue
            score = _lexical_score(text_terms, b.proposition)
            if score <= 0:
                continue
            refs.append(EvidenceRef(
                ref_id=f"vbst_{b.state_id}",
                modality="state",
                source_id=b.state_id,
                time_span=(b.time_anchor, b.time_anchor) if b.time_anchor is not None else None,
                entities=[b.holder_entity],
                provenance="inferred" if b.inferred else "observed",
                confidence=min(1.0, score / 4.0) * b.confidence * self.config.state_weight,
                text=b.proposition,
                meta={"store": "state.social", "polarity": b.polarity},
            ))
        # Spatial
        for sp in self.memory.state.spatial.values():
            if ent_filter and sp.entity_id not in ent_filter:
                continue
            score = _lexical_score(text_terms, sp.location)
            if score <= 0:
                continue
            refs.append(EvidenceRef(
                ref_id=f"vsp_{sp.state_id}",
                modality="state",
                source_id=sp.state_id,
                time_span=(sp.time_anchor, sp.time_anchor) if sp.time_anchor is not None else None,
                entities=[sp.entity_id],
                provenance="inferred" if sp.inferred else "observed",
                confidence=min(1.0, score / 4.0) * sp.confidence * self.config.state_weight,
                text=sp.location,
                meta={"store": "state.spatial", "visibility": sp.visibility},
            ))
        return refs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dedup(self, refs: List[EvidenceRef]) -> List[EvidenceRef]:
        out: List[EvidenceRef] = []
        seen_ids: Dict[str, EvidenceRef] = {}
        for r in refs:
            key = self._dedup_key(r)
            if key in seen_ids:
                # Keep the higher-confidence one; merge entities
                kept = seen_ids[key]
                if r.confidence > kept.confidence:
                    kept.confidence = r.confidence
                kept.entities = list(set(kept.entities) | set(r.entities))
            else:
                seen_ids[key] = r
                out.append(r)
        return out

    def _dedup_key(self, r: EvidenceRef) -> str:
        if r.source_id:
            return f"src:{r.source_id}"
        if r.time_span is not None:
            lo, hi = r.time_span
            tol = self.config.dedup_overlap_seconds
            return f"ts:{r.modality}:{round(lo / tol)}:{round(hi / tol)}"
        return f"id:{r.ref_id}"

    def _collect_contradictions(self, refs: List[EvidenceRef]) -> List[EvidenceRef]:
        ref_source_ids = {r.source_id for r in refs if r.source_id is not None}
        out: List[EvidenceRef] = []
        for a, b, reason in self.memory.contradicts:
            if a in ref_source_ids:
                # The other side is the contradiction
                stored = self.memory.evidence.get(b)
                if stored is not None:
                    out.append(stored)
                else:
                    out.append(EvidenceRef(
                        ref_id=f"vc_{b}", modality="memory_node",
                        source_id=b, provenance="observed", confidence=0.5,
                        meta={"contradicts": a, "reason": reason},
                    ))
            elif b in ref_source_ids:
                stored = self.memory.evidence.get(a)
                if stored is not None:
                    out.append(stored)
                else:
                    out.append(EvidenceRef(
                        ref_id=f"vc_{a}", modality="memory_node",
                        source_id=a, provenance="observed", confidence=0.5,
                        meta={"contradicts": b, "reason": reason},
                    ))
        return out

    def _compute_coverage(self, refs: List[EvidenceRef], query: RetrievalQuery) -> Dict[str, Any]:
        ents = sorted({e for r in refs for e in r.entities})
        time_spans = [r.time_span for r in refs if r.time_span is not None]
        return {
            "entities": ents,
            "n_with_time": len(time_spans),
            "store_split": _store_split(refs),
            "perspective": query.perspective,
        }

    def _sufficiency_hint(self, refs: List[EvidenceRef], query: RetrievalQuery) -> float:
        if not refs:
            return 0.0
        # Coverage of required entities is the main sufficiency signal.
        if query.entity_filter:
            covered = {e for r in refs for e in r.entities} & set(query.entity_filter)
            return min(1.0, len(covered) / len(query.entity_filter))
        return min(1.0, len(refs) / max(query.k, 1))


# ---------------------------------------------------------------------------
# Tokenization / scoring
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return [t for t in out if t]


def _lexical_score(terms: Iterable[str], doc: str) -> float:
    if not doc:
        return 0.0
    doc_lc = doc.lower()
    return float(sum(1 for t in terms if t and t in doc_lc))


def _store_split(refs: List[EvidenceRef]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in refs:
        store = r.meta.get("store") if isinstance(r.meta, dict) else None
        if store is None:
            store = r.modality
        out[store] = out.get(store, 0) + 1
    return out


def _clone_ref_with_score(
    src: EvidenceRef,
    score: float,
    base_confidence: float,
    query: RetrievalQuery,
) -> EvidenceRef:
    bumped = max(0.0, min(1.0, base_confidence * (0.5 + min(score / 4.0, 0.5))))
    return EvidenceRef(
        ref_id=src.ref_id,
        modality=src.modality,
        source_id=src.source_id,
        time_span=src.time_span,
        entities=list(src.entities),
        provenance=src.provenance,
        confidence=max(src.confidence, bumped),
        text=src.text,
        locator=src.locator,
        meta={**src.meta, "retrieval_score": score, "retrieved_for": query.query_id},
    )

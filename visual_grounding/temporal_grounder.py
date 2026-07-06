"""Temporal grounding (typed).

Builds :class:`TemporalRelation` rows over a sequence of typed events:

* ``before`` / ``after``  — time-ordered pairs.
* ``overlap``             — intervals that intersect.
* ``causes`` / ``enables`` — only added when the heuristic is
                              reasonably confident (and explicitly
                              tagged ``inferred``).

Per the visual-grounding plan §7: causal candidates are added only when
grounded enough — by default we *don't* propose ``causes`` edges; users
who want them can opt-in with ``allow_causal=True``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from visual_grounding.grounding_schemas import (
    EventSpan,
    TemporalRelation,
    new_grounding_id,
)


class TemporalGrounder:
    """Construct ordered + overlap (+ optional causal) temporal links."""

    def __init__(
        self,
        *,
        allow_causal: bool = False,
        max_pairs: int = 4096,
        causal_max_gap_seconds: float = 5.0,
    ) -> None:
        self.allow_causal = allow_causal
        self.max_pairs = max_pairs
        self.causal_max_gap_seconds = causal_max_gap_seconds

    def build_relations(
        self, events: Sequence[EventSpan],
    ) -> List[TemporalRelation]:
        if not events:
            return []
        ordered = sorted(
            [e for e in events if e.start_time is not None],
            key=lambda e: (e.start_time or 0.0, e.event_type),
        )
        out: List[TemporalRelation] = []
        for i, ei in enumerate(ordered):
            si = ei.start_time or 0.0
            eii = ei.end_time if ei.end_time is not None else si
            for ej in ordered[i + 1:]:
                if len(out) >= self.max_pairs:
                    return out
                sj = ej.start_time or 0.0
                ejj = ej.end_time if ej.end_time is not None else sj
                if ejj <= si or sj >= eii:  # disjoint or touching
                    out.append(TemporalRelation(
                        relation_id=new_grounding_id("trel"),
                        lhs_event_id=ei.event_id,
                        rhs_event_id=ej.event_id,
                        relation_type="before",
                        evidence_refs=list(ei.evidence_refs)
                        + list(ej.evidence_refs),
                        confidence=0.95,
                        source_type="observed",
                    ))
                else:
                    out.append(TemporalRelation(
                        relation_id=new_grounding_id("trel"),
                        lhs_event_id=ei.event_id,
                        rhs_event_id=ej.event_id,
                        relation_type="overlap",
                        evidence_refs=list(ei.evidence_refs)
                        + list(ej.evidence_refs),
                        confidence=0.85,
                        source_type="observed",
                    ))

                if self.allow_causal:
                    gap = sj - eii
                    if 0 <= gap <= self.causal_max_gap_seconds and (
                        set(ei.participants) & set(ej.participants)
                    ):
                        out.append(TemporalRelation(
                            relation_id=new_grounding_id("trel"),
                            lhs_event_id=ei.event_id,
                            rhs_event_id=ej.event_id,
                            relation_type="enables",
                            evidence_refs=list(ei.evidence_refs)
                            + list(ej.evidence_refs),
                            confidence=0.4,
                            source_type="inferred",
                        ))
        return out


__all__ = ["TemporalGrounder"]

"""Memory Procedure Registry.

The fixed v1 catalog of memory-management operations
(``infra_plans/02_memory/agentic_memory_design.md`` §0.2 + lifecycle table).

Properties enforced here:

- **Stable.** Same input → same effect across calls.
- **Manually versioned.** No trace-driven promotion. To change a procedure's
  semantics, edit this module and bump :data:`PROCEDURE_REGISTRY_VERSION`.
- **Closed catalog.** :data:`PROCEDURE_NAMES` is the only allowed write
  surface for memory.
- **Audit trail.** Every invocation appends to
  :class:`MemoryProcedureRegistry.audit_log`.

Reasoning skills **never write memory directly** — they request a procedure
via the harness, which calls the registry. That keeps the write path
auditable and prevents bank churn from corrupting the substrate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..contracts import EvidenceRef, GroundedWindow, new_id, now_ts
from .stores import (
    BeliefState,
    EntityProfile,
    EpisodicEvent,
    EpisodicThread,
    Memory,
    SemanticSummary,
    SpatialState,
)


PROCEDURE_REGISTRY_VERSION = "v1.0"

PROCEDURE_NAMES = (
    "open_episode_thread",
    "append_grounded_event",
    "update_entity_profile",
    "refresh_state_memory",
    "compress_episode_cluster",
    "attach_evidence_ref",
    "resolve_entity_alias",
    "revise_belief_state",
    "mark_memory_conflict",
)


# Default lifecycle thresholds (§Lifecycle implementation table). These are
# fixed-by-hand for v1 and are not auto-tuned.
DEFAULT_TAU_GROUNDING = 0.5
DEFAULT_SEMANTIC_CLUSTER_MIN = 3
DEFAULT_BELIEF_DECAY_HALFLIFE_S = 600.0


@dataclass
class ProcedureCallRecord:
    """One audit-log entry per procedure invocation."""

    call_id: str
    procedure: str
    args: Dict[str, Any]
    returned_ids: List[str]
    timestamp: float
    caller: Optional[str] = None  # skill_id or controller stage that requested


class MemoryProcedureRegistry:
    """Registry of the nine fixed memory-management procedures.

    Construct with a :class:`Memory` instance; the registry then exposes
    each procedure as a method (and via :meth:`call`).
    """

    version = PROCEDURE_REGISTRY_VERSION

    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.audit_log: List[ProcedureCallRecord] = []
        self._dispatch: Dict[str, Callable[..., Any]] = {
            "open_episode_thread": self.open_episode_thread,
            "append_grounded_event": self.append_grounded_event,
            "update_entity_profile": self.update_entity_profile,
            "refresh_state_memory": self.refresh_state_memory,
            "compress_episode_cluster": self.compress_episode_cluster,
            "attach_evidence_ref": self.attach_evidence_ref,
            "resolve_entity_alias": self.resolve_entity_alias,
            "revise_belief_state": self.revise_belief_state,
            "mark_memory_conflict": self.mark_memory_conflict,
        }

    # ------------------------------------------------------------------
    # Dispatch / audit
    # ------------------------------------------------------------------

    def call(self, procedure: str, *, caller: Optional[str] = None, **kwargs: Any) -> Any:
        """Invoke ``procedure`` by name; logs to audit_log unconditionally."""
        if procedure not in self._dispatch:
            raise KeyError(
                f"unknown memory procedure {procedure!r}. "
                f"Allowed: {PROCEDURE_NAMES}"
            )
        result = self._dispatch[procedure](**kwargs)
        returned_ids = self._extract_ids(result)
        self.audit_log.append(
            ProcedureCallRecord(
                call_id=new_id("memcall"),
                procedure=procedure,
                args={k: _summarize(v) for k, v in kwargs.items()},
                returned_ids=returned_ids,
                timestamp=now_ts(),
                caller=caller,
            )
        )
        return result

    @staticmethod
    def _extract_ids(result: Any) -> List[str]:
        if isinstance(result, str):
            return [result]
        if isinstance(result, (list, tuple)):
            return [r for r in result if isinstance(r, str)]
        for attr in ("event_id", "thread_id", "summary_id", "state_id", "entity_id", "ref_id"):
            v = getattr(result, attr, None)
            if isinstance(v, str):
                return [v]
        return []

    # ------------------------------------------------------------------
    # 1. open_episode_thread
    # ------------------------------------------------------------------

    def open_episode_thread(
        self,
        *,
        clip_id: str,
        time_span: Optional[Tuple[float, float]] = None,
        thread_id: Optional[str] = None,
    ) -> EpisodicThread:
        """Create a new episodic thread for a clip / window.

        Idempotency: if a thread already exists for ``clip_id`` it is returned
        unchanged.
        """
        for t in self.memory.episodic.threads.values():
            if t.clip_id == clip_id:
                return t
        thread = EpisodicThread(
            thread_id=thread_id or new_id("ethr"),
            clip_id=clip_id,
            time_span=time_span,
        )
        self.memory.episodic.add_thread(thread)
        return thread

    # ------------------------------------------------------------------
    # 2. append_grounded_event
    # ------------------------------------------------------------------

    def append_grounded_event(
        self,
        *,
        window: GroundedWindow,
        tau_grounding: float = DEFAULT_TAU_GROUNDING,
    ) -> List[EpisodicEvent]:
        """Append every event of a ``GroundedWindow`` to episodic memory.

        Skips windows whose grounding confidence is below ``tau_grounding``
        (per the lifecycle write trigger).
        """
        if window.confidence < tau_grounding:
            return []
        if not window.events:
            return []
        thread = self.open_episode_thread(
            clip_id=window.clip_id,
            time_span=window.time_span,
        )
        # Attach window evidence (keyframes + provenance) once per call.
        ev_refs: List[str] = []
        for kf in window.keyframes:
            ref = EvidenceRef(
                ref_id=new_id("ev"),
                modality="frame",
                source_id=window.window_id,
                time_span=(kf.timestamp, kf.timestamp),
                provenance="observed",
                confidence=window.confidence,
                locator=kf.locator,
                meta={"frame_id": kf.frame_id, **window.provenance},
            )
            self.memory.evidence.add(ref)
            ev_refs.append(ref.ref_id)

        appended: List[EpisodicEvent] = []
        for ev in window.events:
            # collision check — same (entities, time_span) flips into a contradicts edge
            collisions: List[str] = []
            if ev.time_span is not None:
                for prior in self.memory.episodic.events_in_time(ev.time_span):
                    if set(prior.participants) == set(ev.participants) and prior.event_type != ev.event_type:
                        collisions.append(prior.event_id)

            event_id = new_id("eepi")
            # Per-event anchor ref so downstream verifiers can align via
            # source_id / entities (the keyframe-level refs only cover frames).
            anchor = EvidenceRef(
                ref_id=new_id("ev"),
                modality="frame",
                source_id=event_id,
                time_span=ev.time_span,
                entities=list(ev.participants),
                provenance="observed" if not window.inferred else "inferred",
                text=ev.description,
                confidence=ev.confidence,
                meta={"event_type": ev.event_type, "window_id": window.window_id},
            )
            self.memory.evidence.add(anchor)

            event = EpisodicEvent(
                event_id=event_id,
                thread_id=thread.thread_id,
                clip_id=window.clip_id,
                window_id=window.window_id,
                event_type=ev.event_type,
                description=ev.description,
                participants=list(ev.participants),
                time_span=ev.time_span,
                evidence_ref_ids=[anchor.ref_id, *ev_refs],
                contradicts=collisions,
                confidence=ev.confidence,
                inferred=window.inferred,
            )
            self.memory.episodic.add_event(event)
            for cid in collisions:
                self.memory.contradicts.append(
                    (event.event_id, cid, "same_entities_different_event_type")
                )
            appended.append(event)
        return appended

    # ------------------------------------------------------------------
    # 3. update_entity_profile
    # ------------------------------------------------------------------

    def update_entity_profile(
        self,
        *,
        entity_id: str,
        canonical_name: Optional[str] = None,
        alias: Optional[str] = None,
        appearance_evidence_id: Optional[str] = None,
        voice_evidence_id: Optional[str] = None,
        seen_at: Optional[float] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> EntityProfile:
        """Apply a detection / equivalence update to an ``EntityProfile``.

        Creates the profile on first detection. Aliases append-only.
        """
        profile = self.memory.entities.profiles.get(entity_id)
        if profile is None:
            profile = EntityProfile(
                entity_id=entity_id,
                canonical_name=canonical_name,
                first_seen=seen_at,
                last_seen=seen_at,
            )
            self.memory.entities.register(profile)
        else:
            if canonical_name and not profile.canonical_name:
                profile.canonical_name = canonical_name

        if alias and alias not in profile.aliases and alias not in profile.aliases_pending:
            profile.aliases_pending.append(alias)
        if appearance_evidence_id and appearance_evidence_id not in profile.appearance_evidence_ids:
            profile.appearance_evidence_ids.append(appearance_evidence_id)
        if voice_evidence_id and voice_evidence_id not in profile.voice_evidence_ids:
            profile.voice_evidence_ids.append(voice_evidence_id)
        if seen_at is not None:
            profile.first_seen = seen_at if profile.first_seen is None else min(profile.first_seen, seen_at)
            profile.last_seen = seen_at if profile.last_seen is None else max(profile.last_seen, seen_at)
        if attributes:
            profile.attributes.update(attributes)
        return profile

    # ------------------------------------------------------------------
    # 4. refresh_state_memory
    # ------------------------------------------------------------------

    def refresh_state_memory(
        self,
        *,
        time_anchor: float,
        entities: Optional[List[str]] = None,
        decay_halflife_s: float = DEFAULT_BELIEF_DECAY_HALFLIFE_S,
    ) -> Dict[str, int]:
        """Recompute social / spatial state for a query at time ``t``.

        Decays old beliefs via half-life, deactivates beliefs whose decayed
        confidence falls below 0.1, and returns a stats dict of how many
        rows were touched.
        """
        import math

        decayed = 0
        deactivated = 0
        target_holders = set(entities) if entities else None
        for b in self.memory.state.beliefs.values():
            if not b.is_active:
                continue
            if target_holders is not None and b.holder_entity not in target_holders:
                continue
            if b.time_anchor is None:
                continue
            age = max(0.0, time_anchor - b.time_anchor)
            factor = math.pow(0.5, age / max(decay_halflife_s, 1e-6))
            new_conf = b.confidence * factor
            if new_conf < 0.1:
                b.is_active = False
                deactivated += 1
            else:
                b.confidence = new_conf
                decayed += 1
        return {"decayed": decayed, "deactivated": deactivated}

    # ------------------------------------------------------------------
    # 5. compress_episode_cluster
    # ------------------------------------------------------------------

    def compress_episode_cluster(
        self,
        *,
        thread_id: str,
        subject: str,
        text: Optional[str] = None,
        min_cluster: int = DEFAULT_SEMANTIC_CLUSTER_MIN,
    ) -> Optional[SemanticSummary]:
        """Produce a per-episode summary, retaining ``source_ids``.

        Only runs if the thread has at least ``min_cluster`` events. The
        summary is written to ``SemanticStore``; the thread's
        ``summary_id`` is set so subsequent compressions roll up versions.
        """
        thread = self.memory.episodic.threads.get(thread_id)
        if thread is None or len(thread.event_ids) < min_cluster:
            return None
        events = [self.memory.episodic.events[eid] for eid in thread.event_ids]
        # Default v1 summarizer is deterministic concatenation. The plan
        # explicitly allows a per-release upgrade to a real summarizer.
        if text is None:
            text = (
                f"{len(events)} events for clip {thread.clip_id}: "
                + "; ".join(e.description for e in events)
            )
        prior = (
            self.memory.semantic.summaries.get(thread.summary_id)
            if thread.summary_id
            else None
        )
        summary = SemanticSummary(
            summary_id=new_id("esem"),
            subject=subject,
            text=text,
            source_episode_ids=[e.event_id for e in events],
            version=(prior.version + 1) if prior else 1,
            parent_version_id=thread.summary_id,
            confidence=min((e.confidence for e in events), default=1.0),
            inferred=True,
        )
        self.memory.semantic.add(summary)
        if thread.summary_id is not None:
            self.memory.semantic.archive(thread.summary_id)
        thread.summary_id = summary.summary_id
        return summary

    # ------------------------------------------------------------------
    # 6. attach_evidence_ref
    # ------------------------------------------------------------------

    def attach_evidence_ref(
        self,
        *,
        record_id: str,
        evidence: EvidenceRef,
    ) -> EvidenceRef:
        """Bind an ``EvidenceRef`` to an episodic / state record.

        ``record_id`` may be an episodic event_id, a belief state_id, or a
        spatial state_id. The ref is registered in the evidence store and
        appended to the record's ``evidence_ref_ids`` list.
        """
        self.memory.evidence.add(evidence)
        if record_id in self.memory.episodic.events:
            ep = self.memory.episodic.events[record_id]
            if evidence.ref_id not in ep.evidence_ref_ids:
                ep.evidence_ref_ids.append(evidence.ref_id)
        elif record_id in self.memory.state.beliefs:
            b = self.memory.state.beliefs[record_id]
            if evidence.ref_id not in b.evidence_ref_ids:
                b.evidence_ref_ids.append(evidence.ref_id)
        elif record_id in self.memory.state.spatial:
            sp = self.memory.state.spatial[record_id]
            if evidence.ref_id not in sp.evidence_ref_ids:
                sp.evidence_ref_ids.append(evidence.ref_id)
        else:
            raise KeyError(f"no record found for record_id={record_id!r}")
        return evidence

    # ------------------------------------------------------------------
    # 7. resolve_entity_alias
    # ------------------------------------------------------------------

    def resolve_entity_alias(
        self,
        *,
        entity_id: str,
        alias: str,
        merge_with: Optional[str] = None,
    ) -> EntityProfile:
        """Move an alias from ``aliases_pending`` to bound ``aliases``.

        If ``merge_with`` is supplied, also union-finds the two profiles.
        """
        profile = self.memory.entities.profiles.get(entity_id)
        if profile is None:
            raise KeyError(f"unknown entity {entity_id!r}")
        if alias in profile.aliases_pending:
            profile.aliases_pending.remove(alias)
        if alias not in profile.aliases:
            profile.aliases.append(alias)
        if merge_with is not None:
            self.memory.entities.merge(entity_id, merge_with)
        return profile

    # ------------------------------------------------------------------
    # 8. revise_belief_state
    # ------------------------------------------------------------------

    def revise_belief_state(
        self,
        *,
        holder_entity: str,
        proposition: str,
        polarity: str = "true",
        confidence: float = 1.0,
        time_anchor: Optional[float] = None,
        evidence_ref_ids: Optional[List[str]] = None,
        source_step_id: Optional[str] = None,
        supersedes: Optional[str] = None,
    ) -> BeliefState:
        """Apply a verifier-passed belief update with ``supersedes`` linkage."""
        belief = BeliefState(
            state_id=new_id("bst"),
            holder_entity=holder_entity,
            proposition=proposition,
            polarity=polarity,
            time_anchor=time_anchor,
            evidence_ref_ids=list(evidence_ref_ids or []),
            source_step_id=source_step_id,
            supersedes=supersedes,
            confidence=confidence,
            inferred=True,
        )
        self.memory.state.add_belief(belief)
        return belief

    # ------------------------------------------------------------------
    # 9. mark_memory_conflict
    # ------------------------------------------------------------------

    def mark_memory_conflict(
        self,
        *,
        record_id_a: str,
        record_id_b: str,
        reason: str,
    ) -> Tuple[str, str, str]:
        """Record a ``contradicts`` edge between two memory records."""
        edge = (record_id_a, record_id_b, reason)
        if edge not in self.memory.contradicts:
            self.memory.contradicts.append(edge)
        return edge


def _summarize(v: Any) -> Any:
    """Make a JSON-friendly summary of an audit arg without dumping payloads."""
    if isinstance(v, GroundedWindow):
        return {
            "type": "GroundedWindow",
            "window_id": v.window_id,
            "clip_id": v.clip_id,
            "n_events": len(v.events),
            "n_keyframes": len(v.keyframes),
        }
    if isinstance(v, EvidenceRef):
        return {"type": "EvidenceRef", "ref_id": v.ref_id, "modality": v.modality}
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (str, int, float, bool)):
        return list(v)
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return f"<{type(v).__name__}>"

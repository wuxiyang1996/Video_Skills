"""Curated v1 starter atomic skill inventory.

Implements the 12 atomics named in
``infra_plans/05_skills/skill_extraction_bank.md`` §4.8:

1.  identify_question_target
2.  decompose_into_subgoals
3.  retrieve_relevant_episode
4.  ground_entity_reference
5.  ground_event_span
6.  infer_observation_access
7.  order_two_events
8.  check_state_change
9.  check_causal_support
10. update_belief_state
11. check_evidence_sufficiency
12. decide_answer_or_abstain

Plus the second-pass additions also called out by §4.8:
13. check_alternative_hypothesis
14. locate_counterevidence

Each skill is implemented as a pure Python function that produces a
:class:`SkillOutput`. Skills do not write memory directly; if they need to
persist a belief or attach evidence, they request a memory procedure via
``SkillOutput.requested_memory_writes`` (the harness then routes the call
through the :class:`MemoryProcedureRegistry`).

These v1 implementations are **deterministic, evidence-honest, and
intentionally conservative**. They are designed so that:

- the harness produces valid ``ReasoningTrace`` objects end-to-end,
- the verifier's six named checks have something concrete to operate on,
- the controller's rule-based planner has correct routing,

so that later phases can swap any single skill for an LLM-call without
changing the contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..contracts import (
    EvidenceBundle,
    EvidenceRef,
    HopGoal,
    RetrievalQuery,
    TriggerSpec,
    VerificationCheckSpec,
    new_id,
)
from .bank import (
    AtomicSkill,
    ReasoningSkillBank,
    SkillContext,
    SkillOutput,
    SkillRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_evidence_confidence(bundle: Optional[EvidenceBundle]) -> float:
    if bundle is None or bundle.is_empty():
        return 0.0
    return sum(r.confidence for r in bundle.refs) / len(bundle.refs)


def _entity_canonicalize(memory: Any, mention: str) -> Optional[str]:
    """Try to resolve a free-text entity mention to a canonical entity_id."""
    return memory.entities.resolve(mention)


# ---------------------------------------------------------------------------
# 1. identify_question_target
# ---------------------------------------------------------------------------


def _exec_identify_question_target(ctx: SkillContext) -> SkillOutput:
    q = ctx.inputs.get("question_text") or ctx.hop_goal.goal_text
    target = (ctx.inputs.get("target_entities") or ctx.hop_goal.required_entities or [])
    return SkillOutput(
        output={
            "question_text": q,
            "target_entities": list(target),
            "perspective_anchor": ctx.hop_goal.perspective_anchor,
        },
        output_type="meta",
        confidence=1.0,
        inferred=False,
    )


# ---------------------------------------------------------------------------
# 2. decompose_into_subgoals
# ---------------------------------------------------------------------------


def _exec_decompose_into_subgoals(ctx: SkillContext) -> SkillOutput:
    """Heuristic decomposition by claim type. Real version: LLM-mediated."""
    target_type = ctx.hop_goal.target_claim_type
    targets = ctx.hop_goal.required_entities
    subgoals: List[str] = []
    if target_type == "ordering" and len(targets) >= 2:
        subgoals = [
            f"ground event reference for {targets[0]}",
            f"ground event reference for {targets[1]}",
            f"compare time spans of {targets[0]} and {targets[1]}",
        ]
    elif target_type == "belief":
        holder = ctx.hop_goal.perspective_anchor or (targets[0] if targets else "agent")
        subgoals = [
            f"ground the proposition referenced in the goal",
            f"determine whether {holder} had observation access",
            f"update {holder}'s belief state",
        ]
    elif target_type == "presence":
        if targets:
            subgoals = [f"check whether {t} is in the time window" for t in targets]
    elif target_type == "causal":
        subgoals = [
            "ground the candidate cause event",
            "ground the candidate effect event",
            "check that cause precedes effect with non-trivial dependency",
        ]
    else:
        subgoals = [ctx.hop_goal.goal_text]
    return SkillOutput(
        output={"subgoals": subgoals},
        output_type="meta",
        confidence=1.0,
        inferred=True,
    )


# ---------------------------------------------------------------------------
# 3. retrieve_relevant_episode
# ---------------------------------------------------------------------------


def _exec_retrieve_relevant_episode(ctx: SkillContext) -> SkillOutput:
    query = RetrievalQuery(
        query_id=new_id("rq"),
        text=ctx.hop_goal.goal_text,
        entity_filter=list(ctx.hop_goal.required_entities),
        time_filter=ctx.hop_goal.required_time_scope,
        perspective=ctx.hop_goal.perspective_anchor,
        store_filter="episodic",
        k=8,
        mode="lexical",
    )
    bundle = ctx.retriever.retrieve(query)
    return SkillOutput(
        output={"episode_refs": [r.ref_id for r in bundle.refs]},
        output_type="evidence_set",
        confidence=bundle.confidence,
        inferred=False,
        used_evidence=bundle,
        failure_mode=None if not bundle.is_empty() else "empty_evidence",
    )


# ---------------------------------------------------------------------------
# 4. ground_entity_reference
# ---------------------------------------------------------------------------


def _exec_ground_entity_reference(ctx: SkillContext) -> SkillOutput:
    mention = (
        ctx.inputs.get("mention")
        or (ctx.hop_goal.required_entities[0] if ctx.hop_goal.required_entities else "")
    )
    canonical = _entity_canonicalize(ctx.memory, mention)
    if canonical is None:
        # No matching profile — request creation as a pending alias for the
        # next pass; the v1 path leaves resolution to the harness/operator.
        return SkillOutput(
            output={"mention": mention, "entity_id": None},
            output_type="entity_ref",
            confidence=0.0,
            inferred=True,
            failure_mode="missing_input",
        )
    profile = ctx.memory.entities.profiles[canonical]
    refs = ctx.memory.evidence.for_entity(canonical)
    bundle = EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs[:8],
        query=RetrievalQuery(
            query_id=new_id("rq"), text=f"entity:{mention}", entity_filter=[canonical],
            store_filter="any", k=8, mode="lexical",
        ),
        sufficiency_hint=1.0 if refs else 0.0,
        confidence=1.0 if refs else 0.5,
    )
    return SkillOutput(
        output={
            "mention": mention,
            "entity_id": canonical,
            "canonical_name": profile.canonical_name,
        },
        output_type="entity_ref",
        confidence=1.0,
        inferred=False,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 5. ground_event_span
# ---------------------------------------------------------------------------


def _exec_ground_event_span(ctx: SkillContext) -> SkillOutput:
    """Return time span(s) of episodic events matching the goal description.

    Match heuristic (v1): an event is a candidate if either
    (a) any non-stopword token from the description appears in the event's
        description / participant list, OR
    (b) the event's participants overlap the hop's ``required_entities``.

    The richer matcher avoids brittleness from verbose goal_text strings.
    """
    description = ctx.inputs.get("description") or ctx.hop_goal.goal_text
    stopwords = {
        "for", "the", "of", "and", "to", "in", "on", "at",
        "ground", "event", "span", "determine", "ordering",
    }
    tokens = [
        t for t in description.lower().replace(",", " ").split()
        if t and t not in stopwords and len(t) > 1
    ]
    required = {e.lower() for e in ctx.hop_goal.required_entities}
    candidates = []
    refs: List[EvidenceRef] = []
    for ev in ctx.memory.episodic.events.values():
        desc_lc = ev.description.lower()
        parts_lc = [p.lower() for p in ev.participants]
        token_hit = any(t in desc_lc for t in tokens) or any(
            t in p for t in tokens for p in parts_lc
        )
        entity_hit = bool(required & set(parts_lc))
        if token_hit or entity_hit:
            candidates.append(ev)
            for ref_id in ev.evidence_ref_ids:
                ref = ctx.memory.evidence.get(ref_id)
                if ref is not None:
                    refs.append(ref)
    if not candidates:
        return SkillOutput(
            output={"description": description, "event_ids": [], "time_spans": []},
            output_type="span",
            confidence=0.0,
            inferred=True,
            failure_mode="empty_evidence",
        )
    bundle = EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs[:16],
        query=RetrievalQuery(
            query_id=new_id("rq"), text=description, store_filter="episodic",
            k=16, mode="lexical",
        ),
        sufficiency_hint=min(1.0, len(refs) / 4.0),
        confidence=min(1.0, sum(c.confidence for c in candidates) / len(candidates)),
    )
    return SkillOutput(
        output={
            "description": description,
            "event_ids": [e.event_id for e in candidates],
            "time_spans": [e.time_span for e in candidates if e.time_span],
        },
        output_type="span",
        confidence=bundle.confidence,
        inferred=False,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 6. infer_observation_access
# ---------------------------------------------------------------------------


def _exec_infer_observation_access(ctx: SkillContext) -> SkillOutput:
    """Decide whether a holder agent had perceptual access to an event.

    V1 logic: the holder was a participant in the event, OR the holder's
    spatial visibility overlapped the event's time window. Real version:
    72B grounding tool call.
    """
    holder = ctx.inputs.get("holder") or ctx.hop_goal.perspective_anchor
    event_ids = ctx.inputs.get("event_ids") or []
    if holder is None or not event_ids:
        return SkillOutput(
            output={"holder": holder, "event_ids": event_ids, "access": False},
            output_type="claim",
            confidence=0.0,
            inferred=True,
            failure_mode="missing_input",
        )
    holder = ctx.memory.entities.resolve(holder) or holder
    refs: List[EvidenceRef] = []
    access = False
    for eid in event_ids:
        ev = ctx.memory.episodic.events.get(eid)
        if ev is None:
            continue
        if holder in ev.participants:
            access = True
            for ref_id in ev.evidence_ref_ids:
                r = ctx.memory.evidence.get(ref_id)
                if r is not None:
                    refs.append(r)
        elif ev.time_span is not None:
            for sp in ctx.memory.state.spatial_for(holder):
                if sp.visibility == "visible" and sp.time_anchor is not None:
                    es, ee = ev.time_span
                    if es <= sp.time_anchor <= ee:
                        access = True
                        for ref_id in sp.evidence_ref_ids:
                            r = ctx.memory.evidence.get(ref_id)
                            if r is not None:
                                refs.append(r)
    bundle = EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs[:8],
        query=RetrievalQuery(
            query_id=new_id("rq"),
            text=f"observation_access(holder={holder})",
            entity_filter=[holder],
            store_filter="any",
            mode="lexical",
        ),
        sufficiency_hint=1.0 if refs else 0.0,
        confidence=1.0 if access and refs else (0.6 if access else 0.4),
    )
    return SkillOutput(
        output={"holder": holder, "event_ids": event_ids, "access": access},
        output_type="claim",
        confidence=bundle.confidence,
        inferred=True,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 7. order_two_events
# ---------------------------------------------------------------------------


def _exec_order_two_events(ctx: SkillContext) -> SkillOutput:
    a_id = ctx.inputs.get("event_a_id")
    b_id = ctx.inputs.get("event_b_id")
    if a_id is None or b_id is None:
        # Fall back to the first two events from a prior ground_event_span step.
        spans: List[str] = []
        for prior in reversed(ctx.trace_so_far):
            if prior.output_type == "span" and prior.output.get("event_ids"):
                spans.extend(prior.output["event_ids"])
                if len(spans) >= 2:
                    break
        if len(spans) >= 2:
            a_id, b_id = spans[0], spans[1]
    if a_id is None or b_id is None:
        return SkillOutput(
            output={"order": None}, output_type="ordering",
            confidence=0.0, inferred=True, failure_mode="missing_input",
        )
    a = ctx.memory.episodic.events.get(a_id)
    b = ctx.memory.episodic.events.get(b_id)
    if a is None or b is None or a.time_span is None or b.time_span is None:
        return SkillOutput(
            output={"order": None, "a": a_id, "b": b_id},
            output_type="ordering",
            confidence=0.0, inferred=True, failure_mode="missing_input",
        )
    ae, be = a.time_span[1], b.time_span[1]
    if ae < b.time_span[0]:
        order = "a_before_b"
    elif be < a.time_span[0]:
        order = "b_before_a"
    else:
        order = "overlapping"
    refs = [
        ctx.memory.evidence.get(r) for r in (a.evidence_ref_ids + b.evidence_ref_ids)
    ]
    refs = [r for r in refs if r is not None]
    bundle = EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs[:8],
        query=RetrievalQuery(
            query_id=new_id("rq"), text=f"order({a_id},{b_id})",
            store_filter="episodic", mode="lexical",
        ),
        sufficiency_hint=1.0 if refs else 0.0,
        confidence=min(a.confidence, b.confidence),
    )
    return SkillOutput(
        output={"a": a_id, "b": b_id, "order": order},
        output_type="ordering",
        confidence=bundle.confidence,
        inferred=False,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 8. check_state_change
# ---------------------------------------------------------------------------


def _exec_check_state_change(ctx: SkillContext) -> SkillOutput:
    """Detect whether a predicate flipped between t0 and t1.

    Compares belief states (or spatial states) across the time window.
    """
    holder = ctx.inputs.get("holder") or ctx.hop_goal.perspective_anchor
    proposition = ctx.inputs.get("proposition") or ctx.hop_goal.goal_text
    if holder is None:
        return SkillOutput(
            output={"changed": None}, output_type="claim",
            confidence=0.0, inferred=True, failure_mode="missing_input",
        )
    matches = [
        b
        for b in ctx.memory.state.beliefs_for(holder, active_only=False)
        if proposition.lower() in b.proposition.lower()
    ]
    matches.sort(key=lambda b: b.time_anchor or 0.0)
    if len(matches) < 2:
        return SkillOutput(
            output={"changed": False, "n_states": len(matches)},
            output_type="claim",
            confidence=0.5,
            inferred=True,
        )
    changed = any(matches[i].polarity != matches[i + 1].polarity for i in range(len(matches) - 1))
    refs: List[EvidenceRef] = []
    for m in matches:
        for ref_id in m.evidence_ref_ids:
            r = ctx.memory.evidence.get(ref_id)
            if r is not None:
                refs.append(r)
    bundle = EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs[:8],
        query=RetrievalQuery(
            query_id=new_id("rq"), text=f"state_change({holder},{proposition})",
            entity_filter=[holder], store_filter="state", mode="lexical",
        ),
        sufficiency_hint=1.0 if refs else 0.0,
        confidence=0.7 if refs else 0.4,
    )
    return SkillOutput(
        output={"changed": changed, "holder": holder, "proposition": proposition},
        output_type="claim",
        confidence=bundle.confidence,
        inferred=True,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 9. check_causal_support
# ---------------------------------------------------------------------------


def _exec_check_causal_support(ctx: SkillContext) -> SkillOutput:
    """Score evidence supporting one cause→effect link.

    V1: support is high if cause precedes effect AND both share at least one
    participant. Real version: 72B reasoner.
    """
    cause_id = ctx.inputs.get("cause_id")
    effect_id = ctx.inputs.get("effect_id")
    if cause_id is None or effect_id is None:
        return SkillOutput(
            output={"supported": None}, output_type="claim",
            confidence=0.0, inferred=True, failure_mode="missing_input",
        )
    c = ctx.memory.episodic.events.get(cause_id)
    e = ctx.memory.episodic.events.get(effect_id)
    if c is None or e is None or c.time_span is None or e.time_span is None:
        return SkillOutput(
            output={"supported": None}, output_type="claim",
            confidence=0.0, inferred=True, failure_mode="missing_input",
        )
    precedes = c.time_span[1] <= e.time_span[0]
    shared = set(c.participants) & set(e.participants)
    supported = bool(precedes and shared)
    refs: List[EvidenceRef] = []
    for ref_id in c.evidence_ref_ids + e.evidence_ref_ids:
        r = ctx.memory.evidence.get(ref_id)
        if r is not None:
            refs.append(r)
    bundle = EvidenceBundle(
        bundle_id=new_id("eb"),
        refs=refs[:8],
        query=RetrievalQuery(
            query_id=new_id("rq"), text=f"causal({cause_id},{effect_id})",
            store_filter="episodic", mode="lexical",
        ),
        sufficiency_hint=1.0 if refs else 0.0,
        confidence=0.7 if supported else 0.3,
    )
    return SkillOutput(
        output={
            "cause_id": cause_id,
            "effect_id": effect_id,
            "supported": supported,
            "precedes": precedes,
            "shared_participants": sorted(shared),
        },
        output_type="claim",
        confidence=bundle.confidence,
        inferred=True,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 10. update_belief_state
# ---------------------------------------------------------------------------


def _exec_update_belief_state(ctx: SkillContext) -> SkillOutput:
    """Apply an evidence-justified update to an agent's belief.

    Per the boundary rule (skill never writes memory directly), this skill
    requests a ``revise_belief_state`` procedure call rather than mutating
    state directly. The harness honors the request after verification.
    """
    holder = ctx.inputs.get("holder") or ctx.hop_goal.perspective_anchor
    proposition = ctx.inputs.get("proposition") or ctx.hop_goal.goal_text
    polarity = ctx.inputs.get("polarity", "true")
    if holder is None or not proposition:
        return SkillOutput(
            output={"updated": False},
            output_type="belief",
            confidence=0.0,
            inferred=True,
            failure_mode="missing_input",
        )
    evidence = ctx.evidence
    confidence = _aggregate_evidence_confidence(evidence) or 0.5
    return SkillOutput(
        output={
            "updated": True,
            "holder": holder,
            "proposition": proposition,
            "polarity": polarity,
            "confidence": confidence,
        },
        output_type="belief",
        confidence=confidence,
        inferred=True,
        used_evidence=evidence,
        requested_memory_writes=[
            {
                "procedure": "revise_belief_state",
                "args": {
                    "holder_entity": holder,
                    "proposition": proposition,
                    "polarity": polarity,
                    "confidence": confidence,
                    "evidence_ref_ids": [r.ref_id for r in (evidence.refs if evidence else [])],
                },
            }
        ],
    )


# ---------------------------------------------------------------------------
# 11. check_evidence_sufficiency
# ---------------------------------------------------------------------------


def _exec_check_evidence_sufficiency(ctx: SkillContext) -> SkillOutput:
    """Decide whether the cited bundle covers the claim's entities + scope."""
    bundle = ctx.evidence
    required_ents = set(ctx.hop_goal.required_entities)
    if bundle is None or bundle.is_empty():
        return SkillOutput(
            output={"sufficient": False, "reason": "no_bundle"},
            output_type="claim",
            confidence=0.0,
            inferred=True,
            failure_mode="empty_evidence",
        )
    covered = set()
    for r in bundle.refs:
        covered.update(r.entities)
    missing = required_ents - covered
    sufficient = not missing
    return SkillOutput(
        output={
            "sufficient": sufficient,
            "missing_entities": sorted(missing),
            "n_refs": len(bundle.refs),
        },
        output_type="claim",
        confidence=0.9 if sufficient else 0.4,
        inferred=False,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# 12. decide_answer_or_abstain
# ---------------------------------------------------------------------------


def _exec_decide_answer_or_abstain(ctx: SkillContext) -> SkillOutput:
    """Aggregate per-hop confidences and emit answer / abstain decision."""
    hop_steps = ctx.trace_so_far
    if not hop_steps:
        return SkillOutput(
            output={"decision": "abstain", "reason": "no_steps", "score": 0.0},
            output_type="decision",
            confidence=1.0,
            inferred=False,
        )
    score = sum(s.confidence for s in hop_steps) / len(hop_steps)
    threshold = float(ctx.inputs.get("answer_threshold", 0.6))
    decision = "answer" if score >= threshold else "abstain"
    return SkillOutput(
        output={
            "decision": decision,
            "score": score,
            "threshold": threshold,
            "n_steps_considered": len(hop_steps),
        },
        output_type="decision",
        confidence=1.0,
        inferred=False,
    )


# ---------------------------------------------------------------------------
# 13. check_alternative_hypothesis (second-pass)
# ---------------------------------------------------------------------------


def _exec_check_alternative_hypothesis(ctx: SkillContext) -> SkillOutput:
    """Compare the chosen claim against the strongest alternative.

    V1 logic: looks for ``contradicts`` edges touching any cited evidence /
    record; if any found, treats them as the strongest alternative.
    """
    primary_claim = ctx.inputs.get("claim") or {}
    contradicts = ctx.memory.contradicts
    relevant = [edge for edge in contradicts if any(
        rid in str(primary_claim) or rid in (
            (ctx.evidence.bundle_id if ctx.evidence else "")
        )
        for rid in (edge[0], edge[1])
    )]
    has_alt = bool(relevant)
    return SkillOutput(
        output={
            "has_alternative": has_alt,
            "alternatives": [{"a": e[0], "b": e[1], "reason": e[2]} for e in relevant[:5]],
        },
        output_type="claim",
        confidence=0.8 if not has_alt else 0.5,
        inferred=True,
        used_evidence=ctx.evidence,
    )


# ---------------------------------------------------------------------------
# 14. locate_counterevidence (second-pass)
# ---------------------------------------------------------------------------


def _exec_locate_counterevidence(ctx: SkillContext) -> SkillOutput:
    claim = ctx.inputs.get("claim") or {}
    bundle = ctx.retriever.retrieve_counter(claim, ctx.trace_so_far)
    return SkillOutput(
        output={
            "counter_refs": [r.ref_id for r in bundle.refs],
            "n_counter": len(bundle.refs),
        },
        output_type="evidence_set",
        confidence=bundle.confidence,
        inferred=False,
        used_evidence=bundle,
    )


# ---------------------------------------------------------------------------
# Bank construction
# ---------------------------------------------------------------------------


def _record(
    *,
    skill_id: str,
    name: str,
    family: str,
    output_type: str,
    inputs: Dict[str, Dict[str, Any]],
    outputs: Dict[str, Dict[str, Any]],
    failure_modes: List[str],
    required_memory_fields: List[str],
    verification_rule: List[VerificationCheckSpec],
    triggers: Optional[List[TriggerSpec]] = None,
    required_primitives: Optional[List[str]] = None,
    protocol_steps: Optional[List[str]] = None,
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        type="atomic",
        family=family,
        trigger_conditions=triggers or [],
        input_schema=inputs,
        output_schema=outputs,
        output_type=output_type,
        verification_rule=verification_rule,
        failure_modes=failure_modes,
        required_memory_fields=required_memory_fields,
        required_primitives=required_primitives or [],
        protocol_steps=protocol_steps or [],
    )


def register_starter_skills(bank: ReasoningSkillBank) -> None:
    """Register the v1 starter inventory into ``bank``."""

    # 1. identify_question_target ------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.identify_question_target",
            name="identify_question_target",
            family="question_parsing",
            output_type="meta",
            inputs={
                "question_text": {"type": "str", "required": False},
                "target_entities": {"type": "list[str]", "required": False},
            },
            outputs={
                "question_text": {"type": "str", "required": True},
                "target_entities": {"type": "list[str]", "required": True},
                "perspective_anchor": {"type": "str|null", "required": True},
            },
            failure_modes=["missing_input"],
            required_memory_fields=[],
            verification_rule=[VerificationCheckSpec(
                name="entity_consistency", inputs=["target_entities"],
                predicate="all targets are valid mentions",
                on_fail="continue",
            )],
        ),
        executable=_exec_identify_question_target,
    ))

    # 2. decompose_into_subgoals -------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.decompose_into_subgoals",
            name="decompose_into_subgoals",
            family="question_parsing",
            output_type="meta",
            inputs={"hop_goal": {"type": "HopGoal", "required": True}},
            outputs={"subgoals": {"type": "list[str]", "required": True}},
            failure_modes=[],
            required_memory_fields=[],
            verification_rule=[VerificationCheckSpec(
                name="claim_evidence_alignment", inputs=["subgoals"],
                predicate="non-empty list", on_fail="retry",
            )],
        ),
        executable=_exec_decompose_into_subgoals,
    ))

    # 3. retrieve_relevant_episode -----------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.retrieve_relevant_episode",
            name="retrieve_relevant_episode",
            family="retrieval_grounding",
            output_type="evidence_set",
            inputs={"hop_goal": {"type": "HopGoal", "required": True}},
            outputs={"episode_refs": {"type": "list[str]", "required": True}},
            failure_modes=["empty_evidence"],
            required_memory_fields=["episodic.events"],
            required_primitives=["search_memory"],
            verification_rule=[VerificationCheckSpec(
                name="evidence_sufficiency", inputs=["episode_refs"],
                predicate="len(refs) > 0", on_fail="broaden",
            )],
        ),
        executable=_exec_retrieve_relevant_episode,
    ))

    # 4. ground_entity_reference -------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.ground_entity_reference",
            name="ground_entity_reference",
            family="retrieval_grounding",
            output_type="entity_ref",
            inputs={"mention": {"type": "str", "required": True}},
            outputs={
                "mention": {"type": "str", "required": True},
                "entity_id": {"type": "str|null", "required": True},
            },
            failure_modes=["missing_input"],
            required_memory_fields=["entities"],
            verification_rule=[VerificationCheckSpec(
                name="entity_consistency", inputs=["entity_id"],
                predicate="entity_id resolves in registry",
                on_fail="switch_skill",
            )],
        ),
        executable=_exec_ground_entity_reference,
    ))

    # 5. ground_event_span -------------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.ground_event_span",
            name="ground_event_span",
            family="retrieval_grounding",
            output_type="span",
            inputs={"description": {"type": "str", "required": False}},
            outputs={
                "event_ids": {"type": "list[str]", "required": True},
                "time_spans": {"type": "list[tuple]", "required": True},
            },
            failure_modes=["empty_evidence"],
            required_memory_fields=["episodic.events"],
            verification_rule=[VerificationCheckSpec(
                name="evidence_sufficiency", inputs=["event_ids"],
                predicate="len(events) > 0", on_fail="broaden",
            )],
        ),
        executable=_exec_ground_event_span,
    ))

    # 6. infer_observation_access ------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.infer_observation_access",
            name="infer_observation_access",
            family="social_belief",
            output_type="claim",
            inputs={
                "holder": {"type": "str", "required": False},
                "event_ids": {"type": "list[str]", "required": True},
            },
            outputs={"access": {"type": "bool", "required": True}},
            failure_modes=["missing_input"],
            required_memory_fields=["episodic.events", "state.spatial"],
            verification_rule=[VerificationCheckSpec(
                name="perspective_consistency", inputs=["holder", "access"],
                predicate="holder cited from perspective thread", on_fail="broaden",
            )],
        ),
        executable=_exec_infer_observation_access,
    ))

    # 7. order_two_events --------------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.order_two_events",
            name="order_two_events",
            family="temporal",
            output_type="ordering",
            inputs={
                "event_a_id": {"type": "str", "required": False},
                "event_b_id": {"type": "str", "required": False},
            },
            outputs={"order": {"type": "str", "required": True}},
            failure_modes=["missing_input"],
            required_memory_fields=["episodic.events"],
            verification_rule=[VerificationCheckSpec(
                name="temporal_consistency", inputs=["a", "b", "order"],
                predicate="time_spans support claimed order", on_fail="abstain",
            )],
        ),
        executable=_exec_order_two_events,
    ))

    # 8. check_state_change ------------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.check_state_change",
            name="check_state_change",
            family="temporal",
            output_type="claim",
            inputs={
                "holder": {"type": "str", "required": False},
                "proposition": {"type": "str", "required": False},
            },
            outputs={"changed": {"type": "bool|null", "required": True}},
            failure_modes=["missing_input"],
            required_memory_fields=["state.social"],
            verification_rule=[VerificationCheckSpec(
                name="temporal_consistency", inputs=["changed"],
                predicate="state history orders correctly",
                on_fail="broaden",
            )],
        ),
        executable=_exec_check_state_change,
    ))

    # 9. check_causal_support ----------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.check_causal_support",
            name="check_causal_support",
            family="causal",
            output_type="claim",
            inputs={
                "cause_id": {"type": "str", "required": True},
                "effect_id": {"type": "str", "required": True},
            },
            outputs={"supported": {"type": "bool", "required": True}},
            failure_modes=["missing_input"],
            required_memory_fields=["episodic.events"],
            verification_rule=[VerificationCheckSpec(
                name="temporal_consistency", inputs=["supported"],
                predicate="cause precedes effect", on_fail="abstain",
            )],
        ),
        executable=_exec_check_causal_support,
    ))

    # 10. update_belief_state ----------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.update_belief_state",
            name="update_belief_state",
            family="social_belief",
            output_type="belief",
            inputs={
                "holder": {"type": "str", "required": False},
                "proposition": {"type": "str", "required": False},
                "polarity": {"type": "str", "required": False},
            },
            outputs={"updated": {"type": "bool", "required": True}},
            failure_modes=["missing_input", "empty_evidence"],
            required_memory_fields=["state.social"],
            required_primitives=["mem.revise_belief_state"],
            verification_rule=[VerificationCheckSpec(
                name="claim_evidence_alignment", inputs=["proposition"],
                predicate="evidence supports update", on_fail="abstain",
            )],
        ),
        executable=_exec_update_belief_state,
    ))

    # 11. check_evidence_sufficiency ---------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.check_evidence_sufficiency",
            name="check_evidence_sufficiency",
            family="verification",
            output_type="claim",
            inputs={"bundle": {"type": "EvidenceBundle", "required": True}},
            outputs={"sufficient": {"type": "bool", "required": True}},
            failure_modes=["empty_evidence"],
            required_memory_fields=[],
            verification_rule=[VerificationCheckSpec(
                name="evidence_sufficiency", inputs=["sufficient"],
                predicate="all required entities present in refs",
                on_fail="broaden",
            )],
        ),
        executable=_exec_check_evidence_sufficiency,
    ))

    # 12. decide_answer_or_abstain ----------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.decide_answer_or_abstain",
            name="decide_answer_or_abstain",
            family="decision",
            output_type="decision",
            inputs={"answer_threshold": {"type": "float", "required": False}},
            outputs={
                "decision": {"type": "str", "required": True},
                "score": {"type": "float", "required": True},
            },
            failure_modes=[],
            required_memory_fields=[],
            verification_rule=[VerificationCheckSpec(
                name="evidence_sufficiency", inputs=["score"],
                predicate="score >= answer_threshold OR decision==abstain",
                on_fail="continue",
            )],
        ),
        executable=_exec_decide_answer_or_abstain,
    ))

    # 13. check_alternative_hypothesis ------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.check_alternative_hypothesis",
            name="check_alternative_hypothesis",
            family="verification",
            output_type="claim",
            inputs={"claim": {"type": "dict", "required": True}},
            outputs={"has_alternative": {"type": "bool", "required": True}},
            failure_modes=[],
            required_memory_fields=[],
            verification_rule=[VerificationCheckSpec(
                name="claim_evidence_alignment", inputs=["has_alternative"],
                predicate="alternatives ranked vs primary", on_fail="continue",
            )],
        ),
        executable=_exec_check_alternative_hypothesis,
    ))

    # 14. locate_counterevidence ------------------------------------------
    bank.register(AtomicSkill(
        record=_record(
            skill_id="atom.locate_counterevidence",
            name="locate_counterevidence",
            family="retrieval_grounding",
            output_type="evidence_set",
            inputs={"claim": {"type": "dict", "required": True}},
            outputs={"counter_refs": {"type": "list[str]", "required": True}},
            failure_modes=[],
            required_memory_fields=["episodic.events"],
            required_primitives=["search_memory"],
            verification_rule=[VerificationCheckSpec(
                name="counterevidence", inputs=["counter_refs"],
                predicate="any counter ref outscores supporting refs",
                on_fail="continue",
            )],
        ),
        executable=_exec_locate_counterevidence,
    ))


def build_starter_bank() -> ReasoningSkillBank:
    """Convenience constructor used by the loop / tests."""
    bank = ReasoningSkillBank()
    register_starter_skills(bank)
    return bank

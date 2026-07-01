"""Executable Reasoning Graph Assembly Skills."""

from __future__ import annotations

import re
from typing import Any

from ..common import find_nodes, lexical_score, make_result, normalize_time_span, spans_overlap, stable_id


def _evidence_text(graph: dict[str, Any], refs: list[str]) -> str:
    by_id = {node.get("node_id"): node for node in graph.get("nodes", [])}
    return " ".join(
        str(by_id[ref].get("text") or by_id[ref].get("event_description") or by_id[ref].get("state_value") or "")
        for ref in refs
        if ref in by_id
    )


def _node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node.get("node_id"): node for node in graph.get("nodes", []) if node.get("node_id")}


def parse_question_target(question_text: str, options: list[dict[str, Any]] | None = None) -> Any:
    entities = re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", question_text)
    time_words = [word for word in ("before", "after", "later", "earlier", "when", "while") if word in question_text.lower()]
    focus = "causal" if re.search(r"\bwhy|because|cause", question_text, re.I) else "descriptive"
    if re.search(r"inconsistent|contradict|alibi|but", question_text, re.I):
        focus = "social_contradiction"
    answer_format = "multiple_choice" if options else "free_text"
    outputs = {
        "target_entities": list(dict.fromkeys(entities)),
        "target_events": [],
        "constraints": {"time_words": time_words},
        "answer_format": answer_format,
        "question_focus": focus,
        "options": options or [],
    }
    return make_result("parse_question_target", outputs)


def propose_evidence_roles(
    question_text: str,
    parsed_target: dict[str, Any],
    task_family: str | None = None,
) -> Any:
    focus = parsed_target.get("question_focus") or task_family or "generic"
    if focus == "social_contradiction":
        roles = ["stated_claim", "later_action", "temporal_order", "contradiction_evidence"]
        shape = "claim_then_counterevidence"
    elif focus == "causal":
        roles = ["cause_candidate", "effect_observation", "context_evidence"]
        shape = "cause_to_effect"
    else:
        roles = ["question_anchor", "supporting_evidence"]
        shape = "support_chain"
    return make_result(
        "propose_evidence_roles",
        {
            "evidence_roles": roles,
            "role_constraints": [{"role": role, "must_cite_evidence": True} for role in roles],
            "expected_chain_shape": shape,
        },
    )


def retrieve_by_event(
    evidence_graph: dict[str, Any],
    *,
    event_description: str,
    time_range: dict[str, Any] | None = None,
    entity_filter: str | None = None,
) -> Any:
    nodes = find_nodes(evidence_graph, text_query=event_description, time_range=time_range, entity_id=entity_filter)
    nodes = [node for node in nodes if node.get("node_type") in {"event", "observation", "dialogue_span", "state"}]
    scored = sorted(
        ((lexical_score(event_description, node.get("text") or node.get("event_description") or ""), node) for node in nodes),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [node for score, node in scored if score > 0][:10] or nodes[:5]
    refs = [node["node_id"] for node in selected]
    return make_result(
        "retrieve_by_event",
        {"event_nodes": selected, "evidence_refs": refs, "retrieval_scores": {node["node_id"]: lexical_score(event_description, node.get("text") or "") for node in selected}},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_event_match",
    )


def retrieve_by_entity(
    evidence_graph: dict[str, Any],
    *,
    entity_id: str,
    time_range: dict[str, Any] | None = None,
    predicate_filter: str | None = None,
) -> Any:
    nodes = []
    for node in evidence_graph.get("nodes", []):
        text = " ".join(str(node.get(k, "")) for k in ("text", "canonical_name", "event_description", "state_predicate"))
        refs = node.get("entity_refs", []) or []
        if entity_id == node.get("entity_id") or entity_id in refs or entity_id.lower() in text.lower():
            if time_range and not spans_overlap(node.get("time_span"), time_range):
                continue
            if predicate_filter and lexical_score(predicate_filter, text) == 0:
                continue
            nodes.append(node)
    refs = [node["node_id"] for node in nodes]
    return make_result(
        "retrieve_by_entity",
        {"entity_timeline": nodes, "evidence_refs": refs},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_entity_match",
    )


def retrieve_by_time(
    evidence_graph: dict[str, Any],
    *,
    anchor_event_or_time: str | dict[str, Any],
    window_before: float,
    window_after: float,
) -> Any:
    if isinstance(anchor_event_or_time, str):
        anchor = _node_map(evidence_graph).get(anchor_event_or_time, {})
        span = normalize_time_span(anchor.get("time_span")) or {"start_s": 0.0, "end_s": 0.0}
    else:
        span = normalize_time_span(anchor_event_or_time) or {"start_s": 0.0, "end_s": 0.0}
    query_range = {"start_s": max(0.0, span["start_s"] - window_before), "end_s": span["end_s"] + window_after}
    nodes = [node for node in evidence_graph.get("nodes", []) if spans_overlap(node.get("time_span"), query_range)]
    refs = [node["node_id"] for node in nodes]
    return make_result(
        "retrieve_by_time",
        {"neighbor_events": nodes, "evidence_refs": refs, "time_range": query_range},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_time_overlap",
    )


def retrieve_by_relation(
    evidence_graph: dict[str, Any],
    *,
    source_node: str,
    relation_type: str,
    hop_limit: int = 1,
) -> Any:
    frontier = {source_node}
    seen = {source_node}
    path_edges = []
    for _ in range(max(1, hop_limit)):
        next_frontier = set()
        for edge in evidence_graph.get("edges", []):
            if edge.get("edge_type") != relation_type:
                continue
            if edge.get("src") in frontier and edge.get("dst") not in seen:
                next_frontier.add(edge["dst"])
                path_edges.append(edge)
            if edge.get("dst") in frontier and edge.get("src") not in seen:
                next_frontier.add(edge["src"])
                path_edges.append(edge)
        seen.update(next_frontier)
        frontier = next_frontier
    related = [_node_map(evidence_graph)[nid] for nid in seen if nid != source_node and nid in _node_map(evidence_graph)]
    refs = [node["node_id"] for node in related]
    return make_result(
        "retrieve_by_relation",
        {"related_nodes": related, "path_edges": path_edges, "evidence_refs": refs},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_relation_path",
    )


def localize_clue(
    candidate_evidence: list[dict[str, Any]],
    *,
    role_constraint: str,
    question_context: str,
) -> Any:
    query = f"{role_constraint} {question_context}"
    ranked = sorted(
        candidate_evidence,
        key=lambda node: lexical_score(query, node.get("text") or node.get("event_description") or ""),
        reverse=True,
    )
    selected = ranked[:3]
    refs = [node["node_id"] for node in selected if node.get("node_id")]
    spans = [node.get("time_span") for node in selected if node.get("time_span")]
    confidence = lexical_score(query, selected[0].get("text") or selected[0].get("event_description") or "") if selected else 0.0
    return make_result(
        "localize_clue",
        {"clue_refs": refs, "clue_spans": spans, "confidence": confidence},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_clue_candidate",
        confidence=confidence,
    )


def extract_claim(
    evidence_graph: dict[str, Any],
    *,
    evidence_ref: str,
    speaker_hint: str | None = None,
    claim_query: str | None = None,
) -> Any:
    node = _node_map(evidence_graph).get(evidence_ref)
    if not node:
        return make_result("extract_claim", ok=False, failure_code="missing_evidence_ref")
    text = node.get("text") or node.get("event_description") or ""
    if claim_query and lexical_score(claim_query, text) == 0:
        return make_result("extract_claim", ok=False, failure_code="claim_query_not_supported")
    claim_id = stable_id("claim", evidence_ref, speaker_hint, text)
    return make_result(
        "extract_claim",
        {"claim_id": claim_id, "claim_text": text, "speaker": speaker_hint or node.get("speaker"), "evidence_ref": evidence_ref},
        [evidence_ref],
        confidence=0.9,
    )


def assign_evidence_role(
    evidence_graph: dict[str, Any],
    *,
    evidence_ref: str,
    role_schema: str,
    question_context: str,
) -> Any:
    node = _node_map(evidence_graph).get(evidence_ref)
    if not node:
        return make_result("assign_evidence_role", ok=False, failure_code="missing_evidence_ref")
    text = node.get("text") or node.get("event_description") or ""
    confidence = max(0.35, lexical_score(f"{role_schema} {question_context}", text))
    labeled = {"evidence_ref": evidence_ref, "role": role_schema, "text": text, "confidence": confidence}
    return make_result(
        "assign_evidence_role",
        {"role_labeled_evidence": labeled, "role_confidence": confidence},
        [evidence_ref],
        confidence=confidence,
    )


def compose_evidence_chain(
    role_labeled_evidence: list[dict[str, Any]],
    *,
    dependency_template: str,
) -> Any:
    present_roles = {item.get("role") for item in role_labeled_evidence}
    required_roles = [part.strip() for part in re.split(r"->|,", dependency_template) if part.strip()]
    missing = [role for role in required_roles if role not in present_roles]
    refs = [item["evidence_ref"] for item in role_labeled_evidence if item.get("evidence_ref")]
    chain = {
        "chain_id": stable_id("chain", dependency_template, refs),
        "dependency_template": dependency_template,
        "items": role_labeled_evidence,
        "evidence_refs": refs,
    }
    edges = [
        {"src_role": a.get("role"), "dst_role": b.get("role"), "edge_type": "evidence_dependency"}
        for a, b in zip(role_labeled_evidence, role_labeled_evidence[1:])
    ]
    return make_result(
        "compose_evidence_chain",
        {"evidence_chain": chain, "chain_edges": edges, "missing_roles": missing},
        refs,
        ok=not missing and bool(refs),
        failure_code=None if not missing and refs else "missing_required_roles",
    )


def detect_missing_role(evidence_chain: dict[str, Any], *, required_roles: list[str]) -> Any:
    present = {item.get("role") for item in evidence_chain.get("items", [])}
    missing = [role for role in required_roles if role not in present]
    queries = [f"find evidence for {role}" for role in missing]
    return make_result("detect_missing_role", {"missing_roles": missing, "suggested_queries": queries}, evidence_chain.get("evidence_refs", []), ok=bool(missing) is False, failure_code="roles_missing" if missing else None)


def search_counterevidence(
    evidence_graph: dict[str, Any],
    *,
    claim: dict[str, Any],
    supporting_evidence: list[str],
    search_scope: str,
) -> Any:
    claim_text = claim.get("claim_text") or claim.get("text") or ""
    negation_words = {"not", "never", "no", "left", "changed", "instead", "but", "however"}
    candidates = []
    for node in evidence_graph.get("nodes", []):
        if node.get("node_id") in supporting_evidence:
            continue
        text = node.get("text") or node.get("event_description") or ""
        if lexical_score(search_scope or claim_text, text) > 0 or any(word in text.lower() for word in negation_words):
            candidates.append(node)
    refs = [node["node_id"] for node in candidates[:5]]
    return make_result(
        "search_counterevidence",
        {"counterevidence_refs": refs, "counter_claims": [{"text": node.get("text") or node.get("event_description"), "evidence_ref": node["node_id"]} for node in candidates[:5]]},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_counterevidence",
    )


def infer_temporal_relation(event_refs: list[str], *, evidence_graph: dict[str, Any]) -> Any:
    nodes = [_node_map(evidence_graph).get(ref) for ref in event_refs]
    nodes = [node for node in nodes if node]
    if len(nodes) < 2:
        return make_result("infer_temporal_relation", ok=False, failure_code="too_few_events")
    ordered = sorted(nodes, key=lambda node: (normalize_time_span(node.get("time_span")) or {"start_s": 0.0})["start_s"])
    relation = "before" if ordered[0]["node_id"] == event_refs[0] else "after"
    return make_result(
        "infer_temporal_relation",
        {"temporal_relation": relation, "ordered_event_refs": [node["node_id"] for node in ordered], "supporting_evidence": event_refs},
        event_refs,
    )


def infer_state_change(
    evidence_graph: dict[str, Any],
    *,
    entity_or_object: str,
    state_predicate: str,
    before_after_refs: list[str],
) -> Any:
    nodes = [_node_map(evidence_graph).get(ref) for ref in before_after_refs]
    nodes = [node for node in nodes if node]
    ordered = sorted(nodes, key=lambda node: (normalize_time_span(node.get("time_span")) or {"start_s": 0.0})["start_s"])
    if len(ordered) < 2:
        return make_result("infer_state_change", ok=False, failure_code="too_few_state_refs")
    claim = f"{entity_or_object} changed {state_predicate} from {ordered[0].get('text')} to {ordered[-1].get('text')}"
    return make_result(
        "infer_state_change",
        {"state_change_claim": claim, "before_state": ordered[0], "after_state": ordered[-1]},
        [node["node_id"] for node in ordered],
        confidence=0.75,
    )


def infer_causal_relation(candidate_cause: str, candidate_effect: str, *, evidence_chain: dict[str, Any]) -> Any:
    refs = evidence_chain.get("evidence_refs", [])
    claim = f"{candidate_cause} plausibly caused {candidate_effect}"
    return make_result(
        "infer_causal_relation",
        {"causal_claim": claim, "supporting_roles": [item.get("role") for item in evidence_chain.get("items", [])]},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "empty_evidence_chain",
        confidence=0.65,
    )


def infer_intention_or_motive(agent: str, actions: list[str], *, context_evidence: list[str]) -> Any:
    claim = f"{agent}'s likely intention is inferred from actions: {', '.join(actions)}"
    alternatives = [f"{agent} may have another unstated reason"]
    return make_result(
        "infer_intention_or_motive",
        {"intention_claim": claim, "alternatives": alternatives, "supporting_roles": context_evidence},
        context_evidence,
        ok=bool(context_evidence),
        failure_code=None if context_evidence else "missing_context_evidence",
        confidence=0.6,
    )


def infer_social_contradiction(
    claim_or_alibi: dict[str, Any],
    *,
    evidence_chain: dict[str, Any],
    counterevidence: list[str] | None = None,
) -> Any:
    refs = list(dict.fromkeys((evidence_chain.get("evidence_refs", []) or []) + (counterevidence or [])))
    text = claim_or_alibi.get("claim_text") or claim_or_alibi.get("text") or "the stated claim"
    contradiction = f"{text} is inconsistent with later or counter evidence."
    return make_result(
        "infer_social_contradiction",
        {"contradiction_claim": contradiction, "supporting_evidence": refs},
        refs,
        ok=bool(refs),
        failure_code=None if refs else "missing_contradiction_evidence",
        confidence=0.7,
    )


def verify_claim_support(claim: dict[str, Any] | str, *, evidence_chain: dict[str, Any], support_policy: dict[str, Any] | None = None) -> Any:
    refs = evidence_chain.get("evidence_refs", [])
    min_refs = int((support_policy or {}).get("min_evidence_refs", 1))
    text = claim if isinstance(claim, str) else claim.get("claim_text") or claim.get("text") or str(claim)
    passed = len(refs) >= min_refs and bool(text)
    score = min(1.0, len(refs) / max(min_refs, 1)) if text else 0.0
    return make_result(
        "verify_claim_support",
        {"verification_score": score, "passed": passed, "failure_code": None if passed else "insufficient_evidence", "messages": [], "verified_claim": {"claim_id": stable_id("claim.verified", text, refs), "text": text, "claim_status": "verified" if passed else "insufficient", "supported_by_refs": refs}},
        refs,
        ok=passed,
        failure_code=None if passed else "insufficient_evidence",
        confidence=score,
    )


def commit_answer(
    verified_claim: dict[str, Any],
    *,
    options: list[dict[str, Any]] | None = None,
    answer_format: str = "free_text",
    support_chain: dict[str, Any],
) -> Any:
    claim_text = verified_claim.get("text") or verified_claim.get("claim_text") or ""
    if verified_claim.get("claim_status") not in {None, "verified"}:
        return make_result("commit_answer", ok=False, failure_code="claim_not_verified")
    final_answer = claim_text
    if answer_format == "multiple_choice" and options:
        best = max(options, key=lambda opt: lexical_score(claim_text, f"{opt.get('label', '')} {opt.get('text', '')}"))
        final_answer = best.get("label") or best.get("text") or claim_text
    refs = support_chain.get("evidence_refs", [])
    return make_result(
        "commit_answer",
        {"final_answer": final_answer, "answer_support_chain": support_chain, "confidence": verified_claim.get("confidence", 0.8)},
        refs,
        ok=bool(final_answer and refs),
        failure_code=None if final_answer and refs else "invalid_answer_commit",
    )

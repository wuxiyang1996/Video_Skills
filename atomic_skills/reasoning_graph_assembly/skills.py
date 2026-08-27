"""Executable Reasoning Graph Assembly Skills."""

from __future__ import annotations

import re
from typing import Any

from ..common import find_nodes, lexical_score, make_result, normalize_time_span, spans_overlap, stable_id


def _evidence_text(graph: dict[str, Any], refs: list[str]) -> str:
    by_id = {node.get("node_id"): node for node in graph.get("nodes", [])}
    diagnostic_types = {"question_requirement", "required_modality", "answerability_gap", "l2_repair_reminder"}
    return " ".join(
        str(by_id[ref].get("text") or by_id[ref].get("event_description") or by_id[ref].get("state_value") or "")
        for ref in refs
        if ref in by_id
        and by_id[ref].get("node_type") not in diagnostic_types
    )


def _node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node.get("node_id"): node for node in graph.get("nodes", []) if node.get("node_id")}


def _coerce_hypothesis(hypothesis: dict[str, Any] | list[Any] | str) -> dict[str, Any] | str:
    if isinstance(hypothesis, list):
        return next((item for item in hypothesis if isinstance(item, dict)), hypothesis[0] if hypothesis else "")
    return hypothesis


_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "because",
    "does",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "there",
    "this",
    "to",
    "type",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_TOKEN_SYNONYMS = {
    "automobile": {"car", "vehicle"},
    "back": {"return", "returns", "returned", "previously", "earlier"},
    "broadcast": {"telecast", "tv", "program"},
    "car": {"automobile", "vehicle"},
    "chef": {"cook", "cooking", "kitchen"},
    "cooking": {"cook", "preparing", "food", "kitchen", "meal", "dish"},
    "echoes": {"repeats", "returns", "reappears", "same"},
    "earlier": {"previously", "before", "original"},
    "food": {"meal", "dish", "pasta", "cooking", "preparing"},
    "instrument": {"guitar", "piano", "drum", "violin", "music", "musical"},
    "kitchen": {"cooking", "cook", "food", "chef"},
    "location": {"place", "position"},
    "meal": {"food", "dish", "cooking"},
    "musical": {"music", "instrument"},
    "original": {"previously", "earlier", "place", "position"},
    "pasta": {"food", "dish", "meal", "cooking"},
    "place": {"location", "position"},
    "position": {"place", "location", "original"},
    "preparing": {"cooking", "cook", "food", "kitchen", "meal", "dish"},
    "previously": {"earlier", "before", "original"},
    "reappears": {"returns", "again", "same"},
    "reading": {"book", "read"},
    "repeated": {"same", "again", "reappears"},
    "returns": {"back", "returned", "reappears", "original"},
    "rv": {"vehicle", "van"},
    "sport": {"sports", "game", "match", "broadcast"},
    "sports": {"sport", "game", "match", "broadcast"},
    "van": {"vehicle", "rv"},
    "vehicle": {"automobile", "car", "rv", "truck", "van"},
    "walked": {"moved", "went", "returns"},
}

_SEMANTIC_GROUPS = {
    "cooking_food": {
        "claim": {"cook", "cooking", "preparing", "prepare", "food", "meal", "dish"},
        "evidence": {
            "chef",
            "cook",
            "cooking",
            "food",
            "kitchen",
            "meal",
            "dish",
            "pasta",
            "lemon",
            "lemons",
            "bowl",
            "pot",
            "strainer",
            "tongs",
            "stove",
            "countertop",
            "parsley",
        },
    },
    "driving_vehicle": {
        "claim": {"driving", "drive", "vehicle", "car", "truck", "van"},
        "evidence": {"driving", "drive", "vehicle", "car", "truck", "van", "road", "steering", "wheel"},
    },
    "music_instrument": {
        "claim": {"playing", "musical", "music", "instrument", "guitar", "piano", "drum", "violin"},
        "evidence": {"musical", "music", "instrument", "guitar", "piano", "drum", "violin", "microphone", "stage"},
    },
    "reading_book": {
        "claim": {"reading", "read", "book", "page", "pages"},
        "evidence": {"reading", "read", "book", "page", "pages", "library"},
    },
    "sports_broadcast": {
        "claim": {"sport", "sports", "broadcast", "game", "match"},
        "evidence": {"sport", "sports", "broadcast", "game", "match", "scoreboard", "field", "court", "team", "athlete"},
    },
}


def _content_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", (text or "").lower()) if tok not in _QUESTION_STOPWORDS}


def _expanded_tokens(text: str) -> set[str]:
    tokens = _content_tokens(text)
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_TOKEN_SYNONYMS.get(token, set()))
    return expanded


def _target_alignment_score(question_text: str, evidence_text: str) -> float:
    question_tokens = _expanded_tokens(question_text)
    evidence_tokens = _expanded_tokens(evidence_text)
    if not question_tokens or not evidence_tokens:
        return 0.0
    generic_targets = {"doing", "moment", "scene", "shown", "happening", "person", "people"}
    if question_tokens <= generic_targets:
        return 1.0
    return len(question_tokens & evidence_tokens) / max(1, len(question_tokens))


def _semantic_group_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = _expanded_tokens(claim_text)
    evidence_tokens = _expanded_tokens(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    best = 0.0
    for group in _SEMANTIC_GROUPS.values():
        claim_hit = bool(claim_tokens & group["claim"])
        evidence_hits = len(evidence_tokens & group["evidence"])
        if claim_hit and evidence_hits:
            best = max(best, min(1.0, 0.35 + 0.15 * evidence_hits))
    return best


def _content_lexical_score(query: str, text: str) -> float:
    q = _expanded_tokens(query)
    t = _expanded_tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def _support_score(claim_text: str, evidence_text: str) -> float:
    return max(_content_lexical_score(claim_text, evidence_text), _semantic_group_score(claim_text, evidence_text))


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


def generate_answer_hypotheses(
    question_text: str,
    *,
    options: list[dict[str, Any]] | None = None,
    parsed_target: dict[str, Any] | None = None,
) -> Any:
    """Turn answer options into explicit candidate claims for option-level reasoning."""
    hypotheses = []
    for idx, option in enumerate(options or []):
        label = str(option.get("label") or option.get("id") or chr(ord("A") + idx))
        text = str(option.get("text") or option.get("answer") or label)
        hypotheses.append(
            {
                "hypothesis_id": stable_id("hypothesis", question_text, label, text),
                "option_label": label,
                "claim_text": text,
                "question_text": question_text,
                "question_focus": (parsed_target or {}).get("question_focus"),
                "source": "answer_option",
            }
        )
    if not hypotheses:
        hypotheses.append(
            {
                "hypothesis_id": stable_id("hypothesis", question_text),
                "option_label": None,
                "claim_text": question_text,
                "question_text": question_text,
                "question_focus": (parsed_target or {}).get("question_focus"),
                "source": "question_claim",
            }
        )
    return make_result(
        "generate_answer_hypotheses",
        {"hypotheses": hypotheses},
        confidence=1.0,
    )


def retrieve_evidence_for_hypothesis(
    evidence_graph: dict[str, Any],
    *,
    hypothesis: dict[str, Any] | list[Any] | str,
    max_refs: int = 6,
) -> Any:
    """Retrieve L1 evidence for one candidate hypothesis."""
    hypothesis = _coerce_hypothesis(hypothesis)
    claim_text = hypothesis if isinstance(hypothesis, str) else hypothesis.get("claim_text") or hypothesis.get("text") or ""
    question_text = "" if isinstance(hypothesis, str) else str(hypothesis.get("question_text") or "")
    query_text = " ".join(part for part in [question_text, claim_text] if part).strip() or claim_text
    scored = []
    for node in evidence_graph.get("nodes", []):
        text = node.get("text") or node.get("event_description") or node.get("state_value") or ""
        score = lexical_score(query_text, text)
        if score > 0:
            scored.append((score, node))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [node for _, node in scored[:max_refs]]
    refs = [node["node_id"] for node in selected if node.get("node_id")]
    return make_result(
        "retrieve_evidence_for_hypothesis",
        {
            "support_refs": refs,
            "weak_refs": [],
            "missing_refs": [] if refs else [claim_text],
            "retrieval_scores": {node["node_id"]: score for score, node in scored[:max_refs] if node.get("node_id")},
        },
        refs,
        ok=bool(refs),
        failure_code=None if refs else "no_hypothesis_evidence",
        confidence=scored[0][0] if scored else 0.0,
    )


def score_hypothesis_support(
    hypothesis: dict[str, Any] | list[Any] | str,
    *,
    support_evidence: list[str] | dict[str, Any],
    counterevidence: list[str] | None = None,
    evidence_graph: dict[str, Any] | None = None,
) -> Any:
    """Assign a comparable score to one hypothesis from support and contradiction evidence."""
    hypothesis = _coerce_hypothesis(hypothesis)
    support_refs = support_evidence.get("support_refs", []) if isinstance(support_evidence, dict) else support_evidence
    support_refs = support_refs or []
    counter_refs = counterevidence or []
    claim_text = hypothesis if isinstance(hypothesis, str) else hypothesis.get("claim_text") or hypothesis.get("text") or ""
    if evidence_graph is not None and support_refs:
        evidence_text = _evidence_text(evidence_graph, support_refs)
        lexical_support = _support_score(claim_text, evidence_text)
        coverage_support = min(0.25, 0.05 * len(support_refs)) if lexical_support > 0 else 0.0
        support_score = min(1.0, lexical_support + coverage_support)
    else:
        support_score = min(1.0, 0.25 * len(support_refs))
    contradiction_score = min(1.0, 0.35 * len(counter_refs))
    score = max(0.0, support_score - contradiction_score)
    scored = {
        "hypothesis": hypothesis,
        "claim_text": claim_text,
        "option_label": None if isinstance(hypothesis, str) else hypothesis.get("option_label"),
        "support_refs": support_refs,
        "counterevidence_refs": counter_refs,
        "support_score": support_score,
        "contradiction_score": contradiction_score,
        "overall_score": score,
        "missing_reason": None if support_refs else "missing_support_evidence",
    }
    return make_result(
        "score_hypothesis_support",
        {"scored_hypothesis": scored},
        support_refs + counter_refs,
        ok=bool(support_refs),
        failure_code=None if support_refs else "missing_support_evidence",
        confidence=score,
    )


def _pad_scored_hypotheses_with_options(
    scored: list[dict[str, Any]],
    options: list[Any] | None,
) -> list[dict[str, Any]]:
    """Ensure every MCQ label appears so force_explore is not stuck on size-1 pools."""
    out = [dict(item) for item in scored if isinstance(item, dict)]
    if not options:
        return out
    seen: set[str] = set()
    for item in out:
        hyp = item.get("hypothesis") if isinstance(item.get("hypothesis"), dict) else {}
        label = str(item.get("option_label") or hyp.get("option_label") or "").strip()
        if label:
            seen.add(label)
    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label") or opt.get("id") or "").strip()
        if not label or label in seen:
            continue
        text = str(opt.get("text") or label)
        out.append(
            {
                "option_label": label,
                "claim_text": text,
                "overall_score": 0.0,
                "support_score": 0.0,
                "support_refs": [],
                "counterevidence_refs": [],
                "hypothesis": {"option_label": label, "claim_text": text},
                "padded_option": True,
            }
        )
        seen.add(label)
    return out


def compare_hypotheses(
    scored_hypotheses: list[dict[str, Any]],
    *,
    decision_policy: dict[str, Any] | None = None,
) -> Any:
    """Compare option-level hypotheses and choose the strongest supported candidate."""
    policy = decision_policy or {}
    force_explore = bool(policy.get("force_explore", False))
    candidates = [item for item in scored_hypotheses if isinstance(item, dict)]
    if force_explore and policy.get("options"):
        candidates = _pad_scored_hypotheses_with_options(candidates, policy.get("options"))
    if not candidates:
        return make_result("compare_hypotheses", ok=False, failure_code="no_hypotheses")
    margin = float(policy.get("min_margin", 0.0))
    ranked = sorted(
        candidates,
        key=lambda item: item.get("overall_score", item.get("support_score", 0.0)),
        reverse=True,
    )
    best = ranked[0]
    best_score = float(best.get("overall_score", best.get("support_score", 0.0)) or 0.0)
    # Exploration for GRPO K-samples via explore_seed (typically grpo_seed).
    # - default: rotate among near-tied candidates within tie_epsilon
    # - force_explore: always rotate among top-k (guarantees label diversity)
    # - if top-k pool size==1, fall back to all ranked options
    explore_seed = policy.get("explore_seed")
    tie_eps = float(policy.get("tie_epsilon", 0.15))
    explore_top_k = max(1, int(policy.get("explore_top_k", 2)))
    decision_reason = "highest_support_margin"
    if explore_seed is not None and len(ranked) > 1:
        if force_explore:
            pool = ranked[: min(explore_top_k, len(ranked))]
            if len(pool) <= 1:
                pool = ranked
        elif tie_eps > 0:
            pool = [
                item
                for item in ranked
                if best_score - float(item.get("overall_score", item.get("support_score", 0.0)) or 0.0)
                <= tie_eps
            ]
        else:
            pool = [ranked[0]]
        if len(pool) > 1:
            best = pool[int(explore_seed) % len(pool)]
            best_score = float(best.get("overall_score", best.get("support_score", 0.0)) or 0.0)
            decision_reason = "force_explore_seed" if force_explore else "near_tie_explore_seed"
    second_score = (
        ranked[1].get("overall_score", ranked[1].get("support_score", 0.0)) if len(ranked) > 1 else 0.0
    )
    if best is not ranked[0] and len(ranked) > 1:
        # Recompute margin vs the strongest non-chosen candidate.
        others = [item for item in ranked if item is not best]
        second_score = max(
            (float(item.get("overall_score", item.get("support_score", 0.0)) or 0.0) for item in others),
            default=0.0,
        )
    refs = list(
        dict.fromkeys(
            ref
            for item in ranked
            for ref in item.get("support_refs", []) + item.get("counterevidence_refs", [])
        )
    )
    eliminated = [
        {
            "option_label": item.get("option_label"),
            "claim_text": item.get("claim_text"),
            "reason": "lower_support_or_more_counterevidence",
            "overall_score": item.get("overall_score", 0.0),
        }
        for item in ranked
        if item is not best
    ]
    return make_result(
        "compare_hypotheses",
        {
            "best_hypothesis": best,
            "eliminated_hypotheses": eliminated,
            "decision_reason": decision_reason,
            "score_margin": best_score - float(second_score or 0.0),
        },
        refs,
        ok=best_score > 0 and (best_score - float(second_score or 0.0)) >= margin,
        failure_code=None
        if best_score > 0 and (best_score - float(second_score or 0.0)) >= margin
        else "ambiguous_hypotheses",
        confidence=max(0.0, min(1.0, best_score)),
    )


def bridge_evidence_hops(
    evidence_graph: dict[str, Any],
    *,
    source_evidence: list[str] | str,
    target_hypothesis: dict[str, Any] | list[Any] | str,
    allowed_hop_types: list[str] | None = None,
    max_hops: int = 2,
) -> Any:
    """Build a small multi-hop chain from evidence refs toward a target hypothesis."""
    target_hypothesis = _coerce_hypothesis(target_hypothesis)
    sources = [source_evidence] if isinstance(source_evidence, str) else list(source_evidence or [])
    allowed = set(allowed_hop_types or [])
    claim_text = target_hypothesis if isinstance(target_hypothesis, str) else target_hypothesis.get("claim_text") or ""
    nodes = _node_map(evidence_graph)
    frontier = set(sources)
    seen = set(sources)
    chain_edges = []
    for _ in range(max(1, max_hops)):
        next_frontier = set()
        for edge in evidence_graph.get("edges", []):
            if allowed and edge.get("edge_type") not in allowed:
                continue
            if edge.get("src") in frontier and edge.get("dst") not in seen:
                next_frontier.add(edge["dst"])
                chain_edges.append(edge)
            elif edge.get("dst") in frontier and edge.get("src") not in seen:
                next_frontier.add(edge["src"])
                chain_edges.append(edge)
        seen.update(next_frontier)
        frontier = next_frontier
    lexical_bridges = [
        node["node_id"]
        for node in evidence_graph.get("nodes", [])
        if node.get("node_id") not in seen
        and lexical_score(claim_text, node.get("text") or node.get("event_description") or "") > 0
    ][:3]
    chain_refs = list(dict.fromkeys([*sources, *seen, *lexical_bridges]))
    chain_refs = [ref for ref in chain_refs if ref in nodes]
    return make_result(
        "bridge_evidence_hops",
        {"multi_hop_chain": {"evidence_refs": chain_refs, "path_edges": chain_edges, "target_claim": claim_text}},
        chain_refs,
        ok=len(chain_refs) >= 2,
        failure_code=None if len(chain_refs) >= 2 else "no_bridge_path",
        confidence=min(1.0, len(chain_refs) / 4),
    )


def verify_temporal_social_consistency(
    evidence_chain: dict[str, Any],
    *,
    hypothesis: dict[str, Any] | list[Any] | str,
    evidence_graph: dict[str, Any] | None = None,
) -> Any:
    """Check generic temporal ordering and social-plausibility signals without motif rules."""
    hypothesis = _coerce_hypothesis(hypothesis)
    refs = evidence_chain.get("evidence_refs", []) or []
    nodes = [_node_map(evidence_graph or {}).get(ref) for ref in refs]
    nodes = [node for node in nodes if node]
    spans = [normalize_time_span(node.get("time_span")) for node in nodes]
    spans = [span for span in spans if span]
    spans = sorted(spans, key=lambda span: span["start_s"])
    temporal_ok = all(span["start_s"] <= span["end_s"] for span in spans) if spans else bool(refs)
    claim_text = hypothesis if isinstance(hypothesis, str) else hypothesis.get("claim_text") or hypothesis.get("text") or ""
    social_terms = {"person", "man", "woman", "boy", "girl", "friend", "talk", "look", "give", "take", "help", "angry"}
    evidence_text = _evidence_text(evidence_graph or {}, refs)
    social_plausibility_ok = bool(set(re.findall(r"[A-Za-z]+", f"{claim_text} {evidence_text}".lower())) & social_terms) or bool(refs)
    conflicts = []
    if not temporal_ok:
        conflicts.append("temporal_order_uncertain")
    if not social_plausibility_ok:
        conflicts.append("social_context_uncertain")
    return make_result(
        "verify_temporal_social_consistency",
        {
            "temporal_ok": temporal_ok,
            "social_plausibility_ok": social_plausibility_ok,
            "conflicts": conflicts,
        },
        refs,
        ok=not conflicts,
        failure_code=None if not conflicts else "consistency_conflict",
        confidence=1.0 if not conflicts else 0.4,
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


def verify_claim_support(
    claim: dict[str, Any] | str,
    *,
    evidence_chain: dict[str, Any],
    support_policy: dict[str, Any] | None = None,
    evidence_graph: dict[str, Any] | None = None,
    question_text: str | None = None,
) -> Any:
    refs = evidence_chain.get("evidence_refs", [])
    policy = support_policy or {}
    min_refs = int(policy.get("min_evidence_refs", 1))
    text = claim if isinstance(claim, str) else claim.get("claim_text") or claim.get("text") or str(claim)
    nested = {} if isinstance(claim, str) else claim.get("hypothesis") if isinstance(claim.get("hypothesis"), dict) else {}
    option_label = None if isinstance(claim, str) else claim.get("option_label") or nested.get("option_label")
    question_context = question_text or (None if isinstance(claim, str) else claim.get("question_text") or nested.get("question_text"))
    evidence_text = _evidence_text(evidence_graph or {}, refs) if evidence_graph is not None else ""
    claim_score = _support_score(text, evidence_text) if evidence_text else (1.0 if refs else 0.0)
    target_score = _target_alignment_score(str(question_context or ""), evidence_text) if question_context and evidence_text else 1.0
    min_claim_score = float(policy.get("min_claim_score", 0.05 if evidence_graph is not None else 0.0))
    min_target_score = float(policy.get("min_target_score", 0.05 if question_context and evidence_graph is not None else 0.0))
    refs_ok = len(refs) >= min_refs
    claim_ok = claim_score >= min_claim_score
    target_ok = target_score >= min_target_score
    passed = refs_ok and bool(text) and claim_ok and target_ok
    score = min(1.0, 0.4 * min(1.0, len(refs) / max(min_refs, 1)) + 0.4 * claim_score + 0.2 * target_score) if text else 0.0
    messages = []
    if not refs_ok:
        messages.append("not enough evidence refs")
    if not claim_ok:
        messages.append("evidence text does not support claim text")
    if not target_ok:
        messages.append("evidence text is not aligned with question target")
    return make_result(
        "verify_claim_support",
        {
            "verification_score": score,
            "passed": passed,
            "failure_code": None if passed else "insufficient_evidence",
            "messages": messages,
            "claim_support_score": claim_score,
            "target_alignment_score": target_score,
            "verified_claim": {
                "claim_id": stable_id("claim.verified", text, refs),
                "text": text,
                "option_label": option_label,
                "question_text": question_context,
                "claim_status": "verified" if passed else "insufficient",
                "supported_by_refs": refs,
            },
        },
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
    decision_policy: dict[str, Any] | None = None,
) -> Any:
    claim_text = verified_claim.get("text") or verified_claim.get("claim_text") or ""
    if verified_claim.get("claim_status") not in {None, "verified"}:
        return make_result("commit_answer", ok=False, failure_code="claim_not_verified")
    final_answer = claim_text
    commit_explore_used = False
    if answer_format == "multiple_choice" and options:
        label = str(verified_claim.get("option_label") or "").strip()
        by_label = {
            str(opt.get("label") or opt.get("id") or "").strip(): opt
            for opt in options
            if isinstance(opt, dict)
        }
        best = by_label.get(label) if label else None
        policy = decision_policy or {}
        # Backup explore when compare was skipped / singleton: rotate MCQ labels by seed.
        if (
            policy.get("force_explore")
            and policy.get("explore_seed") is not None
            and by_label
            and (policy.get("commit_explore") or not label or bool(policy.get("commit_explore_always")))
        ):
            labels = list(by_label.keys())
            pick = labels[int(policy["explore_seed"]) % len(labels)]
            best = by_label[pick]
            commit_explore_used = True
        if best is None:
            best = max(
                options,
                key=lambda opt: lexical_score(
                    claim_text, f"{opt.get('label', '')} {opt.get('text', '')}"
                ),
            )
        final_answer = best.get("label") or best.get("text") or claim_text
    refs = support_chain.get("evidence_refs", [])
    return make_result(
        "commit_answer",
        {
            "final_answer": final_answer,
            "answer_support_chain": support_chain,
            "confidence": verified_claim.get("confidence", 0.8),
            "commit_explore_used": commit_explore_used,
        },
        refs,
        ok=bool(final_answer and refs),
        failure_code=None if final_answer and refs else "invalid_answer_commit",
    )

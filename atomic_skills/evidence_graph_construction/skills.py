"""Executable Evidence Graph Construction Skills.

These functions are deliberately lightweight: they turn dataset annotations,
captions, subtitles, or model/tool outputs into typed graph records that later
reasoning skills can cite. Heavy perception models can be plugged in upstream.
"""

from __future__ import annotations

import re
from typing import Any

from ..common import (
    add_edge_once,
    add_node_once,
    ensure_graph,
    make_result,
    node_ids,
    normalize_time_span,
    stable_id,
    text_tokens,
)


ALLOWED_MEMORY_EDGES = {
    "temporal_next",
    "entity_mention",
    "derived_from",
    "causal_hint",
    "same_entity",
    "state_of",
    "dialogue_speaker",
    "located_in",
}


def segment_video_or_select_clip(
    graph: dict[str, Any] | None,
    *,
    video_id: str,
    clip_policy: dict[str, Any],
    observation_end_s: float | None = None,
) -> Any:
    graph = ensure_graph(graph)
    strategy = clip_policy.get("strategy", "whole_video")
    duration_s = float(clip_policy.get("duration_s", observation_end_s or 0.0))
    if observation_end_s is not None:
        duration_s = min(duration_s, float(observation_end_s)) if duration_s else float(observation_end_s)

    spans: list[dict[str, float]] = []
    if strategy == "whole_video":
        spans = [{"start_s": 0.0, "end_s": max(duration_s, 0.0)}]
    elif strategy in {"fixed_window", "hierarchical", "coarse_only"}:
        window_key = "coarse_window_s" if strategy in {"hierarchical", "coarse_only"} else "window_s"
        window_s = float(clip_policy.get(window_key, clip_policy.get("window_s", 30.0)))
        overlap_s = float(clip_policy.get("overlap_s", 0.0))
        if window_s <= 0 or overlap_s >= window_s:
            return make_result(
                "segment_video_or_select_clip",
                ok=False,
                failure_code="invalid_clip_policy",
                messages=["window_s must be positive and overlap_s must be smaller than window_s"],
            )
        cursor = 0.0
        limit = max(duration_s, window_s)
        while cursor < limit:
            end = min(cursor + window_s, limit)
            if observation_end_s is None or end <= observation_end_s:
                spans.append({"start_s": cursor, "end_s": end})
            cursor += window_s - overlap_s
    else:
        return make_result(
            "segment_video_or_select_clip",
            ok=False,
            failure_code="invalid_clip_policy",
            messages=[f"unsupported clip policy: {strategy}"],
        )

    clip_nodes = []
    for index, span in enumerate(spans):
        node = {
            "node_id": stable_id("evidence.clip", video_id, strategy, index, span),
            "node_type": "clip",
            "video_id": video_id,
            "clip_policy": strategy,
            "granularity": "coarse" if strategy in {"hierarchical", "coarse_only"} else "fine",
            "time_span": span,
            "source_ids": [video_id],
            "provenance": {"created_by": "segment_video_or_select_clip"},
        }
        add_node_once(graph, node)
        clip_nodes.append(node)

    return make_result(
        "segment_video_or_select_clip",
        {"graph": graph, "clip_nodes": clip_nodes, "time_spans": spans},
        [node["node_id"] for node in clip_nodes],
    )


def extract_observation(
    graph: dict[str, Any] | None,
    *,
    clip_or_text_ref: str,
    modality: str,
    text: str,
    time_span: dict[str, Any] | None = None,
    observation_query: str | None = None,
) -> Any:
    graph = ensure_graph(graph)
    if not text.strip():
        return make_result(
            "extract_observation",
            ok=False,
            failure_code="empty_observation",
            messages=["observation text is empty"],
        )
    node = {
        "node_id": stable_id("evidence.observation", clip_or_text_ref, modality, text, time_span),
        "node_type": "observation",
        "source_ids": [clip_or_text_ref],
        "modality": modality,
        "text": text,
        "observation_query": observation_query,
        "time_span": normalize_time_span(time_span),
        "provenance": {"created_by": "extract_observation"},
    }
    add_node_once(graph, node)
    if clip_or_text_ref in node_ids(graph):
        add_edge_once(
            graph,
            {
                "edge_id": stable_id("mem.edge", clip_or_text_ref, node["node_id"], "derived_from"),
                "src": node["node_id"],
                "dst": clip_or_text_ref,
                "edge_type": "derived_from",
            },
        )
    return make_result(
        "extract_observation",
        {"graph": graph, "observation_nodes": [node], "evidence_refs": [node["node_id"]]},
        [node["node_id"]],
    )


def extract_dialogue_span(
    graph: dict[str, Any] | None,
    *,
    subtitle_or_asr_ref: str,
    text: str,
    time_span: dict[str, Any],
    speaker_hint: str | None = None,
) -> Any:
    graph = ensure_graph(graph)
    speaker = speaker_hint
    utterance = text
    match = re.match(r"\s*([^:：]{1,40})[:：]\s*(.+)", text)
    if match:
        speaker = speaker or match.group(1).strip()
        utterance = match.group(2).strip()

    if not utterance:
        return make_result(
            "extract_dialogue_span",
            ok=False,
            failure_code="empty_dialogue",
            messages=["dialogue utterance is empty"],
        )
    node = {
        "node_id": stable_id("evidence.dialogue", subtitle_or_asr_ref, speaker, utterance, time_span),
        "node_type": "dialogue_span",
        "source_ids": [subtitle_or_asr_ref],
        "speaker": speaker,
        "speaker_mention": speaker,
        "text": utterance,
        "time_span": normalize_time_span(time_span),
        "provenance": {"created_by": "extract_dialogue_span"},
    }
    add_node_once(graph, node)
    return make_result(
        "extract_dialogue_span",
        {"graph": graph, "dialogue_span_node": node, "speaker_mention": speaker, "evidence_ref": node["node_id"]},
        [node["node_id"]],
    )


def detect_entity_mention(
    graph: dict[str, Any] | None,
    *,
    observation_ref: str,
    text: str | None = None,
    entity_type: str | None = None,
) -> Any:
    graph = ensure_graph(graph)
    source = next((node for node in graph.get("nodes", []) if node.get("node_id") == observation_ref), None)
    source_text = text if text is not None else (source or {}).get("text", "")
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", source_text)
    if not candidates:
        # Fallback for non-English text: keep unique longer tokens as weak mentions.
        candidates = [tok for tok in text_tokens(source_text) if len(tok) >= 2][:5]

    mention_nodes = []
    for surface in dict.fromkeys(candidates):
        node = {
            "node_id": stable_id("evidence.mention", observation_ref, surface, entity_type),
            "node_type": "entity_mention",
            "surface_form": surface,
            "entity_type": entity_type or "unknown",
            "source_ids": [observation_ref],
            "time_span": (source or {}).get("time_span"),
            "text": surface,
            "provenance": {"created_by": "detect_entity_mention"},
        }
        add_node_once(graph, node)
        if observation_ref in node_ids(graph):
            add_edge_once(
                graph,
                {
                    "edge_id": stable_id("mem.edge", observation_ref, node["node_id"], "entity_mention"),
                    "src": observation_ref,
                    "dst": node["node_id"],
                    "edge_type": "entity_mention",
                },
            )
        mention_nodes.append(node)

    return make_result(
        "detect_entity_mention",
        {
            "graph": graph,
            "mention_nodes": mention_nodes,
            "surface_forms": [node["surface_form"] for node in mention_nodes],
            "time_spans": [node.get("time_span") for node in mention_nodes],
        },
        [node["node_id"] for node in mention_nodes],
        ok=bool(mention_nodes),
        failure_code=None if mention_nodes else "no_entity_mentions",
    )


def resolve_entity_coreference(
    graph: dict[str, Any] | None,
    *,
    mention_nodes: list[str],
    context_edges: list[str] | None = None,
) -> Any:
    graph = ensure_graph(graph)
    by_id = {node.get("node_id"): node for node in graph.get("nodes", [])}
    mentions = [by_id[mid] for mid in mention_nodes if mid in by_id]
    if not mentions:
        return make_result(
            "resolve_entity_coreference",
            ok=False,
            failure_code="missing_mentions",
            messages=["no mention node ids resolve in the graph"],
        )
    canonical = sorted((m.get("surface_form") or m.get("text") or m["node_id"] for m in mentions), key=len)[0]
    entity_id = stable_id("evidence.entity", canonical.lower())
    entity_node = {
        "node_id": entity_id,
        "node_type": "entity",
        "entity_id": entity_id,
        "canonical_name": canonical,
        "mention_refs": [m["node_id"] for m in mentions],
        "context_edges": context_edges or [],
        "provenance": {"created_by": "resolve_entity_coreference"},
    }
    add_node_once(graph, entity_node)
    same_entity_edges = []
    for mention in mentions:
        edge = {
            "edge_id": stable_id("mem.edge", mention["node_id"], entity_id, "same_entity"),
            "src": mention["node_id"],
            "dst": entity_id,
            "edge_type": "same_entity",
        }
        add_edge_once(graph, edge)
        same_entity_edges.append(edge)
    return make_result(
        "resolve_entity_coreference",
        {"graph": graph, "entity_node": entity_node, "same_entity_edges": same_entity_edges, "confidence": 0.8},
        [entity_id],
        confidence=0.8,
    )


def create_event_node(
    graph: dict[str, Any] | None,
    *,
    observation_refs: list[str],
    event_description: str,
    time_span: dict[str, Any],
    event_type: str = "event",
) -> Any:
    graph = ensure_graph(graph)
    refs = [ref for ref in observation_refs if ref in node_ids(graph)]
    if not refs:
        return make_result(
            "create_event_node",
            ok=False,
            failure_code="missing_observation_refs",
            messages=["event must be grounded in existing observation refs"],
        )
    node = {
        "node_id": stable_id("evidence.event", event_description, time_span, refs),
        "node_type": "event",
        "event_type": event_type,
        "event_description": event_description,
        "text": event_description,
        "source_ids": refs,
        "evidence_refs": refs,
        "time_span": normalize_time_span(time_span),
        "provenance": {"created_by": "create_event_node"},
    }
    add_node_once(graph, node)
    for ref in refs:
        add_edge_once(
            graph,
            {
                "edge_id": stable_id("mem.edge", node["node_id"], ref, "derived_from"),
                "src": node["node_id"],
                "dst": ref,
                "edge_type": "derived_from",
            },
        )
    return make_result("create_event_node", {"graph": graph, "event_node": node, "event_type": event_type, "evidence_refs": refs}, [node["node_id"], *refs])


def create_state_node(
    graph: dict[str, Any] | None,
    *,
    entity_ref: str,
    state_predicate: str,
    evidence_refs: list[str],
    state_value: str,
    time_span: dict[str, Any] | None = None,
) -> Any:
    graph = ensure_graph(graph)
    ids = node_ids(graph)
    refs = [ref for ref in evidence_refs if ref in ids]
    if entity_ref not in ids or not refs:
        return make_result(
            "create_state_node",
            ok=False,
            failure_code="missing_state_grounding",
            messages=["state requires an existing entity_ref and at least one evidence_ref"],
        )
    node = {
        "node_id": stable_id("evidence.state", entity_ref, state_predicate, state_value, time_span, refs),
        "node_type": "state",
        "entity_ref": entity_ref,
        "entity_refs": [entity_ref],
        "state_predicate": state_predicate,
        "state_value": state_value,
        "text": f"{entity_ref} {state_predicate} {state_value}",
        "evidence_refs": refs,
        "time_span": normalize_time_span(time_span),
        "confidence": 0.8,
        "provenance": {"created_by": "create_state_node"},
    }
    add_node_once(graph, node)
    add_edge_once(
        graph,
        {
            "edge_id": stable_id("mem.edge", node["node_id"], entity_ref, "state_of"),
            "src": node["node_id"],
            "dst": entity_ref,
            "edge_type": "state_of",
        },
    )
    return make_result("create_state_node", {"graph": graph, "state_node": node, "state_value": state_value, "confidence": 0.8}, [node["node_id"], *refs], confidence=0.8)


def link_graph_relation(
    graph: dict[str, Any] | None,
    *,
    source_node: str,
    target_node: str,
    edge_type: str,
    evidence_refs: list[str] | None = None,
    confidence: float = 1.0,
) -> Any:
    graph = ensure_graph(graph)
    ids = node_ids(graph)
    if source_node not in ids or target_node not in ids:
        return make_result(
            "link_graph_relation",
            ok=False,
            failure_code="missing_edge_endpoint",
            messages=["source_node and target_node must exist"],
        )
    if edge_type not in ALLOWED_MEMORY_EDGES:
        return make_result(
            "link_graph_relation",
            ok=False,
            failure_code="invalid_edge_type",
            messages=[f"unsupported edge type: {edge_type}"],
        )
    edge = {
        "edge_id": stable_id("mem.edge", source_node, target_node, edge_type),
        "src": source_node,
        "dst": target_node,
        "edge_type": edge_type,
        "evidence_refs": evidence_refs or [],
        "confidence": confidence,
    }
    add_edge_once(graph, edge)
    return make_result("link_graph_relation", {"graph": graph, "memory_edge": edge, "confidence": confidence}, evidence_refs or [], confidence=confidence)


def assign_provenance_trust(
    graph: dict[str, Any] | None,
    *,
    node_or_edge_ref: str,
    source_ref: str,
    mode: str,
    trust_policy: dict[str, Any],
) -> Any:
    graph = ensure_graph(graph)
    candidates = graph.get("nodes", []) + graph.get("edges", [])
    target = next(
        (
            item
            for item in candidates
            if item.get("node_id") == node_or_edge_ref or item.get("edge_id") == node_or_edge_ref
        ),
        None,
    )
    if target is None:
        return make_result(
            "assign_provenance_trust",
            ok=False,
            failure_code="missing_target",
            messages=[f"unknown node_or_edge_ref: {node_or_edge_ref}"],
        )
    trust_level = "weak"
    for level in ("gold", "strong", "weak", "model_labeled"):
        key = f"{level}_sources"
        if source_ref in trust_policy.get(key, []):
            trust_level = level
            break
    provenance = {
        "source_ref": source_ref,
        "mode": mode,
        "trust_level": trust_level,
        "discovery_status": "hidden_supervision" if mode == "expert_demo" and trust_level == "gold" else "visible",
    }
    target.setdefault("provenance", {}).update(provenance)
    return make_result("assign_provenance_trust", {"graph": graph, "provenance": provenance, "trust_level": trust_level, "discovery_status": provenance["discovery_status"]}, [node_or_edge_ref])

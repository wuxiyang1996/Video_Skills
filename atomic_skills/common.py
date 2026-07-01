"""Shared data helpers for atomic skill execution."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable


SCHEMA_VERSION = "video-skills-relaunch/v0.1"


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    skill_set: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    verifier_focus: str
    failure_codes: list[str]
    fn: Callable[..., "SkillResult"] | None = None

    def as_ontology_record(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_set": self.skill_set,
            "purpose": self.purpose,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "verifier_focus": self.verifier_focus,
            "failure_codes": self.failure_codes,
        }


@dataclass
class SkillResult:
    ok: bool
    skill_id: str
    outputs: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    failure_code: str | None = None
    messages: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_node_payload(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "outputs": self.outputs,
            "evidence_refs": self.evidence_refs,
            "status": "verified" if self.ok else "failed",
            "confidence": self.confidence,
            "failure_code": self.failure_code,
            "messages": self.messages,
        }


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts if part is not None)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe_prefix = prefix.strip(":")
    return f"{safe_prefix}:{digest}"


def ensure_graph(graph: dict[str, Any] | None = None) -> dict[str, Any]:
    graph = graph or {}
    graph.setdefault("schema_version", SCHEMA_VERSION)
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])
    return graph


def node_ids(graph: dict[str, Any]) -> set[str]:
    return {node.get("node_id") for node in graph.get("nodes", []) if node.get("node_id")}


def edge_ids(graph: dict[str, Any]) -> set[str]:
    return {edge.get("edge_id") for edge in graph.get("edges", []) if edge.get("edge_id")}


def add_node_once(graph: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ensure_graph(graph)
    existing = node_ids(graph)
    if node.get("node_id") not in existing:
        graph["nodes"].append(node)
    return node


def add_edge_once(graph: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any]:
    ensure_graph(graph)
    if edge.get("src") not in node_ids(graph) or edge.get("dst") not in node_ids(graph):
        raise ValueError(f"edge endpoints must exist before linking: {edge}")
    if edge.get("edge_id") not in edge_ids(graph):
        graph["edges"].append(edge)
    return edge


def normalize_time_span(time_span: dict[str, Any] | None = None) -> dict[str, float] | None:
    if not time_span:
        return None
    start = float(time_span.get("start_s", 0.0))
    end = float(time_span.get("end_s", start))
    if end < start:
        start, end = end, start
    return {"start_s": start, "end_s": end}


def spans_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    a = normalize_time_span(a)
    b = normalize_time_span(b)
    if not a or not b:
        return False
    return a["start_s"] <= b["end_s"] and b["start_s"] <= a["end_s"]


def text_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text or "")}


def lexical_score(query: str, text: str) -> float:
    q = text_tokens(query)
    t = text_tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def find_nodes(
    graph: dict[str, Any],
    *,
    node_type: str | None = None,
    text_query: str | None = None,
    entity_id: str | None = None,
    time_range: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if node_type and node.get("node_type") != node_type:
            continue
        if entity_id:
            refs = node.get("entity_refs", []) or []
            if entity_id != node.get("entity_id") and entity_id not in refs:
                continue
        if time_range and not spans_overlap(node.get("time_span"), time_range):
            continue
        if text_query:
            text = " ".join(
                str(node.get(field, ""))
                for field in ("text", "description", "event_description", "state_value")
            )
            if lexical_score(text_query, text) <= 0:
                continue
        matches.append(node)
    return matches


def make_result(
    skill_id: str,
    outputs: dict[str, Any] | None = None,
    evidence_refs: list[str] | None = None,
    *,
    ok: bool = True,
    failure_code: str | None = None,
    messages: list[str] | None = None,
    confidence: float = 1.0,
) -> SkillResult:
    return SkillResult(
        ok=ok,
        skill_id=skill_id,
        outputs=outputs or {},
        evidence_refs=evidence_refs or [],
        failure_code=failure_code,
        messages=messages or [],
        confidence=confidence,
    )

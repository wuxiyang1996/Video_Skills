#!/usr/bin/env python3
"""Generate expert-demo graph labels with gpt-5-mini.

The script asks an expert model to convert a compact video QA example into:

- perception records
- query / reasoning chain
- clue-memory graph
- skill graph rollout over the frozen atomic ontology

It validates structural invariants locally. The model may label and fit traces,
but it is not allowed to create new skill ids or schema fields.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5-mini"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_skills import export_skill_ontology  # noqa: E402
from atomic_skills.evidence_graph_construction import (  # noqa: E402
    assign_provenance_trust,
    create_event_node,
    detect_entity_mention,
    extract_observation,
    link_graph_relation,
    resolve_entity_coreference,
)


TOY_EXPERT_INPUT = {
    "demo_id": "toy_social_contradiction:gpt5mini:001",
    "dataset": "toy",
    "video_id": "toy_library_bus",
    "question_id": "toy_q1",
    "question": "Why is Alice's earlier statement inconsistent with what happens later?",
    "answer_format": "free_text",
    "gold_answer": "Alice said she would stay in the library until noon, but later she left and boarded a bus.",
    "perceived_video_notes": [
        {
            "source_id": "obs_001",
            "time_span": {"start_s": 2.0, "end_s": 7.0},
            "modality": "subtitle",
            "text": "Alice tells Bob: I will stay in the library until noon to study.",
        },
        {
            "source_id": "obs_002",
            "time_span": {"start_s": 18.0, "end_s": 24.0},
            "modality": "visual_caption",
            "text": "Alice puts a notebook into her backpack and leaves the library.",
        },
        {
            "source_id": "obs_003",
            "time_span": {"start_s": 31.0, "end_s": 37.0},
            "modality": "visual_caption",
            "text": "Alice boards a bus outside the campus gate while Bob watches.",
        },
        {
            "source_id": "obs_004",
            "time_span": {"start_s": 41.0, "end_s": 46.0},
            "modality": "subtitle",
            "text": "Bob says: She said she would stay, but she just left.",
        },
    ],
}


def parse_video_holmes_time(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    value = value.strip()
    if "-" in value:
        start, end = value.split("-", 1)
    else:
        start = end = value

    def to_seconds(part: str) -> float:
        pieces = [float(p) for p in part.strip().split(":")]
        if len(pieces) == 3:
            return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
        if len(pieces) == 2:
            return pieces[0] * 60 + pieces[1]
        return pieces[0]

    start_s = to_seconds(start)
    end_s = to_seconds(end)
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    if start_s == end_s:
        end_s += 1.0
    return {"start_s": start_s, "end_s": end_s}


def _first_annotation_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def find_video_holmes_annotation(video_id: str, dataset_root: Path) -> tuple[Path | None, dict[str, Any]]:
    benchmark = dataset_root / "Video-Holmes" / "Benchmark"
    for folder in ("annotations", "annotation_training"):
        path = benchmark / folder / f"{video_id}.json"
        if path.exists():
            return path, _first_annotation_payload(path)
    return None, {}


def normalize_video_holmes_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    segments = annotation.get("Segment Description") or annotation.get("SegmentDescription") or []
    relationships = annotation.get("Key Relationships") or annotation.get("KeyRelationships") or []
    inference = annotation.get("Inference Shots") or annotation.get("InferenceScenes") or []
    supernatural = annotation.get("Supernatural Elements") or annotation.get("SupernaturalElements") or {}
    theme = annotation.get("Core Theme") or annotation.get("MainIdea")
    return {
        "segment_descriptions": segments,
        "key_relationships": relationships,
        "inference_shots": inference,
        "supernatural_elements": supernatural,
        "core_theme": theme,
    }


def load_video_holmes_example(
    *,
    dataset_root: Path,
    split: str,
    index: int = 0,
    video_id: str | None = None,
    question_id: str | None = None,
) -> dict[str, Any]:
    benchmark = dataset_root / "Video-Holmes" / "Benchmark"
    qa_path = benchmark / f"{split}_Video-Holmes.json"
    records = json.loads(qa_path.read_text(encoding="utf-8"))
    selected = None
    if video_id is not None:
        for item in records:
            if item.get("video ID") != video_id:
                continue
            if question_id is None or str(item.get("Question ID")) == str(question_id):
                selected = item
                break
        if selected is None:
            raise ValueError(f"Video-Holmes record not found: split={split} video_id={video_id} question_id={question_id}")
    else:
        selected = records[index]

    vid = selected["video ID"]
    annotation_path, raw_annotation = find_video_holmes_annotation(vid, dataset_root)
    annotation = normalize_video_holmes_annotation(raw_annotation)
    options = selected.get("Options") or {}
    answer_label = selected.get("Answer")
    answer_text = options.get(answer_label) if isinstance(options, dict) else None

    perceived_notes: list[dict[str, Any]] = []
    for i, seg in enumerate(annotation["segment_descriptions"], start=1):
        perceived_notes.append(
            {
                "source_id": f"vh_segment_{i:03d}",
                "source_kind": "segment_description",
                "time_span": parse_video_holmes_time(seg.get("TimeRange")),
                "modality": "dataset_annotation",
                "text": seg.get("Description", ""),
            }
        )
    for i, shot in enumerate(annotation["inference_shots"], start=1):
        text = shot.get("Clue", "")
        if shot.get("Conclusion"):
            text = f"{text} Conclusion: {shot['Conclusion']}"
        perceived_notes.append(
            {
                "source_id": f"vh_inference_{i:03d}",
                "source_kind": "inference_shot",
                "time_span": parse_video_holmes_time(shot.get("Time")),
                "modality": "dataset_annotation",
                "text": text,
            }
        )
    for i, rel in enumerate(annotation["key_relationships"], start=1):
        text = " ".join(
            str(rel.get(field, ""))
            for field in ("Combination", "Relation", "Reason")
            if rel.get(field)
        )
        if text and text.lower() != "none none":
            perceived_notes.append(
                {
                    "source_id": f"vh_relationship_{i:03d}",
                    "source_kind": "key_relationship",
                    "time_span": None,
                    "modality": "dataset_annotation",
                    "text": text,
                }
            )
    if annotation["supernatural_elements"]:
        perceived_notes.append(
            {
                "source_id": "vh_supernatural_001",
                "source_kind": "supernatural_elements",
                "time_span": None,
                "modality": "dataset_annotation",
                "text": json.dumps(annotation["supernatural_elements"], ensure_ascii=False),
            }
        )
    if annotation["core_theme"]:
        perceived_notes.append(
            {
                "source_id": "vh_theme_001",
                "source_kind": "core_theme",
                "time_span": None,
                "modality": "dataset_annotation",
                "text": annotation["core_theme"],
            }
        )

    return {
        "demo_id": f"video_holmes:{split}:{vid}:q{selected.get('Question ID')}",
        "dataset": "Video-Holmes",
        "split": split,
        "video_id": vid,
        "question_id": str(selected.get("Question ID")),
        "question_type": selected.get("Question Type"),
        "question": selected.get("Question"),
        "options": [{"label": k, "text": v} for k, v in options.items()],
        "answer_format": "multiple_choice",
        "gold_answer": {"label": answer_label, "text": answer_text},
        "explanation": selected.get("Explanation"),
        "annotation_path": str(annotation_path) if annotation_path else None,
        "source_supervision": annotation,
        "perceived_video_notes": perceived_notes,
    }


def build_seed_clue_memory_graph(example: dict[str, Any]) -> dict[str, Any]:
    graph: dict[str, Any] = {"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []}
    event_refs: list[str] = []
    mention_refs_by_surface: dict[str, list[str]] = {}

    for note in example.get("perceived_video_notes", []):
        result = extract_observation(
            graph,
            clip_or_text_ref=note["source_id"],
            modality=note.get("modality", "dataset_annotation"),
            text=note.get("text", ""),
            time_span=note.get("time_span"),
            observation_query=note.get("source_kind"),
        )
        if not result.ok:
            continue
        graph = result.outputs["graph"]
        obs_ref = result.evidence_refs[0]

        trust = {
            "gold_sources": ["segment_description", "inference_shot", "key_relationship"],
            "strong_sources": ["supernatural_elements", "core_theme"],
            "weak_sources": [],
            "model_labeled_sources": [],
        }
        assign_provenance_trust(
            graph,
            node_or_edge_ref=obs_ref,
            source_ref=note.get("source_kind", "dataset_annotation"),
            mode="expert_demo",
            trust_policy=trust,
        )

        mentions = detect_entity_mention(graph, observation_ref=obs_ref, text=note.get("text", ""))
        graph = mentions.outputs.get("graph", graph)
        for mention in mentions.outputs.get("mention_nodes", []):
            surface = (mention.get("surface_form") or "").lower()
            mention_refs_by_surface.setdefault(surface, []).append(mention["node_id"])

        if note.get("time_span"):
            event = create_event_node(
                graph,
                observation_refs=[obs_ref],
                event_description=note.get("text", ""),
                time_span=note["time_span"],
                event_type=note.get("source_kind", "annotation_event"),
            )
            if event.ok:
                graph = event.outputs["graph"]
                event_refs.append(event.outputs["event_node"]["node_id"])

    for surface, refs in mention_refs_by_surface.items():
        if surface and len(refs) >= 2:
            result = resolve_entity_coreference(graph, mention_nodes=refs)
            graph = result.outputs.get("graph", graph) if result.ok else graph

    timed_events = [
        node
        for node in graph.get("nodes", [])
        if node.get("node_id") in event_refs and node.get("time_span")
    ]
    timed_events.sort(key=lambda node: node["time_span"]["start_s"])
    for left, right in zip(timed_events, timed_events[1:]):
        link_graph_relation(
            graph,
            source_node=left["node_id"],
            target_node=right["node_id"],
            edge_type="temporal_next",
            evidence_refs=[left["node_id"], right["node_id"]],
        )

    return graph


def make_expert_input(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "toy":
        example = TOY_EXPERT_INPUT.copy()
        example["provided_clue_memory_graph"] = {}
        return example
    if args.dataset != "video_holmes":
        raise ValueError(f"unsupported dataset for expert demo: {args.dataset}")
    example = load_video_holmes_example(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        index=args.index,
        video_id=args.video_id,
        question_id=args.question_id,
    )
    example["provided_clue_memory_graph"] = build_seed_clue_memory_graph(example)
    return example


def load_openrouter_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]

    keys_path = WORKSPACE_ROOT / "keys.py"
    if not keys_path.exists():
        raise RuntimeError("OPENROUTER_API_KEY not found and keys.py is missing")

    spec = importlib.util.spec_from_file_location("local_keys", keys_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import keys from {keys_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = getattr(module, "OPENROUTER_API_KEY", None)
    if not key:
        raise RuntimeError("keys.py does not define OPENROUTER_API_KEY")
    return key


def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    ontology = export_skill_ontology()
    reasoning_skill_ids = [
        item["skill_id"] for item in ontology["reasoning_graph_assembly"]
    ]
    evidence_skill_ids = [
        item["skill_id"] for item in ontology["evidence_graph_construction"]
    ]

    system = """You are an expert video reasoning trace labeler.
Return JSON only. Do not include markdown.

Your job is to fit the given video QA example to a frozen atomic skill ontology.
You may create perception labels, query plans, evidence roles, clue-memory graph
nodes, and a reasoning skill rollout, but you must not invent new atomic skills.
Do not reveal hidden chain-of-thought. Provide concise step summaries and typed
arguments only."""

    user_payload = {
        "task": "Create an expert demo for trace-to-skill fitting.",
        "example": example,
        "frozen_atomic_ontology": ontology,
        "allowed_reasoning_skill_ids": reasoning_skill_ids,
        "allowed_evidence_graph_skill_ids": evidence_skill_ids,
        "required_reasoning_chain": [
            "parse_question_target",
            "propose_evidence_roles",
            "retrieve_by_event",
            "retrieve_by_entity",
            "retrieve_by_time",
            "localize_clue",
            "extract_claim",
            "assign_evidence_role",
            "compose_evidence_chain",
            "infer_temporal_relation",
            "infer_social_contradiction",
            "verify_claim_support",
            "commit_answer",
        ],
        "hard_rules": [
            "Use only allowed skill ids.",
            "Every reasoning node after parse_question_target must cite at least one provided clue-memory node id.",
            "If provided_clue_memory_graph is non-empty, reuse its node ids and edges; do not create alternative evidence ids.",
            "The clue_memory_graph stores perceived facts, entities, events, captions, dialogue, and relations.",
            "The skill_graph_rollout stores executable reasoning actions over clue_memory_graph evidence.",
            "Do not invent video facts beyond perceived_video_notes.",
            "Use source_supervision, explanation, gold_answer, and annotation clues only as expert-demo supervision.",
            "evidence_refs must contain clue_memory_graph node_id values only. Do not put edge_id or claim_id values in evidence_refs.",
            "The final answer must be grounded in verify_claim_support and commit_answer.",
        ],
        "output_schema": {
            "demo_id": "string",
            "model_role": "expert_trace_labeler",
            "perception": [
                {
                    "source_id": "obs_001",
                    "modality": "subtitle|visual_caption|annotation",
                    "time_span": {"start_s": 0.0, "end_s": 1.0},
                    "observed_fact": "concise grounded fact",
                }
            ],
            "query_chain": [
                {
                    "step_id": "q1",
                    "query": "retrieval or reasoning query",
                    "intended_skill_id": "one allowed reasoning skill id",
                    "expected_evidence_role": "role name",
                }
            ],
            "clue_memory_graph": {
                "nodes": [
                    {
                        "node_id": "evidence.*",
                        "node_type": "clip|observation|dialogue_span|entity|event|state|semantic_memory",
                        "text": "grounded content",
                        "source_ids": ["obs_001"],
                        "time_span": {"start_s": 0.0, "end_s": 1.0},
                    }
                ],
                "edges": [
                    {
                        "edge_id": "mem_e1",
                        "src": "evidence.*",
                        "dst": "evidence.*",
                        "edge_type": "temporal_next|entity_mention|derived_from|same_entity|causal_hint",
                    }
                ],
            },
            "skill_graph_rollout": {
                "nodes": [
                    {
                        "node_id": "n1",
                        "skill_id": "one allowed reasoning skill id",
                        "args": {},
                        "outputs": {},
                        "evidence_refs": ["evidence.*"],
                        "status": "verified|needs_review",
                    }
                ],
                "edges": [
                    {"edge_id": "skill_e1", "src": "n1", "dst": "n2", "edge_type": "data|temporal|causal|evidence|claim_support|control"}
                ],
                "claims": [
                    {"claim_id": "claim:*", "text": "verified claim", "claim_status": "verified", "supported_by_refs": ["evidence.*"]}
                ],
                "final_answer": {"text": "answer", "confidence": 0.0},
            },
            "verifier_notes": {
                "needs_review": False,
                "known_weaknesses": [],
            },
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def call_openrouter(model: str, messages: list[dict[str, str]], temperature: float) -> str:
    key = load_openrouter_key()
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/wuxiyang1996/Video_Skills",
            "X-Title": "Video Skills Expert Demo Generator",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_demo(demo: dict[str, Any]) -> dict[str, Any]:
    """Repair harmless model-format drift while preserving typed content."""
    demo = json.loads(json.dumps(demo, ensure_ascii=False))
    memory = demo.get("clue_memory_graph", {})
    rollout = demo.get("skill_graph_rollout", {})
    evidence_ids = {node.get("node_id") for node in memory.get("nodes", []) if node.get("node_id")}
    edge_support: dict[str, list[str]] = {}

    valid_edges = []
    for edge in memory.get("edges", []):
        src = edge.get("src")
        dst = edge.get("dst")
        edge_id = edge.get("edge_id")
        refs = [ref for ref in edge.get("evidence_refs", []) if ref in evidence_ids]
        if src in evidence_ids and dst in evidence_ids:
            valid_edges.append(edge)
            edge_support[edge_id] = refs or [src, dst]
        elif edge_id:
            edge_support[edge_id] = refs + [ref for ref in (src, dst) if ref in evidence_ids]
    memory["edges"] = valid_edges

    claim_support = {
        claim.get("claim_id"): [
            ref for ref in claim.get("supported_by_refs", []) if ref in evidence_ids
        ]
        for claim in rollout.get("claims", [])
        if claim.get("claim_id")
    }

    def normalize_refs(refs: list[str]) -> list[str]:
        normalized: list[str] = []
        for ref in refs or []:
            if ref in evidence_ids:
                normalized.append(ref)
            elif ref in edge_support:
                normalized.extend(edge_support[ref])
            elif ref in claim_support:
                normalized.extend(claim_support[ref])
        return list(dict.fromkeys(normalized))

    for node in rollout.get("nodes", []):
        node["evidence_refs"] = normalize_refs(node.get("evidence_refs", []))
    for claim in rollout.get("claims", []):
        claim["supported_by_refs"] = normalize_refs(claim.get("supported_by_refs", []))
    if "answer_support_chain" in rollout:
        for step in rollout.get("answer_support_chain", []):
            step["evidence_refs"] = normalize_refs(step.get("evidence_refs", []))

    return demo


def validate_demo(demo: dict[str, Any]) -> dict[str, Any]:
    ontology = export_skill_ontology()
    allowed_reasoning = {item["skill_id"] for item in ontology["reasoning_graph_assembly"]}
    memory = demo.get("clue_memory_graph", {})
    rollout = demo.get("skill_graph_rollout", {})
    errors: list[str] = []
    warnings: list[str] = []

    evidence_ids = {node.get("node_id") for node in memory.get("nodes", []) if node.get("node_id")}
    skill_ids = {node.get("node_id") for node in rollout.get("nodes", []) if node.get("node_id")}
    claim_ids = {claim.get("claim_id") for claim in rollout.get("claims", []) if claim.get("claim_id")}

    if not evidence_ids:
        errors.append("clue_memory_graph.nodes is empty")
    if not skill_ids:
        errors.append("skill_graph_rollout.nodes is empty")

    for edge in memory.get("edges", []):
        if edge.get("src") not in evidence_ids or edge.get("dst") not in evidence_ids:
            errors.append(f"memory edge has unresolved endpoint: {edge.get('edge_id')}")

    for edge in rollout.get("edges", []):
        if edge.get("src") not in skill_ids or edge.get("dst") not in skill_ids:
            errors.append(f"skill edge has unresolved endpoint: {edge.get('edge_id')}")

    for node in rollout.get("nodes", []):
        if node.get("skill_id") not in allowed_reasoning:
            errors.append(f"unknown reasoning skill_id: {node.get('skill_id')}")
        if node.get("skill_id") != "parse_question_target" and not node.get("evidence_refs"):
            errors.append(f"reasoning node lacks evidence_refs: {node.get('node_id')}")
        for ref in node.get("evidence_refs", []) or []:
            if ref not in evidence_ids:
                errors.append(f"reasoning node cites unknown evidence_ref: {ref}")

    for claim in rollout.get("claims", []):
        for ref in claim.get("supported_by_refs", []) or []:
            if ref not in evidence_ids:
                errors.append(f"claim cites unknown evidence_ref: {ref}")

    present_skills = {node.get("skill_id") for node in rollout.get("nodes", [])}
    required = {"parse_question_target", "propose_evidence_roles", "verify_claim_support", "commit_answer"}
    missing = sorted(required - present_skills)
    if missing:
        errors.append(f"missing required skill ids: {missing}")
    if not claim_ids:
        warnings.append("no claims emitted")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "perception_items": len(demo.get("perception", [])),
            "query_chain_steps": len(demo.get("query_chain", [])),
            "memory_nodes": len(memory.get("nodes", [])),
            "memory_edges": len(memory.get("edges", [])),
            "skill_nodes": len(rollout.get("nodes", [])),
            "skill_edges": len(rollout.get("edges", [])),
            "claims": len(rollout.get("claims", [])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="video_holmes", choices=["video_holmes", "toy"])
    parser.add_argument("--dataset-root", default=str(WORKSPACE_ROOT / "datasets"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--video-id")
    parser.add_argument("--question-id")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default="experiments/expert_demo_gpt5mini_output.json")
    args = parser.parse_args()

    expert_input = make_expert_input(args)
    messages = build_messages(expert_input)
    raw = call_openrouter(args.model, messages, args.temperature)
    raw_demo = parse_json_response(raw)
    demo = normalize_demo(raw_demo)
    validation = validate_demo(demo)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "expert_demo_gpt5mini",
        "model": args.model,
        "input": expert_input,
        "raw_demo": raw_demo,
        "demo": demo,
        "validation": validation,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"wrote: {output_path}")
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

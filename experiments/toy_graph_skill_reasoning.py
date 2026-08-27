#!/usr/bin/env python3
"""Toy experiment for clue-memory graphs and agent-composed skill graphs.

This script uses a small synthetic "perceived video" record instead of decoding
frames. The model is asked to structure the perceived clues into an
EvidenceMemoryGraph and compose a SkillGraphRollout over that graph. A local
verifier then checks the typed bindings between the two layers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


SCHEMA_VERSION = "video-skills-relaunch/toy-v0.1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


TOY_PERCEPTION_NOTES = [
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
]


QUESTION = {
    "question_id": "toy_q1",
    "question_text": "Why is Alice's earlier statement inconsistent with what happens later?",
    "answer_format": "free_text",
}


OPERATOR_REGISTRY = [
    {
        "operator_id": "extract_caption_evidence",
        "operator_type": "perception",
        "writes": "evidence_memory_graph",
    },
    {
        "operator_id": "link_entity_mentions",
        "operator_type": "indexing",
        "writes": "evidence_memory_graph",
    },
    {
        "operator_id": "build_temporal_edges",
        "operator_type": "indexing",
        "writes": "evidence_memory_graph",
    },
    {
        "operator_id": "parse_question_target",
        "operator_type": "reasoning",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "propose_evidence_roles",
        "operator_type": "reasoning",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "retrieve_event",
        "operator_type": "retrieval",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "extract_dialogue_claim",
        "operator_type": "reasoning",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "order_events",
        "operator_type": "reasoning",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "compose_evidence_chain",
        "operator_type": "reasoning",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "infer_social_contradiction",
        "operator_type": "reasoning",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "verify_evidence_supports_claim",
        "operator_type": "verification",
        "writes": "skill_graph_rollout",
    },
    {
        "operator_id": "repair_by_requery",
        "operator_type": "repair",
        "writes": "skill_graph_rollout",
    },
]


REQUIRED_REASONING_SKILLS = [
    "parse_question_target",
    "propose_evidence_roles",
    "extract_dialogue_claim",
    "retrieve_event",
    "order_events",
    "compose_evidence_chain",
    "infer_social_contradiction",
    "verify_evidence_supports_claim",
]


@dataclass
class VerificationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, int]


def load_openrouter_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]

    repo_root = Path(__file__).resolve().parents[2]
    keys_path = repo_root / "keys.py"
    if not keys_path.exists():
        raise RuntimeError("OPENROUTER_API_KEY not found in env and keys.py is missing")

    spec = importlib.util.spec_from_file_location("local_keys", keys_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import key module from {keys_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = getattr(module, "OPENROUTER_API_KEY", None)
    if not key:
        raise RuntimeError("keys.py does not define OPENROUTER_API_KEY")
    return key


def build_prompt() -> list[dict[str, str]]:
    system = """You convert perceived video clues into typed graphs.

Return JSON only. Do not include markdown.
Use only the provided perception notes; do not invent video facts.

There are two layers:
1. evidence_memory_graph: organizes perceived clues and memory. Its nodes should
   use ids such as evidence.caption:obs_001, evidence.event:leave_library,
   evidence.entity:alice. Its edges should describe memory/index structure such
   as temporal_next, entity_mention, derived_from, and same_entity.
2. skill_graph_rollout: the agent-composed skill graph for multi-hop reasoning.
   Its nodes are executable operator invocations from the registry. The skill
   graph reads and binds evidence from the evidence_memory_graph; it does not
   rewrite perceived facts.

Composed skills may be mentioned only as motifs in metadata. The executable
rollout must expand into atomic operator nodes.
"""

    user_payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "Build both the clue-memory graph and the agent-composed skill graph.",
        "question": QUESTION,
        "perception_notes": TOY_PERCEPTION_NOTES,
        "operator_registry": OPERATOR_REGISTRY,
        "hard_constraints": [
            "The evidence_memory_graph must organize perceived clues, not reasoning steps.",
            "The skill_graph_rollout must contain only agent-executable operator invocations.",
            "The skill graph must be multi-hop, not a direct one-step answer.",
            "The skill graph must include every skill_id in required_reasoning_skills.",
            "Every non-meta skill node after parse_question_target must cite at least one evidence node.",
            "The final answer must be produced through verify_evidence_supports_claim.",
        ],
        "required_reasoning_skills": REQUIRED_REASONING_SKILLS,
        "required_output_shape": {
            "schema_version": SCHEMA_VERSION,
            "evidence_memory_graph": {
                "nodes": [
                    {
                        "node_id": "evidence.*",
                        "node_type": "clip|caption|event|entity|semantic_memory",
                        "source_ids": ["obs_*"],
                        "time_span": {"start_s": 0.0, "end_s": 1.0},
                        "text": "grounded content",
                        "provenance": {"created_by": "model_structuring_from_perception_notes"},
                    }
                ],
                "edges": [
                    {
                        "edge_id": "mem_e1",
                        "src": "evidence.*",
                        "dst": "evidence.*",
                        "edge_type": "temporal_next|entity_mention|derived_from|same_entity",
                    }
                ],
            },
            "skill_graph_rollout": {
                "rollout_id": "toy_rollout_001",
                "input_mode": "expert_demo",
                "nodes": [
                    {
                        "node_id": "n1",
                        "skill_id": "one operator_id from registry",
                        "operator_type": "reasoning|retrieval|verification|repair",
                        "args": {},
                        "outputs": {},
                        "evidence_refs": ["evidence.*"],
                        "claim_ids": [],
                        "status": "verified",
                    }
                ],
                "edges": [
                    {
                        "edge_id": "skill_e1",
                        "src": "n1",
                        "dst": "n2",
                        "edge_type": "data|temporal|causal|evidence|claim_support|control",
                    }
                ],
                "claims": [
                    {
                        "claim_id": "claim:toy_q1:answer",
                        "text": "answer claim",
                        "claim_status": "verified",
                        "supported_by_refs": ["evidence.*"],
                    }
                ],
                "answer_support_chain": [
                    {
                        "node_id": "n_last",
                        "claim_id": "claim:toy_q1:answer",
                        "evidence_refs": ["evidence.*"],
                    }
                ],
                "final_answer": {"text": "short final answer", "confidence": 0.0},
            },
            "cross_layer_links": [
                {
                    "source": "reasoning node id or claim id",
                    "target": "evidence node id",
                    "edge_type": "uses_evidence|supported_by|verified_by",
                }
            ],
            "motifs_used": [],
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]


def call_openrouter(model: str, temperature: float) -> str:
    key = load_openrouter_key()
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/wuxiyang1996/Video_Skills",
            "X-Title": "Video Skills Toy Graph Experiment",
        },
        json={
            "model": model,
            "messages": build_prompt(),
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=90,
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


def verify_result(result: dict[str, Any]) -> VerificationResult:
    errors: list[str] = []
    warnings: list[str] = []

    allowed_ops = {item["operator_id"]: item for item in OPERATOR_REGISTRY}
    memory = result.get("evidence_memory_graph", {})
    rollout = result.get("skill_graph_rollout", {})
    cross_links = result.get("cross_layer_links", [])

    memory_nodes = memory.get("nodes", [])
    memory_edges = memory.get("edges", [])
    skill_nodes = rollout.get("nodes", [])
    skill_edges = rollout.get("edges", [])
    claims = rollout.get("claims", [])

    evidence_ids = {node.get("node_id") for node in memory_nodes}
    evidence_ids.discard(None)
    skill_ids = {node.get("node_id") for node in skill_nodes}
    skill_ids.discard(None)
    claim_ids = {claim.get("claim_id") for claim in claims}
    claim_ids.discard(None)

    if not memory_nodes:
        errors.append("evidence_memory_graph.nodes is empty")
    if not skill_nodes:
        errors.append("skill_graph_rollout.nodes is empty")

    if len(evidence_ids) != len(memory_nodes):
        errors.append("evidence_memory_graph has duplicate or missing node_id")
    if len(skill_ids) != len(skill_nodes):
        errors.append("skill_graph_rollout has duplicate or missing node_id")

    for edge in memory_edges:
        if edge.get("src") not in evidence_ids:
            errors.append(f"memory edge {edge.get('edge_id')} has unknown src")
        if edge.get("dst") not in evidence_ids:
            errors.append(f"memory edge {edge.get('edge_id')} has unknown dst")

    for node in skill_nodes:
        op_id = node.get("skill_id")
        if op_id not in allowed_ops:
            errors.append(f"unknown skill_id/operator_id: {op_id}")
            continue
        if allowed_ops[op_id]["writes"] != "skill_graph_rollout":
            errors.append(f"skill graph used non-reasoning graph builder operator: {op_id}")
        for ref in node.get("evidence_refs", []) or []:
            if ref not in evidence_ids:
                errors.append(f"node {node.get('node_id')} cites unknown evidence_ref {ref}")

    for edge in skill_edges:
        if edge.get("src") not in skill_ids:
            errors.append(f"skill edge {edge.get('edge_id')} has unknown src")
        if edge.get("dst") not in skill_ids:
            errors.append(f"skill edge {edge.get('edge_id')} has unknown dst")

    for claim in claims:
        for ref in claim.get("supported_by_refs", []) or []:
            if ref not in evidence_ids:
                errors.append(f"claim {claim.get('claim_id')} cites unknown evidence_ref {ref}")

    for step in rollout.get("answer_support_chain", []) or []:
        if step.get("node_id") not in skill_ids:
            errors.append(f"answer support step has unknown node_id {step.get('node_id')}")
        if step.get("claim_id") not in claim_ids:
            errors.append(f"answer support step has unknown claim_id {step.get('claim_id')}")
        for ref in step.get("evidence_refs", []) or []:
            if ref not in evidence_ids:
                errors.append(f"answer support step cites unknown evidence_ref {ref}")

    for link in cross_links:
        source = link.get("source")
        target = link.get("target")
        if source not in skill_ids and source not in claim_ids:
            errors.append(f"cross_layer link has unknown source {source}")
        if target not in evidence_ids:
            errors.append(f"cross_layer link has unknown target {target}")

    present_skills = {node.get("skill_id") for node in skill_nodes}
    missing_required = [
        skill_id for skill_id in REQUIRED_REASONING_SKILLS if skill_id not in present_skills
    ]
    if missing_required:
        errors.append(f"skill graph is missing required multi-hop skills: {missing_required}")

    if len(skill_nodes) < len(REQUIRED_REASONING_SKILLS):
        errors.append("skill graph is too short for the required multi-hop reasoning chain")

    for node in skill_nodes:
        if node.get("skill_id") == "parse_question_target":
            continue
        if not node.get("evidence_refs"):
            errors.append(f"node {node.get('node_id')} has no evidence_refs")

    if len(rollout.get("answer_support_chain", []) or []) == 0:
        warnings.append("answer_support_chain is empty")
    else:
        final_support_node = (rollout.get("answer_support_chain") or [{}])[-1].get("node_id")
        verifier_nodes = {
            node.get("node_id")
            for node in skill_nodes
            if node.get("skill_id") == "verify_evidence_supports_claim"
        }
        if final_support_node not in verifier_nodes:
            errors.append("final answer support must come from verify_evidence_supports_claim")

    return VerificationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        stats={
            "memory_nodes": len(memory_nodes),
            "memory_edges": len(memory_edges),
            "skill_nodes": len(skill_nodes),
            "skill_edges": len(skill_edges),
            "claims": len(claims),
            "cross_layer_links": len(cross_links),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="experiments/toy_graph_skill_reasoning_output.json",
        help="Path relative to the video_skills_relaunched repo root.",
    )
    args = parser.parse_args()

    raw_text = call_openrouter(model=args.model, temperature=args.temperature)
    result = parse_json_response(raw_text)
    verification = verify_result(result)

    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "experiment": "toy_graph_skill_reasoning",
        "model": args.model,
        "question": QUESTION,
        "perception_notes": TOY_PERCEPTION_NOTES,
        "operator_registry": OPERATOR_REGISTRY,
        "model_result": result,
        "verification": {
            "passed": verification.passed,
            "errors": verification.errors,
            "warnings": verification.warnings,
            "stats": verification.stats,
        },
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(json.dumps(output_payload["verification"], indent=2))
    print(f"wrote: {output_path}")
    return 0 if verification.passed else 2


if __name__ == "__main__":
    sys.exit(main())

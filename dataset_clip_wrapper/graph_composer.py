"""Compose clue-memory graphs with gpt-oss-120B over frozen graph-crafting atomic skills."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_skills import export_skill_ontology  # noqa: E402
from atomic_skills.evidence_graph_construction import (  # noqa: E402
    assign_provenance_trust,
    create_event_node,
    create_state_node,
    detect_entity_mention,
    extract_dialogue_span,
    extract_observation,
    link_graph_relation,
    resolve_entity_coreference,
    segment_video_or_select_clip,
)

from .graph_plan_validator import (
    ALLOWED_MEMORY_EDGES,
    build_skill_plan_response_schema,
    executable_skill_ids,
    resolve_plan_value,
    validate_skill_plan,
)
from .openrouter_client import OpenRouterClient
from .schemas import GraphComposerConfig, RuntimeMode

SKILL_EXECUTORS = {
    "segment_video_or_select_clip": segment_video_or_select_clip,
    "extract_observation": extract_observation,
    "extract_dialogue_span": extract_dialogue_span,
    "detect_entity_mention": detect_entity_mention,
    "resolve_entity_coreference": resolve_entity_coreference,
    "create_event_node": create_event_node,
    "create_state_node": create_state_node,
    "link_graph_relation": link_graph_relation,
    "assign_provenance_trust": assign_provenance_trust,
}

EXECUTABLE_SKILL_IDS = executable_skill_ids(SKILL_EXECUTORS)
_ALLOWED_EDGE_TYPES = ", ".join(sorted(ALLOWED_MEMORY_EDGES))
VLM_L1_EDGE_TYPES = {
    "temporal_next",
    "entity_mention",
    "derived_from",
    "same_entity",
    "same_object",
    "same_place",
    "reappears",
    "before_after",
    "state_change",
    "supports_observation",
    "contrasts_observation",
    "located_in",
    "causal_hint",
    "social_cue",
}

GRAPH_COMPOSE_PROMPT = f"""You compose a clue-memory / perception graph using ONLY the executable
Evidence Graph Construction atomic skills listed in allowed_skill_ids.

This is a MULTIPLE-CHOICE skill selection task:
- skill_id MUST be exactly one value from allowed_skill_ids (no new names, no synonyms).
- Do NOT output free-form skill names or invented skills such as create_relation_node.

Return JSON only:
{{
  "skill_plan": [
    {{
      "step_id": "s1",
      "skill_id": "segment_video_or_select_clip",
      "args": {{
        "video_id": "$bindings.video_id",
        "clip_policy": "$bindings.clip_policy"
      }},
      "depends_on": []
    }},
    {{
      "step_id": "s2",
      "skill_id": "extract_observation",
      "args": {{
        "clip_or_text_ref": "$step.s1.evidence_refs.0",
        "modality": "visual",
        "text": "grounded observation text from clip schema"
      }},
      "depends_on": ["s1"]
    }}
  ],
  "notes": "short summary"
}}

Rules:
1. skill_id is multiple-choice from allowed_skill_ids only.
2. Reference prior step outputs ONLY with $step.<step_id>.<field> or $step.<step_id>.evidence_refs.N
   Never invent placeholder ids like obs:s2, mention:s3, entity:s4, edge:s8.
3. For node ID references (observation_ref, entity_ref, source_node, target_node, clip_or_text_ref,
   node_or_edge_ref, subtitle_or_asr_ref), ALWAYS use $step.<step_id>.evidence_refs.N which resolves
   to a node_id string. Do NOT use $step.X.observation_nodes.0 or $step.X.mention_nodes.0 (these
   resolve to full node objects, not strings).
4. For mention_nodes (list of node IDs), use [$step.<step_id>.evidence_refs.0, $step.<other>.evidence_refs.0].
5. link_graph_relation edge_type MUST be one of: {_ALLOWED_EDGE_TYPES}.
6. assign_provenance_trust trust_policy MUST be an object with gold_sources / model_labeled_sources lists.
7. Prefer: segment -> extract_observation / extract_dialogue_span -> detect_entity_mention ->
   create_event_node -> link_graph_relation(temporal_next) -> assign_provenance_trust.
8. Keep the plan compact; skip coreference unless clearly needed.
9. Do not output chain-of-thought.
"""

VLM_L1_GRAPH_PROMPT = """You build a video-only L1 clue-memory graph directly from clip schemas.

Return JSON only:
{
  "nodes": [
    {
      "node_id": "optional stable local id",
      "node_type": "observation|entity_mention|event|state|dialogue_span|clue",
      "clip_id": "clip id from input",
      "time_span": {"start_s": number, "end_s": number},
      "text": "grounded clue text",
      "modality": "visual|audio|subtitle|ocr|mixed",
      "confidence": 0.0
    }
  ],
  "edges": [
    {
      "edge_id": "optional stable local id",
      "src": "node_id",
      "dst": "node_id",
      "edge_type": "same_object|same_place|reappears|before_after|state_change|supports_observation|contrasts_observation|located_in|causal_hint|social_cue|temporal_next|derived_from",
      "evidence_refs": ["node_id"],
      "text": "short grounded reason"
    }
  ],
  "notes": "short quality note"
}

Rules:
1. Use only visible/spoken/subtitle/OCR evidence from clip_schemas.
2. Do not use official answers, hidden clues, labels, or dataset supervision.
3. Prefer reasoning-useful clues over generic captions.
4. Cross-clip semantic edges must be model-judged from the clip evidence, not string matching.
5. If evidence is weak, include uncertainty in node text and keep confidence low.
6. Every node must reference an input clip_id and time_span when possible.
7. Every edge src/dst must refer to nodes you output.
8. Do not output chain-of-thought.
"""


def build_vlm_l1_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vlm_l1_clue_graph",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "node_id": {"type": "string"},
                                "node_type": {"type": "string"},
                                "clip_id": {"type": "string"},
                                "time_span": {"type": "object", "additionalProperties": True},
                                "text": {"type": "string"},
                                "modality": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["node_type", "text"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "edge_id": {"type": "string"},
                                "src": {"type": "string"},
                                "dst": {"type": "string"},
                                "edge_type": {"type": "string"},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                "text": {"type": "string"},
                            },
                            "required": ["src", "dst", "edge_type"],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["nodes", "edges", "notes"],
            },
        },
    }


class GraphComposer:
    def __init__(self, config: GraphComposerConfig, client: OpenRouterClient):
        self.config = config
        self.client = client
        full_ontology = export_skill_ontology()["evidence_graph_construction"]
        executable = set(EXECUTABLE_SKILL_IDS)
        self.ontology = [item for item in full_ontology if item["skill_id"] in executable]
        self.allowed_skill_ids = executable

    @staticmethod
    def _stable_id(prefix: str, *parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"

    def plan_skill_graph(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> dict[str, Any]:
        allowed = sorted(self.allowed_skill_ids)
        payload = {
            "task": "compose_clue_memory_graph",
            "example_id": example_id,
            "video_id": video_id,
            "mode": mode.value,
            "layer": "clue_memory",
            "clip_policy": clip_policy,
            "clip_schemas": clip_schemas,
            "segments": segments,
            "allowed_skill_ids": allowed,
            "allowed_edge_types": sorted(ALLOWED_MEMORY_EDGES),
            "ontology": self.ontology,
            "instructions": GRAPH_COMPOSE_PROMPT,
        }
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert graph-crafting planner. "
                        "Choose skills only from allowed_skill_ids (multiple choice). "
                        "Never invent skill ids or placeholder node ids."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=build_skill_plan_response_schema(allowed),
        )
        response["model"] = self.config.model
        response["composer"] = "gpt_oss_graph_composer"
        validation_errors = validate_skill_plan(
            response.get("skill_plan") or [],
            allowed_skill_ids=self.allowed_skill_ids,
            clip_schemas=clip_schemas,
        )
        if validation_errors:
            response["validation_errors"] = validation_errors
            response["skill_plan"] = []
            response["notes"] = (
                (response.get("notes") or "")
                + " Plan rejected: skill selection must be multiple-choice from allowed_skill_ids with $step refs."
            ).strip()
        return response

    def execute_skill_plan(
        self,
        *,
        graph: dict[str, Any] | None,
        skill_plan: list[dict[str, Any]],
        bindings: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        graph = graph or {"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []}
        trace: list[dict[str, Any]] = []
        step_outputs: dict[str, Any] = {}

        for step in skill_plan:
            step_id = step.get("step_id")
            skill_id = step.get("skill_id")
            args = dict(step.get("args") or {})
            if skill_id not in SKILL_EXECUTORS:
                trace.append(
                    {
                        "step_id": step_id,
                        "skill_id": skill_id,
                        "ok": False,
                        "failure_code": "unknown_skill_id",
                    }
                )
                continue

            try:
                resolved_args = resolve_plan_value(args, bindings, step_outputs)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                trace.append(
                    {
                        "step_id": step_id,
                        "skill_id": skill_id,
                        "ok": False,
                        "failure_code": "invalid_step_reference",
                        "messages": [str(exc)],
                    }
                )
                continue
            try:
                result = SKILL_EXECUTORS[skill_id](graph, **resolved_args)
            except TypeError as exc:
                trace.append(
                    {
                        "step_id": step_id,
                        "skill_id": skill_id,
                        "ok": False,
                        "failure_code": "invalid_skill_args",
                        "messages": [str(exc)],
                    }
                )
                continue
            graph = result.outputs.get("graph", graph)
            trace.append(
                {
                    "step_id": step_id,
                    "skill_id": skill_id,
                    "ok": result.ok,
                    "failure_code": result.failure_code,
                    "evidence_refs": result.evidence_refs,
                }
            )
            if step_id:
                step_outputs[step_id] = {**result.outputs, "evidence_refs": result.evidence_refs}
        return graph, trace

    def compose_vlm_l1_graph(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> dict[str, Any]:
        visible_schemas = [schema for schema in clip_schemas if not schema.get("model_error")]
        payload = {
            "task": "compose_video_only_vlm_l1_clue_graph",
            "example_id": example_id,
            "video_id": video_id,
            "mode": mode.value,
            "clip_policy": clip_policy,
            "clip_schemas": visible_schemas,
            "segments": segments,
            "allowed_edge_types": sorted(VLM_L1_EDGE_TYPES),
            "instructions": VLM_L1_GRAPH_PROMPT,
        }
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a grounded video clue-graph composer. "
                        "Create semantic L1 nodes and edges only from the supplied clip schemas."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=build_vlm_l1_response_schema(),
        )
        response["model"] = self.config.model
        response["composer"] = "vlm_l1_graph_composer"
        return response

    def _graph_from_vlm_l1_response(
        self,
        *,
        response: dict[str, Any],
        base_graph: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        graph = base_graph
        clip_by_id = {schema.get("clip_id"): schema for schema in clip_schemas if schema.get("clip_id")}
        existing_node_ids = {node.get("node_id") for node in graph.get("nodes", [])}
        id_map: dict[str, str] = {}

        for index, raw_node in enumerate(response.get("nodes") or []):
            if not isinstance(raw_node, dict):
                continue
            text = str(raw_node.get("text") or raw_node.get("description") or "").strip()
            if not text:
                continue
            node_type = str(raw_node.get("node_type") or "observation").strip() or "observation"
            clip_id = str(raw_node.get("clip_id") or "").strip()
            schema = clip_by_id.get(clip_id) or {}
            time_span = raw_node.get("time_span") if isinstance(raw_node.get("time_span"), dict) else schema.get("time_span")
            proposed_id = str(raw_node.get("node_id") or "").strip()
            node_id = proposed_id if proposed_id and proposed_id not in existing_node_ids else self._stable_id(
                f"evidence.{node_type}",
                clip_id,
                time_span,
                text,
                index,
            )
            id_map[proposed_id] = node_id
            existing_node_ids.add(node_id)
            graph.setdefault("nodes", []).append(
                {
                    **raw_node,
                    "node_id": node_id,
                    "node_type": node_type,
                    "text": text,
                    "clip_id": clip_id or schema.get("clip_id"),
                    "time_span": time_span,
                    "modality": raw_node.get("modality") or "mixed",
                    "source_type": raw_node.get("source_type") or "vlm_l1",
                    "producer": "vlm_l1_graph_composer",
                    "visibility": {"hidden_supervision": False, "mode": mode.value},
                }
            )
            trace.append({"skill_id": "vlm_l1_create_node", "ok": True, "node_id": node_id})

        valid_node_ids = {node.get("node_id") for node in graph.get("nodes", [])}
        for index, raw_edge in enumerate(response.get("edges") or []):
            if not isinstance(raw_edge, dict):
                continue
            src = id_map.get(str(raw_edge.get("src") or ""), raw_edge.get("src"))
            dst = id_map.get(str(raw_edge.get("dst") or ""), raw_edge.get("dst"))
            if src not in valid_node_ids or dst not in valid_node_ids:
                trace.append(
                    {
                        "skill_id": "vlm_l1_skip_edge",
                        "ok": True,
                        "reason": "missing_endpoint",
                        "src": src,
                        "dst": dst,
                    }
                )
                continue
            edge_type = str(raw_edge.get("edge_type") or "supports_observation").strip()
            if edge_type not in VLM_L1_EDGE_TYPES:
                edge_type = "supports_observation"
            refs = [
                id_map.get(str(ref), str(ref))
                for ref in raw_edge.get("evidence_refs") or [src, dst]
                if id_map.get(str(ref), str(ref)) in valid_node_ids
            ]
            edge_id = str(raw_edge.get("edge_id") or "").strip() or self._stable_id(
                "edge",
                src,
                dst,
                edge_type,
                index,
            )
            graph.setdefault("edges", []).append(
                {
                    **raw_edge,
                    "edge_id": edge_id,
                    "src": src,
                    "dst": dst,
                    "edge_type": edge_type,
                    "evidence_refs": refs or [src, dst],
                    "producer": "vlm_l1_graph_composer",
                    "visibility": {"hidden_supervision": False, "mode": mode.value},
                }
            )
            trace.append({"skill_id": "vlm_l1_create_edge", "ok": True, "edge_id": edge_id, "edge_type": edge_type})

        if not any(node.get("producer") == "vlm_l1_graph_composer" for node in graph.get("nodes", [])):
            raise ValueError("VLM L1 composer produced no usable graph nodes")
        return graph, trace

    def compose_from_clip_schemas(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        mode: RuntimeMode,
        duration_s: float,
        observation_end_s: float | None = None,
    ) -> dict[str, Any]:
        composer_mode = self.config.composer_mode
        if not self.config.use_llm_planner:
            composer_mode = "deterministic"

        plan_payload: dict[str, Any]
        if composer_mode == "vlm_l1":
            plan_payload = {"nodes": [], "edges": [], "notes": "vlm_l1 graph compose not attempted"}
            skill_plan = []
        elif composer_mode == "skill_plan":
            try:
                plan_payload = self.plan_skill_graph(
                    example_id=example_id,
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    mode=mode,
                )
                skill_plan = plan_payload.get("skill_plan") or []
            except Exception as exc:
                plan_payload = {
                    "skill_plan": [],
                    "notes": "planner failed; deterministic fallback used",
                    "planner_error": str(exc),
                    "model": self.config.model,
                    "composer": "gpt_oss_graph_composer",
                }
                skill_plan = []
        else:
            plan_payload = {"skill_plan": [], "notes": "deterministic debug composer"}
            skill_plan = []

        bindings = {
            "video_id": video_id,
            "clip_policy": {**clip_policy, "duration_s": duration_s},
            "observation_end_s": observation_end_s,
            "mode": mode.value,
        }

        graph: dict[str, Any] = {"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []}
        trace: list[dict[str, Any]] = []
        used_deterministic = False

        if composer_mode == "vlm_l1":
            seg_result = segment_video_or_select_clip(
                graph,
                video_id=video_id,
                clip_policy={**clip_policy, "duration_s": duration_s},
                observation_end_s=observation_end_s,
            )
            graph = seg_result.outputs["graph"]
            trace.append(
                {
                    "skill_id": "segment_video_or_select_clip",
                    "ok": seg_result.ok,
                    "source": "vlm_l1_plumbing",
                }
            )
            try:
                plan_payload = self.compose_vlm_l1_graph(
                    example_id=example_id,
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    mode=mode,
                )
                graph, vlm_trace = self._graph_from_vlm_l1_response(
                    response=plan_payload,
                    base_graph=graph,
                    clip_schemas=clip_schemas,
                    mode=mode,
                )
                trace.extend(vlm_trace)
            except Exception as exc:
                plan_payload = {
                    "nodes": [],
                    "edges": [],
                    "notes": "VLM L1 composer failed; deterministic debug fallback used",
                    "planner_error": str(exc),
                    "model": self.config.model,
                    "composer": "vlm_l1_graph_composer",
                }
                graph, deterministic_trace = self._compose_deterministically(
                    graph={"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []},
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    duration_s=duration_s,
                    observation_end_s=observation_end_s,
                    mode=mode,
                )
                trace.append(
                    {
                        "skill_id": "deterministic_fallback",
                        "ok": True,
                        "reason": "vlm_l1_failed",
                    }
                )
                trace.extend(deterministic_trace)
                used_deterministic = True
        elif skill_plan:
            graph, trace = self.execute_skill_plan(graph=graph, skill_plan=skill_plan, bindings=bindings)
            failed_steps = [step for step in trace if step.get("ok") is False]
            successful_graph_steps = [
                step
                for step in trace
                if step.get("ok") and step.get("skill_id") != "segment_video_or_select_clip"
            ]
            if failed_steps or not successful_graph_steps:
                graph, deterministic_trace = self._compose_deterministically(
                    graph={"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []},
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    duration_s=duration_s,
                    observation_end_s=observation_end_s,
                    mode=mode,
                )
                trace.append(
                    {
                        "skill_id": "deterministic_fallback",
                        "ok": True,
                        "reason": "llm_plan_invalid_or_failed",
                        "failed_steps": len(failed_steps),
                    }
                )
                trace.extend(deterministic_trace)
                used_deterministic = True
        else:
            graph, deterministic_trace = self._compose_deterministically(
                graph=graph,
                video_id=video_id,
                clip_policy=clip_policy,
                clip_schemas=clip_schemas,
                segments=segments,
                duration_s=duration_s,
                observation_end_s=observation_end_s,
                mode=mode,
            )
            trace.extend(deterministic_trace)
            used_deterministic = True

        return {
            "graph": graph,
            "skill_plan": plan_payload,
            "execution_trace": trace,
            "composer_model": self.config.model,
            "composer_mode": composer_mode,
            "used_deterministic_fallback": used_deterministic,
        }

    def _compose_deterministically(
        self,
        *,
        graph: dict[str, Any],
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        duration_s: float,
        observation_end_s: float | None,
        mode: RuntimeMode,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        seg_result = segment_video_or_select_clip(
            graph,
            video_id=video_id,
            clip_policy={**clip_policy, "duration_s": duration_s},
            observation_end_s=observation_end_s,
        )
        graph = seg_result.outputs["graph"]
        trace.append({"skill_id": "segment_video_or_select_clip", "ok": seg_result.ok})

        clip_ref_by_id = {node["node_id"]: node for node in seg_result.outputs.get("clip_nodes", [])}
        clip_nodes = list(clip_ref_by_id.keys())
        for schema in clip_schemas:
            if schema.get("model_error"):
                trace.append(
                    {
                        "skill_id": "skip_failed_clip_schema",
                        "ok": True,
                        "clip_id": schema.get("clip_id"),
                        "source": "model_error",
                    }
                )
                continue
            clip_id = schema.get("clip_id")
            clip_ref = clip_id if clip_id in clip_ref_by_id else (clip_nodes[0] if clip_nodes else clip_id)
            scene = schema.get("scene_description") or ""
            if scene:
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="visual_caption",
                    text=scene,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "scene_description"})
                mentions = detect_entity_mention(graph, observation_ref=obs.evidence_refs[0])
                graph = mentions.outputs["graph"]
                trace.append({"skill_id": "detect_entity_mention", "ok": mentions.ok})

            for fact in schema.get("observable_facts") or []:
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality=fact.get("modality", "visual"),
                    text=fact.get("text", ""),
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "observable_fact"})

            for obj in schema.get("salient_objects") or []:
                if not isinstance(obj, dict):
                    continue
                attributes = ", ".join(str(item) for item in obj.get("attributes") or [])
                phrases = ", ".join(str(item) for item in obj.get("searchable_phrases") or [])
                text = " ".join(
                    part
                    for part in [
                        str(obj.get("surface_form") or "").strip(),
                        attributes,
                        phrases,
                    ]
                    if part
                )
                if not text:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="object_clue",
                    text=text,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "salient_object"})
                mention = detect_entity_mention(
                    graph,
                    observation_ref=obs.evidence_refs[0],
                    text=str(obj.get("surface_form") or text),
                    entity_type="object",
                )
                graph = mention.outputs["graph"]
                trace.append({"skill_id": "detect_entity_mention", "ok": mention.ok, "source": "salient_object"})

            place = schema.get("place") or {}
            if isinstance(place, dict):
                phrases = place.get("searchable_phrases") or []
                text = " ".join(
                    part
                    for part in [
                        str(place.get("description") or "").strip(),
                        " ".join(str(item) for item in phrases),
                    ]
                    if part
                )
                if text:
                    obs = extract_observation(
                        graph,
                        clip_or_text_ref=clip_ref,
                        modality="place_clue",
                        text=text,
                        time_span=schema.get("time_span"),
                    )
                    graph = obs.outputs["graph"]
                    trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "place"})

            for phrase in schema.get("searchable_phrases") or []:
                text = str(phrase).strip()
                if not text:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="searchable_phrase",
                    text=text,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "searchable_phrase"})

            for cue in schema.get("cross_clip_cues") or []:
                if not isinstance(cue, dict):
                    continue
                text = " ".join(
                    part
                    for part in [
                        str(cue.get("cue_type") or "").strip(),
                        str(cue.get("description") or "").strip(),
                    ]
                    if part
                )
                if not text:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="cross_clip_cue",
                    text=text,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "cross_clip_cue"})

            for mention_payload in schema.get("entity_mentions") or []:
                if not isinstance(mention_payload, dict):
                    continue
                surface = str(mention_payload.get("surface_form") or "").strip()
                if not surface:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="entity_schema",
                    text=surface,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "entity_schema"})
                mention = detect_entity_mention(
                    graph,
                    observation_ref=obs.evidence_refs[0],
                    text=surface,
                    entity_type=mention_payload.get("entity_type"),
                )
                graph = mention.outputs["graph"]
                trace.append({"skill_id": "detect_entity_mention", "ok": mention.ok, "source": "entity_schema"})

            for dialogue in schema.get("dialogue_spans") or []:
                dia = extract_dialogue_span(
                    graph,
                    subtitle_or_asr_ref=clip_ref,
                    text=dialogue.get("text", ""),
                    time_span=dialogue.get("time_span") or schema.get("time_span") or {"start_s": 0.0, "end_s": 1.0},
                    speaker_hint=dialogue.get("speaker"),
                )
                graph = dia.outputs["graph"]
                trace.append({"skill_id": "extract_dialogue_span", "ok": dia.ok})

            for event in schema.get("events") or []:
                obs_refs = [node["node_id"] for node in graph.get("nodes", []) if node.get("node_type") == "observation"][-1:]
                if not obs_refs:
                    continue
                ev = create_event_node(
                    graph,
                    observation_refs=obs_refs[:1],
                    event_description=event.get("description", ""),
                    time_span=event.get("time_span") or schema.get("time_span") or {"start_s": 0.0, "end_s": 1.0},
                )
                graph = ev.outputs["graph"]
                trace.append({"skill_id": "create_event_node", "ok": ev.ok})

        event_nodes = [node for node in graph.get("nodes", []) if node.get("node_type") == "event" and node.get("time_span")]
        event_nodes.sort(key=lambda node: node["time_span"]["start_s"])
        for left, right in zip(event_nodes, event_nodes[1:]):
            rel = link_graph_relation(
                graph,
                source_node=left["node_id"],
                target_node=right["node_id"],
                edge_type="temporal_next",
                evidence_refs=[left["node_id"], right["node_id"]],
            )
            graph = rel.outputs["graph"]
            trace.append({"skill_id": "link_graph_relation", "ok": rel.ok})

        mentions_by_surface: dict[str, list[dict[str, Any]]] = {}
        for node in graph.get("nodes", []):
            if node.get("node_type") != "entity_mention":
                continue
            surface = str(node.get("surface_form") or node.get("text") or "").strip().lower()
            if len(surface) < 2:
                continue
            mentions_by_surface.setdefault(surface, []).append(node)
        for mentions in mentions_by_surface.values():
            mentions.sort(key=lambda node: (node.get("time_span") or {}).get("start_s", 0.0))
            for left, right in zip(mentions, mentions[1:]):
                if left["node_id"] == right["node_id"]:
                    continue
                rel = link_graph_relation(
                    graph,
                    source_node=left["node_id"],
                    target_node=right["node_id"],
                    edge_type="same_entity",
                    evidence_refs=[left["node_id"], right["node_id"]],
                )
                graph = rel.outputs["graph"]
                trace.append({"skill_id": "link_graph_relation", "ok": rel.ok, "source": "same_entity_surface"})

        trust_policy = {
            "gold_sources": ["segment_description", "inference_shot", "clue_interval", "reasoning_process_step"],
            "strong_sources": ["video_summary", "subtitle_span"],
            "weak_sources": [],
            "model_labeled_sources": ["visual_caption", "scene_description"],
        }
        for node in graph.get("nodes", []):
            if node.get("node_type") in {"observation", "event", "dialogue_span"}:
                prov = assign_provenance_trust(
                    graph,
                    node_or_edge_ref=node["node_id"],
                    source_ref=node.get("modality") or node.get("source_type") or "model_labeled_span",
                    mode=mode.value,
                    trust_policy=trust_policy,
                )
                graph = prov.outputs["graph"]
        trace.append({"skill_id": "assign_provenance_trust", "ok": True})
        return graph, trace

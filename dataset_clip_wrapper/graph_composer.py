"""Compose clue-memory graphs with gpt-oss-120B over frozen graph-crafting atomic skills."""

from __future__ import annotations

import json
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


class GraphComposer:
    def __init__(self, config: GraphComposerConfig, client: OpenRouterClient):
        self.config = config
        self.client = client
        full_ontology = export_skill_ontology()["evidence_graph_construction"]
        executable = set(EXECUTABLE_SKILL_IDS)
        self.ontology = [item for item in full_ontology if item["skill_id"] in executable]
        self.allowed_skill_ids = executable

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
        plan_payload: dict[str, Any]
        if self.config.use_llm_planner:
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
            plan_payload = {"skill_plan": [], "notes": "deterministic fallback"}
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

        if skill_plan:
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

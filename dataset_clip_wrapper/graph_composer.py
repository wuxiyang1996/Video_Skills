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
    detect_entity_mention,
    extract_dialogue_span,
    extract_observation,
    link_graph_relation,
    resolve_entity_coreference,
    segment_video_or_select_clip,
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
    "link_graph_relation": link_graph_relation,
    "assign_provenance_trust": assign_provenance_trust,
}

GRAPH_COMPOSE_PROMPT = """You compose a clue-memory / perception graph using ONLY the frozen
Evidence Graph Construction atomic skills.

Return JSON only:
{
  "skill_plan": [
    {
      "step_id": "s1",
      "skill_id": "extract_observation",
      "args": {},
      "depends_on": []
    }
  ],
  "notes": "short summary"
}

Rules:
1. Use only allowed skill ids from the ontology.
2. Every non-segment step must ground to an existing clip_id, observation id, or dialogue id.
3. Prefer this order when possible:
   segment_video_or_select_clip -> extract_observation / extract_dialogue_span ->
   detect_entity_mention -> resolve_entity_coreference -> create_event_node ->
   link_graph_relation -> assign_provenance_trust
4. Do not invent new skill ids.
5. Do not output free-form chain-of-thought.
6. Keep the plan compact but sufficient to organize the provided clip schemas.
"""


class GraphComposer:
    def __init__(self, config: GraphComposerConfig, client: OpenRouterClient):
        self.config = config
        self.client = client
        self.ontology = export_skill_ontology()["evidence_graph_construction"]
        self.allowed_skill_ids = {item["skill_id"] for item in self.ontology}

    def plan_skill_graph(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        question: dict[str, Any],
        mode: RuntimeMode,
    ) -> dict[str, Any]:
        payload = {
            "task": "compose_clue_memory_graph",
            "example_id": example_id,
            "video_id": video_id,
            "mode": mode.value,
            "clip_policy": clip_policy,
            "question": question,
            "clip_schemas": clip_schemas,
            "segments": segments,
            "allowed_skill_ids": sorted(self.allowed_skill_ids),
            "ontology": self.ontology,
            "instructions": GRAPH_COMPOSE_PROMPT,
        }
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "You are an expert graph-crafting planner for video perception graphs.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
        response["model"] = self.config.model
        response["composer"] = "gpt_oss_graph_composer"
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

            resolved_args = self._resolve_args(args, bindings, step_outputs)
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
                step_outputs[step_id] = result.outputs
        return graph, trace

    def _resolve_args(self, args: dict[str, Any], bindings: dict[str, Any], step_outputs: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$bindings."):
                resolved[key] = bindings.get(value.split(".", 1)[1])
            elif isinstance(value, str) and value.startswith("$step."):
                _, rest = value.split(".", 1)
                step_id, field = rest.split(".", 1)
                resolved[key] = step_outputs.get(step_id, {}).get(field)
            else:
                resolved[key] = value
        return resolved

    def compose_from_clip_schemas(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        question: dict[str, Any],
        mode: RuntimeMode,
        duration_s: float,
        observation_end_s: float | None = None,
    ) -> dict[str, Any]:
        if self.config.use_llm_planner:
            try:
                plan_payload = self.plan_skill_graph(
                    example_id=example_id,
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    question=question,
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

        if skill_plan:
            graph, trace = self.execute_skill_plan(graph=graph, skill_plan=skill_plan, bindings=bindings)
            successful_graph_steps = [
                step
                for step in trace
                if step.get("ok") and step.get("skill_id") != "segment_video_or_select_clip"
            ]
            if not successful_graph_steps:
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
                        "reason": "llm_plan_had_no_successful_graph_steps",
                    }
                )
                trace.extend(deterministic_trace)
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

        return {
            "graph": graph,
            "skill_plan": plan_payload,
            "execution_trace": trace,
            "composer_model": self.config.model,
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
        for schema in clip_schemas:
            clip_id = schema.get("clip_id")
            clip_ref = clip_id if clip_id in clip_ref_by_id else next(iter(clip_ref_by_id), clip_id)
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

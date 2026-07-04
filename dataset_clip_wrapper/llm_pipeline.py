"""Two-stage LLM pipeline: Qwen clip schemas + gpt-oss graph composition."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from .adapters.base import RawDatasetItem
from .adapters import get_adapter
from .clip_policy import segment_coarse_index, segment_perception_clips, segment_video
from .clip_retrieval import retrieve_coarse_clips
from .clip_schema import QwenClipSchemaProducer
from .graph_composer import GraphComposer
from .openrouter_client import OpenRouterClient, load_openrouter_api_key
from .clue_memory import extract_clue_memory_graph
from .pipeline import _clip_id, build_canonical_example
from .reasoning_rollout import build_reasoning_rollout
from .schemas import ClipPolicyConfig, ClipSpan, RuntimeMode, WrapperConfig
from .video_tool_backend import VideoToolConfig, VideoToolPerceptionBackend


def _subtitle_context_for_clip(segments: list[dict[str, Any]], clip_span: dict[str, float]) -> str:
    start_s = clip_span["start_s"]
    end_s = clip_span["end_s"]
    texts: list[str] = []
    for seg in segments:
        span = seg.get("time_span")
        if not span:
            continue
        if span["end_s"] < start_s or span["start_s"] > end_s:
            continue
        text = seg.get("text")
        if text:
            texts.append(text)
    return " | ".join(texts)


def _derived_clips_for_spans(
    *,
    video_id: str,
    primary_path: str,
    spans: list[ClipSpan],
) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": _clip_id(video_id, span.clip_index, span.granularity),
            "path": primary_path,
            "source_span": span.to_dict(),
            "granularity": span.granularity,
            "parent_index": span.parent_index,
        }
        for span in spans
    ]


def _clip_ref_node(*, video_id: str, primary_path: str, span: ClipSpan, level: str) -> dict[str, Any]:
    return {
        "node_id": _clip_id(video_id, span.clip_index, span.granularity),
        "node_type": "clip",
        "video_id": video_id,
        "media_ref": {"path": primary_path, "video_id": video_id},
        "granularity": span.granularity,
        "level": level,
        "parent_index": span.parent_index,
        "time_span": span.to_dict(),
        "provenance": {"created_by": "dataset_clip_wrapper.coarse_fine_reference"},
    }


def _temporal_edges(nodes: list[dict[str, Any]], *, level: str) -> list[dict[str, Any]]:
    return [
        {
            "edge_id": f"edge:{left['node_id']}->{right['node_id']}:temporal_next",
            "src": left["node_id"],
            "dst": right["node_id"],
            "edge_type": "temporal_next",
            "level": level,
        }
        for left, right in zip(nodes, nodes[1:])
    ]


def _parse_time_anchors_s(text: str) -> list[float]:
    anchors: list[float] = []
    for minutes, seconds in re.findall(r"\b(\d{1,2}):(\d{2})\b", text):
        anchors.append(float(int(minutes) * 60 + int(seconds)))
    for value in re.findall(r"\b(?:at|around|near|after|before)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text, re.I):
        anchors.append(float(value))
    return list(dict.fromkeys(anchors))


def _coarse_indices_for_time_anchors(coarse_spans: list[ClipSpan], anchors_s: list[float]) -> list[int]:
    selected: list[int] = []
    for anchor in anchors_s:
        for index, span in enumerate(coarse_spans):
            if span.start_s <= anchor <= span.end_s:
                selected.append(index)
                if index > 0:
                    selected.append(index - 1)
                if index + 1 < len(coarse_spans):
                    selected.append(index + 1)
                break
    return list(dict.fromkeys(selected))


def _schema_text(schema: dict[str, Any]) -> str:
    if schema.get("model_error"):
        return ""
    parts: list[str] = []
    for key in ("scene_description", "uncertainty"):
        value = schema.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("observable_facts", "dialogue_spans", "entity_mentions", "events", "salient_objects", "cross_clip_cues"):
        value = schema.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                for field in ("text", "description", "surface_form", "cue_type"):
                    field_value = item.get(field)
                    if isinstance(field_value, str) and field_value.strip():
                        parts.append(field_value.strip())
                for field in ("attributes", "searchable_phrases"):
                    field_value = item.get(field)
                    if isinstance(field_value, list):
                        parts.extend(str(part).strip() for part in field_value if str(part).strip())
    place = schema.get("place")
    if isinstance(place, dict):
        for field in ("description", "searchable_phrases"):
            value = place.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, list):
                parts.extend(str(part).strip() for part in value if str(part).strip())
    phrases = schema.get("searchable_phrases")
    if isinstance(phrases, list):
        parts.extend(str(part).strip() for part in phrases if str(part).strip())
    return " ".join(dict.fromkeys(parts))


def _spans_overlap(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not left or not right:
        return False
    return not (float(left["end_s"]) < float(right["start_s"]) or float(right["end_s"]) < float(left["start_s"]))


def _coarse_to_fine_links(
    *,
    video_id: str,
    coarse_spans: list[ClipSpan],
    fine_spans: list[ClipSpan],
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for fine in fine_spans:
        if fine.parent_index is None or fine.parent_index >= len(coarse_spans):
            continue
        coarse = coarse_spans[fine.parent_index]
        fine_id = _clip_id(video_id, fine.clip_index, fine.granularity)
        coarse_id = _clip_id(video_id, coarse.clip_index, coarse.granularity)
        links.append(
            {
                "edge_id": f"edge:{fine_id}->{coarse_id}:refines",
                "src": fine_id,
                "dst": coarse_id,
                "edge_type": "refines",
                "coarse_index": fine.parent_index,
            }
        )
    return links


def _build_coarse_fine_reference_graph(
    *,
    video_id: str,
    primary_path: str,
    duration_s: float,
    clip_policy: ClipPolicyConfig,
    regime,
    perception_spans: list[ClipSpan],
    perception_meta: dict[str, Any],
    clip_schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose the video-reference layer: full coarse coverage + retrieved fine refs."""
    if clip_policy.strategy == "hierarchical":
        coarse_spans = segment_coarse_index(duration_s, clip_policy, regime=regime)
    else:
        coarse_spans = []
    fine_spans = [span for span in perception_spans if span.granularity == "fine"]
    if not fine_spans:
        fine_spans = perception_spans

    coarse_nodes = [
        _clip_ref_node(video_id=video_id, primary_path=primary_path, span=span, level="coarse")
        for span in coarse_spans
    ]
    fine_nodes = [
        _clip_ref_node(video_id=video_id, primary_path=primary_path, span=span, level="fine")
        for span in fine_spans
    ]
    schema_by_clip = {schema.get("clip_id"): schema for schema in clip_schemas}
    fine_schema_nodes = []
    for node in fine_nodes:
        schema = schema_by_clip.get(node["node_id"])
        if not schema or schema.get("model_error"):
            continue
        text = schema.get("scene_description")
        if not isinstance(text, str) or not text.strip():
            continue
        fine_schema_nodes.append(
            {
                "node_id": f"obs:fine:{node['node_id']}",
                "node_type": "observation",
                "level": "fine",
                "source_ids": [node["node_id"]],
                "time_span": node.get("time_span"),
                "text": text,
                "producer": schema.get("producer"),
                "model": schema.get("model"),
                "provenance": {"created_by": "dataset_clip_wrapper.clip_schema"},
            }
        )
    fine_schema_edges = [
        {
            "edge_id": f"edge:{obs['node_id']}->{obs['source_ids'][0]}:derived_from",
            "src": obs["node_id"],
            "dst": obs["source_ids"][0],
            "edge_type": "derived_from",
            "level": "fine",
        }
        for obs in fine_schema_nodes
    ]
    coarse_summary_nodes: list[dict[str, Any]] = []
    coarse_summary_edges: list[dict[str, Any]] = []
    for coarse_node in coarse_nodes:
        overlapping = [
            schema
            for schema in clip_schemas
            if _spans_overlap(schema.get("time_span"), coarse_node.get("time_span"))
        ]
        summary_text = " ".join(_schema_text(schema) for schema in overlapping).strip()
        if not summary_text:
            span = coarse_node["time_span"]
            summary_text = (
                f"Unexpanded coarse clip reference from {span['start_s']:.2f}s to {span['end_s']:.2f}s; "
                "fine perception is not available until this neighborhood is retrieved."
            )
        summary = {
            "node_id": f"summary:coarse:{coarse_node['node_id']}",
            "node_type": "coarse_summary",
            "level": "coarse",
            "source_ids": [coarse_node["node_id"]],
            "time_span": coarse_node.get("time_span"),
            "text": summary_text,
            "expanded": bool(overlapping),
            "provenance": {"created_by": "dataset_clip_wrapper.coarse_summary"},
        }
        coarse_summary_nodes.append(summary)
        coarse_summary_edges.append(
            {
                "edge_id": f"edge:{summary['node_id']}->{coarse_node['node_id']}:derived_from",
                "src": summary["node_id"],
                "dst": coarse_node["node_id"],
                "edge_type": "derived_from",
                "level": "coarse",
            }
        )

    links = _coarse_to_fine_links(
        video_id=video_id,
        coarse_spans=coarse_spans,
        fine_spans=fine_spans,
    )
    retrieval = perception_meta.get("retrieval") or {}
    return {
        "schema_version": "video-skills-relaunch/coarse-fine-reference/v0.1",
        "purpose": "video_clip_reference_layer",
        "video_id": video_id,
        "duration_s": duration_s,
        "strategy": clip_policy.strategy,
        "coarse_graph": {
            "coverage": "full_video" if coarse_nodes else "not_applicable_for_short_video",
            "nodes": coarse_nodes + coarse_summary_nodes,
            "edges": _temporal_edges(coarse_nodes, level="coarse") + coarse_summary_edges,
        },
        "fine_graph": {
            "coverage": "retrieved_neighborhoods" if coarse_nodes else "full_video",
            "nodes": fine_nodes + fine_schema_nodes,
            "edges": _temporal_edges(fine_nodes, level="fine") + fine_schema_edges,
        },
        "coarse_to_fine_links": links,
        "retrieval": retrieval,
        "selected_coarse_indices": retrieval.get("selected_coarse_indices", []),
        "counts": {
            "coarse_nodes": len(coarse_nodes),
            "coarse_summary_nodes": len(coarse_summary_nodes),
            "expanded_coarse_summary_nodes": sum(1 for node in coarse_summary_nodes if node["expanded"]),
            "fine_clip_nodes": len(fine_nodes),
            "fine_observation_nodes": len(fine_schema_nodes),
            "coarse_to_fine_links": len(links),
        },
    }


def _resolve_perception_spans(
    *,
    duration_s: float,
    clip_policy: ClipPolicyConfig,
    regime,
    retrieval_config,
    question_text: str,
    visible_segments: list[dict[str, Any]],
    mode: RuntimeMode,
) -> tuple[list[ClipSpan], dict[str, Any]]:
    """Select fine perception clips; long video uses retrieve-gated coarse → fine."""
    meta: dict[str, Any] = {}
    retrieval_query = question_text if mode == RuntimeMode.EXPERT_DEMO or retrieval_config.query_in_video_only else ""

    if clip_policy.strategy == "hierarchical" and clip_policy.index_fine_expansion == "retrieval_gated":
        coarse = segment_coarse_index(duration_s, clip_policy, regime=regime)
        meta["coarse_index_count"] = len(coarse)
        time_anchors_s = _parse_time_anchors_s(question_text) if retrieval_config.expand_time_anchors else []
        time_anchor_indices = _coarse_indices_for_time_anchors(coarse, time_anchors_s)

        if retrieval_config.enabled:
            retrieval = retrieve_coarse_clips(
                coarse_spans=coarse,
                query_text=retrieval_query,
                segments=visible_segments,
                topk=retrieval_config.topk,
                threshold=retrieval_config.threshold,
                observation_end_s=clip_policy.observation_end_s,
                mode=retrieval_config.mode if retrieval_query else "sequential",
            )
            selected = retrieval["selected_coarse_indices"]
            if time_anchor_indices:
                selected = list(dict.fromkeys(selected + time_anchor_indices))[: max(retrieval_config.topk, len(time_anchor_indices))]
                retrieval["selected_coarse_indices"] = selected
                retrieval["time_anchor_seconds"] = time_anchors_s
                retrieval["time_anchor_coarse_indices"] = time_anchor_indices
            meta["retrieval"] = retrieval
        else:
            selected = list(range(min(retrieval_config.topk, len(coarse))))
            if time_anchor_indices:
                selected = list(dict.fromkeys(selected + time_anchor_indices))
            meta["retrieval"] = {"enabled": False, "selected_coarse_indices": selected}
            if time_anchor_indices:
                meta["retrieval"]["time_anchor_seconds"] = time_anchors_s
                meta["retrieval"]["time_anchor_coarse_indices"] = time_anchor_indices

        perception = segment_perception_clips(
            duration_s,
            clip_policy,
            regime=regime,
            selected_coarse_indices=selected,
        )
        fine_spans = [span for span in perception if span.granularity == "fine"]
        meta["perception_clip_count"] = len(fine_spans)
        return fine_spans, meta

    spans = segment_video(duration_s, clip_policy, regime=regime, fine_expansion="all")
    fine_spans = [span for span in spans if span.granularity == "fine"]
    perception = fine_spans if fine_spans else spans
    meta["perception_clip_count"] = len(perception)
    return perception, meta


def _produce_clip_schemas(
    *,
    item: RawDatasetItem,
    config: WrapperConfig,
    spans: list[ClipSpan],
    derived_clips: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.clip_schema.backend == "video_tools":
        producer = VideoToolPerceptionBackend(
            VideoToolConfig(request_frames=config.clip_schema.request_frames)
        )
        question_context = item.question.get("question_text") if config.mode == RuntimeMode.EXPERT_DEMO else None
        schemas: list[dict[str, Any]] = []
        budget = config.clip_schema.max_clips
        for i, (clip, derived) in enumerate(zip(spans, derived_clips)):
            if budget is not None and i >= budget:
                break
            schema = producer.build_clip_schema(
                clip_id=derived["clip_id"],
                clip=clip,
                video_path=item.video_path,
                subtitle_context=_subtitle_context_for_clip(visible_segments, clip.to_dict()),
                question_context=question_context,
            )
            schemas.append(schema)
        return schemas

    keys_py = config.clip_schema.keys_py_path or config.backbone.keys_py_path
    api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.clip_schema.api_key_env)
    client = OpenRouterClient(
        model=config.clip_schema.model,
        api_key=api_key,
        api_base=config.clip_schema.api_base,
        temperature=config.clip_schema.temperature,
        max_tokens=config.clip_schema.max_tokens,
        reasoning={"effort": config.clip_schema.reasoning_effort, "exclude": True}
        if config.clip_schema.reasoning_effort
        else None,
        timeout_s=config.clip_schema.timeout_s,
    )
    producer = QwenClipSchemaProducer(config.clip_schema, client)
    question_context = item.question.get("question_text") if config.mode == RuntimeMode.EXPERT_DEMO else None
    schemas: list[dict[str, Any]] = []
    budget = config.clip_schema.max_clips
    for i, (clip, derived) in enumerate(zip(spans, derived_clips)):
        if budget is not None and i >= budget:
            break
        if not item.video_path or not item.video_path.exists():
            continue
        schema = producer.build_clip_schema(
            clip_id=derived["clip_id"],
            clip=clip,
            video_path=item.video_path,
            subtitle_context=_subtitle_context_for_clip(visible_segments, clip.to_dict()),
            question_context=question_context,
        )
        schemas.append(schema)
    return schemas


def _build_skill_executor(api_key: str, config: WrapperConfig):
    """Create a SkillExecutor for LLM/VLM-backed skill dispatch.

    Model allocation (teacher-student architecture):
    - Teacher mode (expert_demo): gpt-oss-120b generates expert trajectories
      via GraphComposerConfig.model for L1/L2 planning. Skill execution uses
      Qwen3.5-9B (cheaper, faster, target student model).
    - Student mode (inference after distillation): Qwen3.5-9B does everything.
      The planner model would also be switched to qwen3.5-9b.

    Currently both modes use qwen3.5-9b for skill execution, since even in
    teacher mode the planner (gpt-oss) only generates the plan — execution
    is done by the same model that will be deployed.
    """
    from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
    from atomic_skills.skill_model_client import SkillModelClient
    from atomic_skills.skill_executor import SkillExecutor

    skill_cfg = config.skill_execution

    llm_client = None
    vlm_client = None

    if skill_cfg.enable_llm_skills:
        llm_client = SkillModelClient(
            model=skill_cfg.skill_model,
            api_key=api_key,
            api_base=skill_cfg.skill_api_base,
            max_tokens=skill_cfg.skill_max_tokens_llm,
            temperature=skill_cfg.skill_temperature,
            timeout_s=skill_cfg.skill_timeout_s,
        )

    if skill_cfg.enable_vlm_skills:
        vlm_client = SkillModelClient(
            model=skill_cfg.skill_model,
            api_key=api_key,
            api_base=skill_cfg.skill_api_base,
            max_tokens=skill_cfg.skill_max_tokens_vlm,
            temperature=skill_cfg.skill_temperature,
            timeout_s=skill_cfg.skill_timeout_s,
        )

    backend_config = SkillBackendConfig(default_mode=SkillBackendMode.LLM)
    return SkillExecutor(llm_client=llm_client, vlm_client=vlm_client, config=backend_config)


def build_llm_enriched_example(
    item: RawDatasetItem,
    *,
    config: WrapperConfig,
) -> dict[str, Any]:
    example = build_canonical_example(
        item,
        config=config,
        backbone=None,
    )
    if not (config.run_clip_schema or config.run_graph_compose):
        return example

    duration_s = float(example["video"].get("duration_s") or 0.0)
    clip_policy = config.resolved_clip_policy(duration_s)
    visible_segments = example["video"]["segments"]
    question_text = item.question.get("question_text") or ""

    perception_spans, perception_meta = _resolve_perception_spans(
        duration_s=duration_s,
        clip_policy=clip_policy,
        regime=config.regime,
        retrieval_config=config.retrieval,
        question_text=question_text,
        visible_segments=visible_segments,
        mode=config.mode,
    )
    primary_path = str(item.video_path) if item.video_path else ""
    perception_derived = _derived_clips_for_spans(
        video_id=item.video_id,
        primary_path=primary_path,
        spans=perception_spans,
    )
    example["metadata"]["perception"] = perception_meta

    clip_schemas: list[dict[str, Any]] = []
    if config.run_clip_schema:
        clip_schemas = _produce_clip_schemas(
            item=item,
            config=config,
            spans=perception_spans,
            derived_clips=perception_derived,
            visible_segments=visible_segments,
        )
        example["metadata"]["clip_schemas"] = clip_schemas
        example["metadata"]["clip_schema_model"] = (
            config.clip_schema.model if config.clip_schema.backend == "qwen" else "local-video-tools"
        )
        example["metadata"]["clip_schema_backend"] = config.clip_schema.backend

    example["metadata"]["coarse_fine_graph"] = _build_coarse_fine_reference_graph(
        video_id=item.video_id,
        primary_path=primary_path,
        duration_s=duration_s,
        clip_policy=clip_policy,
        regime=config.regime,
        perception_spans=perception_spans,
        perception_meta=perception_meta,
        clip_schemas=clip_schemas,
    )

    if config.run_graph_compose:
        api_key: str | None = None
        if config.graph_composer.use_llm_planner or config.run_l2_llm_planner:
            keys_py = config.graph_composer.keys_py_path or config.backbone.keys_py_path
            api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.graph_composer.api_key_env)
            client = OpenRouterClient(
                model=config.graph_composer.model,
                api_key=api_key,
                api_base=config.graph_composer.api_base,
                temperature=config.graph_composer.temperature,
                max_tokens=config.graph_composer.max_tokens,
                reasoning={"effort": config.graph_composer.reasoning_effort, "exclude": True}
                if config.graph_composer.reasoning_effort
                else None,
            )
        else:
            client = OpenRouterClient(model="offline", api_key="offline")
        composer = GraphComposer(config.graph_composer, client)
        composed = composer.compose_from_clip_schemas(
            example_id=example["example_id"],
            video_id=item.video_id,
            clip_policy=clip_policy.to_dict(),
            clip_schemas=clip_schemas,
            segments=visible_segments,
            mode=config.mode,
            duration_s=duration_s,
            observation_end_s=clip_policy.observation_end_s,
        )
        graph = composed["graph"]
        example["evidence_index"]["nodes"] = graph.get("nodes", [])
        example["evidence_index"]["edges"] = graph.get("edges", [])
        example["evidence_index"]["graph_composer"] = config.graph_composer.to_dict()
        example["evidence_index"]["retrieval"] = config.retrieval.to_dict()
        example["metadata"]["graph_compose"] = {
            "composer_model": composed.get("composer_model"),
            "execution_trace": composed.get("execution_trace"),
            "skill_plan": composed.get("skill_plan"),
        }

        for node in graph.get("nodes", []):
            if node.get("node_type") != "observation" or not node.get("text"):
                continue
            example["evidence_candidates"].append(
                {
                    "evidence_id": f"ev:{node['node_id']}",
                    "source_type": "caption_span",
                    "time_span": node.get("time_span"),
                    "text": node.get("text"),
                    "trust_level": "model_labeled",
                    "provenance": {
                        "created_by": "dataset_clip_wrapper.graph_composer",
                        "composer_model": config.graph_composer.model,
                    },
                    "discovery_status": "discovered_runtime",
                }
            )

        clue_graph = extract_clue_memory_graph(example, mode=config.mode)
        example["metadata"]["clue_memory_graph"] = clue_graph

        if config.run_l2_llm_planner:
            from .reasoning_planner import build_llm_reasoning_rollout
            if api_key is None:
                keys_py = config.graph_composer.keys_py_path or config.backbone.keys_py_path
                api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.graph_composer.api_key_env)
            l2_client = OpenRouterClient(
                model=config.graph_composer.model if config.graph_composer else "openai/gpt-oss-120b",
                api_key=api_key,
                max_tokens=1800,
                reasoning={"effort": "minimal", "exclude": True},
            )
            skill_exec = _build_skill_executor(api_key, config) if config.run_l2_llm_planner else None
            reasoning_rollout = build_llm_reasoning_rollout(example, clue_graph, client=l2_client, skill_executor=skill_exec)
        else:
            reasoning_rollout = build_reasoning_rollout(example, clue_graph, rollout_source="llm_pipeline")
        example["metadata"]["reasoning_rollout"] = reasoning_rollout
        example["metadata"]["reasoning_rollout_shell"] = reasoning_rollout

    example["metadata"]["llm_pipeline"] = {
        "clip_schema": config.clip_schema.to_dict() if config.run_clip_schema else None,
        "graph_composer": config.graph_composer.to_dict() if config.run_graph_compose else None,
        "retrieval": config.retrieval.to_dict(),
    }
    return example


def iter_llm_enriched_examples(config: WrapperConfig) -> Iterator[dict[str, Any]]:
    adapter = get_adapter(config.dataset, Path(config.dataset_root), split=config.split)
    for item in adapter.iter_items(limit=config.limit):
        yield build_llm_enriched_example(item, config=config)

"""Two-stage LLM pipeline: Qwen clip schemas + gpt-oss graph composition."""

from __future__ import annotations

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
    retrieval_query = question_text if mode == RuntimeMode.EXPERT_DEMO else ""

    if clip_policy.strategy == "hierarchical" and clip_policy.index_fine_expansion == "retrieval_gated":
        coarse = segment_coarse_index(duration_s, clip_policy, regime=regime)
        meta["coarse_index_count"] = len(coarse)

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
            meta["retrieval"] = retrieval
        else:
            selected = list(range(min(retrieval_config.topk, len(coarse))))
            meta["retrieval"] = {"enabled": False, "selected_coarse_indices": selected}

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

    if config.run_graph_compose:
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
            l2_client = OpenRouterClient(
                model=config.graph_composer.model if config.graph_composer else "openai/gpt-oss-120b",
                api_key=api_key,
                max_tokens=1800,
                reasoning={"effort": "minimal", "exclude": True},
            )
            reasoning_rollout = build_llm_reasoning_rollout(example, clue_graph, client=l2_client)
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

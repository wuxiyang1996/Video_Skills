"""Two-stage LLM pipeline: Qwen clip schemas + gpt-oss graph composition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .adapters.base import RawDatasetItem
from .adapters import get_adapter
from .clip_policy import segment_video
from .clip_schema import QwenClipSchemaProducer
from .graph_composer import GraphComposer
from .openrouter_client import OpenRouterClient, load_openrouter_api_key
from .pipeline import build_canonical_example


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


def _produce_clip_schemas(
    *,
    item: RawDatasetItem,
    config: WrapperConfig,
    spans,
    derived_clips: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys_py = config.clip_schema.keys_py_path or config.backbone.keys_py_path
    api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.clip_schema.api_key_env)
    client = OpenRouterClient(
        model=config.clip_schema.model,
        api_key=api_key,
        api_base=config.clip_schema.api_base,
        temperature=config.clip_schema.temperature,
    )
    producer = QwenClipSchemaProducer(config.clip_schema, client)
    question_context = item.question.get("question_text")
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
    spans = segment_video(duration_s, clip_policy, regime=config.regime)
    derived_clips = example["video"]["derived_clips"]
    visible_segments = example["video"]["segments"]

    clip_schemas: list[dict[str, Any]] = []
    if config.run_clip_schema:
        clip_schemas = _produce_clip_schemas(
            item=item,
            config=config,
            spans=spans,
            derived_clips=derived_clips,
            visible_segments=visible_segments,
        )
        example["metadata"]["clip_schemas"] = clip_schemas
        example["metadata"]["clip_schema_model"] = config.clip_schema.model

    if config.run_graph_compose:
        keys_py = config.graph_composer.keys_py_path or config.backbone.keys_py_path
        api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.graph_composer.api_key_env)
        client = OpenRouterClient(
            model=config.graph_composer.model,
            api_key=api_key,
            api_base=config.graph_composer.api_base,
            temperature=config.graph_composer.temperature,
        )
        composer = GraphComposer(config.graph_composer, client)
        composed = composer.compose_from_clip_schemas(
            example_id=example["example_id"],
            video_id=item.video_id,
            clip_policy=clip_policy.to_dict(),
            clip_schemas=clip_schemas,
            segments=visible_segments,
            question=item.question,
            mode=config.mode,
            duration_s=duration_s,
            observation_end_s=clip_policy.observation_end_s,
        )
        graph = composed["graph"]
        example["evidence_index"]["nodes"] = graph.get("nodes", [])
        example["evidence_index"]["edges"] = graph.get("edges", [])
        example["evidence_index"]["graph_composer"] = config.graph_composer.to_dict()
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

    example["metadata"]["llm_pipeline"] = {
        "clip_schema": config.clip_schema.to_dict() if config.run_clip_schema else None,
        "graph_composer": config.graph_composer.to_dict() if config.run_graph_compose else None,
    }
    return example


def iter_llm_enriched_examples(config: WrapperConfig) -> Iterator[dict[str, Any]]:
    adapter = get_adapter(config.dataset, Path(config.dataset_root), split=config.split)
    for item in adapter.iter_items(limit=config.limit):
        yield build_llm_enriched_example(item, config=config)

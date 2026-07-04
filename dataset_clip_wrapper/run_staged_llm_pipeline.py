#!/usr/bin/env python3
"""Staged/resumable runner for Qwen L1 and GPT-OSS L2 graph creation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .clip_schema import QwenClipSchemaProducer
from .clue_memory import extract_clue_memory_graph
from .graph_composer import GraphComposer
from .llm_pipeline import (
    _build_coarse_fine_reference_graph,
    _build_skill_executor,
    _derived_clips_for_spans,
    _resolve_perception_spans,
    _subtitle_context_for_clip,
)
from .openrouter_client import OpenRouterClient, load_openrouter_api_key
from .pipeline import build_canonical_example
from .reasoning_rollout import build_reasoning_rollout
from .schemas import (
    BackboneConfig,
    ClipPolicyConfig,
    ClipRetrievalConfig,
    ClipSchemaConfig,
    GraphComposerConfig,
    RuntimeMode,
    SkillExecutionConfig,
    VideoRegime,
    WrapperConfig,
)
from .video_tool_backend import VideoToolConfig, VideoToolPerceptionBackend


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a staged/resumable Qwen + GPT-OSS graph pipeline.")
    parser.add_argument("--dataset", required=True, choices=["video_holmes", "cg_bench", "vrbench", "siv_bench"])
    parser.add_argument("--dataset-root", default="/fs/gamma-projects/vlm-robot/datasets")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--regime", default=None, choices=["short", "long", "streaming"])
    parser.add_argument("--mode", default="video_only", choices=["expert_demo", "video_only"])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", default="dataset_clip_wrapper/output/staged_llm_pipeline.jsonl")
    parser.add_argument("--stage-dir", default="dataset_clip_wrapper/output/staged")
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--force", action="store_true", help="Ignore cached stage files and rebuild.")
    parser.add_argument(
        "--rebuild-from-stages",
        action="store_true",
        help="Ignore cached final_example.json but reuse cached clip schemas without retrying model_error rows.",
    )
    parser.add_argument(
        "--no-fill-missing-clip-schemas",
        action="store_true",
        help="When rebuilding from stages, do not call the clip-schema backend for missing clip ids.",
    )
    parser.add_argument(
        "--retry-failed-clip-schemas",
        action="store_true",
        help="Reuse good cached clip schemas but discard cached model_error rows and retry those clips.",
    )

    parser.add_argument("--clip-schema-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--clip-schema-backend", default="qwen", choices=["qwen", "video_tools"])
    parser.add_argument("--clip-schema-max-clips", type=int, default=999)
    parser.add_argument("--clip-schema-frames", type=int, default=1)
    parser.add_argument("--clip-schema-max-tokens", type=int, default=700)
    parser.add_argument("--clip-schema-timeout-s", type=int, default=45)

    parser.add_argument("--graph-model", default="openai/gpt-oss-120b")
    parser.add_argument("--graph-max-tokens", type=int, default=3500)
    parser.add_argument("--graph-composer-mode", default="vlm_l1", choices=["vlm_l1", "skill_plan", "deterministic"])
    parser.add_argument("--graph-deterministic", action="store_true")
    parser.add_argument("--skip-l2-planner", action="store_true")
    parser.add_argument("--disable-llm-skills", action="store_true")
    parser.add_argument("--disable-vlm-skills", action="store_true")

    parser.add_argument("--retrieval-topk", type=int, default=2)
    parser.add_argument("--retrieval-mode", default="lexical", choices=["lexical", "sequential"])
    parser.add_argument("--query-time-retrieval", action="store_true")
    parser.add_argument("--no-time-anchor-expansion", action="store_true")
    parser.add_argument("--index-fine-expansion", default=None, choices=["none", "all", "retrieval_gated"])
    return parser


def _config_from_args(args: argparse.Namespace) -> WrapperConfig:
    regime = VideoRegime(args.regime) if args.regime else None
    dataset_regime = regime or {
        "video_holmes": VideoRegime.SHORT,
        "siv_bench": VideoRegime.SHORT,
        "cg_bench": VideoRegime.LONG,
        "vrbench": VideoRegime.LONG,
    }[args.dataset]

    clip_policy = ClipPolicyConfig.dataset_default(args.dataset, dataset_regime)
    if args.index_fine_expansion:
        clip_policy.index_fine_expansion = args.index_fine_expansion  # type: ignore[assignment]

    return WrapperConfig(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        regime=dataset_regime,
        mode=RuntimeMode(args.mode),
        clip_policy=clip_policy,
        retrieval=ClipRetrievalConfig(
            topk=args.retrieval_topk,
            mode=args.retrieval_mode,  # type: ignore[arg-type]
            query_in_video_only=args.query_time_retrieval,
            expand_time_anchors=not args.no_time_anchor_expansion,
        ),
        backbone=BackboneConfig(keys_py_path=args.keys_py),
        clip_schema=ClipSchemaConfig(
            backend=args.clip_schema_backend,
            model=args.clip_schema_model,
            keys_py_path=args.keys_py,
            max_clips=args.clip_schema_max_clips,
            request_frames=args.clip_schema_frames,
            max_tokens=args.clip_schema_max_tokens,
            timeout_s=args.clip_schema_timeout_s,
        ),
        graph_composer=GraphComposerConfig(
            model=args.graph_model,
            keys_py_path=args.keys_py,
            use_llm_planner=not args.graph_deterministic,
            composer_mode="deterministic" if args.graph_deterministic else args.graph_composer_mode,
            max_tokens=args.graph_max_tokens,
        ),
        skill_execution=SkillExecutionConfig(
            skill_model=args.clip_schema_model,
            enable_llm_skills=not args.disable_llm_skills,
            enable_vlm_skills=not args.disable_vlm_skills,
        ),
        split=args.split,
        limit=args.limit,
        run_clip_schema=True,
        run_graph_compose=True,
        run_l2_llm_planner=not args.skip_l2_planner,
    )


def _clip_schema_producer(config: WrapperConfig) -> Any:
    if config.clip_schema.backend == "video_tools":
        return VideoToolPerceptionBackend(VideoToolConfig(request_frames=config.clip_schema.request_frames))
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
    return QwenClipSchemaProducer(config.clip_schema, client)


def _produce_or_resume_clip_schemas(
    *,
    item: Any,
    config: WrapperConfig,
    spans: list[Any],
    derived_clips: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
    stage_path: Path,
    force: bool,
    retry_failed: bool,
    fill_missing: bool,
) -> list[dict[str, Any]]:
    if force and stage_path.exists():
        stage_path.unlink()

    cached = _read_jsonl(stage_path)
    if retry_failed and cached:
        cached = [row for row in cached if not row.get("model_error")]
        _write_jsonl(stage_path, cached)
    by_clip_id = {row.get("clip_id"): row for row in cached if row.get("clip_id")}
    producer = _clip_schema_producer(config)
    budget = config.clip_schema.max_clips

    for index, (span, derived) in enumerate(zip(spans, derived_clips)):
        if budget is not None and index >= budget:
            break
        clip_id = derived["clip_id"]
        if clip_id in by_clip_id:
            continue
        if not fill_missing:
            continue
        schema = producer.build_clip_schema(
            clip_id=clip_id,
            clip=span,
            video_path=item.video_path,
            subtitle_context=_subtitle_context_for_clip(visible_segments, span.to_dict()),
            question_context=item.question.get("question_text") if config.mode == RuntimeMode.EXPERT_DEMO else None,
        )
        _append_jsonl(stage_path, schema)
        by_clip_id[clip_id] = schema

    return [by_clip_id[derived["clip_id"]] for derived in derived_clips if derived["clip_id"] in by_clip_id]


def _compose_l1_and_l2(
    *,
    example: dict[str, Any],
    item: Any,
    config: WrapperConfig,
    clip_policy: ClipPolicyConfig,
    clip_schemas: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
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
            reasoning={"effort": config.graph_composer.reasoning_effort, "exclude": True},
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
        "composer_mode": composed.get("composer_mode"),
        "used_deterministic_fallback": composed.get("used_deterministic_fallback"),
        "execution_trace": composed.get("execution_trace"),
        "skill_plan": composed.get("skill_plan"),
    }

    for node in graph.get("nodes", []):
        if node.get("node_type") == "observation" and node.get("text"):
            example["evidence_candidates"].append(
                {
                    "evidence_id": f"ev:{node['node_id']}",
                    "source_type": "caption_span",
                    "time_span": node.get("time_span"),
                    "text": node.get("text"),
                    "trust_level": "model_labeled",
                    "provenance": {"created_by": "dataset_clip_wrapper.staged_graph_composer"},
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
            model=config.graph_composer.model,
            api_key=api_key,
            max_tokens=1800,
            reasoning={"effort": "minimal", "exclude": True},
        )
        skill_exec = _build_skill_executor(api_key, config)
        rollout = build_llm_reasoning_rollout(example, clue_graph, client=l2_client, skill_executor=skill_exec)
    else:
        rollout = build_reasoning_rollout(example, clue_graph, rollout_source="staged_llm_pipeline")
    example["metadata"]["reasoning_rollout"] = rollout
    example["metadata"]["reasoning_rollout_shell"] = rollout
    return example


def _run_item(
    item: Any,
    *,
    config: WrapperConfig,
    root_stage_dir: Path,
    force: bool,
    rebuild_from_stages: bool,
    retry_failed_clip_schemas: bool,
    fill_missing_clip_schemas: bool,
) -> dict[str, Any]:
    example_stage_dir = root_stage_dir / _safe_name(item.example_id)
    example_stage_dir.mkdir(parents=True, exist_ok=True)

    final_path = example_stage_dir / "final_example.json"
    if final_path.exists() and not force and not rebuild_from_stages and not retry_failed_clip_schemas:
        return _read_json(final_path)

    example = build_canonical_example(item, config=config, backbone=None)
    duration_s = float(example["video"].get("duration_s") or 0.0)
    clip_policy = config.resolved_clip_policy(duration_s)
    visible_segments = example["video"]["segments"]
    perception_spans, perception_meta = _resolve_perception_spans(
        duration_s=duration_s,
        clip_policy=clip_policy,
        regime=config.regime,
        retrieval_config=config.retrieval,
        question_text=item.question.get("question_text") or "",
        visible_segments=visible_segments,
        mode=config.mode,
    )
    primary_path = str(item.video_path) if item.video_path else ""
    derived = _derived_clips_for_spans(video_id=item.video_id, primary_path=primary_path, spans=perception_spans)
    example["metadata"]["perception"] = perception_meta
    _write_json(example_stage_dir / "00_shell.json", example)
    _write_json(
        example_stage_dir / "01_perception_spans.json",
        {"perception": perception_meta, "derived_clips": derived},
    )

    clip_schemas = _produce_or_resume_clip_schemas(
        item=item,
        config=config,
        spans=perception_spans,
        derived_clips=derived,
        visible_segments=visible_segments,
        stage_path=example_stage_dir / "02_clip_schemas.jsonl",
        force=force,
        retry_failed=retry_failed_clip_schemas,
        fill_missing=fill_missing_clip_schemas,
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
    _write_json(example_stage_dir / "03_l1_inputs.json", example)

    example = _compose_l1_and_l2(
        example=example,
        item=item,
        config=config,
        clip_policy=clip_policy,
        clip_schemas=clip_schemas,
        visible_segments=visible_segments,
        duration_s=duration_s,
    )
    _write_json(example_stage_dir / "04_l1_example.json", {**example, "metadata": {**example["metadata"], "reasoning_rollout": None}})
    _write_json(example_stage_dir / "05_l2_rollout.json", example["metadata"].get("reasoning_rollout"))
    example["metadata"]["llm_pipeline"] = {
        "clip_schema": config.clip_schema.to_dict(),
        "graph_composer": config.graph_composer.to_dict(),
        "retrieval": config.retrieval.to_dict(),
        "staged_dir": str(example_stage_dir),
    }
    _write_json(final_path, example)
    return example


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    adapter = get_adapter(config.dataset, Path(config.dataset_root), split=config.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output_path.exists():
        output_path.unlink()

    written = 0
    for item in adapter.iter_items(limit=config.limit):
        example = _run_item(
            item,
            config=config,
            root_stage_dir=Path(args.stage_dir),
            force=args.force,
            rebuild_from_stages=args.rebuild_from_stages,
            retry_failed_clip_schemas=args.retry_failed_clip_schemas,
            fill_missing_clip_schemas=not args.no_fill_missing_clip_schemas,
        )
        _append_jsonl(output_path, example)
        written += 1
        print(
            json.dumps(
                {
                    "example_id": example["example_id"],
                    "dataset": example["dataset"],
                    "stage_dir": example["metadata"]["llm_pipeline"]["staged_dir"],
                    "clip_schema_count": len(example["metadata"].get("clip_schemas") or []),
                    "output": str(output_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(json.dumps({"written": written, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

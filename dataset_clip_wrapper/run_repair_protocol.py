#!/usr/bin/env python3
"""Run targeted L1/L2 repair for examples flagged by the quality report.

The repair protocol is deliberately narrow:
1. read the L1/L2 quality report,
2. expand the coarse windows around the reported failure,
3. ask the VLM for focused repair clip schemas,
4. write a non-destructive graph patch, and
5. verify answer claims with the existing atomic verifier.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from atomic_skills.common import stable_id
    from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
    from atomic_skills.skill_executor import SkillExecutor
    from atomic_skills.skill_model_client import SkillModelClient
    from .clip_policy import segment_coarse_index, segment_perception_clips
    from .clip_retrieval import retrieve_coarse_clips
    from .clip_schema import QwenClipSchemaProducer
    from .openrouter_client import OpenRouterClient, load_openrouter_api_key
    from .schemas import ClipPolicyConfig, ClipSchemaConfig, ClipSpan, VideoRegime
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from atomic_skills.common import stable_id
    from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
    from atomic_skills.skill_executor import SkillExecutor
    from atomic_skills.skill_model_client import SkillModelClient
    from dataset_clip_wrapper.clip_policy import segment_coarse_index, segment_perception_clips
    from dataset_clip_wrapper.clip_retrieval import retrieve_coarse_clips
    from dataset_clip_wrapper.clip_schema import QwenClipSchemaProducer
    from dataset_clip_wrapper.openrouter_client import OpenRouterClient, load_openrouter_api_key
    from dataset_clip_wrapper.schemas import ClipPolicyConfig, ClipSchemaConfig, ClipSpan, VideoRegime


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def _load_source_example(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["source_path"]))
    example_id = row.get("example_id")
    for example in _read_jsonl(path):
        if example.get("example_id") == example_id:
            return example
    raise ValueError(f"Could not find example_id={example_id} in {path}")


def _option_text(question: dict[str, Any]) -> str:
    options = question.get("options") or []
    return "\n".join(f"{opt.get('label')}. {opt.get('text')}" for opt in options)


def _gap_types(row: dict[str, Any]) -> list[str]:
    hints = row.get("repair_hints") or {}
    gaps: list[str] = []
    gaps.extend(str(item) for item in hints.get("missing_requirements") or [])
    commonsense = hints.get("commonsense_repair") or {}
    gaps.extend(str(item) for item in commonsense.get("missing_requirements") or [])
    if not gaps and row.get("verifier_reason"):
        gaps.append(str(row.get("verifier_reason")))
    return list(dict.fromkeys(gaps))


def _strategy_for_gaps(gaps: list[str]) -> str:
    if "discriminative_visual_evidence" in gaps:
        return "visual_disambiguation_retrieval"
    if {"social_intent_or_affect", "causal_explanation"} & set(gaps):
        return "social_causal_commonsense_bridge"
    return "evidence_sufficiency_retrieval"


def _expanded_indices(selected: list[int], coarse_count: int, *, radius: int) -> list[int]:
    out: list[int] = []
    for idx in selected:
        for j in range(idx - radius, idx + radius + 1):
            if 0 <= j < coarse_count:
                out.append(j)
    return list(dict.fromkeys(out))


def _repair_query(example: dict[str, Any], row: dict[str, Any], gaps: list[str], clue_spec: dict[str, Any] | None = None) -> str:
    question = example.get("question") or {}
    commonsense = ((row.get("repair_hints") or {}).get("commonsense_repair") or {})
    hypotheses = commonsense.get("top_hypotheses") or []
    hyp_text = "\n".join(
        f"- {h.get('label')}: {h.get('text')} (commonsense_score={h.get('commonsense_score')})"
        for h in hypotheses[:4]
    )
    clue_text = json.dumps(clue_spec or {}, ensure_ascii=False, indent=2)
    return (
        "Visible QA repair context. Do not use hidden answer labels.\n"
        "STRICT VIDEO-ONLY RULES:\n"
        "- Use visual frames only. Ignore audio, ASR, subtitle, narration, and the wording of this prompt as evidence.\n"
        "- Do not copy the question or options into observable_facts.\n"
        "- If the requested target/clue is not visible in the clip, explicitly say visual evidence insufficient.\n"
        "- Negative evidence such as no visible target is useful and should be recorded as visual uncertainty.\n"
        f"Question: {question.get('question_text')}\n"
        f"Options:\n{_option_text(question)}\n"
        f"Current missing requirements: {', '.join(gaps) or 'evidence_sufficiency'}\n"
        f"Candidate commonsense hypotheses to check against visual evidence:\n{hyp_text}\n"
        f"Clue need spec:\n{clue_text}\n"
        "Inspect the clip only for the visual targets, attributes, actions, and exclusion criteria in the clue need spec. "
        "Return grounded visual observations or a clear visual-insufficient note."
    )


def _prior_negative_notes(schemas: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for schema in schemas:
        text = " ".join(_text_items(schema))
        if not text or not _has_negative_target_evidence(text):
            continue
        notes.append(
            {
                "clip_id": schema.get("clip_id"),
                "time_span": schema.get("time_span"),
                "negative_visual_note": text[:500],
            }
        )
        if len(notes) >= limit:
            break
    return notes


def _fallback_clue_need_spec(example: dict[str, Any], row: dict[str, Any], gaps: list[str]) -> dict[str, Any]:
    question = example.get("question") or {}
    options = question.get("options") or []
    return {
        "planner_backend": "fallback_no_api",
        "visual_target": question.get("question_text") or "",
        "must_find_visual_evidence": [question.get("question_text") or ""],
        "visual_attributes_to_resolve": [str(opt.get("text") or "") for opt in options],
        "forbidden_modalities": ["audio", "asr", "subtitle", "dialogue"],
        "positive_evidence_criteria": [
            "The target object/event/action is visible in the frames.",
            "The answer attribute is directly visible or explicitly absent.",
        ],
        "insufficient_evidence_rule": "If the target or attribute is not visible, mark visual evidence insufficient.",
        "coarse_search_queries": _query_variants(example, row, gaps, clue_spec=None),
    }


def _clue_need_spec_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "clue_need_spec",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "visual_target": {"type": "string"},
                    "must_find_visual_evidence": {"type": "array", "items": {"type": "string"}},
                    "visual_attributes_to_resolve": {"type": "array", "items": {"type": "string"}},
                    "temporal_or_action_cues": {"type": "array", "items": {"type": "string"}},
                    "negative_evidence_to_exclude": {"type": "array", "items": {"type": "string"}},
                    "forbidden_modalities": {"type": "array", "items": {"type": "string"}},
                    "positive_evidence_criteria": {"type": "array", "items": {"type": "string"}},
                    "insufficient_evidence_rule": {"type": "string"},
                    "coarse_search_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "role": {"type": "string"},
                                "query": {"type": "string"},
                            },
                            "required": ["role", "query"],
                        },
                    },
                    "clip_inspection_instruction": {"type": "string"},
                },
                "required": [
                    "visual_target",
                    "must_find_visual_evidence",
                    "visual_attributes_to_resolve",
                    "forbidden_modalities",
                    "positive_evidence_criteria",
                    "insufficient_evidence_rule",
                    "coarse_search_queries",
                    "clip_inspection_instruction",
                ],
            },
        },
    }


def _coarse_selector_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "coarse_window_selection",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "selected_coarse_indices": {"type": "array", "items": {"type": "integer"}},
                    "selection_rounds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "role": {"type": "string"},
                                "query_or_need": {"type": "string"},
                                "selected_after_exclusion": {"type": "array", "items": {"type": "integer"}},
                                "reason": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                        },
                    },
                    "missing_clue_diagnosis": {"type": "string"},
                },
                "required": ["selected_coarse_indices", "selection_rounds", "missing_clue_diagnosis"],
            },
        },
    }


def _valid_clue_need_spec(spec: dict[str, Any]) -> bool:
    return bool(
        isinstance(spec, dict)
        and spec.get("visual_target")
        and isinstance(spec.get("coarse_search_queries"), list)
        and any(isinstance(item, dict) and item.get("query") for item in spec.get("coarse_search_queries") or [])
        and spec.get("clip_inspection_instruction")
    )


def _build_clue_need_spec(
    example: dict[str, Any],
    row: dict[str, Any],
    gaps: list[str],
    *,
    prior_schemas: list[dict[str, Any]],
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fallback = _fallback_clue_need_spec(example, row, gaps)
    if not api_key or args.disable_llm_clue_planner or args.dry_run:
        return fallback

    question = example.get("question") or {}
    prompt = {
        "task": "Create a video-only clue seeking specification for long-video repair.",
        "rules": [
            "Use only the visible question/options and prior negative visual notes.",
            "Do not use hidden answer labels or dataset annotations.",
            "The downstream VLM may only use frames; audio, ASR, subtitles, narration, and dialogue are forbidden evidence.",
            "Describe what visual clue must be found and what would count as insufficient evidence.",
            "Generate concrete search queries for coarse visual summaries.",
        ],
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
            "answer_format": question.get("answer_format"),
        },
        "gap_types": gaps,
        "repair_hints": row.get("repair_hints") or {},
        "prior_negative_visual_notes": _prior_negative_notes(prior_schemas),
        "required_json_shape": _clue_need_spec_response_schema()["json_schema"]["schema"]["properties"],
    }
    client = OpenRouterClient(
        model=args.clue_planner_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.clue_planner_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.clue_planner_timeout_s,
    )
    try:
        spec = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You output JSON only. Plan grounded visual clue searches for video-only QA repair. "
                        "Do not include analysis, markdown, or prose outside JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format=_clue_need_spec_response_schema(),
        )
    except Exception as exc:
        if args.allow_lexical_fallback:
            fallback["planner_error"] = str(exc)
            return fallback
        raise RuntimeError(f"LLM clue planner failed and heuristic fallback is disabled: {exc}") from exc
    if not _valid_clue_need_spec(spec):
        if args.allow_lexical_fallback:
            fallback["planner_error"] = "invalid clue_need_spec"
            fallback["invalid_planner_payload"] = spec
            return fallback
        raise RuntimeError(f"LLM clue planner returned invalid clue_need_spec: {json.dumps(spec, ensure_ascii=False)[:800]}")
    spec["planner_backend"] = args.clue_planner_model
    spec.setdefault("forbidden_modalities", ["audio", "asr", "subtitle", "dialogue"])
    return spec


def _coarse_schema_segments(example: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, schema in enumerate(((example.get("metadata") or {}).get("coarse_clip_schemas") or [])):
        if not isinstance(schema, dict):
            continue
        text_items = _text_items(schema)
        if not text_items:
            continue
        time_span = schema.get("time_span") or {}
        segments.append(
            {
                "segment_id": f"repair.coarse_schema:{idx:04d}",
                "source_type": "coarse_visual_summary",
                "time_span": time_span,
                "text": " ".join(text_items),
            }
        )
    return segments


def _query_variants(
    example: dict[str, Any],
    row: dict[str, Any],
    gaps: list[str],
    clue_spec: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if clue_spec:
        planned = []
        for item in clue_spec.get("coarse_search_queries") or []:
            if isinstance(item, dict) and item.get("query"):
                planned.append({"role": str(item.get("role") or "planned_clue_retrieval"), "query": str(item["query"])})
        if planned:
            return planned
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    option_text = " ".join(str(opt.get("text") or "") for opt in question.get("options") or [])
    commonsense = ((row.get("repair_hints") or {}).get("commonsense_repair") or {})
    top_hypotheses = " ".join(str(h.get("text") or "") for h in (commonsense.get("top_hypotheses") or [])[:4])
    variants = [
        {"role": "target_retrieval", "query": question_text},
        {"role": "attribute_retrieval", "query": f"{question_text} {option_text}"},
        {"role": "temporal_context_retrieval", "query": f"before after context event setup payoff {question_text}"},
    ]
    if {"social_intent_or_affect", "causal_explanation"} & set(gaps):
        variants.append(
            {
                "role": "social_causal_bridge_retrieval",
                "query": f"{question_text} intention motive reason social affect causal context {top_hypotheses}",
            }
        )
    if "discriminative_visual_evidence" in gaps:
        variants.append(
            {
                "role": "visual_disambiguation_retrieval",
                "query": f"{question_text} discriminative visual attribute color object moving direction animation {option_text}",
            }
        )
    return [variant for variant in variants if variant["query"].strip()]


def _negative_parent_indices_from_schemas(example: dict[str, Any], schemas: list[dict[str, Any]]) -> list[int]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    coarse = segment_coarse_index(duration_s, policy, regime=VideoRegime.LONG)
    negative_indices: list[int] = []
    for schema in schemas:
        text = " ".join(_text_items(schema)).lower()
        if not text or not _has_negative_target_evidence(text):
            continue
        span = schema.get("time_span") or {}
        try:
            midpoint = (float(span.get("start_s")) + float(span.get("end_s"))) / 2.0
        except (TypeError, ValueError):
            continue
        for idx, coarse_span in enumerate(coarse):
            if coarse_span.start_s <= midpoint <= coarse_span.end_s:
                negative_indices.append(idx)
                break
    return list(dict.fromkeys(negative_indices))


def _coarse_summaries_for_prompt(example: dict[str, Any], *, max_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, schema in enumerate(((example.get("metadata") or {}).get("coarse_clip_schemas") or [])):
        if not isinstance(schema, dict):
            continue
        text = " ".join(_text_items(schema))
        rows.append(
            {
                "coarse_index": idx,
                "time_span": schema.get("time_span") or {},
                "visual_summary": text[:max_chars],
            }
        )
    return rows


def _llm_select_coarse_indices(
    example: dict[str, Any],
    *,
    clue_spec: dict[str, Any],
    negative_indices: set[int],
    api_key: str | None,
    args: argparse.Namespace,
) -> tuple[list[int], list[dict[str, Any]]]:
    if not api_key or args.disable_llm_reroute_selector or args.dry_run:
        return [], []
    question = example.get("question") or {}
    coarse_rows = _coarse_summaries_for_prompt(example, max_chars=args.coarse_summary_prompt_chars)
    if not coarse_rows:
        return [], []
    prompt = {
        "task": "Select coarse video windows to inspect for the needed visual clue.",
        "rules": [
            "Use only the coarse visual summaries as search evidence.",
            "Do not infer from audio, subtitles, hidden labels, or the gold answer.",
            "Do not select excluded coarse indices unless every other window is worse.",
            "Prefer windows likely to contain positive visible evidence, not windows that merely mention the target is absent.",
            "If no summary appears to contain the target, return low confidence and explain the missing clue.",
        ],
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
        },
        "clue_need_spec": clue_spec,
        "excluded_negative_coarse_indices": sorted(negative_indices),
        "max_indices": args.reroute_topk,
        "coarse_summaries": coarse_rows,
        "required_json_shape": _coarse_selector_response_schema()["json_schema"]["schema"]["properties"],
    }
    client = OpenRouterClient(
        model=args.clue_planner_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.clue_selector_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.clue_planner_timeout_s,
    )
    try:
        payload = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You output JSON only. Select long-video coarse windows for visual clue discovery. "
                        "Do not include analysis, markdown, or prose outside JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format=_coarse_selector_response_schema(),
        )
    except Exception as exc:
        if args.allow_lexical_fallback:
            return [], [{"role": "model_clue_selector_error", "reason": str(exc), "selected_after_exclusion": []}]
        raise RuntimeError(f"LLM reroute selector failed and heuristic fallback is disabled: {exc}") from exc
    selected: list[int] = []
    max_index = len(coarse_rows) - 1
    for item in payload.get("selected_coarse_indices") or []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx <= max_index and idx not in negative_indices and idx not in selected:
            selected.append(idx)
        if len(selected) >= args.reroute_topk:
            break
    rounds = payload.get("selection_rounds") if isinstance(payload.get("selection_rounds"), list) else []
    if rounds:
        for row in rounds:
            if isinstance(row, dict):
                row["selector_backend"] = args.clue_planner_model
    else:
        rounds = [
            {
                "role": "model_clue_selector",
                "query_or_need": clue_spec.get("visual_target"),
                "selected_after_exclusion": selected,
                "reason": payload.get("missing_clue_diagnosis") or "",
                "confidence": 0.0,
                "selector_backend": args.clue_planner_model,
                "selector_abstained": not bool(selected),
            }
        ]
    return selected, rounds


def _select_rerouted_repair_spans(
    example: dict[str, Any],
    row: dict[str, Any],
    *,
    gaps: list[str],
    clue_spec: dict[str, Any],
    prior_schemas: list[dict[str, Any]],
    max_repair_clips: int,
    reroute_topk: int,
    reroute_topk_per_query: int,
    api_key: str | None,
    args: argparse.Namespace,
) -> tuple[list[ClipSpan], dict[str, Any]]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    coarse = segment_coarse_index(duration_s, policy, regime=VideoRegime.LONG)
    segments = _coarse_schema_segments(example)
    negative_indices = set(_negative_parent_indices_from_schemas(example, prior_schemas))
    rounds: list[dict[str, Any]] = []
    combined_scores: dict[int, float] = {}

    selected, llm_rounds = _llm_select_coarse_indices(
        example,
        clue_spec=clue_spec,
        negative_indices=negative_indices,
        api_key=api_key,
        args=args,
    )
    if selected:
        rounds.extend(llm_rounds)

    selector_abstained = bool(
        api_key
        and not selected
        and not args.allow_lexical_fallback
        and not args.disable_llm_reroute_selector
        and not args.dry_run
    )

    for variant in ([] if selected or selector_abstained else _query_variants(example, row, gaps, clue_spec=clue_spec)):
        retrieval = retrieve_coarse_clips(
            coarse_spans=coarse,
            query_text=variant["query"],
            segments=segments,
            topk=max(reroute_topk_per_query + len(negative_indices), reroute_topk_per_query),
            threshold=0.0,
            mode="lexical",
        )
        kept_scores: list[dict[str, Any]] = []
        for score_row in retrieval.get("scores") or []:
            idx = int(score_row["coarse_index"])
            if idx in negative_indices:
                continue
            score = float(score_row.get("score") or 0.0)
            kept_scores.append(score_row)
            combined_scores[idx] = combined_scores.get(idx, 0.0) + score + 0.05
            if len(kept_scores) >= reroute_topk_per_query:
                break
        rounds.append(
            {
                "role": variant["role"],
                "query": variant["query"],
                "fallback_reason": retrieval.get("fallback_reason"),
                "selected_before_exclusion": retrieval.get("selected_coarse_indices") or [],
                "selected_after_exclusion": [int(row["coarse_index"]) for row in kept_scores],
                "scores": kept_scores,
            }
        )

    if not selected:
        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        selected = [idx for idx, _ in ranked[:reroute_topk]]
    if not selected:
        if selector_abstained:
            return [], {
                "mode": "reroute",
                "duration_s": duration_s,
                "negative_coarse_indices": sorted(negative_indices),
                "selected_coarse_indices": [],
                "candidate_span_count": 0,
                "candidate_fine_span_count": 0,
                "fresh_span_count": 0,
                "chosen_span_count": 0,
                "retrieval_rounds": rounds,
                "clue_planner_backend": clue_spec.get("planner_backend"),
                "reroute_selector_backend": args.clue_planner_model,
                "selector_abstained": True,
            }
        selected = [idx for idx in range(min(reroute_topk, len(coarse))) if idx not in negative_indices]

    spans = segment_perception_clips(
        duration_s,
        policy,
        regime=VideoRegime.LONG,
        selected_coarse_indices=selected,
    )
    fine_spans = [span for span in spans if span.granularity == "fine"]
    existing_ids = {
        str(schema.get("clip_id"))
        for schema in ((example.get("metadata") or {}).get("clip_schemas") or [])
        if isinstance(schema, dict) and schema.get("clip_id")
    }
    video_id = (example.get("video") or {}).get("video_id")
    fresh = [
        span
        for span in fine_spans
        if f"clip:{video_id}:fine:{span.clip_index:04d}" not in existing_ids
    ]
    chosen = (fresh or fine_spans or spans)[:max_repair_clips]
    return chosen, {
        "mode": "reroute",
        "duration_s": duration_s,
        "negative_coarse_indices": sorted(negative_indices),
        "selected_coarse_indices": selected,
        "candidate_span_count": len(spans),
        "candidate_fine_span_count": len(fine_spans),
        "fresh_span_count": len(fresh),
        "chosen_span_count": len(chosen),
        "retrieval_rounds": rounds,
        "clue_planner_backend": clue_spec.get("planner_backend"),
        "reroute_selector_backend": args.clue_planner_model if rounds and rounds[0].get("selector_backend") else "lexical_fallback",
        "selector_abstained": False,
    }


def _select_repair_spans(
    example: dict[str, Any],
    row: dict[str, Any],
    *,
    radius: int,
    max_repair_clips: int,
) -> tuple[list[ClipSpan], dict[str, Any]]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    cf_graph = ((example.get("metadata") or {}).get("coarse_fine_graph") or {})
    coarse_count = int(((cf_graph.get("counts") or {}).get("coarse_nodes")) or 0)
    selected = [int(i) for i in ((row.get("L1_quality") or {}).get("selected_coarse_indices") or [])]
    if not selected:
        selected = list(range(min(coarse_count, 3)))
    expanded = _expanded_indices(selected, coarse_count or max(selected, default=0) + 1, radius=radius)
    spans = segment_perception_clips(
        duration_s,
        policy,
        regime=VideoRegime.LONG,
        selected_coarse_indices=expanded,
    )
    fine_spans = [span for span in spans if span.granularity == "fine"]

    existing_ids = {
        str(schema.get("clip_id"))
        for schema in ((example.get("metadata") or {}).get("clip_schemas") or [])
        if isinstance(schema, dict) and schema.get("clip_id")
    }
    fresh = [
        span
        for span in fine_spans
        if f"clip:{(example.get('video') or {}).get('video_id')}:fine:{span.clip_index:04d}" not in existing_ids
    ]
    chosen = (fresh or fine_spans or spans)[:max_repair_clips]
    return chosen, {
        "mode": "local",
        "duration_s": duration_s,
        "selected_coarse_indices": selected,
        "expanded_coarse_indices": expanded,
        "candidate_span_count": len(spans),
        "candidate_fine_span_count": len(fine_spans),
        "fresh_span_count": len(fresh),
        "chosen_span_count": len(chosen),
    }


def _build_producer(args: argparse.Namespace, api_key: str) -> QwenClipSchemaProducer:
    reasoning = None if args.clip_schema_reasoning_effort in {"", "none", "off"} else {"effort": args.clip_schema_reasoning_effort}
    if args.clip_schema_reasoning_effort == "none":
        reasoning = {"effort": "none", "exclude": True}
    client = OpenRouterClient(
        model=args.clip_schema_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.clip_schema_max_tokens,
        reasoning=reasoning,
        timeout_s=args.clip_schema_timeout_s,
    )
    cfg = ClipSchemaConfig(
        model=args.clip_schema_model,
        keys_py_path=str(args.keys_py) if args.keys_py else None,
        request_frames=args.request_frames,
        max_tokens=args.clip_schema_max_tokens,
        timeout_s=args.clip_schema_timeout_s,
    )
    return QwenClipSchemaProducer(cfg, client)


def _schema_for_span(
    producer: QwenClipSchemaProducer,
    *,
    example: dict[str, Any],
    span: ClipSpan,
    repair_query: str,
) -> dict[str, Any]:
    video = example.get("video") or {}
    video_id = str(video.get("video_id") or "video")
    clip_id = f"clip:{video_id}:fine:{span.clip_index:04d}"
    return producer.build_clip_schema(
        clip_id=clip_id,
        clip=span,
        video_path=Path(str(video.get("primary_path"))) if video.get("primary_path") else None,
        subtitle_context=None,
        question_context=repair_query,
    )


def _audio_or_subtitle_like(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in ("'modality': 'audio'", '"modality": "audio"', "'modality': 'subtitle'", '"modality": "subtitle"')
    )


def _text_items(schema: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("scene_description", "uncertainty"):
        text = str(schema.get(key) or "").strip()
        if text and not _audio_or_subtitle_like(text):
            items.append(text)
    for key in ("observable_facts", "events", "visual_social_cues", "cross_clip_cues", "searchable_phrases"):
        value = schema.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    if not _audio_or_subtitle_like(item):
                        items.append(item)
                elif isinstance(item, dict):
                    modality = str(item.get("modality") or "visual").lower()
                    if modality in {"audio", "subtitle", "asr"}:
                        continue
                    text = item.get("text") or item.get("event_description") or item.get("description") or item.get("cue")
                    if text and not _audio_or_subtitle_like(str(text)):
                        items.append(str(text))
    return [item.strip() for item in items if item and item.strip()]


def _has_negative_target_evidence(text: str) -> bool:
    lowered = text.lower()
    negative_terms = (
        "no vehicle",
        "no animated vehicle",
        "no animation",
        "not visible",
        "cannot determine",
        "does not provide enough information",
        "no visible",
    )
    return any(term in lowered for term in negative_terms)


def _negative_target_count(patch: dict[str, Any]) -> int:
    count = 0
    for node in patch.get("nodes") or []:
        if node.get("node_type") in {"l2_repair_reminder", "answerability_gap"}:
            continue
        text = str(node.get("text") or "").lower()
        if _has_negative_target_evidence(text):
            count += 1
    return count


def _build_l1_patch(example: dict[str, Any], row: dict[str, Any], schemas: list[dict[str, Any]], gaps: list[str]) -> dict[str, Any]:
    video_id = str((example.get("video") or {}).get("video_id") or "")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    gap_id = stable_id("repair.gap", example.get("example_id"), gaps)
    nodes.append(
        {
            "node_id": gap_id,
            "node_type": "l2_repair_reminder",
            "text": f"Repair needed: {', '.join(gaps) or row.get('verifier_reason')}",
            "source_type": "quality_report",
            "producer": "run_repair_protocol",
            "visibility": {"mode": "video_only", "hidden_supervision": False},
        }
    )
    for schema in schemas:
        if schema.get("model_error"):
            continue
        clip_id = str(schema.get("clip_id") or "")
        time_span = schema.get("time_span") or {}
        for text in _text_items(schema)[:10]:
            node_type = "visual_social_cue" if any(gap in {"social_intent_or_affect", "causal_explanation"} for gap in gaps) else "observation"
            node_id = stable_id("repair.obs", example.get("example_id"), clip_id, text)
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "text": text,
                    "modality": "visual",
                    "confidence": 0.72,
                    "clip_id": clip_id,
                    "time_span": time_span,
                    "source_type": "repair_clip_schema",
                    "producer": "qwen_repair_clip_schema",
                    "video_id": video_id,
                    "visibility": {"mode": "video_only", "hidden_supervision": False},
                    "provenance": {"created_by": "dataset_clip_wrapper.run_repair_protocol"},
                }
            )
            edges.append(
                {
                    "edge_id": stable_id("repair.edge", gap_id, node_id),
                    "src": node_id,
                    "dst": gap_id,
                    "edge_type": "repair_candidate_for",
                    "text": "Repair candidate evidence for the reported L2/L1 gap.",
                    "confidence": 0.6,
                    "evidence_refs": [node_id, gap_id],
                    "producer": "run_repair_protocol",
                    "visibility": {"mode": "video_only", "hidden_supervision": False},
                }
            )
    return {
        "schema_version": "video-skills-relaunch/repair-v0.1",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "strategy": _strategy_for_gaps(gaps),
        "gap_types": gaps,
        "nodes": nodes,
        "edges": edges,
        "counts": {"nodes": len(nodes), "edges": len(edges), "repair_observation_nodes": max(0, len(nodes) - 1)},
    }


def _merge_patch_graph(example: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    base = ((example.get("metadata") or {}).get("clue_memory_graph") or {"nodes": [], "edges": []})
    graph = dict(base)
    graph["nodes"] = list(base.get("nodes") or []) + list(patch.get("nodes") or [])
    graph["edges"] = list(base.get("edges") or []) + list(patch.get("edges") or [])
    graph.setdefault("metadata", {})
    graph["metadata"] = dict(graph.get("metadata") or {})
    graph["metadata"]["repair_patch_schema_version"] = patch.get("schema_version")
    graph["metadata"]["repair_gap_types"] = patch.get("gap_types") or []
    return graph


def _candidate_refs_for_option(graph: dict[str, Any], option: dict[str, Any], question_text: str, *, limit: int) -> list[str]:
    query_words = set(re.findall(r"[a-zA-Z0-9]+", f"{question_text} {option.get('text', '')}".lower()))
    scored: list[tuple[float, str]] = []
    for node in graph.get("nodes") or []:
        node_id = node.get("node_id")
        if not node_id or node.get("node_type") in {"l2_repair_reminder", "answerability_gap", "question_requirement"}:
            continue
        if node.get("source_type") not in {"repair_clip_schema", "observation"} and node.get("producer") != "qwen_repair_clip_schema":
            continue
        text = str(node.get("text") or "")
        words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        overlap = len(query_words & words)
        bonus = 2 if node.get("source_type") == "repair_clip_schema" else 0
        if overlap or bonus:
            scored.append((overlap + bonus, str(node_id)))
    scored.sort(reverse=True)
    return [ref for _, ref in scored[:limit]]


def _verify_options(
    example: dict[str, Any],
    graph: dict[str, Any],
    *,
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    options = question.get("options") or []
    executor: SkillExecutor
    if api_key and not args.skip_gptoss_verifier:
        llm_client = SkillModelClient(
            model=args.verifier_model,
            api_key=api_key,
            max_tokens=args.verifier_max_tokens,
            temperature=0.0,
            timeout_s=args.verifier_timeout_s,
        )
        executor = SkillExecutor(
            llm_client=llm_client,
            config=SkillBackendConfig(default_mode=SkillBackendMode.RULE, llm_skills={"verify_claim_support"}),
        )
        backend = "gptoss_verifier"
    else:
        executor = SkillExecutor(config=SkillBackendConfig(default_mode=SkillBackendMode.RULE))
        backend = "rule_verifier"

    verifications: list[dict[str, Any]] = []
    for opt in options:
        refs = _candidate_refs_for_option(graph, opt, question_text, limit=args.max_verify_refs)
        claim = {
            "claim_text": f"{opt.get('label')}: {opt.get('text')}",
            "option_label": opt.get("label"),
            "question_text": question_text,
        }
        result = executor.execute(
            "verify_claim_support",
            args={
                "claim": claim,
                "question_text": question_text,
                "evidence_chain": {"evidence_refs": refs},
                "support_policy": {
                    "min_evidence_refs": args.min_verify_refs,
                    "min_claim_score": 0.05,
                    "min_target_score": 0.05,
                },
            },
            graph=graph,
        )
        verifications.append(
            {
                "option_label": opt.get("label"),
                "option_text": opt.get("text"),
                "evidence_refs": refs,
                "ok": bool(result.ok),
                "failure_code": result.failure_code,
                "confidence": result.confidence,
                "outputs": result.outputs,
            }
        )

    passed = [row for row in verifications if row["ok"]]
    if passed:
        passed.sort(key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
        status = "resolved_strong" if len(passed[0].get("evidence_refs") or []) >= args.min_verify_refs else "resolved_weak"
        best = passed[0]
    else:
        status = "needs_more_evidence"
        best = max(verifications, key=lambda row: float(row.get("confidence") or 0.0), default={})
    return {
        "schema_version": "video-skills-relaunch/repair-l2-v0.1",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "backend": backend,
        "repair_status": status,
        "best_option": {
            "label": best.get("option_label"),
            "text": best.get("option_text"),
            "confidence": best.get("confidence"),
        },
        "option_verifications": verifications,
    }


def _build_report(plan: dict[str, Any], patch: dict[str, Any], l2: dict[str, Any]) -> dict[str, Any]:
    strong = l2.get("repair_status") == "resolved_strong"
    negative_count = _negative_target_count(patch)
    span_selection = plan.get("span_selection") or {}
    missing = set(plan.get("gap_types") or [])
    if strong:
        next_action = "commit repaired evidence pack"
        failure_type = "resolved"
    elif span_selection.get("selector_abstained"):
        next_action = "mark visual-only evidence missing; do not use heuristic fallback"
        failure_type = "visual_only_benchmark_limitation"
    elif negative_count or span_selection.get("negative_coarse_indices"):
        next_action = "reroute coarse retrieval because repair clips contain negative target evidence"
        failure_type = "l1_target_coverage_failure"
    elif {"social_intent_or_affect", "causal_explanation"} & missing:
        next_action = "keep visual-only boundary and try bridge verification with more target-aligned context"
        failure_type = "l1_context_partial_l2_bridge_needed"
    else:
        next_action = "expand another fine-pass around visually relevant context"
        failure_type = "l1_attribute_or_evidence_resolution_failure"
    return {
        "example_id": plan.get("example_id"),
        "dataset": plan.get("dataset"),
        "strategy": plan.get("strategy"),
        "repair_mode": span_selection.get("mode"),
        "gap_types": plan.get("gap_types") or [],
        "failure_type": failure_type,
        "repair_status": l2.get("repair_status"),
        "best_option": l2.get("best_option"),
        "patch_counts": patch.get("counts") or {},
        "negative_target_evidence_nodes": negative_count,
        "negative_coarse_indices": span_selection.get("negative_coarse_indices") or [],
        "selected_coarse_indices": span_selection.get("selected_coarse_indices") or [],
        "retrieval_round_count": len(span_selection.get("retrieval_rounds") or []),
        "selector_abstained": bool(span_selection.get("selector_abstained")),
        "verifier_backend": l2.get("backend"),
        "repair_needed_after_round": not strong,
        "verifier_reason": "strong repair evidence verified" if strong else "repair evidence remains weak or insufficient",
        "recommended_next_action": next_action,
        "artifact_paths": plan.get("artifact_paths") or {},
    }


def _process_row(row: dict[str, Any], args: argparse.Namespace, api_key: str | None) -> dict[str, Any]:
    example = _load_source_example(row)
    gaps = _gap_types(row)
    strategy = _strategy_for_gaps(gaps)
    out_dir = args.stage_dir / _safe_name(f"{row.get('dataset')}_{row.get('example_id')}")
    plan_path = out_dir / "repair_01_plan.json"
    schemas_path = out_dir / "repair_02_clip_schemas.jsonl"
    patch_path = out_dir / "repair_03_l1_patch.json"
    l2_path = out_dir / "repair_04_l2_verifier.json"
    report_path = out_dir / "repair_05_report.json"

    prior_schemas = _read_jsonl(schemas_path) if schemas_path.exists() else []
    prior_negative_count = sum(1 for schema in prior_schemas if _has_negative_target_evidence(" ".join(_text_items(schema))))
    clue_spec = _build_clue_need_spec(
        example,
        row,
        gaps,
        prior_schemas=prior_schemas,
        api_key=api_key,
        args=args,
    )
    use_reroute = args.repair_mode == "reroute" or (
        args.repair_mode == "auto" and prior_negative_count >= args.negative_reroute_threshold
    )
    if use_reroute:
        spans, span_meta = _select_rerouted_repair_spans(
            example,
            row,
            gaps=gaps,
            clue_spec=clue_spec,
            prior_schemas=prior_schemas,
            max_repair_clips=args.max_repair_clips,
            reroute_topk=args.reroute_topk,
            reroute_topk_per_query=args.reroute_topk_per_query,
            api_key=api_key,
            args=args,
        )
    else:
        spans, span_meta = _select_repair_spans(
            example,
            row,
            radius=args.coarse_radius,
            max_repair_clips=args.max_repair_clips,
        )
        span_meta["prior_negative_schema_count"] = prior_negative_count
    query = _repair_query(example, row, gaps, clue_spec=clue_spec)
    plan = {
        "schema_version": "video-skills-relaunch/repair-plan-v0.1",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "source_path": row.get("source_path"),
        "strategy": strategy,
        "repair_mode": span_meta.get("mode"),
        "gap_types": gaps,
        "clue_need_spec": clue_spec,
        "repair_query": query,
        "span_selection": span_meta,
        "spans": [
            {"clip_index": span.clip_index, "parent_index": span.parent_index, "time_span": span.to_dict(), "granularity": span.granularity}
            for span in spans
        ],
        "acceptance_rule": (
            "Repair may only become resolved_strong when verify_claim_support passes "
            "with non-diagnostic visual evidence refs; commonsense hypotheses are reminders, not final evidence."
        ),
        "artifact_paths": {
            "plan": str(plan_path),
            "clip_schemas": str(schemas_path),
            "l1_patch": str(patch_path),
            "l2_verifier": str(l2_path),
            "report": str(report_path),
        },
    }
    _write_json(plan_path, plan)

    if args.dry_run:
        patch = _build_l1_patch(example, row, [], gaps)
        l2 = {"repair_status": "dry_run", "backend": "none", "best_option": {}}
        _write_json(patch_path, patch)
        _write_json(l2_path, l2)
        report = _build_report(plan, patch, l2)
        _write_json(report_path, report)
        return report

    schemas: list[dict[str, Any]]
    if not spans and span_meta.get("selector_abstained"):
        schemas = []
    elif args.skip_api and schemas_path.exists():
        schemas = _read_jsonl(schemas_path)
    elif args.skip_api:
        schemas = []
    else:
        if not api_key:
            raise RuntimeError("OpenRouter API key is required unless --dry-run or --skip-api is used")
        producer = _build_producer(args, api_key)
        schemas = []
        for span in spans:
            schema = _schema_for_span(producer, example=example, span=span, repair_query=query)
            schemas.append(schema)
            _write_jsonl(schemas_path, schemas)
    if schemas and not schemas_path.exists():
        _write_jsonl(schemas_path, schemas)

    patch = _build_l1_patch(example, row, schemas, gaps)
    _write_json(patch_path, patch)
    if not spans and span_meta.get("selector_abstained"):
        l2 = {
            "schema_version": "video-skills-relaunch/repair-l2-v0.1",
            "example_id": example.get("example_id"),
            "dataset": example.get("dataset"),
            "backend": "selector_abstained",
            "repair_status": "needs_more_evidence",
            "best_option": {"label": None, "text": "", "confidence": 0.0},
            "option_verifications": [],
            "missing_clue_diagnosis": (span_meta.get("retrieval_rounds") or [{}])[0].get("reason", ""),
        }
    else:
        repaired_graph = _merge_patch_graph(example, patch)
        l2 = _verify_options(example, repaired_graph, api_key=api_key, args=args)
    _write_json(l2_path, l2)
    report = _build_report(plan, patch, l2)
    _write_json(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run targeted graph repair for L1/L2 quality failures.")
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["cg_bench", "vrbench"])
    parser.add_argument("--keys-py", type=Path)
    parser.add_argument("--clip-schema-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--verifier-model", default="openai/gpt-oss-120b")
    parser.add_argument("--clue-planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--clue-planner-max-tokens", type=int, default=1200)
    parser.add_argument("--clue-selector-max-tokens", type=int, default=1600)
    parser.add_argument("--clue-planner-timeout-s", type=int, default=180)
    parser.add_argument("--coarse-summary-prompt-chars", type=int, default=260)
    parser.add_argument("--disable-llm-clue-planner", action="store_true")
    parser.add_argument("--disable-llm-reroute-selector", action="store_true")
    parser.add_argument(
        "--allow-lexical-fallback",
        action="store_true",
        help="Allow heuristic lexical fallback when the LLM clue planner/selector fails. Off by default for API runs.",
    )
    parser.add_argument("--request-frames", type=int, default=4)
    parser.add_argument("--clip-schema-max-tokens", type=int, default=1200)
    parser.add_argument("--clip-schema-timeout-s", type=int, default=180)
    parser.add_argument("--clip-schema-reasoning-effort", default="none")
    parser.add_argument("--verifier-max-tokens", type=int, default=512)
    parser.add_argument("--verifier-timeout-s", type=int, default=120)
    parser.add_argument(
        "--repair-mode",
        default="local",
        choices=["local", "reroute", "auto"],
        help=(
            "local expands around prior coarse hits; reroute re-ranks the full coarse "
            "index with multiple query roles; auto reroutes when cached repair schemas "
            "contain enough negative target evidence."
        ),
    )
    parser.add_argument("--coarse-radius", type=int, default=1)
    parser.add_argument("--max-repair-clips", type=int, default=12)
    parser.add_argument("--reroute-topk", type=int, default=6)
    parser.add_argument("--reroute-topk-per-query", type=int, default=3)
    parser.add_argument("--negative-reroute-threshold", type=int, default=2)
    parser.add_argument("--max-verify-refs", type=int, default=8)
    parser.add_argument("--min-verify-refs", type=int, default=2)
    parser.add_argument("--skip-gptoss-verifier", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = _read_json(args.quality_report)
    reports = payload.get("reports") if isinstance(payload, dict) else payload
    wanted = set(args.datasets)
    rows = [
        row
        for row in reports
        if row.get("dataset") in wanted and row.get("repair_needed") and str(row.get("video_regime")) == "long"
    ]

    api_key = None
    if not args.dry_run and not args.skip_api:
        api_key = load_openrouter_api_key(keys_py_path=str(args.keys_py) if args.keys_py else None)
    elif not args.dry_run and not args.skip_gptoss_verifier:
        api_key = load_openrouter_api_key(keys_py_path=str(args.keys_py) if args.keys_py else None)

    out_reports = [_process_row(row, args, api_key) for row in rows]
    summary = {
        "examples": len(out_reports),
        "datasets": sorted({str(row.get("dataset")) for row in out_reports}),
        "repair_status_counts": {},
        "repair_needed_after_round": sum(1 for row in out_reports if row.get("repair_needed_after_round")),
    }
    for row in out_reports:
        status = str(row.get("repair_status"))
        summary["repair_status_counts"][status] = summary["repair_status_counts"].get(status, 0) + 1
    final = {"summary": summary, "reports": out_reports}
    _write_json(args.output, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

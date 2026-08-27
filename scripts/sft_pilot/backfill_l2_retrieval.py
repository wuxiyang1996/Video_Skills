#!/usr/bin/env python3
"""Backfill atomic L2 retrieval actions from cached coarse schemas.

This does not rerun video perception. It only calls the bounded coarse selector
for already-correct, quality-gated rollouts that predate selector logging.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from dataset_clip_wrapper.training.l2_retrieval_sft_adapter import (
    _catalog,
    _quality_gate,
    build_l2_retrieval_exports,
)
from dataset_clip_wrapper.training.sft_common import read_json, read_jsonl, write_json, write_jsonl


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "coarse_clip_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_coarse_indices": {"type": "array", "items": {"type": "integer"}},
                "rationale_short": {"type": "string"},
            },
            "required": ["selected_coarse_indices", "rationale_short"],
        },
    },
}


def _repair_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            payload = read_json(path)
            rows = payload.get("reports") if isinstance(payload.get("reports"), list) else [payload]
        except (json.JSONDecodeError, ValueError):
            rows = read_jsonl(path)
        for row in rows:
            if isinstance(row, dict) and row.get("example_id"):
                result[str(row["example_id"])] = row
    return result


def _select(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    coarse = metadata.get("coarse_clip_schemas") if isinstance(metadata.get("coarse_clip_schemas"), list) else []
    topk = int(((metadata.get("perception") or {}).get("retrieval") or {}).get("topk") or 8)
    question = row.get("question") if isinstance(row.get("question"), dict) else {}
    prompt = {
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
        },
        "topk": topk,
        "coarse_summary_catalog": _catalog(coarse),
    }
    last_error = ""
    for attempt in range(1, args.attempts + 1):
        try:
            client = OpenRouterClient(
                model=args.model,
                api_key=load_openrouter_api_key(keys_py_path=str(args.keys_py)),
                temperature=0.0,
                max_tokens=500,
                reasoning={"effort": "minimal", "exclude": True},
                timeout_s=args.timeout,
            )
            payload = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the Video_Skills L2 retrieval controller. Select the coarse video windows "
                            "most likely to contain direct visible evidence. Options are hypotheses, not facts. "
                            "Return the atomic selection action as JSON."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))},
                ],
                response_format=RESPONSE_FORMAT,
            )
            selected: list[int] = []
            for value in payload.get("selected_coarse_indices") or []:
                index = int(value)
                if 0 <= index < len(coarse) and index not in selected:
                    selected.append(index)
                if len(selected) >= topk:
                    break
            if not selected:
                raise ValueError("selector returned no valid indices")
            return {
                "ok": True,
                "mode": "gpt_oss_atomic_select_coarse",
                "selected_coarse_indices": selected,
                "rationale_short": str(payload.get("rationale_short") or "")[:500],
                "llm_usage": client.last_response_metadata,
                "backfilled": True,
            }, ""
        except Exception as exc:  # bounded external retry
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.attempts:
                time.sleep(args.retry_sleep)
    return None, last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--repair-results", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-120b:free")
    parser.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    source_report = read_json(args.source_report)
    repairs = _repair_map(args.repair_results)
    candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    seen: set[str] = set()
    for source in map(Path, source_report.get("source_rollout_jsonl") or []):
        if not source.exists():
            continue
        for row in read_jsonl(source):
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in seen:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            perception = metadata.get("perception") if isinstance(metadata.get("perception"), dict) else {}
            selection = perception.get("retrieval") if isinstance(perception.get("retrieval"), dict) else {}
            coarse = metadata.get("coarse_clip_schemas") if isinstance(metadata.get("coarse_clip_schemas"), list) else []
            if selection.get("mode") == "gpt_oss_atomic_select_coarse":
                excluded["already_atomic"] += 1
                continue
            if not coarse:
                excluded["missing_coarse"] += 1
                continue
            eligible, reason = _quality_gate(row, {"ok": True, "selected_coarse_indices": [0]}, repairs)
            if not eligible:
                excluded[reason] += 1
                continue
            seen.add(example_id)
            candidates.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for index, row in enumerate(candidates, 1):
        selection, error = _select(row, args)
        example_id = str(row.get("example_id"))
        if selection is None:
            failures[example_id] = error
        else:
            payload = copy.deepcopy(row)
            payload.setdefault("metadata", {}).setdefault("perception", {})["retrieval"] = selection
            updated.append(payload)
        print(json.dumps({
            "progress": index,
            "total": len(candidates),
            "example_id": example_id,
            "ok": selection is not None,
            "error": error[:500] if error else "",
        }), flush=True)

    rollout_path = args.output_dir / "backfilled_rollouts.jsonl"
    write_jsonl(rollout_path, updated)
    transitions, chats, export_report = build_l2_retrieval_exports(
        [rollout_path], repair_results_paths=args.repair_results
    )
    write_jsonl(args.output_dir / "l2_retrieval_transitions.jsonl", transitions)
    write_jsonl(args.output_dir / "l2_retrieval_sft.jsonl", chats)
    report = {
        "source_candidates": len(candidates),
        "selector_successes": len(updated),
        "selector_failures": failures,
        "prefilter_excluded": dict(excluded),
        "export": export_report,
    }
    write_json(args.output_dir / "backfill_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

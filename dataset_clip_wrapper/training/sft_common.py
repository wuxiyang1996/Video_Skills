"""Shared helpers for leakage-aware cold-start SFT exporters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


# Qwen3 / Qwen3.5 chat templates must always run with thinking disabled for
# Video_Skills controller SFT and generation gates. `enable_thinking=False`
# still inserts an empty <think></think> block before the answer; that is the
# official "thinking off" prompt shape, not an active reasoning mode.
ENABLE_THINKING = False
_THINK_COMPLETE_RE = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove residual ``<think>…</think>`` (or truncated open) blocks."""
    if not text or "<think>" not in text:
        return text
    result = _THINK_COMPLETE_RE.sub("", text)
    result = _THINK_OPEN_RE.sub("", result)
    return result.strip()


def apply_chat_template_no_think(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    **kwargs: Any,
) -> Any:
    """apply_chat_template with thinking hard-disabled."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=ENABLE_THINKING,
        **kwargs,
    )


FORBIDDEN_PROMPT_KEYS = {
    "answer",
    "correct",
    "correct_answer",
    "final_answer",
    "gold",
    "gold_answer",
    "gold_label",
    "hidden_supervision",
    "official_answer",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def contains_forbidden_prompt_key(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(
            str(key).lower() in FORBIDDEN_PROMPT_KEYS or contains_forbidden_prompt_key(value)
            for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(contains_forbidden_prompt_key(value) for value in payload)
    return False


def compact_visibility(payload: Any) -> Any:
    """Drop provenance fields that can contain supervision or large runtime logs."""
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PROMPT_KEYS or lowered in {"llm_usage", "source_path"}:
                continue
            result[key] = compact_visibility(value)
        return result
    if isinstance(payload, list):
        return [compact_visibility(value) for value in payload]
    return payload

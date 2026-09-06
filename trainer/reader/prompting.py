"""Prompts and targets for the trainable reader — identical to what evaluation sends.

`scripts.eval.measure_answer_chain.direct_messages` builds the exact system/user
messages the evaluation uses (whole catalog, optional pointer, rationale JSON
format).  Training reuses it so that the model is trained and evaluated on the
same distribution of inputs; the only thing training adds is the completion.
"""
from __future__ import annotations

import json
import re
from typing import Any

from scripts.eval.measure_answer_chain import direct_messages, retrieval_catalog

_CLIP = re.compile(r"\bclips?\s+(\d{1,3})(?:\s*(?:-|–|to|and|,)\s*(\d{1,3}))?", re.I)


def reader_messages(example: dict[str, Any], *, rationale: bool = True) -> list[dict[str, str]]:
    """Chat messages (system + user) for the whole-catalog direct prompt, no pointer."""
    schemas, _ = retrieval_catalog(example)
    indices = list(range(len(schemas)))
    system, user = direct_messages(example, indices, rationale=rationale)
    assert isinstance(user, str)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reader_target(reasoning: str, label: str) -> str:
    """The completion the reader is trained to produce: the same JSON the evaluator parses."""
    return json.dumps({"reasoning": reasoning.strip(), "label": str(label).strip()}, ensure_ascii=False)


def parse_reader_output(text: str) -> tuple[str | None, str]:
    """(label, reasoning) from a completion, tolerant of prose around the JSON — mirrors the evaluator."""
    try:
        obj = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0)) or {}
        return (str(obj.get("label")) if obj.get("label") else None), str(obj.get("reasoning") or "")
    except Exception:
        m = re.search(r"\b([A-H])\b", text or "")
        return (m.group(1) if m else None), ""


def cited_ranks(reasoning: str) -> list[int]:
    """1-based clip ranks mentioned in a rationale ('clip 12', 'clips 3-5')."""
    ranks: list[int] = []
    for a, b in _CLIP.findall(reasoning or ""):
        lo, hi = int(a), int(b) if b else int(a)
        if hi < lo or hi - lo > 20:
            hi = lo
        ranks.extend(range(lo, hi + 1))
    return sorted(set(ranks))

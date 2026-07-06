"""Canonical signatures for reusable L1/L2 motif candidates."""

from __future__ import annotations

import re
from typing import Any

from atomic_skills.common import stable_id


_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_TIME_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|min|minute|minutes)\b", re.I)
_OPTION_RE = re.compile(r"\b(?:option\s*)?[A-E]\b", re.I)


def canonical_text(text: str) -> str:
    """Remove surface names, timestamps, and option labels from a signature."""
    out = _TIME_RE.sub("TIME_SPAN", text)
    out = _OPTION_RE.sub("ANSWER_OPTION", out)
    out = _ENTITY_RE.sub("ENTITY", out)
    return re.sub(r"\s+", " ", out).strip().lower()


def canonical_token(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9_:-]+", "_", text)
    return text.strip("_").lower() or "unknown"


def motif_signature(*parts: Any) -> str:
    return "|".join(canonical_token(part) for part in parts)


def motif_id(motif_type: str, signature: str) -> str:
    return stable_id("motif", motif_type, signature)


def role_signature_from_nodes(nodes: list[dict[str, Any]]) -> list[str]:
    roles: list[str] = []
    for node in nodes:
        node_type = canonical_token(node.get("node_type"))
        if node_type:
            roles.append(node_type)
    return roles

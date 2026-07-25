"""Filter frozen L1 / examples by post-training split role."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

VALID_ROLES = ("sft_seed", "opd_pool", "grpo_pool", "dev_tune", "heldout_test")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "videos" not in payload:
        raise ValueError(f"invalid split manifest: {path}")
    return payload


def video_role_index(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Map ``dataset:video_id`` and bare ``video_id`` → role."""
    index: dict[str, str] = {}
    for row in manifest.get("videos") or []:
        if not isinstance(row, Mapping):
            continue
        role = str(row.get("role") or "")
        key = str(row.get("key") or "")
        video_id = str(row.get("video_id") or "")
        dataset = str(row.get("dataset") or "")
        if key:
            index[key] = role
        if dataset and video_id:
            index[f"{dataset}:{video_id}"] = role
        if video_id and video_id not in index:
            index[video_id] = role
    return index


def example_video_key(example: Mapping[str, Any]) -> str:
    meta = example.get("metadata") or {}
    dataset = str(example.get("dataset") or meta.get("dataset") or "")
    video_id = (
        example.get("video_id")
        or meta.get("video_id")
        or (meta.get("clue_memory_graph") or {}).get("video_id")
        or ""
    )
    video_id = str(video_id)
    if dataset and video_id:
        return f"{dataset}:{video_id}"
    return video_id


def filter_examples_by_role(
    examples: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    role: str,
    strict: bool = True,
) -> list[dict[str, Any]]:
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role {role}; expected one of {VALID_ROLES}")
    index = video_role_index(manifest)
    kept: list[dict[str, Any]] = []
    unknown = 0
    for example in examples:
        key = example_video_key(example)
        found = index.get(key)
        if found is None:
            # try bare video id
            bare = key.split(":", 1)[-1]
            found = index.get(bare)
        if found is None:
            unknown += 1
            if strict:
                continue
        if found == role:
            kept.append(dict(example))
    if strict and not kept and unknown:
        raise RuntimeError(
            f"no examples matched role={role}; {unknown} examples missing from split manifest"
        )
    return kept


def assert_role_exclusive(
    examples: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    allowed_roles: Iterable[str],
) -> None:
    allowed = set(allowed_roles)
    index = video_role_index(manifest)
    for example in examples:
        key = example_video_key(example)
        role = index.get(key) or index.get(key.split(":", 1)[-1])
        if role is None:
            raise RuntimeError(f"example video not in split manifest: {key}")
        if role not in allowed:
            raise RuntimeError(f"example {key} has role={role}, allowed={sorted(allowed)}")

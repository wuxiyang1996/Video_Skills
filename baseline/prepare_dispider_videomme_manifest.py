#!/usr/bin/env python3
"""Prepare Dispider's official VideoMME manifest with absolute video paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE = Path(
    "/mnt/is_data/xwu/video_skills/code/Dispider/playground/data/videomme_template.json"
)
DEFAULT_VIDEO_ROOT = Path(
    "/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme/videos"
)


def prepare_manifest(
    template: Path,
    video_root: Path,
    *,
    limit: int | None = None,
    require_all: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(template.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list in {template}")

    prepared: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in payload:
        if limit is not None and len(prepared) >= limit:
            break
        if not isinstance(source, dict):
            continue
        row = dict(source)
        original = Path(str(row.get("video_path") or ""))
        video_id = str(row.get("video_id") or "")
        candidates = [
            video_root / original.name,
            video_root / f"{video_id}.mp4",
        ]
        resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if resolved is None:
            missing.append(video_id or original.name)
            continue
        row["video_path"] = str(resolved)
        prepared.append(row)

    if require_all and missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} VideoMME videos are missing under {video_root}; first entries: {preview}"
        )
    return prepared, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    prepared, missing = prepare_manifest(
        args.template.resolve(),
        args.video_root.resolve(),
        limit=args.limit,
        require_all=not args.allow_missing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "videos": len(prepared),
                "questions": sum(len(row.get("questions") or []) for row in prepared),
                "missing_videos": len(missing),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

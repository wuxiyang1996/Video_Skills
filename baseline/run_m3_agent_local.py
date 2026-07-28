#!/usr/bin/env python3
"""Run official M3 memorization with local embedding and speech backends."""

from __future__ import annotations

import argparse
import glob
import json
import os
import runpy
import sys
from pathlib import Path

from .m3_local_backends import install_into_m3, validate_local_models


def _validate_manifest(path: Path) -> dict[str, int]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    missing_clips = 0
    for row in rows:
        clip_dir = Path(row["clip_path"])
        if not clip_dir.is_dir() or not any(clip_dir.glob("*.mp4")):
            missing_clips += 1
    return {"rows": len(rows), "missing_clip_directories": missing_clips}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m3-repo",
        type=Path,
        default=Path("/mnt/is_data/xwu/video_skills/code/m3-agent"),
    )
    parser.add_argument(
        "--speakerlab-repo",
        type=Path,
        default=Path("/mnt/is_data/xwu/video_skills/code/3D-Speaker"),
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.m3_repo.resolve()
    target = repo / "m3_agent" / "memorization_memory_graphs.py"
    if not target.is_file():
        parser.error(f"official memorization entrypoint not found: {target}")
    if not args.data_file.is_file():
        parser.error(f"manifest not found: {args.data_file}")

    manifest = _validate_manifest(args.data_file.resolve())
    local_models = validate_local_models()
    report = {
        "alignment_class": "official_model_adapted_benchmarks",
        "upstream_entrypoint": str(target),
        "manifest": manifest,
        "local_backends": local_models,
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(manifest["missing_clip_directories"] > 0)

    if manifest["missing_clip_directories"]:
        raise FileNotFoundError(
            f"{manifest['missing_clip_directories']} manifest rows have no materialized clips"
        )
    if os.environ.get("M3_LOCAL_FILES_ONLY", "1") == "1":
        missing = [
            key
            for key in ("embedding_cache_present", "whisper_cache_present")
            if not local_models[key]
        ]
        if missing:
            raise FileNotFoundError(
                "local model cache is incomplete: "
                + ", ".join(missing)
                + "; download under the /mnt Hugging Face cache or set M3_LOCAL_FILES_ONLY=0"
            )

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if not (args.speakerlab_repo / "speakerlab").is_dir():
        raise FileNotFoundError(f"SpeakerLab package not found: {args.speakerlab_repo}")
    sys.path.insert(0, str(args.speakerlab_repo.resolve()))
    backend = install_into_m3()

    # Upstream uses glob order directly. Sorting keeps CLIP_0, CLIP_1, ... in
    # chronological order and is required for streaming entity continuity.
    original_glob = glob.glob
    glob.glob = lambda *a, **kw: sorted(original_glob(*a, **kw))
    print(json.dumps({**report, "installed_backends": backend}, ensure_ascii=False), flush=True)

    sys.argv = [str(target), "--data_file", str(args.data_file.resolve())]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

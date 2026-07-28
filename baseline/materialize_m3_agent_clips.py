#!/usr/bin/env python3
"""Materialize causal 30-second clips for adapted M3-Agent memorization."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _duration_s(path: Path, ffprobe_bin: str | None) -> float:
    if ffprobe_bin and shutil.which(ffprobe_bin):
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ffprobe is unavailable and OpenCV is not installed") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not probe video duration: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata for duration probe: {path}")
    return frames / fps


def _run_ffmpeg(
    *,
    ffmpeg_bin: str,
    source: Path,
    output: Path,
    start_s: float,
    duration_s: float,
    exact_boundary: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    if temporary.exists():
        temporary.unlink()

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_s:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if exact_boundary:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
            ]
        )
    else:
        # Matches the upstream README's regular 30-second segmentation path.
        command.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    command.append(str(temporary))
    subprocess.run(command, check=True)
    temporary.replace(output)


def materialize_row(
    row: dict[str, Any],
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    overwrite: bool,
) -> dict[str, Any]:
    source = Path(row["video_path"]).resolve()
    clip_dir = Path(row["clip_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    source_duration = _duration_s(source, ffprobe_bin)
    cutoff = row.get("observation_end_s")
    requested_end = float(cutoff) if isinstance(cutoff, (int, float)) else source_duration
    effective_end = max(0.0, min(requested_end, source_duration))
    clip_duration = float(row.get("clip_duration_s") or 30.0)

    clip_dir.mkdir(parents=True, exist_ok=True)
    expected = math.ceil(effective_end / clip_duration) if effective_end > 0 else 0
    outputs: list[str] = []
    for clip_id in range(expected):
        start = clip_id * clip_duration
        end = min((clip_id + 1) * clip_duration, effective_end)
        output = clip_dir / f"{clip_id}.mp4"
        outputs.append(str(output))
        if output.exists() and not overwrite:
            continue
        # Re-encode only a final partial clip created by an in-video causal
        # cutoff. This prevents keyframe-level future leakage while keeping the
        # upstream copy path for ordinary full 30-second clips.
        exact_boundary = (
            effective_end < source_duration - 1e-3
            and end >= effective_end - 1e-6
            and effective_end % clip_duration > 1e-6
        )
        _run_ffmpeg(
            ffmpeg_bin=ffmpeg_bin,
            source=source,
            output=output,
            start_s=start,
            duration_s=end - start,
            exact_boundary=exact_boundary,
        )

    metadata = {
        "id": row["id"],
        "source": str(source),
        "source_duration_s": source_duration,
        "requested_observation_end_s": cutoff,
        "effective_observation_end_s": effective_end,
        "clip_duration_s": clip_duration,
        "clip_count": expected,
        "clips": outputs,
        "last_partial_clip_reencoded": bool(
            effective_end < source_duration - 1e-3 and effective_end % clip_duration > 1e-6
        ),
    }
    (clip_dir.parent / "clip_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-plan", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if shutil.which(args.ffmpeg_bin) is None:
        parser.error(f"ffmpeg not found: {args.ffmpeg_bin}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    results = []
    for index, row in enumerate(_iter_jsonl(args.clip_plan)):
        if args.limit is not None and index >= args.limit:
            break
        result = materialize_row(
            row,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            overwrite=args.overwrite,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "graphs_materialized": len(results),
                "clips_materialized_or_reused": sum(row["clip_count"] for row in results),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

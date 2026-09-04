#!/usr/bin/env python3
"""Derive one L1 example per benchmark question from per-video frozen L1.

The layer-1 catalog (clip schemas and clue-memory graph) is built per video and
is question-agnostic: extract_clue_memory_graph takes no question, and in
video_only mode clip schemas carry no question_context.  The staged runner
nevertheless keys its stage directory by example_id, which includes the question
index, and --unique-videos keeps only the first question per video.  For
Video-Holmes that left 270 of 1,837 test questions, 260 of them SR, so every
accuracy measured on it is an SR-subset number rather than the seven-type
leaderboard average.

This copies each video's frozen L1 into one example per question, changing only
the question block, into a lane laid out so --l1-glob picks the files up.  No
captioning is re-run.  Question-conditioned anchor-repass rows, if any, are
dropped from the copy because they belonged to the original question.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import re
from pathlib import Path
from typing import Any, Iterable


def index_frozen_by_video(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    for path in paths:
        example = json.loads(Path(path).read_text(encoding="utf-8"))
        video_id = str((example.get("video") or {}).get("video_id") or example.get("video_id") or "")
        if not video_id:
            match = re.search(r":([^:]+):q\d+$", str(example.get("example_id") or ""))
            video_id = match.group(1) if match else ""
        if video_id and video_id not in by_video:
            by_video[video_id] = example
    return by_video


def derive_example(frozen: dict[str, Any], item: Any) -> dict[str, Any]:
    """Frozen L1 for the video, with this item's identity and question."""
    example = copy.deepcopy(frozen)
    example["example_id"] = str(item.example_id)
    example["question"] = copy.deepcopy(getattr(item, "question", None) or {})
    metadata = example.setdefault("metadata", {})
    # Anchor-repass schemas were conditioned on the original question.
    schemas = metadata.get("clip_schemas") or []
    metadata["clip_schemas"] = [
        s for s in schemas if s.get("schema_attempt_context") != "query_time_anchor_repass"
    ]
    metadata.pop("anchor_repass", None)
    metadata["derived_from_example_id"] = str(frozen.get("example_id") or "")
    metadata["l1_is_question_agnostic_copy"] = True
    return example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", required=True)
    parser.add_argument("--dataset", default="video_holmes")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args(argv)

    from dataset_clip_wrapper.adapters import get_adapter

    by_video = index_frozen_by_video(Path(p) for p in sorted(glob.glob(args.frozen_l1_glob, recursive=True)))
    if args.limit_videos:
        by_video = dict(list(by_video.items())[: args.limit_videos])
    adapter = get_adapter(args.dataset, args.dataset_root)
    adapter.split = args.split
    written = 0
    skipped_no_l1 = 0
    per_video: dict[str, int] = {}
    index: dict[str, dict[str, str]] = {}
    for item in adapter.iter_items():
        video_id = str(item.video_id)
        frozen = by_video.get(video_id)
        if frozen is None:
            skipped_no_l1 += 1
            continue
        example = derive_example(frozen, item)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", example["example_id"])
        out_dir = args.output_root / args.dataset / args.split / "derived" / "stages" / safe
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "04_l1_example.json").write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
        index[example["example_id"]] = {
            "path": str(out_dir / "04_l1_example.json"),
            "question_type": str(example["question"].get("question_type") or "?"),
            "video_id": video_id,
        }
        written += 1
        per_video[video_id] = per_video.get(video_id, 0) + 1
    report = {
        "frozen_videos": len(by_video),
        "examples_written": written,
        "questions_without_l1": skipped_no_l1,
        "questions_per_video_mean": (written / len(per_video)) if per_video else 0.0,
        "output_root": str(args.output_root),
    }
    (args.output_root / "derive_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # A small index so readers need not open every derived file (each carries a
    # multi-thousand-node graph) just to learn an example's type or video.
    (args.output_root / "example_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

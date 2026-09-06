"""Replace a Video-Holmes example's clip catalog with the human annotations.

Every accuracy lever in the answer step is null (frames, voting, reasoning
budget, perfect retrieval); the only one that moved was a stronger reader.
That leaves the L1 descriptions, whose ceiling has never been measured.  This
builds a catalog from the benchmark's own annotation file -- Segment
Description rows (timed narrative) and Inference Shots (timed clue +
conclusion) -- so the same reader answers the same questions over
human-written evidence.  It is a CEILING, not a system: the annotations are
the source the questions were generated from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.adapters.video_holmes import parse_time_range


def annotation_clips(annotation: dict[str, Any], include_inference: bool) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for row in annotation.get("Segment Description") or annotation.get("SegmentDescription") or []:
        span = parse_time_range(row.get("TimeRange"))
        text = str(row.get("Description") or "").strip()
        if span and text:
            clips.append({"clip_id": f"seg:{len(clips)}", "granularity": "segment",
                          "time_span": span, "scene_description": text})
    if include_inference:
        for row in annotation.get("Inference Shots") or annotation.get("InferenceScenes") or []:
            span = parse_time_range(row.get("Time"))
            text = " ".join(str(row.get(k) or "").strip() for k in ("Clue", "Conclusion")).strip()
            if span and text:
                clips.append({"clip_id": f"inf:{len(clips)}", "granularity": "inference_shot",
                              "time_span": span, "scene_description": text})
    clips.sort(key=lambda c: (c["time_span"]["start_s"], c["time_span"]["end_s"]))
    return clips


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-index", type=Path, required=True)
    ap.add_argument("--annotations", type=Path,
                    default=Path("/fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/annotations"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--example-ids", type=Path)
    ap.add_argument("--no-inference-shots", action="store_true",
                    help="Segment Description rows only (the narrative), leaving out the clue/conclusion rows.")
    args = ap.parse_args(argv)

    index = json.loads(args.example_index.read_text())
    wanted = set(args.example_ids.read_text().split()) if args.example_ids else None
    out_index: dict[str, Any] = {}
    written = skipped = 0
    for example_id, meta in index.items():
        if wanted is not None and example_id not in wanted:
            continue
        path = args.annotations / f"{meta['video_id']}.json"
        if not path.exists():
            skipped += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotation = payload[0] if isinstance(payload, list) and payload else payload
        clips = annotation_clips(annotation, not args.no_inference_shots)
        if not clips:
            skipped += 1
            continue
        example = json.loads(Path(meta["path"]).read_text())
        metadata = dict(example.get("metadata") or {})
        metadata["clip_schemas"] = clips
        metadata["coarse_clip_schemas"] = []
        metadata["clip_schema_model"] = "human_annotation"
        example["metadata"] = metadata
        safe = example_id.replace(":", "_")
        out_dir = args.output_root / "stages" / safe
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "04_l1_example.json").write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
        out_index[example_id] = {"path": str(out_dir / "04_l1_example.json"),
                                 "question_type": meta.get("question_type"), "video_id": meta["video_id"]}
        written += 1
    (args.output_root / "example_index.json").write_text(json.dumps(out_index, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"written": written, "skipped": skipped,
                      "mean_clips": round(sum(len(json.loads(Path(v["path"]).read_text())["metadata"]["clip_schemas"])
                                              for v in out_index.values()) / max(written, 1), 1),
                      "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

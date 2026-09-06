"""Append whisper dialogue rows to an existing catalog (no narrative, no re-description).

Isolates the dialogue's share of the narr_px gain: the same clips the reader
already had, plus one row per ~window_s of transcript ("Dialogue 12.2–31.3 s:
..."), so the reader sees what is said without any re-perception.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def dialogue_rows(segments: list[dict[str, Any]], window_s: float = 30.0) -> list[dict[str, Any]]:
    """Group consecutive transcript segments into rows no longer than window_s."""
    rows: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []

    def flush() -> None:
        if not bucket:
            return
        start = float(bucket[0].get("start_s") or 0.0)
        end = float(bucket[-1].get("end_s") or start)
        text = " ".join(str(g.get("text") or "").strip() for g in bucket if str(g.get("text") or "").strip())
        rows.append({"clip_id": f"dialogue:{len(rows) + 1}", "granularity": "dialogue_window",
                     "time_span": {"start_s": start, "end_s": end},
                     "scene_description": f"Dialogue heard {start:.1f}–{end:.1f} s: {text}"})
        bucket.clear()

    for seg in sorted(segments or [], key=lambda g: float(g.get("start_s") or 0.0)):
        if bucket and float(seg.get("end_s") or 0.0) - float(bucket[0].get("start_s") or 0.0) > window_s:
            flush()
        bucket.append(seg)
    flush()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-index", type=Path, required=True)
    ap.add_argument("--example-ids", type=Path)
    ap.add_argument("--asr-dir", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    index = json.loads(args.example_index.read_text())
    wanted = set(args.example_ids.read_text().split()) if args.example_ids else set(index)
    out_index: dict[str, Any] = {}
    n_rows = 0
    for example_id, meta in index.items():
        if example_id not in wanted:
            continue
        example = json.loads(Path(meta["path"]).read_text())
        asr_path = args.asr_dir / f"{meta['video_id']}.json"
        segments = json.loads(asr_path.read_text()).get("segments") if asr_path.exists() else []
        rows = dialogue_rows(segments or [], args.window_s)
        n_rows += len(rows)
        metadata = dict(example.get("metadata") or {})
        metadata["clip_schemas"] = rows + list(metadata.get("clip_schemas") or [])
        metadata["clip_schema_model"] = str(metadata.get("clip_schema_model") or "") + "+asr_rows"
        example["metadata"] = metadata
        out_dir = args.output_root / "stages" / example_id.replace(":", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "04_l1_example.json").write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
        out_index[example_id] = {"path": str(out_dir / "04_l1_example.json"), "question_type": meta.get("question_type"),
                                 "video_id": meta["video_id"]}
    (args.output_root / "example_index.json").write_text(json.dumps(out_index), encoding="utf-8")
    print(json.dumps({"examples": len(out_index), "mean_dialogue_rows": round(n_rows / max(len(out_index), 1), 1),
                      "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

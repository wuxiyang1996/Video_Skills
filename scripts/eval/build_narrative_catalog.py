"""Synthesise a narrative catalog from an example's own clip descriptions.

Human-written Video-Holmes segment rows (3.4 per video) beat our ~61
question-agnostic 4-s clip descriptions by 12 points with the same reader.
Two explanations, with opposite remedies: the information is in our clips but
fragmented (identities and causes do not carry across clips), or it was never
perceived.  This separates them without a GPU: the clips are windowed in time
and a text model writes one narrative paragraph per window *sequentially*,
seeing the previous paragraph and the running cast, so identities persist.
Nothing is looked at again; only the text is re-organised.  If the reader
gains over the synthesised rows, the gap was fragmentation; if not, it is
perception.

Per video the narrative is generated once and cached; every question on that
video reuses it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from scripts.eval.measure_answer_chain import clip_schema_text, retrieval_catalog, sample_clip_frames

NARRATE_SYSTEM = (
    "You turn a sequence of short, independently written clip descriptions from one stretch "
    "of a video into ONE coherent narrative paragraph of what happens in that stretch. Keep "
    "people identifiable across clips with stable descriptors (e.g. 'the woman in the blue "
    "jacket'), reuse the cast list you are given, state actions, reactions, and visible "
    "cause-and-effect in order, and keep anything the clips flag as uncertain. Do not invent "
    "events the clips do not support. Reply with JSON only: "
    '{"narrative": "<paragraph>", "cast": ["<stable descriptor>", ...]}.'
)


ANNOTATE_SYSTEM = (
    "You are annotating one stretch of a short film for a detective-style question set. You are given "
    "evenly spaced frames from that stretch in time order, the dialogue heard in it (if any), the "
    "previous stretch's annotation and the running cast. Write ONE paragraph, the way a careful human "
    "annotator would, of what happens: who is present (stable descriptors reused from the cast, e.g. "
    "'the man with the backpack'), what each person does and says, how they react, what changes, and "
    "what the sequence shows or implies about intentions, relationships and cause-and-effect when the "
    "frames and dialogue make it clear. Quote or paraphrase dialogue in English. Describe events, not "
    "camera work. Do not invent what is not shown or heard. Reply with JSON only: "
    '{"narrative": "<paragraph>", "cast": ["<stable descriptor>", ...]}.'
)


def asr_in_span(segments: list[dict[str, Any]], span: dict[str, Any], pad_s: float = 1.0) -> list[dict[str, Any]]:
    """Transcript segments overlapping the window (padded), as {start_s, end_s, text}."""
    lo, hi = float(span.get("start_s") or 0.0) - pad_s, float(span.get("end_s") or 0.0) + pad_s
    return [{"start_s": g.get("start_s"), "end_s": g.get("end_s"), "text": g.get("text")}
            for g in segments or [] if float(g.get("end_s") or 0.0) > lo and float(g.get("start_s") or 0.0) < hi]


def window_clips(schemas: list[dict[str, Any]], target_s: float = 45.0, min_windows: int = 3) -> list[list[int]]:
    """Consecutive clip indices grouped into ~target_s windows (at least min_windows)."""
    timed = [(i, s) for i, s in enumerate(schemas) if isinstance(s, dict) and isinstance(s.get("time_span"), dict)]
    if not timed:
        return []
    start = min(float(s["time_span"].get("start_s") or 0.0) for _, s in timed)
    end = max(float(s["time_span"].get("end_s") or 0.0) for _, s in timed)
    n = max(min_windows, int(round((end - start) / target_s))) if end > start else min_windows
    width = max((end - start) / n, 1e-6)
    windows: list[list[int]] = [[] for _ in range(n)]
    for i, s in timed:
        mid = (float(s["time_span"].get("start_s") or 0.0) + float(s["time_span"].get("end_s") or 0.0)) / 2.0
        k = min(n - 1, max(0, int((mid - start) / width)))
        windows[k].append(i)
    return [w for w in windows if w]


def narrate_video(client: Any, schemas: list[dict[str, Any]], windows: list[list[int]],
                  per_clip_chars: int = 900, *, video_path: str | None = None, frames_per_window: int = 0,
                  asr_segments: list[dict[str, Any]] | None = None, use_clip_text: bool = True,
                  frame_width: int = 448) -> list[dict[str, Any]]:
    """Sequential synthesis: each window sees the previous narrative and the running cast.

    With `frames_per_window` > 0 the model also *looks* at evenly spaced frames of the window
    (and reads the ASR segments in it), which turns the text re-organiser into a describer.
    """
    rows: list[dict[str, Any]] = []
    previous = ""
    cast: list[str] = []
    looking = frames_per_window > 0 and bool(video_path)
    for k, window in enumerate(windows, start=1):
        clips = []
        for i in window:
            s = schemas[i]
            clips.append({"time_span": s.get("time_span") or {}, "description": clip_schema_text(s)[:per_clip_chars]})
        span = {"start_s": float(schemas[window[0]]["time_span"].get("start_s") or 0.0),
                "end_s": float(schemas[window[-1]]["time_span"].get("end_s") or 0.0)}
        payload: dict[str, Any] = {"window": k, "of": len(windows), "time_span": span,
                                   "previous_narrative": previous[:1500], "cast_so_far": cast[:20]}
        if use_clip_text:
            payload["clips"] = clips
        if asr_segments is not None:
            payload["dialogue"] = asr_in_span(asr_segments, span)
        frames = sample_clip_frames(video_path, span, frames_per_window, width=frame_width) if looking else []
        if frames:
            content: Any = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            for jpeg in frames:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg}"}})
        else:
            content = json.dumps(payload, ensure_ascii=False)
        text = client.chat([
            {"role": "system", "content": ANNOTATE_SYSTEM if looking else NARRATE_SYSTEM},
            {"role": "user", "content": content},
        ])
        narrative = ""
        try:
            parsed = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0)) or {}
            narrative = str(parsed.get("narrative") or "").strip()
            for name in parsed.get("cast") or []:
                name = str(name).strip()
                if name and name not in cast:
                    cast.append(name)
        except Exception:
            narrative = (text or "").strip()
        if not narrative:
            narrative = " ".join(c["description"][:300] for c in clips)   # never drop a window
        rows.append({"clip_id": f"narrative:{k}", "granularity": "narrative_window", "time_span": span,
                     "scene_description": narrative[:2500], "source_clip_count": len(window),
                     "frames_seen": len(frames), "dialogue_lines": len(payload.get("dialogue") or [])})
        previous = narrative
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-index", type=Path, required=True)
    ap.add_argument("--example-ids", type=Path)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    ap.add_argument("--window-s", type=float, default=45.0)
    ap.add_argument("--keep-clips", action="store_true",
                    help="Append the original clips after the narrative rows (hybrid catalog) instead of replacing them.")
    ap.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames-per-window", type=int, default=0,
                    help="Look at this many evenly spaced frames per window (0 = text-only re-organisation).")
    ap.add_argument("--asr-dir", type=Path, help="Per-video whisper JSON (scripts/eval/transcribe_videos.py); dialogue in each window is passed to the model.")
    ap.add_argument("--no-clip-text", action="store_true", help="Do not pass the clip descriptions; frames + dialogue only.")
    args = ap.parse_args(argv)
    if args.no_clip_text and not args.frames_per_window:
        ap.error("--no-clip-text needs --frames-per-window > 0")

    index = json.loads(args.example_index.read_text())
    wanted = set(args.example_ids.read_text().split()) if args.example_ids else set(index)
    client = OpenRouterClient(model=args.model, api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
                              max_tokens=1200, temperature=0.0, reasoning={"effort": "minimal", "exclude": True},
                              timeout_s=300)
    cache_dir = args.output_root / "narratives"
    cache_dir.mkdir(parents=True, exist_ok=True)

    by_video: dict[str, list[str]] = {}
    for example_id, meta in index.items():
        if example_id in wanted:
            by_video.setdefault(meta["video_id"], []).append(example_id)

    def build(video_id: str) -> tuple[str, list[dict[str, Any]]]:
        cache = cache_dir / f"{video_id}.json"
        if cache.exists():
            return video_id, json.loads(cache.read_text())
        example = json.loads(Path(index[by_video[video_id][0]]["path"]).read_text())
        schemas, _ = retrieval_catalog(example)
        asr = None
        if args.asr_dir:
            asr_path = args.asr_dir / f"{video_id}.json"
            asr = (json.loads(asr_path.read_text()).get("segments") or []) if asr_path.exists() else []
        rows = narrate_video(client, schemas, window_clips(schemas, args.window_s),
                             video_path=(example.get("video") or {}).get("primary_path"),
                             frames_per_window=args.frames_per_window, asr_segments=asr,
                             use_clip_text=not args.no_clip_text)
        cache.write_text(json.dumps(rows, ensure_ascii=False))
        return video_id, rows

    from concurrent.futures import ThreadPoolExecutor
    narratives: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for video_id, rows in pool.map(build, sorted(by_video)):
            narratives[video_id] = rows
            print(f"[{len(narratives)}/{len(by_video)}] {video_id}: {len(rows)} narrative rows", flush=True)

    out_index: dict[str, Any] = {}
    for video_id, example_ids in by_video.items():
        rows = narratives.get(video_id) or []
        for example_id in example_ids:
            meta = index[example_id]
            example = json.loads(Path(meta["path"]).read_text())
            schemas, _ = retrieval_catalog(example)
            metadata = dict(example.get("metadata") or {})
            metadata["clip_schemas"] = rows + (list(schemas) if args.keep_clips else [])
            metadata["coarse_clip_schemas"] = []
            metadata["clip_schema_model"] = (f"narrative:{args.model}" + (f"+frames{args.frames_per_window}" if args.frames_per_window else "")
                                             + ("+asr" if args.asr_dir else "") + ("-cliptext" if args.no_clip_text else "")
                                             + ("+clips" if args.keep_clips else ""))
            example["metadata"] = metadata
            out_dir = args.output_root / "stages" / example_id.replace(":", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "04_l1_example.json").write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
            out_index[example_id] = {"path": str(out_dir / "04_l1_example.json"),
                                     "question_type": meta.get("question_type"), "video_id": video_id}
    (args.output_root / "example_index.json").write_text(json.dumps(out_index, ensure_ascii=False), encoding="utf-8")
    mean_rows = sum(len(r) for r in narratives.values()) / max(len(narratives), 1)
    print(json.dumps({"videos": len(narratives), "examples": len(out_index), "mean_narrative_rows": round(mean_rows, 1),
                      "keep_clips": args.keep_clips, "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

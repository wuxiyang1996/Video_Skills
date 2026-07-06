"""End-to-end smoke test: drive each benchmark through GPT-4o.

For every benchmark adapter in :mod:`visual_grounding.pipeline`, this
script takes a single sample video, runs the unified grounding pipeline
with a live VLM backend, and asserts that the result satisfies the
canonical schema contract (``DirectContext`` for short videos,
``SocialVideoGraph`` for long videos).

Unlike the pytest suite in ``tests/visual_grounding/`` (which uses a
deterministic stub VLM), this script is meant to be run manually to
confirm the pipeline actually works against a real model.

Provider selection:
    * ``--provider gpt4o`` (default) → OpenAI GPT-4o / GPT-4o-mini via
      OpenAI or OpenRouter. Current scaffolding.
    * ``--provider claude`` → Anthropic Claude (vision).
    * ``--provider qwen3vl`` → local vLLM server (Qwen3-VL-32B or
      Qwen3-VL-72B). **Production target** — the pipeline doesn't
      change, only the provider binding does.

Each benchmark deliberately uses small sampling parameters to keep
cost/latency bounded: ``fps=0.05``, small ``window_seconds``,
``max_frames_per_window=2``, ``include_scene_changes=False``, and a
``duration=`` override for multi-hour videos.

Usage::

    cd /fs/gamma-projects/vlm-robot/Video_Skills
    # default: GPT-4o via OpenAI / OpenRouter
    python scripts/test_gpt4o_grounding.py --dump out/gpt4o_grounding
    # cheaper scaffolding pass
    python scripts/test_gpt4o_grounding.py --model gpt-4o-mini
    # subset only
    python scripts/test_gpt4o_grounding.py --only video_holmes siv_bench
    # swap to Qwen3-VL-32B on vLLM once the server is up
    python scripts/test_gpt4o_grounding.py \
        --provider qwen3vl --model Qwen/Qwen3-VL-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

# Make the package importable when running as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from visual_grounding import (  # noqa: E402
    DirectContext,
    GroundedWindow,
    SocialVideoGraph,
    build_for_cg_bench,
    build_for_long_video_bench,
    build_for_m3_bench,
    build_for_siv_bench,
    build_for_video_holmes,
    build_for_vrbench,
    make_vlm,
)


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

_REPO = "/fs/gamma-projects/vlm-robot"

BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "video_holmes": {
        "builder": build_for_video_holmes,
        "video": f"{_REPO}/Video_Skills/dataset_examples/video_holmes/0at001QMutY.mp4",
        "subtitle": None,
        "expected_type": DirectContext,
        "kwargs": {
            "window_seconds": 20.0,
            "fps": 0.1,
            "max_frames_per_window": 2,
            "include_scene_changes": False,
        },
    },
    "siv_bench": {
        "builder": build_for_siv_bench,
        "video": f"{_REPO}/datasets/SIV-Bench/origin/boss-employee/video_100.mp4",
        "subtitle": None,
        "expected_type": DirectContext,
        "kwargs": {
            "subtitle_mode": "origin",
            "window_seconds": 20.0,
            "fps": 0.1,
            "max_frames_per_window": 2,
            "include_scene_changes": False,
        },
    },
    "vrbench": {
        "builder": build_for_vrbench,
        "video": f"{_REPO}/datasets/VRBench/v001_360p/00mFFDva8OE.mp4",
        "subtitle": None,
        "expected_type": SocialVideoGraph,
        "kwargs": {
            # VRBench clip is ~90 min; cap so we only run a couple of windows.
            "duration": 60.0,
            "window_seconds": 30.0,
            "fps": 0.05,
            "max_frames_per_window": 2,
            "include_scene_changes": False,
            "top_k": 3,
        },
    },
    "long_video_bench": {
        "builder": build_for_long_video_bench,
        "video": f"{_REPO}/datasets/LongVideoBench/videos/005BeD0c2PA.mp4",
        "subtitle": f"{_REPO}/datasets/LongVideoBench/subtitles/005BeD0c2PA_en.json",
        "expected_type": SocialVideoGraph,
        "kwargs": {
            # Force retrieval mode even for the short sample clip.
            "duration": 200.0,
            "window_seconds": 60.0,
            "fps": 0.05,
            "max_frames_per_window": 2,
            "include_scene_changes": False,
            "top_k": 3,
        },
    },
    "cg_bench": {
        "builder": build_for_cg_bench,
        "video": f"{_REPO}/datasets/CG-Bench/0.mp4",
        "subtitle": None,
        "expected_type": SocialVideoGraph,
        "kwargs": {
            "duration": 60.0,
            "window_seconds": 20.0,
            "fps": 0.1,
            "max_frames_per_window": 2,
            "include_scene_changes": False,
            "top_k": 3,
        },
    },
    "m3_bench": {
        "builder": build_for_m3_bench,
        "video": f"{_REPO}/Video_Skills/dataset_examples/m3_bench/robot/bedroom_01.mp4",
        "subtitle": f"{_REPO}/Video_Skills/dataset_examples/m3_bench/robot/bedroom_01.srt",
        "expected_type": SocialVideoGraph,
        "kwargs": {
            "duration": 60.0,
            "window_seconds": 30.0,
            "fps": 0.05,
            "max_frames_per_window": 2,
            "include_scene_changes": False,
            "top_k": 3,
        },
    },
}


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------


def _validate_grounded_window(gw: GroundedWindow) -> List[str]:
    errs: List[str] = []
    if not isinstance(gw, GroundedWindow):
        errs.append(f"window is {type(gw).__name__}, not GroundedWindow")
        return errs
    if not gw.window_id:
        errs.append("window_id is empty")
    if not (isinstance(gw.time_span, tuple) and len(gw.time_span) == 2
            and gw.time_span[0] <= gw.time_span[1]):
        errs.append(f"bad time_span {gw.time_span!r}")
    for name in ("entities", "interactions", "events",
                 "social_hypotheses", "evidence"):
        if not isinstance(getattr(gw, name), list):
            errs.append(f"{name} is not a list")
    return errs


def _validate_direct_context(ctx: DirectContext) -> Tuple[List[str], Dict[str, Any]]:
    errs: List[str] = []
    if not isinstance(ctx, DirectContext):
        return [f"expected DirectContext got {type(ctx).__name__}"], {}
    if ctx.mode.value != "direct":
        errs.append(f"mode is {ctx.mode.value}, expected 'direct'")
    if not ctx.windows:
        errs.append("no windows produced")
    for gw in ctx.windows:
        errs.extend(_validate_grounded_window(gw))
    try:
        ctx.to_dict()
    except Exception as exc:
        errs.append(f"to_dict() failed: {exc}")
    try:
        _ = ctx.as_reasoner_text()
    except Exception as exc:
        errs.append(f"as_reasoner_text() failed: {exc}")

    totals = {
        "windows": len(ctx.windows),
        "entities": sum(len(w.entities) for w in ctx.windows),
        "interactions": sum(len(w.interactions) for w in ctx.windows),
        "events": sum(len(w.events) for w in ctx.windows),
        "hypotheses": sum(len(w.social_hypotheses) for w in ctx.windows),
        "evidence": sum(len(w.evidence) for w in ctx.windows),
    }
    return errs, totals


def _validate_social_video_graph(
    graph: SocialVideoGraph,
) -> Tuple[List[str], Dict[str, Any]]:
    errs: List[str] = []
    if not isinstance(graph, SocialVideoGraph):
        return [f"expected SocialVideoGraph got {type(graph).__name__}"], {}
    if graph.mode != "retrieval":
        errs.append(f"mode is {graph.mode}, expected 'retrieval'")
    stats = graph.stats()
    if stats.get("total", 0) < 1:
        errs.append("graph has no nodes")
    if stats.get("episodic", 0) < 1:
        errs.append("graph has no episodic backbone nodes")
    try:
        hits = graph.search("person talking", top_k=3)
    except Exception as exc:
        errs.append(f"search() failed: {exc}")
        hits = []
    for node, score in hits:
        if not (hasattr(node, "node_id") and hasattr(node, "node_type")):
            errs.append(f"search() returned non-GroundingNode: {type(node)}")
            break
    return errs, {"stats": stats, "search_hits": len(hits)}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _context_to_preview(ctx: Any) -> Dict[str, Any]:
    """Small JSON-safe preview of the produced context (first window only)."""
    if isinstance(ctx, DirectContext):
        first = ctx.windows[0] if ctx.windows else None
        return {
            "type": "DirectContext",
            "duration": ctx.duration,
            "num_windows": len(ctx.windows),
            "first_window": first.to_dict() if first is not None else None,
        }
    if isinstance(ctx, SocialVideoGraph):
        return {
            "type": "SocialVideoGraph",
            "stats": ctx.stats(),
            "sample_search": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "text": n.text[:120],
                    "score": s,
                }
                for n, s in ctx.search("who is talking", top_k=3)
            ],
        }
    return {"type": type(ctx).__name__, "repr": repr(ctx)[:500]}


def _dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    key: str,
    cfg: Dict[str, Any],
    *,
    vlm_fn: Any,
    frame_cache_dir: str,
    dump_dir: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    print(f"\n=== [{key}] ===")
    video_path = cfg["video"]
    subtitle_path = cfg.get("subtitle")
    builder = cfg["builder"]
    expected_type = cfg["expected_type"]
    kwargs = dict(cfg["kwargs"])

    if not os.path.isfile(video_path):
        print(f"  SKIP: video missing at {video_path}")
        return {"benchmark": key, "status": "skipped",
                "reason": "video_missing", "video": video_path}

    call_kwargs: Dict[str, Any] = {
        "vlm_fn": vlm_fn,
        "frame_cache_dir": os.path.join(frame_cache_dir, key),
        **kwargs,
    }
    # Builders accept subtitle_path only where it is meaningful.
    if builder in (build_for_siv_bench, build_for_long_video_bench, build_for_m3_bench):
        call_kwargs["subtitle_path"] = subtitle_path if (
            subtitle_path and os.path.isfile(subtitle_path)
        ) else None

    start = time.monotonic()
    ctx: Any
    try:
        ctx = builder(video_path, **call_kwargs)
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  FAIL: {exc!r} after {elapsed:.1f}s")
        if verbose:
            traceback.print_exc()
        return {
            "benchmark": key,
            "status": "error",
            "error": repr(exc),
            "elapsed_s": round(elapsed, 2),
        }
    elapsed = time.monotonic() - start

    ok_type = isinstance(ctx, expected_type)
    if isinstance(ctx, DirectContext):
        errs, summary = _validate_direct_context(ctx)
    elif isinstance(ctx, SocialVideoGraph):
        errs, summary = _validate_social_video_graph(ctx)
    else:
        errs, summary = [f"unexpected type {type(ctx).__name__}"], {}

    status = "ok" if (ok_type and not errs) else "bad_schema"
    print(f"  type: {type(ctx).__name__} "
          f"(expected {expected_type.__name__}) → {'OK' if ok_type else 'MISMATCH'}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  summary: {summary}")
    if errs:
        print(f"  schema issues ({len(errs)}):")
        for e in errs:
            print(f"    - {e}")

    preview = _context_to_preview(ctx)
    if dump_dir:
        out_path = os.path.join(dump_dir, f"{key}.json")
        dump_obj: Dict[str, Any] = {
            "benchmark": key,
            "video": video_path,
            "elapsed_s": round(elapsed, 2),
            "status": status,
            "schema_issues": errs,
            "summary": summary,
            "preview": preview,
        }
        # For DirectContext we can also dump the full grounded context.
        if isinstance(ctx, DirectContext):
            dump_obj["full_context"] = ctx.to_dict()
        _dump_json(out_path, dump_obj)
        print(f"  dumped → {out_path}")

    return {
        "benchmark": key,
        "status": status,
        "result_type": type(ctx).__name__,
        "elapsed_s": round(elapsed, 2),
        "schema_issues": errs,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gpt4o",
                        choices=["gpt4o", "claude", "qwen3vl"],
                        help="VLM backend. 'qwen3vl' is the production "
                             "target (local vLLM); 'gpt4o' / 'claude' are "
                             "scaffolding for now.")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=list(BENCHMARK_CONFIGS.keys()),
                        help="Run only a subset of benchmarks.")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: gpt-4o for gpt4o, "
                             "claude-3-5-sonnet-20241022 for claude, "
                             "Qwen/Qwen3-VL-32B-Instruct for qwen3vl.")
    parser.add_argument("--max-frames", type=int, default=2,
                        help="Max images to attach per VLM call.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--image-detail", default="low",
                        choices=["low", "high", "auto"],
                        help="OpenAI/vLLM 'image_url.detail' hint (ignored by Claude).")
    parser.add_argument("--direct-openai", action="store_true",
                        help="Skip OpenRouter even if OPENROUTER_API_KEY is set "
                             "(gpt4o provider only).")
    parser.add_argument("--vllm-base-url", default=None,
                        help="Override VLLM_BASE_URL for --provider qwen3vl.")
    parser.add_argument("--dump", default=None,
                        help="Directory to dump per-benchmark JSON results.")
    parser.add_argument("--frame-cache", default="/tmp/vg_vlm_frames",
                        help="Directory for extracted frame JPEGs.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    default_models = {
        "gpt4o": "gpt-4o",
        "claude": "claude-sonnet-4-5-20250929",
        "qwen3vl": "Qwen/Qwen3-VL-32B-Instruct",
    }
    model = args.model or default_models[args.provider]

    vlm_kwargs: Dict[str, Any] = dict(
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_frames=args.max_frames,
    )
    if args.provider == "gpt4o":
        vlm_kwargs["prefer_openrouter"] = not args.direct_openai
        vlm_kwargs["image_detail"] = args.image_detail
    elif args.provider == "qwen3vl":
        vlm_kwargs["image_detail"] = args.image_detail
        if args.vllm_base_url:
            vlm_kwargs["base_url"] = args.vllm_base_url

    vlm_fn = make_vlm(args.provider, **vlm_kwargs)

    targets = args.only or list(BENCHMARK_CONFIGS.keys())
    print(f"Running {len(targets)} benchmark(s) "
          f"provider={args.provider} model={model} "
          f"max_frames={args.max_frames}")

    results: List[Dict[str, Any]] = []
    for key in targets:
        cfg = BENCHMARK_CONFIGS[key]
        r = run_benchmark(
            key, cfg,
            vlm_fn=vlm_fn,
            frame_cache_dir=args.frame_cache,
            dump_dir=args.dump,
            verbose=args.verbose,
        )
        results.append(r)

    print("\n=== SUMMARY ===")
    for r in results:
        line = f"{r['benchmark']:<18} {r['status']:<10}"
        if "result_type" in r:
            line += f" {r['result_type']}"
        if "elapsed_s" in r:
            line += f"  ({r['elapsed_s']}s)"
        if r.get("schema_issues"):
            line += f"  [{len(r['schema_issues'])} issue(s)]"
        print(line)

    if args.dump:
        _dump_json(os.path.join(args.dump, "summary.json"),
                   {"provider": args.provider, "model": model, "results": results})

    bad = [r for r in results if r["status"] not in ("ok", "skipped")]
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

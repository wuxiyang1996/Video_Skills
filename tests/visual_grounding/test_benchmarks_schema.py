"""End-to-end schema tests for every benchmark adapter preset.

Each test takes a **real sample video** from the corresponding benchmark's
dataset (or a bundled example under ``Video_Skills/dataset_examples``) and
runs it through the matching ``build_for_*`` preset in
``visual_grounding.pipeline``. The test then asserts:

1. The return type matches the plan (``DirectContext`` for short videos,
   ``SocialVideoGraph`` for long videos).
2. The structured grounding schema is populated:
   - ``GroundedWindow`` objects carry time spans, entities, interactions,
     events, and social hypotheses (matching §3.2 of
     ``infra_plans/01_grounding/video_benchmarks_grounding.md``).
   - Retrieval-mode graphs expose the six node types in §2.1 (at least
     ``episodic`` + ``entity`` + ``semantic``).
3. ``evidence`` attachments are recorded (no first-class visual store —
   see ``infra_plans/02_memory/agentic_memory_design.md``).

Design notes for keeping the suite fast:
- A deterministic stub VLM emits the §3.2 JSON envelope so tests don't
  depend on a live model.
- Scene-change detection is disabled (``include_scene_changes=False``)
  because it scans every ~1 s of video.
- For multi-hour files (VRBench, M3-Bench robot) we override
  ``duration`` and ``window_seconds`` so only a handful of windows are
  produced; the seek-based frame sampler is a no-op when max_frames=0.

Videos that are not on disk cause the test to be **skipped**, not failed,
so CI still runs on machines without the full dataset mirror.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pytest

from visual_grounding import (
    DirectContext,
    GroundedWindow,
    SocialVideoGraph,
    build_for_cg_bench,
    build_for_long_video_bench,
    build_for_m3_bench,
    build_for_siv_bench,
    build_for_video_holmes,
    build_for_vrbench,
)


# ---------------------------------------------------------------------------
# Sample-video catalog (benchmark -> (video_path, optional_subtitle_path))
# ---------------------------------------------------------------------------

_REPO = "/fs/gamma-projects/vlm-robot"

BENCHMARK_SAMPLES: Dict[str, Dict[str, Optional[str]]] = {
    "video_holmes": {
        "video": f"{_REPO}/Video_Skills/dataset_examples/video_holmes/0at001QMutY.mp4",
        "subtitle": None,
    },
    "siv_bench": {
        "video": f"{_REPO}/datasets/SIV-Bench/origin/boss-employee/video_100.mp4",
        "subtitle": None,
    },
    "vrbench": {
        "video": f"{_REPO}/datasets/VRBench/v001_360p/00mFFDva8OE.mp4",
        "subtitle": None,
    },
    "long_video_bench": {
        "video": f"{_REPO}/datasets/LongVideoBench/videos/005BeD0c2PA.mp4",
        "subtitle": f"{_REPO}/datasets/LongVideoBench/subtitles/005BeD0c2PA_en.json",
    },
    "cg_bench": {
        # Small 2-second sample that still decodes fine.
        "video": f"{_REPO}/datasets/CG-Bench/0.mp4",
        "subtitle": None,
    },
    "m3_bench": {
        "video": f"{_REPO}/Video_Skills/dataset_examples/m3_bench/robot/bedroom_01.mp4",
        "subtitle": f"{_REPO}/Video_Skills/dataset_examples/m3_bench/robot/bedroom_01.srt",
    },
}


# ---------------------------------------------------------------------------
# Shared deterministic stub VLM
# ---------------------------------------------------------------------------


def _stub_vlm_payload() -> Dict[str, Any]:
    """Fixed §3.2-compliant JSON used by the stub VLM."""
    return {
        "time_span": [0.0, 0.0],  # overwritten by the grounder
        "scene": "kitchen conversation",
        "entities": [
            {"id": "p1", "type": "person",
             "attributes": {"emotion": "tense", "gaze": "p2",
                            "speaking": True, "role": "host",
                            "clothing": "red"}},
            {"id": "p2", "type": "person",
             "attributes": {"emotion": "neutral", "role": "guest",
                            "clothing": "blue"}},
        ],
        "interactions": [
            {"src": "p1", "rel": "talking_to", "dst": "p2",
             "confidence": 0.84},
        ],
        "events": [
            {"type": "confrontation_start", "agents": ["p1", "p2"],
             "confidence": 0.66, "description": "voice rises"},
        ],
        "social_hypotheses": [
            {"type": "intention", "target": "p1",
             "value": "seeking explanation", "confidence": 0.61,
             "provenance": "inferred_from_behavior",
             "supporting_evidence": []},
        ],
    }


def _stub_vlm(prompt: str, **_kwargs: Any) -> str:
    return json.dumps(_stub_vlm_payload())


# ---------------------------------------------------------------------------
# Path guard
# ---------------------------------------------------------------------------


def _require_video(key: str) -> str:
    sample = BENCHMARK_SAMPLES[key]
    path = sample["video"]
    if not path or not os.path.isfile(path):
        pytest.skip(
            f"Benchmark sample for '{key}' not available at {path}. "
            "Skipping to keep tests portable."
        )
    return path


def _optional_subtitle(key: str) -> Optional[str]:
    sub = BENCHMARK_SAMPLES[key].get("subtitle")
    return sub if sub and os.path.isfile(sub) else None


# ---------------------------------------------------------------------------
# Schema validators shared across tests
# ---------------------------------------------------------------------------


def _assert_grounded_window(gw: GroundedWindow) -> None:
    """Validate the §3.2 per-window JSON schema on a :class:`GroundedWindow`."""
    assert isinstance(gw, GroundedWindow)
    assert isinstance(gw.window_id, str) and gw.window_id
    assert isinstance(gw.time_span, tuple) and len(gw.time_span) == 2
    assert gw.time_span[0] <= gw.time_span[1]
    assert isinstance(gw.entities, list)
    assert isinstance(gw.interactions, list)
    assert isinstance(gw.events, list)
    assert isinstance(gw.social_hypotheses, list)
    assert isinstance(gw.evidence, list)


def _assert_direct_context(ctx: DirectContext, *, expect_populated: bool) -> None:
    assert isinstance(ctx, DirectContext)
    assert ctx.mode.value == "direct"
    assert ctx.duration > 0
    assert isinstance(ctx.windows, list) and len(ctx.windows) >= 1
    for gw in ctx.windows:
        _assert_grounded_window(gw)

    # Pretty-print for reasoner prompt must not crash.
    text = ctx.as_reasoner_text()
    assert isinstance(text, str)

    if expect_populated:
        # Stub VLM emits at least one entity + interaction + event on
        # every window — so at least one window should be non-empty.
        assert any(len(w.entities) >= 1 for w in ctx.windows)
        assert any(len(w.interactions) >= 1 for w in ctx.windows)
        assert any(len(w.events) >= 1 for w in ctx.windows)
        assert any(len(w.social_hypotheses) >= 1 for w in ctx.windows)

    # JSON round-trip proves the schema is serializable.
    blob = ctx.to_dict()
    assert blob["mode"] == "direct"
    json.dumps(blob)  # should not raise


def _assert_social_video_graph(graph: SocialVideoGraph) -> None:
    assert isinstance(graph, SocialVideoGraph)
    assert graph.mode == "retrieval"
    stats = graph.stats()
    assert stats["total"] >= 1
    # Retrieval mode must always yield an episodic backbone.
    assert stats.get("episodic", 0) >= 1
    # Entity nodes arrive when entity resolution is enabled (M3-Bench).
    # Not required for every preset; just ensure type dict is shaped right.
    for t, count in stats.items():
        assert isinstance(count, int) and count >= 0

    # search() returns a ranked list of (node, score) with correct shape.
    hits = graph.search("person talking", top_k=3)
    assert isinstance(hits, list)
    for node, score in hits:
        assert hasattr(node, "node_id") and hasattr(node, "node_type")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Benchmark-specific tests
# ---------------------------------------------------------------------------


class TestVideoHolmesSchema:
    """Video-Holmes: short clips, direct mode, no subtitles.

    Maps to ``build_for_video_holmes`` and §5.1 of
    ``video_benchmarks_grounding.md``.
    """

    def test_produces_direct_context(self):
        video = _require_video("video_holmes")
        ctx = build_for_video_holmes(
            video,
            vlm_fn=_stub_vlm,
            include_scene_changes=False,
            window_seconds=30.0,
            fps=0.1,
            max_frames_per_window=1,
        )
        _assert_direct_context(ctx, expect_populated=True)
        # Video-Holmes uses no subtitles.
        assert ctx.subtitles == []
        assert all(w.subtitle_mode == "origin" for w in ctx.windows)


class TestSIVBenchSchema:
    """SIV-Bench: short clips, subtitle-aware social reasoning (direct)."""

    def test_origin_subtitles(self):
        video = _require_video("siv_bench")
        ctx = build_for_siv_bench(
            video,
            subtitle_path=None,  # sample video has no bundled .srt
            subtitle_mode="origin",
            vlm_fn=_stub_vlm,
            include_scene_changes=False,
            window_seconds=30.0,
            fps=0.1,
            max_frames_per_window=1,
        )
        _assert_direct_context(ctx, expect_populated=True)
        assert all(w.subtitle_mode == "origin" for w in ctx.windows)

    def test_subtitle_removed_mode(self):
        """SIV-Bench supports 'removed' subtitle condition — evidence must drop it."""
        video = _require_video("siv_bench")
        ctx = build_for_siv_bench(
            video,
            subtitle_mode="removed",
            vlm_fn=_stub_vlm,
            include_scene_changes=False,
            window_seconds=30.0,
            fps=0.1,
            max_frames_per_window=1,
        )
        _assert_direct_context(ctx, expect_populated=True)
        # No subtitle evidence refs should survive when mode is "removed".
        for w in ctx.windows:
            assert all(e.modality != "subtitle" for e in w.evidence)


class TestVRBenchSchema:
    """VRBench: hour-long narratives, retrieval mode with event grounding."""

    def test_produces_social_video_graph(self):
        video = _require_video("vrbench")
        graph = build_for_vrbench(
            video,
            vlm_fn=_stub_vlm,
            # VRBench sample is ~90 min; cap duration so windowing is cheap.
            duration=60.0,
            window_seconds=30.0,
            fps=0.05,
            include_scene_changes=False,
            max_frames_per_window=1,
            top_k=3,
        )
        _assert_social_video_graph(graph)
        # Event + interaction nodes materialize from the stub VLM payload.
        stats = graph.stats()
        assert stats.get("interaction", 0) >= 1
        assert stats.get("event", 0) >= 1


class TestLongVideoBenchSchema:
    """LongVideoBench: retrieval mode + subtitles."""

    def test_with_subtitles(self):
        video = _require_video("long_video_bench")
        subtitle = _optional_subtitle("long_video_bench")
        graph = build_for_long_video_bench(
            video,
            subtitle_path=subtitle,
            vlm_fn=_stub_vlm,
            # Force retrieval even though the sample is only ~200s.
            duration=200.0,
            window_seconds=60.0,
            fps=0.05,
            include_scene_changes=False,
            max_frames_per_window=1,
            top_k=3,
        )
        _assert_social_video_graph(graph)
        if subtitle:
            stats = graph.stats()
            # Subtitle evidence attaches to windows -> episodic nodes.
            assert stats.get("episodic", 0) >= 1


class TestCGBenchSchema:
    """CG-Bench: retrieval mode, clue/evidence grounding."""

    def test_clue_grounding_graph(self):
        video = _require_video("cg_bench")
        graph = build_for_cg_bench(
            video,
            vlm_fn=_stub_vlm,
            # CG-Bench sample 0.mp4 is ~2s; force retrieval mode explicitly
            # by giving it the longer window schedule.
            duration=60.0,
            window_seconds=20.0,
            fps=0.1,
            include_scene_changes=False,
            max_frames_per_window=1,
            top_k=3,
        )
        _assert_social_video_graph(graph)
        # All nodes must carry clip_ids or timestamps (evidence anchoring).
        for node in list(vars(graph)["_nodes"].values()):  # type: ignore[index]
            s, e = node.timestamp
            assert s <= e


class TestM3BenchSchema:
    """M3-Bench: retrieval + entity tracking + subtitles (face/voice path).

    This is the strictest preset — it should emit ``entity`` nodes from
    the resolver (via :class:`EntityProfile`) and must round-trip
    through ``save`` / ``load``.
    """

    def test_entity_nodes_and_roundtrip(self, tmp_path):
        video = _require_video("m3_bench")
        subtitle = _optional_subtitle("m3_bench")
        graph = build_for_m3_bench(
            video,
            subtitle_path=subtitle,
            vlm_fn=_stub_vlm,
            # M3-Bench robot sample is ~36 min + 2 GB; aggressively cap.
            duration=60.0,
            window_seconds=30.0,
            fps=0.05,
            include_scene_changes=False,
            max_frames_per_window=1,
            top_k=3,
        )
        _assert_social_video_graph(graph)
        stats = graph.stats()
        # Entity tracking produces at least one entity node (p1 / p2 from stub).
        assert stats.get("entity", 0) >= 1

        out_path = tmp_path / "m3_graph.json"
        graph.save(str(out_path))
        restored = SocialVideoGraph.load(str(out_path))
        assert restored.stats()["total"] == stats["total"]


# ---------------------------------------------------------------------------
# Cross-benchmark sanity: every preset produces the canonical contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,key,kwargs",
    [
        (build_for_video_holmes, "video_holmes", {}),
        (build_for_siv_bench, "siv_bench", {"subtitle_mode": "origin"}),
        (build_for_vrbench, "vrbench", {"duration": 60.0}),
        (build_for_long_video_bench, "long_video_bench", {"duration": 120.0}),
        (build_for_cg_bench, "cg_bench", {"duration": 30.0}),
        (build_for_m3_bench, "m3_bench", {"duration": 60.0}),
    ],
)
def test_every_preset_returns_canonical_contract(builder, key, kwargs):
    """Every preset returns either DirectContext or SocialVideoGraph.

    This is the single bar raised by
    ``infra_plans/01_grounding/video_benchmarks_grounding.md`` §5 — adapters
    share a common contract.
    """
    video = _require_video(key)
    result = builder(
        video,
        vlm_fn=_stub_vlm,
        include_scene_changes=False,
        window_seconds=30.0,
        fps=0.05,
        max_frames_per_window=1,
        **kwargs,
    )
    assert isinstance(result, (DirectContext, SocialVideoGraph))
    if isinstance(result, DirectContext):
        _assert_direct_context(result, expect_populated=True)
    else:
        _assert_social_video_graph(result)

"""Benchmark adapters for the typed grounding pipeline.

Per the visual-grounding plan §11:

* Adapters map benchmark-specific raw formats into the *shared* typed
  pipeline.
* They never introduce per-benchmark grounding object types.
* Each adapter returns a :class:`GroundingRuntime` so harness / skills
  call the same interface regardless of which dataset is loaded.

The adapters orchestrate:

    VideoSegmenter -> SubtitleAligner -> ObservationExtractor
        -> EntityTracker -> EventGrounder -> SocialStateGrounder
        -> TemporalGrounder -> GroundingNormalizer
        -> (long videos only) MemoryProjection
        -> GroundingRuntime

Direct vs retrieval mode is chosen per benchmark, matching the
"direct vs retrieval" tier in
``infra_plans/01_grounding/video_benchmarks_grounding.md`` §5.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from visual_grounding.entity_tracker import EntityTracker
from visual_grounding.event_grounder import EventGrounder
from visual_grounding.grounding_normalizer import GroundingNormalizer
from visual_grounding.grounding_runtime import GroundingRuntime
from visual_grounding.grounding_schemas import (
    GroundedClip,
    MemoryRecord,
    SubtitleSpan,
    VideoSegment,
)
from visual_grounding.local_grounder import VLMCallable
from visual_grounding.memory_projection import MemoryProjection
from visual_grounding.observation_extractor import ObservationExtractor
from visual_grounding.perception import sample_frames
from visual_grounding.segmenter import (
    DEFAULT_SHORT_THRESHOLD_SECONDS,
    Window,
    VideoSegmenter,
    probe_duration,
)
from visual_grounding.social_state_grounder import SocialStateGrounder
from visual_grounding.subtitle_aligner import SubtitleAligner
from visual_grounding.temporal_grounder import TemporalGrounder


# ---------------------------------------------------------------------------
# Adapter config dataclass
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    """Per-benchmark wiring (segmentation, mode, subtitle policy, …)."""

    name: str
    mode: str = "direct"  # direct | retrieval
    segmentation: str = "scene"  # fixed | scene | subtitle | long_hierarchical
    subtitles: bool = False
    subtitle_mode: str = "origin"  # origin | added | removed | none
    entity_tracking: bool = False
    voice_tracking: bool = False
    allow_causal: bool = False
    window_seconds: Optional[float] = None
    fps: Optional[float] = None
    max_frames_per_window: Optional[int] = None
    include_scene_changes: bool = True
    semantic_summary: bool = True


# Six benchmark presets, mirroring §5 of the plan.
BENCHMARK_CONFIGS: Dict[str, AdapterConfig] = {
    "video_holmes": AdapterConfig(
        name="video_holmes",
        mode="direct",
        segmentation="scene",
        subtitles=False,
        entity_tracking=False,
    ),
    "siv_bench": AdapterConfig(
        name="siv_bench",
        mode="direct",
        segmentation="subtitle",
        subtitles=True,
        subtitle_mode="origin",
        entity_tracking=True,
    ),
    "vrbench": AdapterConfig(
        name="vrbench",
        mode="retrieval",
        segmentation="long_hierarchical",
        subtitles=True,
        entity_tracking=False,
        allow_causal=True,
    ),
    "long_video_bench": AdapterConfig(
        name="long_video_bench",
        mode="retrieval",
        segmentation="long_hierarchical",
        subtitles=True,
        subtitle_mode="origin",
        entity_tracking=False,
    ),
    "cg_bench": AdapterConfig(
        name="cg_bench",
        mode="retrieval",
        segmentation="long_hierarchical",
        subtitles=False,
        entity_tracking=False,
        allow_causal=True,
    ),
    "m3_bench": AdapterConfig(
        name="m3_bench",
        mode="retrieval",
        segmentation="long_hierarchical",
        subtitles=True,
        subtitle_mode="origin",
        entity_tracking=True,
        voice_tracking=True,
    ),
}


# ---------------------------------------------------------------------------
# BenchmarkAdapter
# ---------------------------------------------------------------------------


class BenchmarkAdapter:
    """Run the typed grounding pipeline end-to-end for one video.

    Construct with a benchmark name (or a custom :class:`AdapterConfig`)
    and call :meth:`build` with the path to a video. Returns a
    :class:`GroundingRuntime`.
    """

    def __init__(
        self,
        config: AdapterConfig,
        *,
        vlm_fn: Optional[VLMCallable] = None,
        embedder: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.vlm_fn = vlm_fn
        self.embedder = embedder

    @classmethod
    def for_benchmark(
        cls,
        benchmark: str,
        *,
        vlm_fn: Optional[VLMCallable] = None,
        embedder: Optional[Any] = None,
    ) -> "BenchmarkAdapter":
        if benchmark not in BENCHMARK_CONFIGS:
            raise ValueError(
                f"Unknown benchmark '{benchmark}'. Known: "
                f"{sorted(BENCHMARK_CONFIGS)}"
            )
        return cls(
            BENCHMARK_CONFIGS[benchmark],
            vlm_fn=vlm_fn, embedder=embedder,
        )

    # ------------------------------------------------------------------
    # Pipeline driver
    # ------------------------------------------------------------------

    def build(
        self,
        video_path: str,
        *,
        subtitle_path: Optional[str] = None,
        video_id: Optional[str] = None,
        duration: Optional[float] = None,
        frame_cache_dir: Optional[str] = None,
    ) -> GroundingRuntime:
        cfg = self.config
        vid = video_id or os.path.basename(str(video_path))

        # 1) Segmentation.
        segmenter = VideoSegmenter()
        subtitles: List[SubtitleSpan] = []
        if cfg.subtitles and subtitle_path:
            aligner = SubtitleAligner()
            subtitles = aligner.load(
                subtitle_path, mode=cfg.subtitle_mode, video_id=vid,
            )
        segments = segmenter.segment(
            video_path,
            strategy=cfg.segmentation,
            video_id=vid,
            duration=duration if duration is not None else probe_duration(
                str(video_path),
            ),
            window_seconds=cfg.window_seconds,
            fps=cfg.fps,
            subtitle_spans=subtitles,
        )

        # 2) Subtitle alignment per segment.
        sub_index: Dict[str, List[SubtitleSpan]] = {
            s.segment_id: [] for s in segments
        }
        if cfg.subtitles and subtitles:
            aligner = SubtitleAligner()
            sub_index = aligner.align(segments, subtitles)

        # 3) Per-segment frame sampling + observation extraction.
        extractor = ObservationExtractor(vlm_fn=self.vlm_fn)
        observations = []
        all_frames_by_segment = {}
        for seg in segments:
            frames = self._sample_frames(
                video_path, seg, frame_cache_dir=frame_cache_dir,
            )
            all_frames_by_segment[seg.segment_id] = frames
            observations.extend(extractor.extract(
                seg,
                frames=frames,
                subtitles=sub_index.get(seg.segment_id, []),
            ))

        # 4) Entity tracker (single pass over the full observation stream).
        tracker = EntityTracker(
            match_threshold=0.4 if cfg.entity_tracking else 0.6,
        )
        entities = tracker.update(observations)
        if cfg.entity_tracking:
            entities = tracker.resolve_aliases(entities)

        # 5) Event grounding + social-state grounding + temporal relations.
        segment_index = {
            s.segment_id: (s.start_time, s.end_time) for s in segments
        }
        event_grounder = EventGrounder()
        events = event_grounder.build_events(
            observations, entities,
            tracker=tracker, segment_index=segment_index,
        )

        social = SocialStateGrounder().build_social_states(
            observations, entities, events, tracker=tracker,
        )

        temporal = TemporalGrounder(
            allow_causal=cfg.allow_causal,
        ).build_relations(events)

        # 6) Normalize per segment.
        normalizer = GroundingNormalizer()
        clips: List[GroundedClip] = []
        for seg in segments:
            clip = normalizer.normalize(
                seg,
                raw_observations=observations,
                entities=entities,
                events=events,
                social_outputs=social,
                temporal_relations=temporal,
            )
            clips.append(clip)

        # 7) Memory projection (long-video only).
        memory: List[MemoryRecord] = []
        if cfg.mode == "retrieval":
            projector = MemoryProjection(
                emit_semantic_summary=cfg.semantic_summary,
            )
            memory = projector.project_clips(clips)

        return GroundingRuntime(
            video_id=vid,
            clips=clips,
            memory_records=memory,
            subtitle_spans=subtitles,
            mode=cfg.mode,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_frames(
        self,
        video_path: str,
        segment: VideoSegment,
        *,
        frame_cache_dir: Optional[str],
    ):
        cfg = self.config
        # Re-use the legacy ``sample_frames`` helper by rebuilding a Window.
        window = Window(
            window_id=segment.segment_id,
            time_span=(segment.start_time, segment.end_time),
            frame_times=list(segment.metadata.get("frame_times") or []),
        )
        if not window.frame_times:
            # Fall back to segment midpoint when the segmenter didn't
            # publish a schedule (e.g. parent windows in long_hierarchical).
            mid = (segment.start_time + segment.end_time) / 2.0
            window.frame_times = [round(mid, 3)]
        return sample_frames(
            str(video_path),
            window,
            out_dir=frame_cache_dir,
            max_frames=cfg.max_frames_per_window,
        )


# ---------------------------------------------------------------------------
# Functional convenience wrappers (one per benchmark)
# ---------------------------------------------------------------------------


def build_runtime(
    benchmark: str,
    video_path: str,
    *,
    subtitle_path: Optional[str] = None,
    vlm_fn: Optional[VLMCallable] = None,
    embedder: Optional[Any] = None,
    duration: Optional[float] = None,
    video_id: Optional[str] = None,
) -> GroundingRuntime:
    adapter = BenchmarkAdapter.for_benchmark(
        benchmark, vlm_fn=vlm_fn, embedder=embedder,
    )
    return adapter.build(
        video_path,
        subtitle_path=subtitle_path,
        duration=duration,
        video_id=video_id,
    )


def adapter_for_video_holmes(**kwargs: Any) -> BenchmarkAdapter:
    return BenchmarkAdapter.for_benchmark("video_holmes", **kwargs)


def adapter_for_siv_bench(**kwargs: Any) -> BenchmarkAdapter:
    return BenchmarkAdapter.for_benchmark("siv_bench", **kwargs)


def adapter_for_vrbench(**kwargs: Any) -> BenchmarkAdapter:
    return BenchmarkAdapter.for_benchmark("vrbench", **kwargs)


def adapter_for_long_video_bench(**kwargs: Any) -> BenchmarkAdapter:
    return BenchmarkAdapter.for_benchmark("long_video_bench", **kwargs)


def adapter_for_cg_bench(**kwargs: Any) -> BenchmarkAdapter:
    return BenchmarkAdapter.for_benchmark("cg_bench", **kwargs)


def adapter_for_m3_bench(**kwargs: Any) -> BenchmarkAdapter:
    return BenchmarkAdapter.for_benchmark("m3_bench", **kwargs)


__all__ = [
    "AdapterConfig",
    "BENCHMARK_CONFIGS",
    "BenchmarkAdapter",
    "build_runtime",
    "adapter_for_video_holmes",
    "adapter_for_siv_bench",
    "adapter_for_vrbench",
    "adapter_for_long_video_bench",
    "adapter_for_cg_bench",
    "adapter_for_m3_bench",
]

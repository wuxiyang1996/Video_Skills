"""Unified ``build_grounded_context`` entry point.

This is the single function that adapters in §5 of
``video_benchmarks_grounding.md`` call. It runs the five-step grounding
pipeline and returns either a :class:`DirectContext` (short videos) or a
:class:`SocialVideoGraph` (long videos).

The pipeline is:

1. ``adaptive_segment(video) -> windows``
2. perceptual grounding (frame sampling, subtitle alignment, optional
   entity detection)
3. local social grounding (VLM) → :class:`GroundedWindow`
4. temporal consolidation (only for retrieval mode)
5A. direct mode → package windows into :class:`DirectContext`
5B. retrieval mode → emit :class:`GroundingNode` rows into a
    :class:`SocialVideoGraph`, add semantic summaries.

The short/long distinction is *purely* about layer 5 — every prior
layer runs the same way. That matches the "Shared Reasoning, Shared
Grounding, Different Context Regimes" key insight in §0 of the plan.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from visual_grounding.consolidator import (
    AttributeEntityResolver,
    EntityResolver,
    distill_semantic_summaries,
    merge_adjacent_windows,
    resolve_entities,
    windows_to_nodes,
)
from visual_grounding.local_grounder import (
    VLMCallable,
    ground_windows_batch,
)
from visual_grounding.perception import (
    EntityDetector,
    SampledFrame,
    align_subtitles_to_window,
    apply_subtitle_mode,
    detect_entities,
    parse_subtitle_file,
    sample_frames,
)
from visual_grounding.schemas import (
    DirectContext,
    EvidenceRef,
    GroundedWindow,
    GroundingMode,
    ModeLike,
    _coerce_mode,
)
from visual_grounding.segmenter import (
    DEFAULT_SHORT_THRESHOLD_SECONDS,
    Window,
    adaptive_segment,
    probe_duration,
)
from visual_grounding.social_video_graph import SocialVideoGraph


# Public alias for callers that don't want to depend on the concrete classes.
GroundedContext = Union[DirectContext, SocialVideoGraph]


# ---------------------------------------------------------------------------
# build_grounded_context
# ---------------------------------------------------------------------------


def build_grounded_context(
    video_path: str,
    *,
    mode: ModeLike = "auto",
    duration: Optional[float] = None,
    window_seconds: Optional[float] = None,
    fps: Optional[float] = None,
    subtitle_path: Optional[str] = None,
    subtitle_mode: str = "origin",
    include_subtitles: bool = True,
    entity_tracking: bool = True,
    voice_tracking: bool = False,
    vlm_fn: Optional[VLMCallable] = None,
    entity_detector: Optional[EntityDetector] = None,
    entity_resolver: Optional[EntityResolver] = None,
    embedder: Optional[Any] = None,
    frame_cache_dir: Optional[str] = None,
    top_k: int = 5,
    short_threshold: float = DEFAULT_SHORT_THRESHOLD_SECONDS,
    semantic_cluster_size: int = 4,
    semantic_summarizer: Optional[Callable[[List[GroundedWindow]], str]] = None,
    include_scene_changes: bool = True,
    max_frames_per_window: Optional[int] = None,
) -> GroundedContext:
    """Run the unified visual-grounding pipeline.

    Args:
        video_path: Path to the video file.
        mode: ``"direct"`` | ``"retrieval"`` | ``"auto"``. ``auto`` picks
            direct for videos shorter than ``short_threshold`` seconds.
        duration: Pre-probed duration (seconds).
        window_seconds, fps: Override sampling schedule defaults.
        subtitle_path: Optional ``.srt`` / ``.vtt`` / ``.json`` subtitle file.
        subtitle_mode: SIV-Bench-style subtitle condition (see §5.1).
        include_subtitles: Global toggle for subtitles.
        entity_tracking: If True, detector output feeds the grounder
            and entity resolution runs in retrieval mode.
        voice_tracking: Reserved; passed through into per-window metadata
            so benchmark adapters can wire speaker diarization.
        vlm_fn: Pluggable VLM callable for local grounding.
        entity_detector: Optional per-window detector.
        entity_resolver: Optional cross-window entity resolver
            (default: attribute-based).
        embedder: Optional embedder for ``SocialVideoGraph`` retrieval.
        frame_cache_dir: If given, extracted frames are written to disk
            (recommended for retrieval-mode long videos).
        top_k: Default retrieval ``k`` for ``SocialVideoGraph.search``.
        short_threshold: Duration boundary for ``mode="auto"``.
        semantic_cluster_size, semantic_summarizer: Control §4 semantic
            distillation in retrieval mode.

    Returns:
        A :class:`DirectContext` (short) or :class:`SocialVideoGraph`
        (long). Both satisfy the ``mode`` attribute so callers can
        branch uniformly.
    """
    dur = duration if duration is not None else probe_duration(video_path)
    chosen_mode = _coerce_mode(mode)
    if chosen_mode == GroundingMode.auto:
        chosen_mode = (
            GroundingMode.direct if 0 < dur <= short_threshold
            else GroundingMode.retrieval
        )
    if dur <= 0:
        # If we can't probe the video we still return an empty context so
        # the caller's reasoning path doesn't blow up.
        dur = 0.0

    # --- 1. adaptive_segment ------------------------------------------------
    subtitles: List[EvidenceRef] = []
    if include_subtitles and subtitle_path:
        subtitles = parse_subtitle_file(subtitle_path)
        subtitles = apply_subtitle_mode(subtitles, subtitle_mode)

    windows = adaptive_segment(
        video_path,
        mode=chosen_mode.value,
        duration=dur,
        window_seconds=window_seconds,
        fps=fps,
        subtitle_refs=subtitles,
        include_scene_changes=include_scene_changes,
    )

    # --- 2. perceptual grounding -------------------------------------------
    frames_by_window: Dict[str, List[SampledFrame]] = {}
    subs_by_window: Dict[str, List[EvidenceRef]] = {}
    hints_by_window: Dict[str, List[Dict[str, Any]]] = {}

    for w in windows:
        frames = sample_frames(
            video_path,
            w,
            out_dir=frame_cache_dir,
            max_frames=max_frames_per_window,
        )
        frames_by_window[w.window_id] = frames
        subs_by_window[w.window_id] = (
            align_subtitles_to_window(w, subtitles) if include_subtitles else []
        )
        if entity_tracking:
            hints_by_window[w.window_id] = detect_entities(
                frames, detector=entity_detector,
            )

    # --- 3. local social grounding -----------------------------------------
    grounded_windows = ground_windows_batch(
        windows,
        frames_by_window,
        subs_by_window,
        entity_hints_by_window=hints_by_window if entity_tracking else None,
        vlm_fn=vlm_fn,
        subtitle_mode=subtitle_mode,
    )

    # Annotate voice-tracking flag for downstream adapters.
    if voice_tracking:
        for gw in grounded_windows:
            gw.metadata.setdefault("voice_tracking", True)

    # --- 5A. Direct mode ----------------------------------------------------
    if chosen_mode == GroundingMode.direct:
        return DirectContext(
            video_path=video_path,
            duration=dur,
            windows=grounded_windows,
            subtitle_mode=subtitle_mode,  # type: ignore[arg-type]
            subtitles=subtitles,
            frame_index_path=frame_cache_dir,
            metadata={
                "window_seconds": window_seconds,
                "fps": fps,
                "entity_tracking": entity_tracking,
                "voice_tracking": voice_tracking,
            },
        )

    # --- 4. temporal consolidation (retrieval mode) ------------------------
    merged_windows = merge_adjacent_windows(grounded_windows)

    resolver = entity_resolver or (AttributeEntityResolver() if entity_tracking else None)
    if resolver is not None:
        _, profiles = resolve_entities(merged_windows, resolver=resolver)
    else:
        profiles = {}

    episodic_and_structural_nodes = windows_to_nodes(merged_windows, profiles)
    semantic_nodes = distill_semantic_summaries(
        merged_windows,
        cluster_size=semantic_cluster_size,
        summarizer=semantic_summarizer,
    )

    # --- 5B. Retrieval mode: build SocialVideoGraph ------------------------
    graph = SocialVideoGraph(
        embedder=embedder,
        top_k=top_k,
        video_path=video_path,
    )
    for prof in profiles.values():
        graph.register_entity_profile(prof)
    # Collect & register evidence before nodes so get_evidence() works.
    for w in merged_windows:
        graph.add_evidence(w.evidence)
    graph.add_nodes(episodic_and_structural_nodes)
    graph.add_nodes(semantic_nodes)
    return graph


# ---------------------------------------------------------------------------
# Benchmark-facing convenience wrappers (mirror §5 adapters)
# ---------------------------------------------------------------------------


def build_for_video_holmes(
    video_path: str,
    *,
    vlm_fn: Optional[VLMCallable] = None,
    **kwargs: Any,
) -> DirectContext:
    """Adapter preset for Video-Holmes (short, direct mode, no subtitles)."""
    ctx = build_grounded_context(
        video_path,
        mode="direct",
        include_subtitles=False,
        entity_tracking=False,
        vlm_fn=vlm_fn,
        **kwargs,
    )
    assert isinstance(ctx, DirectContext)
    return ctx


def build_for_siv_bench(
    video_path: str,
    *,
    subtitle_path: Optional[str] = None,
    subtitle_mode: str = "origin",
    vlm_fn: Optional[VLMCallable] = None,
    **kwargs: Any,
) -> DirectContext:
    """Adapter preset for SIV-Bench (short, direct, subtitle-aware social)."""
    ctx = build_grounded_context(
        video_path,
        mode="direct",
        subtitle_path=subtitle_path,
        subtitle_mode=subtitle_mode,
        include_subtitles=True,
        entity_tracking=True,
        vlm_fn=vlm_fn,
        **kwargs,
    )
    assert isinstance(ctx, DirectContext)
    return ctx


def build_for_vrbench(
    video_path: str,
    *,
    vlm_fn: Optional[VLMCallable] = None,
    embedder: Optional[Any] = None,
    **kwargs: Any,
) -> SocialVideoGraph:
    """Adapter preset for VRBench (long, retrieval, event/narrative grounding)."""
    ctx = build_grounded_context(
        video_path,
        mode="retrieval",
        include_subtitles=True,
        entity_tracking=False,
        vlm_fn=vlm_fn,
        embedder=embedder,
        **kwargs,
    )
    assert isinstance(ctx, SocialVideoGraph)
    return ctx


def build_for_long_video_bench(
    video_path: str,
    *,
    subtitle_path: Optional[str] = None,
    vlm_fn: Optional[VLMCallable] = None,
    embedder: Optional[Any] = None,
    **kwargs: Any,
) -> SocialVideoGraph:
    """Adapter preset for LongVideoBench (long, retrieval + subtitles)."""
    ctx = build_grounded_context(
        video_path,
        mode="retrieval",
        include_subtitles=True,
        subtitle_path=subtitle_path,
        entity_tracking=False,
        vlm_fn=vlm_fn,
        embedder=embedder,
        **kwargs,
    )
    assert isinstance(ctx, SocialVideoGraph)
    return ctx


def build_for_cg_bench(
    video_path: str,
    *,
    vlm_fn: Optional[VLMCallable] = None,
    embedder: Optional[Any] = None,
    **kwargs: Any,
) -> SocialVideoGraph:
    """Adapter preset for CG-Bench (long, retrieval, clue/evidence grounding)."""
    ctx = build_grounded_context(
        video_path,
        mode="retrieval",
        entity_tracking=False,
        vlm_fn=vlm_fn,
        embedder=embedder,
        **kwargs,
    )
    assert isinstance(ctx, SocialVideoGraph)
    return ctx


def build_for_m3_bench(
    video_path: str,
    *,
    subtitle_path: Optional[str] = None,
    vlm_fn: Optional[VLMCallable] = None,
    embedder: Optional[Any] = None,
    entity_resolver: Optional[EntityResolver] = None,
    **kwargs: Any,
) -> SocialVideoGraph:
    """Adapter preset for M3-Bench (long, retrieval, entity + voice tracking)."""
    ctx = build_grounded_context(
        video_path,
        mode="retrieval",
        include_subtitles=True,
        subtitle_path=subtitle_path,
        entity_tracking=True,
        voice_tracking=True,
        vlm_fn=vlm_fn,
        embedder=embedder,
        entity_resolver=entity_resolver,
        **kwargs,
    )
    assert isinstance(ctx, SocialVideoGraph)
    return ctx


__all__ = [
    "GroundedContext",
    "build_grounded_context",
    "build_for_video_holmes",
    "build_for_siv_bench",
    "build_for_vrbench",
    "build_for_long_video_bench",
    "build_for_cg_bench",
    "build_for_m3_bench",
]

"""Adaptive temporal segmentation.

Implements step 1 of the unified grounding pipeline described in
``infra_plans/01_grounding/video_benchmarks_grounding.md`` §3.1 / §3.6:

- Base sampling rate: 1 frame / 2s for short videos, 1 frame / 5s for
  long videos.
- Add keyframes at scene-change boundaries (cheap pixel-diff heuristic).
- Add subtitle-aligned frames when subtitles are available.
- Output ``Window`` descriptors that the local grounder and perception
  primitives consume.

The primitives are *infrastructure*, not first-class reasoning skills
(see ``infra_plans/04_harness/atomic_skills_hop_refactor_execution_plan.md``
"Infrastructure primitives"). They expose a stable function-call surface
for the controller.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from visual_grounding.schemas import EvidenceRef, new_id


# Default thresholds for mode routing.
DEFAULT_SHORT_THRESHOLD_SECONDS = 5 * 60.0   # auto-mode: <5 min -> direct
DEFAULT_WINDOW_SECONDS_SHORT = 5.0           # dense windows for short clips
DEFAULT_WINDOW_SECONDS_LONG = 15.0           # coarser windows for long videos
DEFAULT_FPS_SHORT = 0.5                       # 1 frame / 2s
DEFAULT_FPS_LONG = 0.2                        # 1 frame / 5s


@dataclass
class Window:
    """A single temporal window to be locally grounded.

    ``frame_times`` is the *intended* sampling schedule in seconds;
    perception primitives will materialize actual frames for these times.
    """

    window_id: str
    time_span: Tuple[float, float]
    frame_times: List[float] = field(default_factory=list)
    scene_change: bool = False
    subtitle_refs: List[EvidenceRef] = field(default_factory=list)

    @property
    def duration(self) -> float:
        s, e = self.time_span
        return max(0.0, e - s)


# ---------------------------------------------------------------------------
# Video duration probe
# ---------------------------------------------------------------------------


def probe_duration(video_path: str) -> float:
    """Return video duration in seconds. Tries OpenCV then ffprobe.

    Returns ``0.0`` when the duration cannot be determined — callers
    should treat that as "unknown" and fall back to retrieval mode.
    """
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and nframes > 0:
                return float(nframes) / float(fps)
    except Exception:
        pass

    try:
        import subprocess
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return float(out.decode().strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Scene-change detection (cheap, optional)
# ---------------------------------------------------------------------------


def detect_scene_change_boundaries(
    video_path: str,
    sample_rate: float = 1.0,
    pixel_diff_threshold: float = 0.35,
    max_boundaries: int = 512,
) -> List[float]:
    """Return a list of timestamps (in seconds) where scene changes are detected.

    Uses a lightweight frame-to-frame histogram-difference heuristic; returns
    an empty list if OpenCV is not importable. This is a deliberate fallback
    so that the pipeline stays functional even without perception deps.
    """
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        return []

    step = max(1, int(round(fps / max(sample_rate, 1e-3))))
    boundaries: List[float] = []
    prev_hist = None
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            small = cv2.resize(frame, (64, 64))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hist = hist.flatten()
            if prev_hist is not None:
                diff = float(np.linalg.norm(hist - prev_hist))
                if diff > pixel_diff_threshold:
                    boundaries.append(idx / fps)
                    if len(boundaries) >= max_boundaries:
                        break
            prev_hist = hist
        idx += 1
        ok, frame = cap.read()

    cap.release()
    return boundaries


# ---------------------------------------------------------------------------
# Adaptive segmentation
# ---------------------------------------------------------------------------


def _window_schedule(
    duration: float,
    window_seconds: float,
) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []
    n = max(1, int(math.ceil(duration / window_seconds)))
    spans: List[Tuple[float, float]] = []
    for i in range(n):
        s = i * window_seconds
        e = min(duration, (i + 1) * window_seconds)
        if e - s >= 0.5:  # drop sub-half-second trailing crumbs
            spans.append((s, e))
    return spans


def _frame_schedule(
    span: Tuple[float, float],
    fps: float,
) -> List[float]:
    s, e = span
    if fps <= 0 or e <= s:
        return [s]
    step = 1.0 / fps
    times = []
    t = s
    while t < e - 1e-6:
        times.append(round(t, 3))
        t += step
    if not times or times[-1] < e - step * 0.5:
        times.append(round(min(e, max(s, e - 1e-3)), 3))
    return times


def adaptive_segment(
    video_path: str,
    *,
    mode: str = "auto",
    duration: Optional[float] = None,
    window_seconds: Optional[float] = None,
    fps: Optional[float] = None,
    subtitle_refs: Optional[Sequence[EvidenceRef]] = None,
    include_scene_changes: bool = True,
) -> List[Window]:
    """Produce a list of :class:`Window` descriptors for a video.

    Args:
        video_path: Path to the video file (or a directory of pre-extracted
            frames; duration must then be supplied).
        mode: ``"direct"`` | ``"retrieval"`` | ``"auto"``. Used only to
            pick sensible defaults for ``window_seconds`` / ``fps``.
        duration: Override duration probe; pass it when the video has
            already been inspected upstream.
        window_seconds: Length of each window.
        fps: Target sampling rate (frames per second) inside each window.
        subtitle_refs: Subtitle / ASR spans. Each span's midpoint is added
            as an extra sampling time on the window it falls into.
        include_scene_changes: If True, extra frames are injected at
            detected scene-change boundaries.

    Returns:
        List of :class:`Window` in chronological order.
    """
    dur = duration if duration is not None else probe_duration(video_path)
    if dur <= 0:
        return []

    effective_mode = mode.lower()
    if effective_mode == "auto":
        effective_mode = (
            "direct" if dur <= DEFAULT_SHORT_THRESHOLD_SECONDS else "retrieval"
        )

    ws = window_seconds
    rate = fps
    if effective_mode == "direct":
        ws = ws or DEFAULT_WINDOW_SECONDS_SHORT
        rate = rate or DEFAULT_FPS_SHORT
    else:
        ws = ws or DEFAULT_WINDOW_SECONDS_LONG
        rate = rate or DEFAULT_FPS_LONG

    spans = _window_schedule(dur, ws)
    scene_boundaries: List[float] = []
    if include_scene_changes and os.path.isfile(video_path):
        scene_boundaries = detect_scene_change_boundaries(video_path)

    # Bucket subtitle midpoints by window index.
    sub_bucket: dict[int, list[EvidenceRef]] = {}
    if subtitle_refs:
        for s_ref in subtitle_refs:
            if s_ref.timestamp is None:
                continue
            mid = (s_ref.timestamp[0] + s_ref.timestamp[1]) / 2.0
            idx = min(len(spans) - 1, int(mid // ws))
            sub_bucket.setdefault(idx, []).append(s_ref)

    windows: List[Window] = []
    for idx, span in enumerate(spans):
        times = _frame_schedule(span, rate)
        has_scene_change = any(span[0] <= b < span[1] for b in scene_boundaries)
        # Inject subtitle midpoints + scene-change boundaries as extra frames.
        for b in scene_boundaries:
            if span[0] <= b < span[1]:
                times.append(round(b, 3))
        for s_ref in sub_bucket.get(idx, []):
            mid = (s_ref.timestamp[0] + s_ref.timestamp[1]) / 2.0
            times.append(round(max(span[0], min(span[1] - 1e-3, mid)), 3))

        times = sorted(set(times))

        windows.append(
            Window(
                window_id=new_id("win"),
                time_span=span,
                frame_times=times,
                scene_change=has_scene_change,
                subtitle_refs=list(sub_bucket.get(idx, [])),
            )
        )

    return windows


# ---------------------------------------------------------------------------
# Typed-grounding wrapper (returns ``VideoSegment`` objects)
# ---------------------------------------------------------------------------


class VideoSegmenter:
    """Class wrapper around :func:`adaptive_segment` for the typed pipeline.

    Strategies (per the visual-grounding plan §3):

    * ``fixed``                 — fixed-window segmentation, ignores
                                  scene changes / subtitles. Good for
                                  benchmarks that want strict temporal
                                  coverage.
    * ``scene``                 — fixed windows + scene-change frame
                                  injection (the legacy default).
    * ``subtitle``              — windows aligned to subtitle midpoints
                                  when subtitle spans are supplied.
    * ``long_hierarchical``     — coarser parent windows with smaller
                                  child windows; child windows carry a
                                  ``parent_segment_id`` in metadata so
                                  callers can persist parent/child
                                  links.

    The wrapper returns
    :class:`visual_grounding.grounding_schemas.VideoSegment` objects so
    the typed pipeline does not need to import the legacy ``Window``
    class.
    """

    def __init__(
        self,
        *,
        short_threshold: float = DEFAULT_SHORT_THRESHOLD_SECONDS,
        short_window_seconds: float = DEFAULT_WINDOW_SECONDS_SHORT,
        long_window_seconds: float = DEFAULT_WINDOW_SECONDS_LONG,
        long_parent_seconds: float = 60.0,
        long_overlap_seconds: float = 2.0,
    ) -> None:
        self.short_threshold = short_threshold
        self.short_window_seconds = short_window_seconds
        self.long_window_seconds = long_window_seconds
        self.long_parent_seconds = long_parent_seconds
        self.long_overlap_seconds = long_overlap_seconds

    # -- public --------------------------------------------------------

    def segment(
        self,
        video_input: str,
        *,
        strategy: str = "scene",
        video_id: Optional[str] = None,
        duration: Optional[float] = None,
        window_seconds: Optional[float] = None,
        fps: Optional[float] = None,
        subtitle_spans: Optional[Sequence[Any]] = None,
    ):
        """Return a list of typed ``VideoSegment`` objects.

        ``video_input`` can be a path or an already-probed wrapper; the
        function only relies on path-based duration/scene heuristics
        when those are available. ``video_id`` defaults to the video's
        basename so downstream evidence refs have a stable handle.
        """
        from visual_grounding.grounding_schemas import VideoSegment

        vid = video_id or os.path.basename(str(video_input))
        strategy_l = (strategy or "scene").lower()

        # Convert subtitle_spans (typed) into legacy EvidenceRefs the
        # adaptive segmenter understands.
        legacy_subs: List[EvidenceRef] = []
        for s in subtitle_spans or []:
            ref = getattr(s, "evidence_ref", None)
            if ref is not None:
                legacy_subs.append(ref)
            else:
                legacy_subs.append(
                    EvidenceRef(
                        ref_id=getattr(s, "span_id", new_id("sub")),
                        modality="subtitle",
                        timestamp=(
                            float(getattr(s, "start_time", 0.0)),
                            float(getattr(s, "end_time", 0.0)),
                        ),
                        text=getattr(s, "text", None),
                        video_id=vid,
                    )
                )

        if strategy_l == "long_hierarchical":
            return self._long_hierarchical_segments(
                video_input,
                vid,
                duration=duration,
                fps=fps,
                subtitle_refs=legacy_subs,
            )

        # All other strategies route through adaptive_segment.
        include_scene = strategy_l == "scene"
        windows = adaptive_segment(
            str(video_input),
            mode="auto",
            duration=duration,
            window_seconds=window_seconds or self._window_seconds_for(strategy_l),
            fps=fps,
            subtitle_refs=legacy_subs if strategy_l in ("subtitle", "scene") else None,
            include_scene_changes=include_scene,
        )
        return [self._window_to_segment(w, vid) for w in windows]

    # -- helpers -------------------------------------------------------

    def _window_seconds_for(self, strategy_l: str) -> float:
        if strategy_l == "fixed":
            return self.short_window_seconds
        return self.long_window_seconds

    def _window_to_segment(self, window: "Window", video_id: str):
        from visual_grounding.grounding_schemas import VideoSegment

        return VideoSegment(
            segment_id=window.window_id,
            video_id=video_id,
            start_time=window.time_span[0],
            end_time=window.time_span[1],
            subtitle_span_ids=[r.ref_id for r in window.subtitle_refs],
            metadata={
                "frame_times": list(window.frame_times),
                "scene_change": window.scene_change,
            },
        )

    def _long_hierarchical_segments(
        self,
        video_input: str,
        video_id: str,
        *,
        duration: Optional[float],
        fps: Optional[float],
        subtitle_refs: Sequence[EvidenceRef],
    ):
        from visual_grounding.grounding_schemas import VideoSegment

        dur = duration if duration is not None else probe_duration(str(video_input))
        if dur <= 0:
            return []
        parent_len = self.long_parent_seconds
        child_len = self.long_window_seconds
        overlap = self.long_overlap_seconds
        rate = fps or DEFAULT_FPS_LONG

        # Build parent windows.
        parents: List[VideoSegment] = []
        n_parents = int(math.ceil(dur / parent_len))
        for pi in range(n_parents):
            ps = pi * parent_len
            pe = min(dur, (pi + 1) * parent_len)
            if pe - ps < 0.5:
                continue
            parents.append(
                VideoSegment(
                    segment_id=new_id("parent"),
                    video_id=video_id,
                    start_time=ps,
                    end_time=pe,
                    metadata={"level": "parent", "fps": rate},
                )
            )

        # Build overlapping children inside each parent.
        children: List[VideoSegment] = []
        for parent in parents:
            t = parent.start_time
            while t < parent.end_time - 1e-3:
                cs = t
                ce = min(parent.end_time, t + child_len)
                if ce - cs >= 0.5:
                    sub_ids = [
                        r.ref_id for r in subtitle_refs
                        if r.timestamp is not None
                        and not (r.timestamp[1] < cs or r.timestamp[0] > ce)
                    ]
                    children.append(
                        VideoSegment(
                            segment_id=new_id("child"),
                            video_id=video_id,
                            start_time=cs,
                            end_time=ce,
                            subtitle_span_ids=sub_ids,
                            metadata={
                                "level": "child",
                                "parent_segment_id": parent.segment_id,
                                "fps": rate,
                            },
                        )
                    )
                t += max(child_len - overlap, child_len / 2)
        return parents + children


__all__ = [
    "Window",
    "adaptive_segment",
    "probe_duration",
    "detect_scene_change_boundaries",
    "DEFAULT_SHORT_THRESHOLD_SECONDS",
    "DEFAULT_WINDOW_SECONDS_SHORT",
    "DEFAULT_WINDOW_SECONDS_LONG",
    "DEFAULT_FPS_SHORT",
    "DEFAULT_FPS_LONG",
    "VideoSegmenter",
]

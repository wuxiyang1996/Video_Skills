"""Perceptual grounding primitives.

Step 2 of the pipeline in ``infra_plans/01_grounding/video_benchmarks_grounding.md``
§3.1 — frame sampling, optional face/person detection hooks, subtitle /
ASR alignment. Everything exposed here is an infrastructure primitive
(``observe_segment``, ``detect_entities``, ``align_subtitles``) as
defined in ``atomic_skills_hop_refactor_execution_plan.md`` — never a
bank skill.

Perception can be expensive. Each function accepts a pluggable callable
(e.g. a face detector) and returns gracefully when the backing library
is absent, so this module is always import-safe.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from visual_grounding.schemas import EvidenceRef, new_id
from visual_grounding.segmenter import Window


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


@dataclass
class SampledFrame:
    frame_id: str
    timestamp: float
    array_path: Optional[str] = None        # on-disk path (decoded image)
    # For ad-hoc in-memory use (debug / tests) we allow a ``data`` ref.
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def sample_frames(
    video_path: str,
    window: Window,
    *,
    out_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> List[SampledFrame]:
    """Materialize frames for the sampling times of a window.

    Uses OpenCV when available; otherwise returns lightweight placeholders
    with only timestamps populated so downstream code stays functional on
    machines without decode deps.
    """
    times = list(window.frame_times)
    if max_frames is not None:
        times = times[:max_frames]
    if not times:
        return []

    try:
        import cv2  # type: ignore
    except Exception:
        return [
            SampledFrame(frame_id=new_id("frm"), timestamp=t) for t in times
        ]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [
            SampledFrame(frame_id=new_id("frm"), timestamp=t) for t in times
        ]

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: List[SampledFrame] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, img = cap.read()
        frame_id = new_id("frm")
        fr = SampledFrame(frame_id=frame_id, timestamp=t)
        if ok:
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                p = os.path.join(out_dir, f"{frame_id}.jpg")
                cv2.imwrite(p, img)
                fr.array_path = p
            else:
                fr.data = img
            fr.metadata["frame_index"] = (
                int(round(t * fps)) if fps > 0 else None
            )
        frames.append(fr)

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Entity detection hook
# ---------------------------------------------------------------------------


EntityDetector = Callable[[Sequence[SampledFrame]], List[Dict[str, Any]]]
"""Signature: frames -> list of raw detections.

Each detection is a dict with at minimum ``{"type": "person"|"object",
"bbox": [x, y, w, h], "frame_id": str, "confidence": float}``. The
consolidator is responsible for linking detections across windows into
stable entity IDs.
"""


def detect_entities(
    frames: Sequence[SampledFrame],
    detector: Optional[EntityDetector] = None,
) -> List[Dict[str, Any]]:
    """Run an optional entity detector; return [] when no detector supplied."""
    if detector is None or not frames:
        return []
    try:
        return list(detector(frames))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Subtitle / ASR alignment
# ---------------------------------------------------------------------------


_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})"
)


def _srt_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_subtitle_file(path: str) -> List[EvidenceRef]:
    """Parse a .srt / .vtt / plain-JSON subtitle file into :class:`EvidenceRef`.

    Supports SRT/VTT time codes and a JSON format of
    ``[{"start": float, "end": float, "text": str}, ...]``.
    """
    if not path or not os.path.isfile(path):
        return []
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    refs: List[EvidenceRef] = []
    if path.endswith(".json"):
        try:
            data = json.loads(text)
            for item in data:
                s = float(item.get("start", 0.0))
                e = float(item.get("end", s))
                refs.append(
                    EvidenceRef(
                        ref_id=new_id("sub"),
                        modality="subtitle",
                        timestamp=(s, e),
                        text=str(item.get("text", "")),
                    )
                )
            return refs
        except Exception:
            return []

    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        m = _SRT_TIME.search(block)
        if not m:
            continue
        s = _srt_to_seconds(*m.group(1, 2, 3, 4))
        e = _srt_to_seconds(*m.group(5, 6, 7, 8))
        lines = [ln for ln in block.split("\n") if not _SRT_TIME.search(ln)]
        body = "\n".join(
            ln for ln in lines if ln.strip() and not ln.strip().isdigit()
            and ln.strip().upper() != "WEBVTT"
        ).strip()
        if not body:
            continue
        refs.append(
            EvidenceRef(
                ref_id=new_id("sub"),
                modality="subtitle",
                timestamp=(s, e),
                text=body,
            )
        )
    return refs


def align_subtitles_to_window(
    window: Window,
    subtitles: Iterable[EvidenceRef],
) -> List[EvidenceRef]:
    """Return the subset of ``subtitles`` whose time interval overlaps the window."""
    s0, e0 = window.time_span
    out: List[EvidenceRef] = []
    for ref in subtitles:
        if ref.timestamp is None:
            continue
        s1, e1 = ref.timestamp
        if e1 < s0 or s1 > e0:
            continue
        out.append(ref)
    return out


def apply_subtitle_mode(
    refs: Sequence[EvidenceRef],
    mode: str,
) -> List[EvidenceRef]:
    """Realize SIV-Bench-style ``origin / added / removed / none`` modes.

    - ``origin``: unchanged.
    - ``removed`` / ``none``: return empty list.
    - ``added``: unchanged here (in the real pipeline this is where a
      subtitle-augmentation agent would inject extra lines). The mode
      flag is preserved on the :class:`GroundedWindow` so adapters can
      apply additional logic downstream.
    """
    m = (mode or "origin").lower()
    if m in ("removed", "none"):
        return []
    return list(refs)


__all__ = [
    "SampledFrame",
    "EntityDetector",
    "sample_frames",
    "detect_entities",
    "parse_subtitle_file",
    "align_subtitles_to_window",
    "apply_subtitle_mode",
]

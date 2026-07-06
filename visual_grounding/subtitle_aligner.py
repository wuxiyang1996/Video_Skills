"""Typed subtitle alignment.

A thin wrapper around the existing :mod:`visual_grounding.perception`
SRT/VTT/JSON parsers that returns typed
:class:`visual_grounding.grounding_schemas.SubtitleSpan` objects and
indexes them by typed
:class:`visual_grounding.grounding_schemas.VideoSegment`.

The plan asks for subtitle alignment to be a *first-class module*
because:

* SIV-Bench drives subtitle-aware evidence attribution
  (origin / removed / added).
* LongVideoBench cross-modal retrieval needs subtitle spans linked to
  the same video segments visual evidence is linked to.

Subtitle evidence stays in its own evidence-ref bucket so the runtime
can answer "what subtitle line backs this event?" cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from visual_grounding.grounding_schemas import SubtitleSpan, VideoSegment
from visual_grounding.perception import (
    apply_subtitle_mode,
    parse_subtitle_file,
)
from visual_grounding.schemas import EvidenceRef


# ---------------------------------------------------------------------------
# Typed conversion
# ---------------------------------------------------------------------------


def _ref_to_span(ref: EvidenceRef, video_id: Optional[str]) -> SubtitleSpan:
    s, e = ref.timestamp if ref.timestamp else (0.0, 0.0)
    if video_id and ref.video_id is None:
        ref.video_id = video_id
    return SubtitleSpan(
        span_id=ref.ref_id,
        text=ref.text or "",
        start_time=float(s),
        end_time=float(e),
        speaker=(ref.locator or {}).get("speaker") if ref.locator else None,
        evidence_ref=ref,
    )


def load_subtitles(
    subtitle_path: Optional[str],
    *,
    mode: str = "origin",
    video_id: Optional[str] = None,
) -> List[SubtitleSpan]:
    """Parse a subtitle file into typed :class:`SubtitleSpan` objects.

    ``mode`` accepts SIV-Bench-style ``origin / removed / added / none``
    conditions; ``removed`` / ``none`` returns an empty list so the
    grounding layer never sees subtitle evidence in those conditions.
    """
    if not subtitle_path:
        return []
    refs = parse_subtitle_file(subtitle_path)
    refs = apply_subtitle_mode(refs, mode)
    return [_ref_to_span(r, video_id) for r in refs]


# ---------------------------------------------------------------------------
# SubtitleAligner
# ---------------------------------------------------------------------------


class SubtitleAligner:
    """Index subtitle spans by typed video segment.

    The caller supplies a list of :class:`VideoSegment` and a list of
    :class:`SubtitleSpan` (typically produced by :func:`load_subtitles`);
    :meth:`align` returns ``{segment_id: [SubtitleSpan, …]}`` plus
    optionally writes the subtitle ids back into
    ``segment.subtitle_span_ids`` so the segment description stays
    self-contained.
    """

    def __init__(self, *, write_back: bool = True) -> None:
        self.write_back = write_back

    def load(
        self,
        subtitle_path: Optional[str],
        *,
        mode: str = "origin",
        video_id: Optional[str] = None,
    ) -> List[SubtitleSpan]:
        return load_subtitles(subtitle_path, mode=mode, video_id=video_id)

    def align(
        self,
        segments: Sequence[VideoSegment],
        subtitles: Optional[Sequence[SubtitleSpan]] = None,
        *,
        subtitle_path: Optional[str] = None,
        mode: str = "origin",
    ) -> Dict[str, List[SubtitleSpan]]:
        if subtitles is None:
            video_id = segments[0].video_id if segments else None
            subtitles = self.load(
                subtitle_path, mode=mode, video_id=video_id,
            )
        out: Dict[str, List[SubtitleSpan]] = {seg.segment_id: [] for seg in segments}
        for seg in segments:
            for span in subtitles:
                if span.end_time < seg.start_time or span.start_time > seg.end_time:
                    continue
                out[seg.segment_id].append(span)
            if self.write_back:
                ids = [s.span_id for s in out[seg.segment_id]]
                # Preserve ids from the segmenter while adding the aligned ones.
                merged_ids = list(dict.fromkeys(list(seg.subtitle_span_ids) + ids))
                seg.subtitle_span_ids = merged_ids
        return out


__all__ = [
    "SubtitleAligner",
    "load_subtitles",
]

"""Layer-1 observation extractor.

Produces *low-commitment* :class:`RawObservation` objects from a
:class:`VideoSegment` plus optional subtitles and entity-detector
hints.

Per the visual-grounding plan §5:

* Do **not** collapse everything into one summary string. Each
  observation is a single decomposed unit (caption, action, participant
  mention, speaker turn, event/interaction proposal, subtitle echo, …).
* Large VLMs may be used here, but only as proposers / normalizers /
  distillers — never as the final source of truth.
* Subtitles produce ``observed`` observations; speaker turns also
  ``observed``; event proposals from a captioner are ``inferred`` by
  default.

The extractor wraps the existing
:func:`visual_grounding.local_grounder.ground_window` model call and
converts its §3.2 JSON envelope into a list of typed
:class:`RawObservation` rows. When no VLM call is available, only
subtitle/entity-detection observations are emitted — the pipeline still
returns well-formed (but impoverished) output.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from visual_grounding.grounding_schemas import (
    RawObservation,
    SubtitleSpan,
    VideoSegment,
    new_grounding_id,
)
from visual_grounding.local_grounder import (
    GROUNDING_PROMPT_TEMPLATE,
    VLMCallable,
    _format_entity_hints,
    _format_subtitles,
    _safe_json_loads,
)
from visual_grounding.perception import SampledFrame
from visual_grounding.schemas import EvidenceRef, new_id


def _segment_evidence_ref(segment: VideoSegment) -> EvidenceRef:
    return EvidenceRef(
        ref_id=new_id("clip"),
        modality="clip",
        timestamp=(segment.start_time, segment.end_time),
        video_id=segment.video_id,
        clip_id=segment.segment_id,
        source_type="observed",
        confidence=1.0,
    )


def _frame_evidence_ref(
    frame: SampledFrame, segment: VideoSegment,
) -> EvidenceRef:
    return EvidenceRef(
        ref_id=frame.frame_id,
        modality="frame",
        timestamp=(frame.timestamp, frame.timestamp),
        locator={
            "path": frame.array_path,
            **(frame.metadata or {}),
        },
        video_id=segment.video_id,
        clip_id=segment.segment_id,
        source_type="observed",
        confidence=1.0,
    )


def _subtitle_evidence_ref(
    span: SubtitleSpan, segment: VideoSegment,
) -> EvidenceRef:
    if span.evidence_ref is not None:
        ref = span.evidence_ref
        if ref.video_id is None:
            ref.video_id = segment.video_id
        if ref.clip_id is None:
            ref.clip_id = segment.segment_id
        return ref
    return EvidenceRef(
        ref_id=span.span_id,
        modality="subtitle",
        timestamp=(span.start_time, span.end_time),
        text=span.text,
        video_id=segment.video_id,
        clip_id=segment.segment_id,
        source_type="observed",
    )


# ---------------------------------------------------------------------------
# ObservationExtractor
# ---------------------------------------------------------------------------


class ObservationExtractor:
    r"""Generate :class:`RawObservation`\ s for one segment at a time.

    The VLM call is *optional*. The extractor always emits subtitle and
    detector observations first; VLM proposals (captions, events,
    interactions, hypotheses) are appended on top when ``vlm_fn`` is
    supplied and returns parseable JSON.
    """

    def __init__(
        self,
        *,
        vlm_fn: Optional[VLMCallable] = None,
        source_model: str = "vlm",
        max_frames_in_prompt: Optional[int] = None,
    ) -> None:
        self.vlm_fn = vlm_fn
        self.source_model = source_model
        self.max_frames_in_prompt = max_frames_in_prompt

    # -- entry point ---------------------------------------------------

    def extract(
        self,
        segment: VideoSegment,
        *,
        frames: Sequence[SampledFrame] = (),
        subtitles: Sequence[SubtitleSpan] = (),
        entity_hints: Sequence[Dict[str, Any]] = (),
        audio: Optional[Any] = None,  # reserved
    ) -> List[RawObservation]:
        observations: List[RawObservation] = []
        clip_ref = _segment_evidence_ref(segment)
        frame_refs = [_frame_evidence_ref(f, segment) for f in frames]
        subtitle_refs = [_subtitle_evidence_ref(s, segment) for s in subtitles]

        # 1) subtitle echoes (observed).
        for span, ref in zip(subtitles, subtitle_refs):
            observations.append(
                RawObservation(
                    obs_id=new_grounding_id("obs_sub"),
                    segment_id=segment.segment_id,
                    observation_type="subtitle_echo",
                    payload={
                        "text": span.text,
                        "speaker": span.speaker,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                    },
                    evidence_refs=[ref],
                    confidence=1.0,
                    source_model="subtitle",
                    source_type="observed",
                )
            )
            if span.speaker:
                observations.append(
                    RawObservation(
                        obs_id=new_grounding_id("obs_spk"),
                        segment_id=segment.segment_id,
                        observation_type="speaker_turn",
                        payload={"speaker": span.speaker, "text": span.text},
                        evidence_refs=[ref],
                        confidence=1.0,
                        source_model="subtitle",
                        source_type="observed",
                    )
                )

        # 2) entity-detector hints (observed).
        for hint in entity_hints or []:
            observations.append(
                RawObservation(
                    obs_id=new_grounding_id("obs_det"),
                    segment_id=segment.segment_id,
                    observation_type="entity_mention",
                    payload=dict(hint),
                    evidence_refs=[clip_ref],
                    confidence=float(hint.get("confidence", 1.0) or 1.0),
                    source_model=str(hint.get("source", "detector")),
                    source_type="observed",
                )
            )

        # 3) VLM proposals (caption, events, interactions, hypotheses) —
        #    these are *proposals*; downstream layers decide observed vs
        #    inferred.
        if self.vlm_fn is not None:
            observations.extend(self._vlm_observations(
                segment,
                frames=frames,
                subtitles=subtitles,
                entity_hints=entity_hints,
                clip_ref=clip_ref,
                frame_refs=frame_refs,
                subtitle_refs=subtitle_refs,
            ))
        return observations

    # -- VLM call ------------------------------------------------------

    def _vlm_observations(
        self,
        segment: VideoSegment,
        *,
        frames: Sequence[SampledFrame],
        subtitles: Sequence[SubtitleSpan],
        entity_hints: Sequence[Dict[str, Any]],
        clip_ref: EvidenceRef,
        frame_refs: Sequence[EvidenceRef],
        subtitle_refs: Sequence[EvidenceRef],
    ) -> List[RawObservation]:
        prompt = GROUNDING_PROMPT_TEMPLATE.format(
            t0=segment.start_time,
            t1=segment.end_time,
            n_frames=len(frames),
            timestamps=[round(f.timestamp, 2) for f in frames],
            subtitles=_format_subtitles(subtitle_refs),
            entity_hints=_format_entity_hints(entity_hints),
        )
        frame_payload = [
            {"frame_id": f.frame_id, "timestamp": f.timestamp,
             "path": f.array_path}
            for f in frames
        ]
        if self.max_frames_in_prompt is not None:
            frame_payload = frame_payload[: self.max_frames_in_prompt]

        try:
            raw = self.vlm_fn(prompt, frames=frame_payload)  # type: ignore[misc]
        except TypeError:
            try:
                raw = self.vlm_fn(prompt)  # type: ignore[misc]
            except Exception:
                return []
        except Exception:
            return []

        parsed = _safe_json_loads(raw)
        if not parsed:
            return []

        out: List[RawObservation] = []

        scene = parsed.get("scene")
        if scene:
            out.append(RawObservation(
                obs_id=new_grounding_id("obs_cap"),
                segment_id=segment.segment_id,
                observation_type="caption",
                payload={"scene": scene},
                evidence_refs=[clip_ref, *frame_refs],
                confidence=0.9,
                source_model=self.source_model,
                source_type="observed",
            ))

        for e in parsed.get("entities", []) or []:
            out.append(RawObservation(
                obs_id=new_grounding_id("obs_ent"),
                segment_id=segment.segment_id,
                observation_type="participant_mention",
                payload={
                    "id": e.get("id"),
                    "type": e.get("type", "person"),
                    "attributes": dict(e.get("attributes", {}) or {}),
                },
                evidence_refs=[clip_ref, *frame_refs],
                confidence=0.85,
                source_model=self.source_model,
                source_type="observed",
            ))

        for x in parsed.get("interactions", []) or []:
            out.append(RawObservation(
                obs_id=new_grounding_id("obs_int"),
                segment_id=segment.segment_id,
                observation_type="interaction_proposal",
                payload={
                    "src": x.get("src"),
                    "rel": x.get("rel"),
                    "dst": x.get("dst"),
                    "metadata": dict(x.get("metadata", {}) or {}),
                },
                evidence_refs=[clip_ref, *frame_refs],
                confidence=float(x.get("confidence", 0.7)),
                source_model=self.source_model,
                source_type="observed",
            ))

        for ev in parsed.get("events", []) or []:
            out.append(RawObservation(
                obs_id=new_grounding_id("obs_evt"),
                segment_id=segment.segment_id,
                observation_type="event_proposal",
                payload={
                    "type": ev.get("type"),
                    "agents": list(ev.get("agents", []) or []),
                    "description": ev.get("description"),
                },
                evidence_refs=[clip_ref, *frame_refs],
                confidence=float(ev.get("confidence", 0.7)),
                source_model=self.source_model,
                source_type="observed",
            ))

        for h in parsed.get("social_hypotheses", []) or []:
            target = h.get("target")
            out.append(RawObservation(
                obs_id=new_grounding_id("obs_hyp"),
                segment_id=segment.segment_id,
                observation_type="social_hypothesis_proposal",
                payload={
                    "type": h.get("type") or "intention",
                    "target": target,
                    "value": h.get("value", ""),
                    "polarity": h.get("polarity", "uncertain"),
                    "supporting_evidence": list(
                        h.get("supporting_evidence", []) or []
                    ),
                    "contradicting_evidence": list(
                        h.get("contradicting_evidence", []) or []
                    ),
                    "provenance": h.get("provenance"),
                },
                evidence_refs=[clip_ref, *frame_refs],
                confidence=float(h.get("confidence", 0.5)),
                source_model=self.source_model,
                source_type="inferred",
            ))
        return out


__all__ = [
    "ObservationExtractor",
]

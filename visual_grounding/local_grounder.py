"""Local social grounding of a single window.

Step 3 of the pipeline (``infra_plans/01_grounding/video_benchmarks_grounding.md``
§3.1 / §3.2). Takes frames + subtitles for one window and returns a
:class:`GroundedWindow` populated with entities, interactions, events,
and social hypotheses — each with evidence pointers.

Design:
- The VLM is injected as ``vlm_fn(prompt, frames=...) -> str``. This keeps
  the module independent of any specific model and matches the frozen-tool
  contract in ``actors_reasoning_model.md`` §2.4.
- When no ``vlm_fn`` is supplied, the grounder falls back to the repo's
  ``ask_model`` (text-only) so the pipeline remains importable and
  testable without a real VLM.
- Output parsing is defensive: on any error we return an empty
  :class:`GroundedWindow` rather than raising, so a single bad window
  never poisons the whole video.

The JSON schema emitted by the VLM matches §3.2 exactly.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from visual_grounding.perception import SampledFrame
from visual_grounding.schemas import (
    Entity,
    Event,
    EvidenceRef,
    GroundedWindow,
    Interaction,
    SocialHypothesis,
    new_id,
)
from visual_grounding.segmenter import Window


# A VLM callable: (prompt, *, frames=None, images=None) -> str.
VLMCallable = Callable[..., str]


GROUNDING_PROMPT_TEMPLATE = """You are a social-aware video grounder. You observe a short window of a
video and output structured JSON describing the grounded social state.

Window time span: {t0:.2f}s – {t1:.2f}s
Number of sampled frames: {n_frames}
Sampling timestamps (s): {timestamps}

Subtitles/ASR overlapping this window (may be empty):
{subtitles}

Entity hints from detection (may be empty):
{entity_hints}

Return a single JSON object with this exact schema (no prose, no code
fences). Leave arrays empty when nothing applies. Confidence is in
[0, 1]. Every high-level claim must point at frames or subtitle lines
already listed above.

{{
  "time_span": [{t0:.2f}, {t1:.2f}],
  "scene": "<one short phrase>",
  "entities": [
    {{"id": "p1", "type": "person|object|group|speaker",
      "attributes": {{"emotion": "...", "gaze": "...", "speaking": false}}}}
  ],
  "interactions": [
    {{"src": "p1", "rel": "talking_to|looking_at|helping|blocking|...",
      "dst": "p2", "confidence": 0.8}}
  ],
  "events": [
    {{"type": "enters_room|starts_argument|...",
      "agents": ["p1", "p2"], "confidence": 0.7,
      "description": "short phrase"}}
  ],
  "social_hypotheses": [
    {{"type": "intention|belief|emotion|trust|suspicion|alliance|\
conflict|deception_risk|goal|commitment",
      "target": "p1",
      "value": "short phrase",
      "confidence": 0.6,
      "provenance": "directly_observed|inferred_from_dialogue|\
inferred_from_behavior|inferred_from_absence",
      "supporting_evidence": ["<frame_id or sub_id>"],
      "contradicting_evidence": []}}
  ]
}}
"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _default_vlm_fn(prompt: str, **_kwargs: Any) -> str:
    """Fallback VLM stub using the repo's text-only ``ask_model``.

    It won't see the frames, but it still returns a well-formed JSON
    envelope so the pipeline can be exercised end-to-end without a
    multimodal model.
    """
    try:
        from API_func import ask_model  # type: ignore
        return ask_model(prompt, model="gpt-4o-mini", temperature=0.2,
                         max_tokens=1500)
    except Exception:
        return ""


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    for loader in (json.loads,):
        try:
            return loader(blob)
        except Exception:
            continue
    # last-ditch: strip trailing commas
    try:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", blob)
        return json.loads(cleaned)
    except Exception:
        return None


def _format_subtitles(refs: Sequence[EvidenceRef]) -> str:
    if not refs:
        return "(none)"
    out = []
    for r in refs:
        s, e = r.timestamp if r.timestamp else (0.0, 0.0)
        out.append(f"- [{r.ref_id}] [{s:.2f}-{e:.2f}s] {r.text or ''}")
    return "\n".join(out)


def _format_entity_hints(detections: Sequence[Dict[str, Any]]) -> str:
    if not detections:
        return "(none)"
    out = []
    for d in detections:
        out.append(
            f"- {d.get('type', 'unknown')} at frame "
            f"{d.get('frame_id', '?')} conf={d.get('confidence', 0):.2f}"
        )
    return "\n".join(out)


def ground_window(
    window: Window,
    frames: Sequence[SampledFrame],
    subtitles: Sequence[EvidenceRef],
    *,
    entity_hints: Sequence[Dict[str, Any]] = (),
    vlm_fn: Optional[VLMCallable] = None,
    subtitle_mode: str = "origin",
) -> GroundedWindow:
    """Run local social grounding on a single window.

    Args:
        window: The window descriptor from ``adaptive_segment``.
        frames: Decoded / pre-decoded frames aligned to ``window.frame_times``.
        subtitles: Subtitle / ASR evidence refs overlapping this window.
        entity_hints: Optional detector output used to steer the VLM.
        vlm_fn: Pluggable VLM callable. Must accept ``(prompt, frames=...)``;
            returns the VLM's raw text. Defaults to the repo's ``ask_model``
            when omitted (text-only fallback).
        subtitle_mode: One of ``origin|added|removed|none`` (SIV-Bench style).

    Returns:
        A fully populated :class:`GroundedWindow`. On parse failure the
        window is returned empty (but with evidence refs preserved), so the
        pipeline degrades gracefully.
    """
    t0, t1 = window.time_span
    prompt = GROUNDING_PROMPT_TEMPLATE.format(
        t0=t0,
        t1=t1,
        n_frames=len(frames),
        timestamps=[round(f.timestamp, 2) for f in frames],
        subtitles=_format_subtitles(subtitles),
        entity_hints=_format_entity_hints(entity_hints),
    )

    fn = vlm_fn or _default_vlm_fn
    frame_payload = [
        {"frame_id": f.frame_id, "timestamp": f.timestamp, "path": f.array_path}
        for f in frames
    ]
    try:
        raw = fn(prompt, frames=frame_payload)
    except TypeError:
        # vlm_fn doesn't accept ``frames``; call with prompt only.
        raw = fn(prompt)
    except Exception:
        raw = ""

    parsed = _safe_json_loads(raw)

    evidence = list(subtitles)
    for f in frames:
        evidence.append(
            EvidenceRef(
                ref_id=f.frame_id,
                modality="frame",
                timestamp=(f.timestamp, f.timestamp),
                locator={"path": f.array_path, **f.metadata} if f.array_path
                else dict(f.metadata),
            )
        )

    gw = GroundedWindow(
        window_id=window.window_id,
        time_span=(t0, t1),
        subtitle_mode=subtitle_mode,  # type: ignore[arg-type]
        evidence=evidence,
        frame_indices=[
            int(f.metadata.get("frame_index", i))
            for i, f in enumerate(frames)
        ],
    )

    if not parsed:
        return gw

    gw.scene = parsed.get("scene")
    for e in parsed.get("entities", []) or []:
        try:
            gw.entities.append(Entity(
                id=str(e.get("id") or new_id("p")),
                type=e.get("type") or "person",
                attributes=dict(e.get("attributes", {}) or {}),
            ))
        except Exception:
            continue
    for x in parsed.get("interactions", []) or []:
        try:
            gw.interactions.append(Interaction(
                src=str(x.get("src")),
                rel=str(x.get("rel")),
                dst=str(x.get("dst")),
                confidence=float(x.get("confidence", 1.0)),
                metadata=dict(x.get("metadata", {}) or {}),
            ))
        except Exception:
            continue
    for ev in parsed.get("events", []) or []:
        try:
            gw.events.append(Event(
                type=str(ev.get("type")),
                agents=[str(a) for a in ev.get("agents", []) or []],
                confidence=float(ev.get("confidence", 1.0)),
                description=ev.get("description"),
                metadata=dict(ev.get("metadata", {}) or {}),
            ))
        except Exception:
            continue
    for h in parsed.get("social_hypotheses", []) or []:
        try:
            target = h.get("target")
            gw.social_hypotheses.append(SocialHypothesis(
                type=h.get("type") or "intention",
                target=target if isinstance(target, list) else str(target),
                value=str(h.get("value", "")),
                confidence=float(h.get("confidence", 0.5)),
                provenance=h.get("provenance")
                or "inferred_from_behavior",  # type: ignore[arg-type]
                supporting_evidence=list(h.get("supporting_evidence", []) or []),
                contradicting_evidence=list(
                    h.get("contradicting_evidence", []) or []
                ),
            ))
        except Exception:
            continue
    return gw


def ground_windows_batch(
    windows: Sequence[Window],
    frames_by_window: Dict[str, List[SampledFrame]],
    subtitles_by_window: Dict[str, List[EvidenceRef]],
    *,
    entity_hints_by_window: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    vlm_fn: Optional[VLMCallable] = None,
    subtitle_mode: str = "origin",
) -> List[GroundedWindow]:
    """Ground a sequence of windows (no concurrency — simple driver)."""
    out: List[GroundedWindow] = []
    hints = entity_hints_by_window or {}
    for w in windows:
        out.append(
            ground_window(
                w,
                frames_by_window.get(w.window_id, []),
                subtitles_by_window.get(w.window_id, []),
                entity_hints=hints.get(w.window_id, []),
                vlm_fn=vlm_fn,
                subtitle_mode=subtitle_mode,
            )
        )
    return out


__all__ = [
    "VLMCallable",
    "GROUNDING_PROMPT_TEMPLATE",
    "ground_window",
    "ground_windows_batch",
]

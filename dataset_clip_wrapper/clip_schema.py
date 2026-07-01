"""Structured clip schema generation with a multimodal OpenRouter model."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .openrouter_client import OpenRouterClient
from .schemas import ClipSchemaConfig, ClipSpan

CLIP_SCHEMA_PROMPT = """You convert one video clip span into a structured perception record.

Return JSON only with this shape:
{
  "clip_id": "string",
  "time_span": {"start_s": number, "end_s": number},
  "granularity": "whole|coarse|fine",
  "scene_description": "short grounded scene summary",
  "observable_facts": [
    {"text": "fact grounded in visible or spoken content", "modality": "visual|audio|subtitle|mixed"}
  ],
  "dialogue_spans": [
    {"speaker": "name or unknown", "text": "utterance", "time_span": {"start_s": number, "end_s": number}}
  ],
  "entity_mentions": [
    {"surface_form": "name or object", "entity_type": "person|object|place|other"}
  ],
  "events": [
    {"description": "timestamped event", "time_span": {"start_s": number, "end_s": number}}
  ]
}

Rules:
1. Use only information supported by the clip frames or provided subtitle/context text.
2. Do not invent characters, objects, or events.
3. Keep lists short and precise.
4. If nothing is visible, return empty lists and a cautious scene_description.
"""


class QwenClipSchemaProducer:
    """Produce structured clip schemas from segmented video spans."""

    def __init__(self, config: ClipSchemaConfig, client: OpenRouterClient):
        self.config = config
        self.client = client

    def _sample_frame_jpegs(self, video_path: Path, clip: ClipSpan) -> list[str]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("clip schema generation requires opencv-python") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        frames: list[str] = []
        count = self.config.request_frames
        if count <= 1:
            sample_times = [(clip.start_s + clip.end_s) / 2.0]
        else:
            step = max(clip.end_s - clip.start_s, 0.1) / max(count - 1, 1)
            sample_times = [clip.start_s + i * step for i in range(count)]
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        cap.release()
        return frames

    def build_clip_schema(
        self,
        *,
        clip_id: str,
        clip: ClipSpan,
        video_path: Path | None,
        subtitle_context: str | None = None,
        question_context: str | None = None,
    ) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{CLIP_SCHEMA_PROMPT}\n"
                    f"clip_id={clip_id}\n"
                    f"time_span={clip.to_dict()}\n"
                    f"granularity={clip.granularity}\n"
                    f"subtitle_context={subtitle_context or ''}\n"
                    f"question_context={question_context or ''}"
                ),
            }
        ]
        if video_path and video_path.exists():
            for data in self._sample_frame_jpegs(video_path, clip):
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
                )

        payload = self.client.chat_json(
            [
                {"role": "system", "content": "You are a grounded video perception annotator."},
                {"role": "user", "content": user_content},
            ]
        )
        payload.setdefault("clip_id", clip_id)
        payload.setdefault("time_span", clip.to_dict())
        payload.setdefault("granularity", clip.granularity)
        payload["model"] = self.config.model
        payload["producer"] = "qwen_clip_schema"
        return payload

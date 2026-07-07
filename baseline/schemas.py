"""JSONL-friendly standard records for baseline video QA retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


InputMode = Literal["frames", "video_clip", "text_memory"]


@dataclass(frozen=True)
class VideoQAExample:
    """One question over one video under a stated visibility policy."""

    example_id: str
    dataset: str
    split: str
    video_id: str
    video_path: str
    question_text: str
    options: list[dict[str, str]]
    answer_label: str | None
    answer_text: str | None
    visible_until_s: float | None
    streaming_mode: str
    input_mode: InputMode = "video_clip"
    question_embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoClipRecord:
    """One stable clip span from a canonical wrapper example."""

    row_id: int
    example_id: str
    dataset: str
    video_id: str
    clip_id: str
    video_path: str
    start_s: float
    end_s: float
    granularity: str
    visible_until_s: float | None
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedClip:
    """One FAISS retrieval result."""

    rank: int
    score: float
    row_id: int
    clip: VideoClipRecord

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clip"] = self.clip.to_dict()
        return payload


def visible_until_from_canonical(example: dict[str, Any], default_videomme_cutoff_s: float | None = 60.0) -> float | None:
    """Infer the baseline streaming cutoff from a canonical wrapper example."""

    question = example.get("question") or {}
    video = example.get("video") or {}
    duration = video.get("duration_s")
    duration_s = float(duration) if isinstance(duration, (int, float)) else None
    anchor = question.get("time_anchor_s")
    if isinstance(anchor, (int, float)):
        visible = float(anchor)
    elif example.get("dataset") == "videomme":
        visible = default_videomme_cutoff_s if default_videomme_cutoff_s is not None else duration_s
    else:
        visible = duration_s
    if visible is not None and duration_s is not None:
        visible = max(0.0, min(visible, duration_s))
    return visible


def qa_example_from_canonical(
    example: dict[str, Any],
    *,
    input_mode: InputMode = "video_clip",
    default_videomme_cutoff_s: float | None = 60.0,
    question_embedding: list[float] | None = None,
) -> VideoQAExample:
    question = example.get("question") or {}
    answer = question.get("answer") or {}
    video = example.get("video") or {}
    return VideoQAExample(
        example_id=str(example.get("example_id") or ""),
        dataset=str(example.get("dataset") or ""),
        split=str(example.get("split") or ""),
        video_id=str(video.get("video_id") or ""),
        video_path=str(video.get("primary_path") or ""),
        question_text=str(question.get("question_text") or ""),
        options=list(question.get("options") or []),
        answer_label=answer.get("label"),
        answer_text=answer.get("text"),
        visible_until_s=visible_until_from_canonical(example, default_videomme_cutoff_s),
        streaming_mode=str((example.get("metadata") or {}).get("video_regime") or example.get("task_family") or ""),
        input_mode=input_mode,
        question_embedding=question_embedding,
        metadata={
            "task_family": example.get("task_family"),
            "question_id": question.get("question_id"),
            "answer_format": question.get("answer_format"),
        },
    )


def clip_records_from_canonical(
    example: dict[str, Any],
    *,
    start_row_id: int = 0,
    default_videomme_cutoff_s: float | None = 60.0,
    embeddings: list[list[float]] | None = None,
) -> list[VideoClipRecord]:
    """Convert canonical `video.derived_clips` into baseline clip records."""

    qa = qa_example_from_canonical(example, default_videomme_cutoff_s=default_videomme_cutoff_s)
    records: list[VideoClipRecord] = []
    clips = ((example.get("video") or {}).get("derived_clips") or [])
    for clip in clips:
        span = clip.get("source_span") or {}
        start_s = float(span.get("start_s", 0.0))
        end_s = float(span.get("end_s", start_s))
        if qa.visible_until_s is not None:
            if start_s > qa.visible_until_s:
                continue
            end_s = min(end_s, qa.visible_until_s)
        if end_s <= start_s:
            continue
        clip_id = str(clip.get("clip_id") or f"{qa.video_id}:{len(records)}")
        granularity = str(clip.get("granularity") or "unknown")
        text = " ".join(
            part
            for part in [
                f"dataset={qa.dataset}",
                f"example={qa.example_id}",
                f"video={qa.video_id}",
                f"clip={clip_id}",
                f"time={start_s:.2f}-{end_s:.2f}s",
                f"granularity={granularity}",
                f"question={qa.question_text}",
            ]
            if part
        )
        embedding = embeddings[len(records)] if embeddings is not None and len(records) < len(embeddings) else None
        records.append(
            VideoClipRecord(
                row_id=start_row_id + len(records),
                example_id=qa.example_id,
                dataset=qa.dataset,
                video_id=qa.video_id,
                clip_id=clip_id,
                video_path=str(clip.get("path") or qa.video_path),
                start_s=start_s,
                end_s=end_s,
                granularity=granularity,
                visible_until_s=qa.visible_until_s,
                text=text,
                embedding=embedding,
                metadata={"parent_index": clip.get("parent_index")},
            )
        )
    return records

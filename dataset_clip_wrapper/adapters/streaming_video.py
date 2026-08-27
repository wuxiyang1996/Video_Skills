"""StreamBridge-style OVO-Bench and VideoMME adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetAdapter, RawDatasetItem


def _label_from_index(index: int) -> str:
    return chr(ord("A") + index)


def _normalize_options(options: Any) -> list[dict[str, str]]:
    if isinstance(options, dict):
        return [{"label": str(label), "text": str(text)} for label, text in options.items()]
    if not isinstance(options, list):
        return []

    normalized = []
    for idx, option in enumerate(options):
        label = _label_from_index(idx)
        text = str(option)
        if len(text) >= 3 and text[1:3] in {". ", ") "} and text[0].isalpha():
            label = text[0].upper()
            text = text[3:].strip()
        normalized.append({"label": label, "text": text})
    return normalized


def _answer_from_row(row: dict[str, Any], options: list[dict[str, str]]) -> dict[str, str | None]:
    answer = row.get("answer")
    gt = row.get("gt")
    if isinstance(gt, int) and 0 <= gt < len(options):
        return {"label": options[gt]["label"], "text": options[gt]["text"]}
    if isinstance(answer, str):
        stripped = answer.strip()
        if len(stripped) == 1 and stripped.isalpha():
            label = stripped.upper()
            return {"label": label, "text": next((opt["text"] for opt in options if opt["label"] == label), None)}
        for opt in options:
            if stripped == opt["text"] or stripped.lower() == opt["text"].lower():
                return {"label": opt["label"], "text": opt["text"]}
        return {"label": None, "text": stripped}
    return {"label": None, "text": None}


class OVOBenchAdapter(DatasetAdapter):
    """OVO-Bench format adapter.

    The local workspace currently provides StreamBridge tiny annotations. The
    adapter keeps the same fields as the official StreamBridge OVO path:
    ``video``, ``realtime``, ``question``, ``options``, ``gt`` and ``answer``.
    """

    name = "ovo_bench"

    def _root(self) -> Path:
        return self.dataset_root / "streambridge_tiny"

    def _qa_path(self) -> Path:
        root = self._root()
        preferred = root / "tiny_ovo_bench_50videos.json"
        if preferred.exists():
            return preferred
        return root / "tiny_ovo_bench.json"

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        records = json.loads(self._qa_path().read_text(encoding="utf-8"))
        count = 0
        for row in records:
            video_name = str(row.get("video") or "")
            video_id = Path(video_name).stem
            realtime = row.get("realtime")
            realtime_s = float(realtime) if realtime is not None else None
            qid = str(row.get("id") or f"{video_id}:{count}")
            options = _normalize_options(row.get("options"))
            answer = _answer_from_row(row, options)
            video_path = self._root() / "videos" / video_name

            yield RawDatasetItem(
                dataset=self.name,
                example_id=f"ovo_bench:{qid}",
                split=self.split,
                task_family="streaming_video_realtime_qa",
                video_id=video_id,
                video_path=video_path if video_path.exists() else None,
                duration_s=None,
                question={
                    "question_id": qid,
                    "question_text": row.get("question", ""),
                    "question_type": row.get("task"),
                    "options": options,
                    "answer": answer,
                    "answer_format": "multiple_choice",
                    "time_anchor_s": realtime_s,
                },
                subtitle_paths=[],
                annotation_segments=[],
                evidence_seeds=[],
                hidden_supervision_sources=["official_answer"],
                raw_source_refs=[
                    {
                        "source_name": self._qa_path().name,
                        "source_item_id": qid,
                        "fields_used": ["video", "task", "realtime", "question", "options", "gt", "answer"],
                    }
                ],
                metadata={
                    "benchmark_family": "streaming_video_benchmarks",
                    "benchmark_format": "ovo_bench",
                    "streambridge_compatible": True,
                    "realtime_s": realtime_s,
                },
            )
            count += 1
            if limit is not None and count >= limit:
                break


class VideoMMEAdapter(DatasetAdapter):
    """VideoMME format adapter for StreamBridge-style records."""

    name = "videomme"

    def _root(self) -> Path:
        return self.dataset_root / "streambridge_tiny"

    def _qa_path(self) -> Path:
        return self._root() / "tiny_videomme.json"

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        records = json.loads(self._qa_path().read_text(encoding="utf-8"))
        count = 0
        for row in records:
            video_id = str(row.get("videoID") or "")
            qid = str(row.get("id") or f"{video_id}:{count}")
            options = _normalize_options(row.get("options"))
            answer = _answer_from_row(row, options)
            video_path = self._root() / "videos" / f"{video_id}.mp4"

            yield RawDatasetItem(
                dataset=self.name,
                example_id=f"videomme:{qid}",
                split=self.split,
                task_family="streaming_video_whole_video_qa",
                video_id=video_id,
                video_path=video_path if video_path.exists() else None,
                duration_s=None,
                question={
                    "question_id": qid,
                    "question_text": row.get("question", ""),
                    "question_type": row.get("duration"),
                    "options": options,
                    "answer": answer,
                    "answer_format": "multiple_choice",
                },
                subtitle_paths=[],
                annotation_segments=[],
                evidence_seeds=[],
                hidden_supervision_sources=["official_answer"],
                raw_source_refs=[
                    {
                        "source_name": self._qa_path().name,
                        "source_item_id": qid,
                        "fields_used": ["videoID", "duration", "question", "options", "answer"],
                    }
                ],
                metadata={
                    "benchmark_family": "streaming_video_benchmarks",
                    "benchmark_format": "videomme",
                    "streambridge_compatible": True,
                    "duration_bucket": row.get("duration"),
                },
            )
            count += 1
            if limit is not None and count >= limit:
                break

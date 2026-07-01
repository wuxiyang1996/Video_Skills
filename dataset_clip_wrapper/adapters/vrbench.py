"""VRBench adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetAdapter, RawDatasetItem


def _parse_reasoning_timestamp(text: str) -> dict[str, float] | None:
    match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-~]\s*(\d{1,2}:\d{2}(?::\d{2})?)", text)
    if not match:
        match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", text)
        if not match:
            return None
        start = match.group(1)

        def to_seconds(part: str) -> float:
            pieces = [float(p) for p in part.split(":")]
            if len(pieces) == 3:
                return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
            if len(pieces) == 2:
                return pieces[0] * 60 + pieces[1]
            return pieces[0]

        start_s = to_seconds(start)
        return {"start_s": start_s, "end_s": start_s + 1.0}
    start, end = match.group(1), match.group(2)

    def to_seconds(part: str) -> float:
        pieces = [float(p) for p in part.split(":")]
        if len(pieces) == 3:
            return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
        if len(pieces) == 2:
            return pieces[0] * 60 + pieces[1]
        return pieces[0]

    start_s = to_seconds(start)
    end_s = to_seconds(end)
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    return {"start_s": start_s, "end_s": end_s}


class VRBenchAdapter(DatasetAdapter):
    name = "vrbench"

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        bench = self.dataset_root / "VRBench"
        eval_path = bench / "VRBench_eval.jsonl"
        count = 0
        with eval_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                video_id = row["video_id"]
                rel_path = row.get("video_path") or f"v001_360p/{video_id}.mp4"
                video_path = bench / rel_path
                mcq_map: dict[str, Any] = row.get("mcq") or {}
                for qa_key, qa in mcq_map.items():
                    example_id = f"vrbench:{video_id}:{qa_key}"
                    annotation_segments = []
                    evidence_seeds = []
                    if row.get("video_summary"):
                        annotation_segments.append(
                            {
                                "segment_id": f"vb_summary_{qa_key}",
                                "source_type": "video_summary",
                                "text": row["video_summary"],
                                "provenance": {"field": "video_summary"},
                            }
                        )
                        evidence_seeds.append(
                            {
                                "evidence_id": f"ev:vb:summary:{video_id}:{qa_key}",
                                "source_type": "video_summary",
                                "text": row["video_summary"],
                                "trust_level": "strong",
                                "provenance": {"source_field": "video_summary"},
                            }
                        )
                    for i, step in enumerate(qa.get("reasoning_process") or [], start=1):
                        span = _parse_reasoning_timestamp(step)
                        annotation_segments.append(
                            {
                                "segment_id": f"vb_rp_{qa_key}_{i:02d}",
                                "source_type": "reasoning_process_step",
                                "time_span": span,
                                "text": step,
                                "provenance": {"field": "reasoning_process"},
                            }
                        )
                        evidence_seeds.append(
                            {
                                "evidence_id": f"ev:vb:rp:{video_id}:{qa_key}:{i}",
                                "source_type": "reasoning_process_step",
                                "time_span": span,
                                "text": step,
                                "trust_level": "gold",
                                "provenance": {"source_field": "reasoning_process"},
                            }
                        )
                    options = qa.get("options") or {}
                    answer = qa.get("answer")
                    yield RawDatasetItem(
                        dataset=self.name,
                        example_id=example_id,
                        split=self.split,
                        task_family="long_video_temporal_chain_qa",
                        video_id=video_id,
                        video_path=video_path if video_path.exists() else None,
                        duration_s=None,
                        question={
                            "question_id": qa_key,
                            "question_text": qa.get("question", ""),
                            "options": [{"label": k, "text": v} for k, v in options.items()],
                            "answer": {"label": answer, "text": options.get(answer)},
                            "answer_format": "multiple_choice",
                        },
                        subtitle_paths=[],
                        annotation_segments=annotation_segments,
                        evidence_seeds=evidence_seeds,
                        hidden_supervision_sources=["official_answer", "reasoning_process"],
                        raw_source_refs=[
                            {
                                "source_name": "VRBench_eval.jsonl",
                                "source_item_id": example_id,
                                "fields_used": ["question", "options", "answer", "reasoning_process", "video_summary"],
                            }
                        ],
                        metadata={"reasoning_type": qa.get("reasoning_type"), "video_read_type": row.get("video_read_type")},
                    )
                    count += 1
                    if limit is not None and count >= limit:
                        return
                if limit is not None and count >= limit:
                    return

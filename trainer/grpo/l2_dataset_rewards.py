"""Evaluator-only process supervision for CG-Bench and Video-Holmes.

Hidden benchmark annotations are loaded outside the policy-visible state.  The
resulting components are suitable for OPD labels, terminal reward shaping, and
paper evaluation, but must never be copied into an L2 prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_clip_wrapper.adapters.video_holmes import parse_time_range


VH_PLACEHOLDER_FILTER_VERSION = "unfinished-fill-in-prefix-v1"
RELATIONSHIP_SUPPORT_VERSION = "structured-concept-overlap-v2"


def _span(value: Any) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        try:
            start = float(value.get("start_s"))
            end = float(value.get("end_s"))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        try:
            start, end = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    else:
        return None
    if end < start:
        start, end = end, start
    if end <= start:
        end = start + 1.0
    return {"start_s": start, "end_s": end}


def temporal_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = _span(left)
    b = _span(right)
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"])
    return intersection / union if union > 0 else 0.0


def temporal_hit(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = _span(left)
    b = _span(right)
    return bool(a and b and a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"])


def temporal_retrieval_metrics(
    selected: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    selected_spans = [span for value in selected if (span := _span(value)) is not None]
    gold_spans = [span for value in gold if (span := _span(value)) is not None]
    if not gold_spans:
        return {"recall": 0.0, "precision": 0.0, "mean_best_iou": 0.0, "gold_count": 0.0}
    recall = sum(any(temporal_hit(candidate, target) for candidate in selected_spans) for target in gold_spans)
    precision = sum(any(temporal_hit(candidate, target) for target in gold_spans) for candidate in selected_spans)
    best_ious = [max((temporal_iou(candidate, target) for candidate in selected_spans), default=0.0) for target in gold_spans]
    return {
        "recall": recall / len(gold_spans),
        "precision": precision / max(1, len(selected_spans)),
        "mean_best_iou": sum(best_ious) / len(best_ious),
        "gold_count": float(len(gold_spans)),
    }


def _text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    pieces = [value.get("scene_description"), value.get("uncertainty")]
    pieces.extend(item.get("text") for item in value.get("observable_facts") or [] if isinstance(item, Mapping))
    pieces.extend(item.get("description") for item in value.get("events") or [] if isinstance(item, Mapping))
    pieces.extend(
        item.get("surface_form")
        for item in value.get("entity_mentions") or []
        if isinstance(item, Mapping)
    )
    pieces.extend(
        item.get("description")
        for item in value.get("visual_social_cues") or []
        if isinstance(item, Mapping)
    )
    pieces.extend(
        item.get("description")
        for item in value.get("cross_clip_cues") or []
        if isinstance(item, Mapping)
    )
    pieces.extend(value.get("searchable_phrases") or [])
    place = value.get("place") if isinstance(value.get("place"), Mapping) else {}
    pieces.append(place.get("description"))
    pieces.extend(place.get("searchable_phrases") or [])
    return " ".join(str(piece) for piece in pieces if piece)


_PLACEHOLDER_PREFIX = re.compile(
    r"^\s*(?:please\s+)?fill\s+in\b|^\s*(?:to\s+be\s+filled|placeholder)\b",
    flags=re.IGNORECASE,
)


def is_placeholder_annotation(value: Any) -> bool:
    """Reject unfinished Video-Holmes template fields from evaluator labels."""
    text = str(value or "").strip()
    return not text or bool(_PLACEHOLDER_PREFIX.search(text))


_RELATION_CONCEPTS = {
    "man": "person", "men": "person", "woman": "person", "women": "person",
    "girl": "person", "girls": "person", "boy": "person", "boys": "person",
    "people": "person", "person": "person", "character": "person", "figure": "person",
    "fight": "fight", "fights": "fight", "fighting": "fight", "combat": "fight",
    "attack": "fight", "attacks": "fight", "attacking": "fight", "battle": "fight",
    "projectile": "fight", "weapon": "fight", "firing": "fight", "fires": "fight",
    "phone": "communicate", "calling": "communicate", "talking": "communicate",
    "conversation": "communicate", "dialogue": "communicate",
    "piano": "music", "music": "music", "musical": "music",
}


def _relation_tokens(text: str) -> set[str]:
    output: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if len(raw) < 4 and raw not in _RELATION_CONCEPTS:
            continue
        token = _RELATION_CONCEPTS.get(raw, raw)
        if token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        output.add(_RELATION_CONCEPTS.get(token, token))
    return output


def lexical_support(selected_entries: Sequence[Mapping[str, Any]], references: Sequence[str]) -> float:
    """Deterministic structured concept support for relationship annotations."""
    selected_tokens = _relation_tokens(" ".join(_text(row) for row in selected_entries))
    reference_tokens = _relation_tokens(" ".join(references))
    if not reference_tokens:
        return 0.0
    return min(1.0, len(selected_tokens & reference_tokens) / max(1, min(8, len(reference_tokens))))


def load_dataset_reward_supervision(dataset_root: Path) -> dict[str, dict[str, Any]]:
    """Load train/eval labels into an evaluator-only lookup keyed by example/video."""
    result: dict[str, dict[str, Any]] = {}
    cg_path = dataset_root / "CG-Bench" / "cgbench.json"
    if cg_path.exists():
        for row in json.loads(cg_path.read_text(encoding="utf-8")):
            qid = str(row.get("qid") or "")
            spans = [span for value in row.get("clue_intervals") or [] if (span := _span(value))]
            if qid:
                result[f"cg_bench:{qid}"] = {
                    "dataset": "cg_bench",
                    "clue_spans": spans,
                }

    benchmark = dataset_root / "Video-Holmes" / "Benchmark"
    annotation_dirs = [benchmark / "annotation_training", benchmark / "annotations"]
    seen: set[str] = set()
    for annotation_dir in annotation_dirs:
        if not annotation_dir.exists():
            continue
        for path in sorted(annotation_dir.glob("*.json")):
            if path.stem in seen:
                continue
            seen.add(path.stem)
            payload = json.loads(path.read_text(encoding="utf-8"))
            annotation = payload[0] if isinstance(payload, list) and payload else payload
            if not isinstance(annotation, Mapping):
                continue
            segment_spans = []
            segment_texts = []
            dropped_placeholder_segments = 0
            for row in annotation.get("Segment Description") or annotation.get("SegmentDescription") or []:
                if not isinstance(row, Mapping):
                    continue
                description = str(row.get("Description") or "").strip()
                if is_placeholder_annotation(description):
                    dropped_placeholder_segments += 1
                    continue
                if span := parse_time_range(row.get("TimeRange")):
                    segment_spans.append(span)
                segment_texts.append(description)
            inference_spans = []
            inference_texts = []
            dropped_placeholder_inference_rows = 0
            for row in annotation.get("Inference Shots") or annotation.get("InferenceScenes") or []:
                if not isinstance(row, Mapping):
                    continue
                valid_texts = [
                    str(row[key]).strip()
                    for key in ("Clue", "Conclusion")
                    if row.get(key) and not is_placeholder_annotation(row[key])
                ]
                # A timestamp attached only to unfinished template text is not a
                # gold inference shot (many such rows are the artificial 0--1s span).
                if not valid_texts:
                    dropped_placeholder_inference_rows += 1
                    continue
                if span := parse_time_range(row.get("Time")):
                    inference_spans.append(span)
                inference_texts.extend(valid_texts)
            relationship_texts = []
            dropped_placeholder_relationship_fields = 0
            for row in annotation.get("Key Relationships") or annotation.get("KeyRelationships") or []:
                if isinstance(row, Mapping):
                    for key in ("Combination", "Relation", "Reason"):
                        if not row.get(key):
                            continue
                        if is_placeholder_annotation(row[key]):
                            dropped_placeholder_relationship_fields += 1
                            continue
                        relationship_texts.append(str(row[key]).strip())
            result[f"video_holmes:{path.stem}"] = {
                "dataset": "video_holmes",
                "segment_spans": segment_spans,
                "inference_spans": inference_spans,
                "segment_texts": segment_texts,
                "inference_texts": inference_texts,
                "relationship_texts": relationship_texts,
                "annotation_quality": {
                    "placeholder_filter": VH_PLACEHOLDER_FILTER_VERSION,
                    "dropped_placeholder_segments": dropped_placeholder_segments,
                    "dropped_placeholder_inference_rows": dropped_placeholder_inference_rows,
                    "dropped_placeholder_relationship_fields": dropped_placeholder_relationship_fields,
                },
            }
    return result


def supervision_key(example: Mapping[str, Any]) -> str:
    dataset = str(example.get("dataset") or "")
    if dataset == "cg_bench":
        qid = str((example.get("question") or {}).get("question_id") or "")
        return f"cg_bench:{qid}"
    video = example.get("video") if isinstance(example.get("video"), Mapping) else {}
    video_id = str(video.get("video_id") or "")
    if not video_id:
        parts = str(example.get("example_id") or "").split(":")
        video_id = parts[2] if len(parts) >= 4 and parts[0] == "video_holmes" else ""
    return f"{dataset}:{video_id}"

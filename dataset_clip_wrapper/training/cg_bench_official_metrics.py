#!/usr/bin/env python3
"""CG-Bench official clue-grounding metrics (mIoU, rec.@IoU).

The interval scoring mirrors ``calculate_intervals_iou`` from the benchmark's
reference implementation (CG-Bench/CG-Bench, ``run/utils.py``): both interval
lists are merged, then scored as a set-IoU over the timeline,

    tIoU = |P n G| / (|P| + |G| - |P n G|)

which is *not* the same as a per-gold best-match IoU.  Because the union grows
with every extra predicted interval, the official metric penalises
over-prediction: a 30s clip covering a 9s clue scores 0.30, but five such clips
covering the same clue score 0.06.

The benchmark's prompt caps predictions at five intervals ("You must provide at
least one interval and at most five intervals.  Intervals exceeding five will
NOT be considered valid."), so ``MAX_OFFICIAL_INTERVALS`` is a protocol limit,
not a tuning knob.

Answer-dependent metrics (long-acc, clue-acc, acc.@IoU, CRR) need a QA path and
are deliberately not implemented here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

MAX_OFFICIAL_INTERVALS = 5
REC_IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[list[float]]:
    """Merge overlapping intervals, as the reference implementation does."""
    ordered = sorted(([float(a), float(b)] for a, b in intervals), key=lambda x: x[0])
    if not ordered:
        return []
    merged = [ordered[0]]
    for current in ordered[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    return merged


def intervals_iou(
    predicted: Iterable[Sequence[float]], gold: Iterable[Sequence[float]]
) -> float:
    """Set-IoU between two interval lists, matching CG-Bench's reference."""
    left = merge_intervals(predicted)
    right = merge_intervals(gold)
    length_left = sum(end - start for start, end in left)
    length_right = sum(end - start for start, end in right)
    intersection = 0.0
    for a_start, a_end in left:
        for b_start, b_end in right:
            intersection += max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = length_left + length_right - intersection
    return intersection / union if union > 0 else 0.0


def score_questions(
    per_question: Sequence[tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]],
) -> dict[str, Any]:
    """Return official mIoU and rec.@IoU over (predicted, gold) interval pairs.

    Predictions beyond the five-interval protocol limit are dropped rather than
    silently scored, matching the benchmark's stated validity rule.
    """
    ious = []
    over_limit = 0
    for predicted, gold in per_question:
        predicted = list(predicted)
        if len(predicted) > MAX_OFFICIAL_INTERVALS:
            over_limit += 1
            predicted = predicted[:MAX_OFFICIAL_INTERVALS]
        ious.append(intervals_iou(predicted, gold))
    n = max(1, len(ious))
    recalls = {
        f"rec@{threshold:.1f}": 100.0 * sum(value >= threshold for value in ious) / n
        for threshold in REC_IOU_THRESHOLDS
    }
    return {
        "questions": len(ious),
        "miou": 100.0 * sum(ious) / n,
        "rec@IoU": sum(recalls.values()) / len(recalls),
        **recalls,
        "predictions_truncated_to_limit": over_limit,
    }


def _spans_from_report(report: dict[str, Any], spans: dict[str, dict[int, dict]], top_k: int):
    """Turn a pointwise eval report's per-candidate ranking into official intervals."""
    for result in report["results"]:
        example_id = result["example_id"]
        ranked = [
            int(row["candidate_index"])
            for row in sorted(
                result["ranking"],
                key=lambda row: (-float(row["score"]), int(row["candidate_index"])),
            )
        ]
        by_index = spans.get(example_id) or {}
        predicted = [
            [float(by_index[index]["start_s"]), float(by_index[index]["end_s"])]
            for index in ranked[:top_k]
            if index in by_index
        ]
        yield example_id, predicted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--pointwise-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=list(range(1, MAX_OFFICIAL_INTERVALS + 1)),
        help="Interval budgets to report; the protocol allows at most five.",
    )
    args = parser.parse_args(argv)

    spans: dict[str, dict[int, dict]] = {}
    gold: dict[str, list[list[float]]] = {}
    for line in args.pointwise_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metadata = (json.loads(line).get("metadata") or {})
        example_id = str(metadata.get("source_example_id") or "")
        entry = metadata.get("candidate_entry") or {}
        span = entry.get("time_span")
        if isinstance(span, dict):
            spans.setdefault(example_id, {})[int(metadata["candidate_index"])] = span
        supervision = metadata.get("process_supervision") or {}
        if example_id not in gold and supervision.get("clue_spans"):
            gold[example_id] = [
                [float(s["start_s"]), float(s["end_s"])] for s in supervision["clue_spans"]
            ]

    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": "video-skills/cg-bench-official-grounding-v1",
        "reference": "CG-Bench/CG-Bench run/utils.py calculate_intervals_iou",
        "eval_report": str(args.eval_report),
        "adapter_weight_sha256": report.get("adapter_weight_sha256"),
        "max_official_intervals": MAX_OFFICIAL_INTERVALS,
        "by_top_k": {},
    }
    for top_k in args.top_k:
        pairs = [
            (predicted, gold[example_id])
            for example_id, predicted in _spans_from_report(report, spans, top_k)
            if example_id in gold
        ]
        payload["by_top_k"][str(top_k)] = score_questions(pairs)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

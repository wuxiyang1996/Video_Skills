"""Score a reasoning chain against VRBench's annotated reasoning process.

VRBench gives every question 2-4 reasoning steps, most with a time span
("... [00:07:44->00:08:28]").  A system that cites clips (time spans) for its
answer can be scored on whether those citations land on the annotated steps:

  step recall   = fraction of annotated timed steps overlapped by any cited span
  citation prec = fraction of cited spans that overlap some annotated step
  mean best IoU = mean over steps of the best IoU with a cited span

This is the process-level credit that Video-Holmes cannot give and a single
answer call (no citations) cannot earn.  Rollouts come from
scripts/eval/measure_answer_chain.py --dump-rollouts (graph2: evidence_chain
clip_ranks -> time spans via `indices`; probe_observations carry spans too).

Usage: vrbench_process_score.py --eval-jsonl VRBench_eval.jsonl --rollouts A.rollouts.jsonl
       [--l1-index example_index.json] [--top-options 1]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.eval.measure_answer_chain import retrieval_catalog

_TS = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:->|-|~)\s*(\d{1,2}:\d{2}(?::\d{2})?)")
_T1 = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)")


def _secs(t: str) -> float:
    parts = [float(p) for p in t.split(":")]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]


def step_spans(reasoning_process: Any) -> list[dict[str, float]]:
    """Timed steps of one question, in step order; untimed steps are skipped."""
    out: list[dict[str, float]] = []
    items = reasoning_process.items() if isinstance(reasoning_process, dict) else enumerate(reasoning_process or [])
    for _, text in sorted(items, key=lambda kv: int(str(kv[0])) if str(kv[0]).isdigit() else 0):
        m = _TS.search(str(text))
        if m:
            a, b = _secs(m.group(1)), _secs(m.group(2))
            out.append({"start_s": min(a, b), "end_s": max(a, b)})
            continue
        m = _T1.search(str(text))
        if m:
            s = _secs(m.group(1))
            out.append({"start_s": s, "end_s": s + 1.0})
    return out


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    inter = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"])
    return inter / union if union > 0 else 0.0


def _hit(a: dict[str, float], b: dict[str, float]) -> bool:
    return a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]


def cited_spans(rollout: dict[str, Any], indices: list[int], catalog_spans: list[dict[str, float]],
                top_options: int = 1) -> list[dict[str, float]]:
    """Time spans the chain cites: clip_ranks of the top option(s), plus any probed spans."""
    spans: list[dict[str, float]] = []
    for item in (rollout.get("evidence_chain") or [])[:top_options]:
        for rank in item.get("clip_ranks") or []:
            pos = int(rank) - 1
            if 0 <= pos < len(indices) and 0 <= indices[pos] < len(catalog_spans):
                spans.append(catalog_spans[indices[pos]])
    for obs in rollout.get("probe_observations") or []:
        ts = obs.get("time_span") or {}
        if "start_s" in ts and "end_s" in ts:
            spans.append({"start_s": float(ts["start_s"]), "end_s": float(ts["end_s"])})
    return spans


def score_question(steps: list[dict[str, float]], cited: list[dict[str, float]]) -> dict[str, float]:
    if not steps:
        return {"step_recall": 0.0, "citation_precision": 0.0, "mean_best_iou": 0.0, "timed_steps": 0.0}
    recall = sum(any(_hit(c, s) for c in cited) for s in steps) / len(steps)
    precision = (sum(any(_hit(c, s) for s in steps) for c in cited) / len(cited)) if cited else 0.0
    best = [max((_iou(c, s) for c in cited), default=0.0) for s in steps]
    return {"step_recall": recall, "citation_precision": precision,
            "mean_best_iou": sum(best) / len(best), "timed_steps": float(len(steps))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-jsonl", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--l1-index", type=Path, required=True,
                    help="example_index.json mapping example_id -> {path} of the derived L1 example (for clip spans)")
    ap.add_argument("--top-options", type=int, default=1)
    args = ap.parse_args(argv)

    gold: dict[str, Any] = {}
    for line in args.eval_jsonl.open(encoding="utf-8"):
        row = json.loads(line)
        for key, qa in (row.get("mcq") or {}).items():
            gold[f"vrbench:{row['video_id']}:{key}"] = qa
    index = json.load(args.l1_index.open())

    totals: dict[str, float] = {"step_recall": 0.0, "citation_precision": 0.0, "mean_best_iou": 0.0}
    n = 0; n_cited = 0; correct = 0; n_timed = 0
    for line in args.rollouts.open(encoding="utf-8"):
        rec = json.loads(line)
        eid = rec["example_id"]
        if eid not in gold or eid not in index:
            continue
        example = json.load(open(index[eid]["path"]))
        # the same catalog accessor the answer chain ranks over (coarse+fine as
        # the pipeline exposes them); metadata.clip_schemas alone is only the
        # fine subset and mis-maps the cited ranks
        schemas, _ = retrieval_catalog(example)
        spans = [(s.get("time_span") or {}) if isinstance(s, dict) else {} for s in schemas]
        spans = [{"start_s": float(t.get("start_s") or 0), "end_s": float(t.get("end_s") or 0)} for t in spans]
        steps = step_spans(gold[eid].get("reasoning_process"))
        cited = cited_spans(rec["rollout"], rec.get("indices") or [], spans, args.top_options)
        sc = score_question(steps, cited)
        n += 1; n_cited += bool(cited)
        correct += str((rec["rollout"].get("final_answer") or {}).get("label")) == str(gold[eid].get("answer"))
        if steps:   # process metrics are averaged over questions that have timed steps
            n_timed += 1
            for k in totals:
                totals[k] += sc[k]
    if not n:
        print("no scorable questions"); return 1
    print(json.dumps({"questions": n, "with_citations": n_cited, "accuracy": round(100 * correct / n, 2),
                      "questions_with_timed_steps": n_timed,
                      **{k: round(100 * v / max(n_timed, 1), 2) for k, v in totals.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

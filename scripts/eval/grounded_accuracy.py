"""Reasoning-grounded accuracy on VRBench: a question counts only if the answer is right AND the
cited evidence lands on the annotated reasoning steps.

Answer accuracy alone rewards ungrounded guesses; process metrics alone reward
citations that lead nowhere.  This joins them per question:

  grounded(q) = correct(q) and step_recall(q) >= --min-step-recall   (default 0.5)

and reports it next to accuracy, for chains that cite clips structurally
(`--cite chain`: graph2 evidence_chain clip_ranks + probe spans) or in prose
(`--cite prose`: "clip 12", "clips 3-5" in a rationale's thinking).  A plain
answer call cites nothing and so scores 0 grounded by construction.  Per-question
indicators are written next to the rollouts (`.grounded.jsonl`) for paired tests.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.eval.measure_answer_chain import retrieval_catalog
from scripts.eval.vrbench_process_score import cited_spans, score_question, step_spans

_CLIP = re.compile(r"\bclips?\s+(\d{1,3})(?:\s*(?:-|–|to|and|,)\s*(\d{1,3}))?", re.I)


def ranks_in_prose(text: str) -> list[int]:
    """1-based clip ranks mentioned in prose: 'clip 12', 'clips 3-5', 'clips 4 and 7'."""
    ranks: list[int] = []
    for a, b in _CLIP.findall(text or ""):
        lo, hi = int(a), int(b) if b else int(a)
        if hi < lo or hi - lo > 20:
            hi = lo
        ranks.extend(range(lo, hi + 1))
    return sorted(set(ranks))


def prose_spans(rollout: dict[str, Any], indices: list[int], catalog_spans: list[dict[str, float]]) -> list[dict[str, float]]:
    out = []
    for k in ranks_in_prose(str(rollout.get("thinking") or rollout.get("rationale") or "")):
        if 0 < k <= len(indices) and 0 <= indices[k - 1] < len(catalog_spans):
            out.append(catalog_spans[indices[k - 1]])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-jsonl", type=Path, required=True,
                    help="VRBench_eval.jsonl, or CG-Bench cgbench.json with --gold cgbench")
    ap.add_argument("--gold", choices=["vrbench", "cgbench"], default="vrbench")
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--l1-index", type=Path, required=True)
    ap.add_argument("--cite", choices=["chain", "prose", "none"], default="chain")
    ap.add_argument("--min-step-recall", type=float, default=0.5)
    ap.add_argument("--top-options", type=int, default=1)
    args = ap.parse_args(argv)

    gold: dict[str, Any] = {}
    if args.gold == "cgbench":
        # CG-Bench: gold = clue_intervals [[start, end], ...]; the "steps" are the clue intervals
        for row in json.load(args.eval_jsonl.open(encoding="utf-8")):
            gold[f"cg_bench:{row['qid']}"] = {"answer": row.get("answer"),
                                             "clue_spans": [{"start_s": float(a), "end_s": float(b)}
                                                            for a, b in (row.get("clue_intervals") or []) if b is not None]}
    else:
        for line in args.eval_jsonl.open(encoding="utf-8"):
            row = json.loads(line)
            for key, qa in (row.get("mcq") or {}).items():
                gold[f"vrbench:{row['video_id']}:{key}"] = qa
    index = json.load(args.l1_index.open())
    out_path = args.rollouts.with_suffix(".grounded.jsonl")
    n = n_timed = correct = grounded = 0
    with out_path.open("w", encoding="utf-8") as out:
        for line in args.rollouts.open(encoding="utf-8"):
            rec = json.loads(line)
            eid = rec["example_id"]
            if eid not in gold or eid not in index:
                continue
            example = json.load(open(index[eid]["path"]))
            schemas, _ = retrieval_catalog(example)
            spans = [{"start_s": float((s.get("time_span") or {}).get("start_s") or 0),
                      "end_s": float((s.get("time_span") or {}).get("end_s") or 0)} for s in schemas]
            steps = gold[eid]["clue_spans"] if args.gold == "cgbench" else step_spans(gold[eid].get("reasoning_process"))
            indices = rec.get("indices") or []
            if args.cite == "chain":
                cited = cited_spans(rec["rollout"], indices, spans, args.top_options)
            elif args.cite == "prose":
                cited = prose_spans(rec["rollout"], indices, spans)
            else:
                cited = []
            sc = score_question(steps, cited)
            final = rec["rollout"].get("final_answer") or {}
            if rec.get("gold_label") is not None:      # dumped rollouts carry the option letter the row was scored against
                is_correct = str(final.get("label")) == str(rec["gold_label"])
            else:
                is_correct = str(final.get("label")) == str(gold[eid].get("answer"))
            is_grounded = bool(steps) and is_correct and sc["step_recall"] >= args.min_step_recall
            n += 1; n_timed += bool(steps); correct += is_correct; grounded += is_grounded
            out.write(json.dumps({"example_id": eid, "correct": is_correct, "timed": bool(steps),
                                  "step_recall": sc["step_recall"], "citation_precision": sc["citation_precision"],
                                  "grounded": is_grounded, "cited": len(cited)}) + "\n")
    print(json.dumps({"rollouts": str(args.rollouts.name), "cite": args.cite, "questions": n, "timed": n_timed,
                      "accuracy": round(100 * correct / max(n, 1), 2),
                      "grounded_accuracy_all": round(100 * grounded / max(n, 1), 2),
                      "grounded_accuracy_timed": round(100 * grounded / max(n_timed, 1), 2),
                      "per_question": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

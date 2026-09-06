"""Build reader SFT data from teacher rationales, kept only when verifiably good.

Inputs: an example index (id -> path) of per-question L1 examples with catalogs,
and dumped rollouts from `measure_answer_chain --conditions direct --rationale
--dump-rollouts` (teacher = the 235B).  A row is kept when the teacher's label is
the gold label and, when annotated evidence exists, its citation precision is at
least --min-precision.  Output JSONL rows: {"example_id", "messages", "completion",
"reward"} with the exact evaluation prompt (trainer.reader.prompting).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.reader.prompting import reader_messages, reader_target
from trainer.reader.rewards import reader_reward


def apply_option_perm(example: dict[str, Any], perm: list[str]) -> None:
    """Replay a recorded permutation: position k gets the option that originally had label perm[k]."""
    question = example.get("question") or {}
    by_label = {str(o.get("label")): o for o in question.get("options") or []}
    labels = sorted(by_label)
    gold = str((question.get("answer") or {}).get("label") or "")
    question["options"] = [{**by_label[src], "label": labels[k]} for k, src in enumerate(perm) if src in by_label]
    for k, src in enumerate(perm):
        if src == gold:
            question["answer"] = {**(question.get("answer") or {}), "label": labels[k]}
    question["option_perm"] = list(perm)


def build_rows(index: dict[str, Any], rollouts: list[dict[str, Any]], *, min_precision: float = 0.0,
               max_reasoning_chars: int = 2500) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"rollouts": 0, "no_example": 0, "wrong": 0, "low_precision": 0, "empty_reasoning": 0, "kept": 0}
    for rec in rollouts:
        stats["rollouts"] += 1
        eid = rec.get("example_id")
        meta = index.get(eid)
        if not meta:
            stats["no_example"] += 1
            continue
        example = json.loads(Path(meta["path"]).read_text(encoding="utf-8"))
        if rec.get("option_perm"):
            apply_option_perm(example, rec["option_perm"])
        gold = rec.get("gold_label") or ((example.get("question") or {}).get("answer") or {}).get("label")
        r = rec.get("rollout") or {}
        label = (r.get("final_answer") or {}).get("label")
        reasoning = str(r.get("thinking") or "").strip()[:max_reasoning_chars]
        if not reasoning:
            stats["empty_reasoning"] += 1
            continue
        completion = reader_target(reasoning, str(label or ""))
        score = reader_reward(example, completion, str(gold))
        if not score["correct"]:
            stats["wrong"] += 1
            continue
        if score["citation_precision"] is not None and score["citation_precision"] < min_precision:
            stats["low_precision"] += 1
            continue
        rows.append({"example_id": eid, "messages": reader_messages(example, rationale=True), "completion": completion,
                     "reward": score["reward"], "citation_precision": score["citation_precision"]})
        stats["kept"] += 1
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-index", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-precision", type=float, default=0.0)
    args = ap.parse_args(argv)
    index = json.loads(args.example_index.read_text(encoding="utf-8"))
    rollouts = [json.loads(l) for p in args.rollouts for l in p.open(encoding="utf-8") if l.strip()]
    rows, stats = build_rows(index, rollouts, min_precision=args.min_precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

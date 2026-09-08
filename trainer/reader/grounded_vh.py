"""Grounded accuracy for reader runs on Video-Holmes: right answer AND citation precision >= threshold.

Reads a measure_answer_chain rollouts file (rationale runs: `thinking` + `final_answer`), scores each
rollout with trainer.reader.rewards.reader_reward against the annotated inference shots (supervision
index), and prints accuracy, grounded accuracy and mean citation precision; with --pair, a paired
bootstrap against a second rollouts file on the same questions.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from trainer.reader.prompting import reader_target
from trainer.reader.rewards import reader_reward


def score_file(path: Path, index: dict) -> dict[str, dict]:
    out = {}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        eid = rec["example_id"]
        if eid not in index:
            continue
        example = json.loads(Path(index[eid]["path"]).read_text(encoding="utf-8"))
        r = rec.get("rollout") or {}
        completion = reader_target(str(r.get("thinking") or ""), str((r.get("final_answer") or {}).get("label") or ""))
        s = reader_reward(example, completion, str(rec.get("gold_label") or ""))
        out[eid] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-index", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--pair", type=Path, default=None, help="second rollouts file (B); prints B − A paired CIs")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    index = json.loads(args.example_index.read_text())
    a = score_file(args.rollouts, index)

    def metrics(d):
        ids = sorted(d); n = len(ids)
        acc = sum(d[i]["correct"] for i in ids) / n
        gr = sum(1 for i in ids if d[i]["correct"] and (d[i]["citation_precision"] or 0) >= args.threshold) / n
        prec = [d[i]["citation_precision"] for i in ids if d[i]["citation_precision"] is not None]
        return {"n": n, "accuracy": round(100 * acc, 1), "grounded_accuracy": round(100 * gr, 1),
                "mean_citation_precision": round(100 * sum(prec) / max(len(prec), 1), 1), "with_gold": len(prec)}
    print(json.dumps({"A": str(args.rollouts.name), **metrics(a)}))
    if args.pair:
        b = score_file(args.pair, index)
        ids = sorted(set(a) & set(b)); m = len(ids); rng = random.Random(2026)
        print(json.dumps({"B": str(args.pair.name), **metrics({i: b[i] for i in ids})}))
        for name, f in [("accuracy", lambda s: s["correct"]), ("grounded_accuracy", lambda s: s["correct"] and (s["citation_precision"] or 0) >= args.threshold)]:
            d = [int(bool(f(b[i]))) - int(bool(f(a[i]))) for i in ids]
            bs = sorted(100 * sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(5000))
            print(f"  {name:18s} B−A {100*sum(d)/m:+.1f} [{bs[125]:+.1f}, {bs[4875]:+.1f}] (n={m})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

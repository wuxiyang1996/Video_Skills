#!/usr/bin/env python3
"""Measure the Video-Holmes answer chain on a reserved slice of the heldout set.

End-to-end accuracy on these benchmarks is (completion rate) x (accuracy when
answered), and an abstention scores as wrong.  On the cached GRPO rollouts the
system abstained 60.6% of the time while being right on 76.4% of what it did
answer -- but that cache is a biased sample, selected for reward variance over 25
examples.  This re-measures on heldout examples the policy never trained on.

Two retrieval conditions separate the answer chain from the retriever:
``model`` uses the ranking a checkpoint produced, ``oracle`` uses the candidates
that actually overlap the gold spans.  If accuracy barely moves between them,
retrieval quality is not what limits answers.

Only a seeded subsample is consumed so the rest of the heldout stays unread.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.l2_dataset_rewards import temporal_hit
from trainer.build_l2_dataset_opd import load_dataset_reward_supervision, supervision_key
from trainer.grpo.train_l2_terminal_on_policy import filter_example_for_retrieval, retrieval_catalog


def oracle_indices(example: dict[str, Any], supervision: dict[str, Any], top_k: int) -> list[int]:
    """Candidates that actually overlap a gold span, best-effort up to top_k."""
    schemas, _ = retrieval_catalog(example)
    gold = (supervision or {}).get("segment_spans") or []
    hits = [
        index
        for index, schema in enumerate(schemas)
        if isinstance(schema, dict)
        and isinstance(schema.get("time_span"), dict)
        and any(temporal_hit(schema["time_span"], span) for span in gold)
    ]
    return hits[:top_k]


def model_indices(example_id: str, rankings: dict[str, list[int]], top_k: int) -> list[int]:
    return (rankings.get(example_id) or [])[:top_k]


def score(rollout: dict[str, Any], gold_label: str) -> dict[str, Any]:
    label = (rollout.get("final_answer") or {}).get("label")
    committed = bool(label)
    correct = committed and str(label).strip().upper() == str(gold_label).strip().upper()
    return {
        "committed": committed,
        "correct": bool(correct),
        "acceptance_status": rollout.get("acceptance_status"),
        "failure_reasons": list(rollout.get("failure_reasons") or [])[:3],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-glob", required=True)
    parser.add_argument("--eval-report", type=Path, help="ranking source for the 'model' condition")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b:free")
    parser.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from trainer.grpo.train_l2_terminal_on_policy import load_openrouter_api_key

    paths = [Path(p) for p in sorted(glob.glob(args.l1_glob, recursive=True))]
    examples = {
        str(e.get("example_id") or ""): e
        for e in load_frozen_l1_examples(paths)
        if e.get("example_id")
    }
    chosen = sorted(examples)
    random.Random(args.seed).shuffle(chosen)
    chosen = chosen[: max(1, args.sample)]

    # Hidden supervision is loaded evaluator-side; it never enters a prompt.
    supervision_index = load_dataset_reward_supervision(args.dataset_root)

    rankings: dict[str, list[int]] = {}
    if args.eval_report and args.eval_report.exists():
        report = json.loads(args.eval_report.read_text(encoding="utf-8"))
        for result in report.get("results") or []:
            rankings[str(result["example_id"])] = [
                int(row["candidate_index"])
                for row in sorted(
                    result["ranking"], key=lambda r: (-float(r["score"]), int(r["candidate_index"]))
                )
            ]

    client = OpenRouterClient(
        model=args.planner_model,
        api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
        max_tokens=1800,
        temperature=0.0,
        reasoning={"effort": "minimal", "exclude": True},
        timeout_s=args.timeout_s,
    )

    def run(example_id: str, condition: str) -> dict[str, Any] | None:
        example = examples[example_id]
        question = example.get("question") or {}
        gold_label = (question.get("answer") or {}).get("label")
        if not gold_label:
            return None
        supervision = supervision_index.get(supervision_key(example)) or {}
        indices = (
            oracle_indices(example, supervision, args.top_k)
            if condition == "oracle"
            else model_indices(example_id, rankings, args.top_k)
        )
        if not indices:
            return None
        isolated, graph = filter_example_for_retrieval(example, indices)
        try:
            rollout = build_llm_reasoning_rollout(isolated, graph, client=client, motif_enabled=False)
        except Exception as exc:  # a transport failure is not an abstention
            return {"example_id": example_id, "condition": condition, "error": type(exc).__name__}
        return {"example_id": example_id, "condition": condition, **score(rollout, gold_label)}

    rows: list[dict[str, Any]] = []
    jobs = [(e, c) for e in chosen for c in ("model", "oracle")]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, e, c): (e, c) for e, c in jobs}
        for done in as_completed(futures):
            row = done.result()
            if row:
                rows.append(row)
            print(f"[{len(rows)}/{len(jobs)}]", flush=True)

    summary: dict[str, Any] = {}
    for condition in ("model", "oracle"):
        subset = [r for r in rows if r.get("condition") == condition and "error" not in r]
        n = max(1, len(subset))
        committed = sum(bool(r["committed"]) for r in subset)
        correct = sum(bool(r["correct"]) for r in subset)
        reasons: collections.Counter[str] = collections.Counter()
        for r in subset:
            if not r["committed"]:
                reasons.update(str(x)[:60] for x in r["failure_reasons"])
        summary[condition] = {
            "examples": len(subset),
            "errors": sum(1 for r in rows if r.get("condition") == condition and "error" in r),
            "completion_rate": 100.0 * committed / n,
            "accuracy_end_to_end": 100.0 * correct / n,
            "accuracy_when_answered": (100.0 * correct / committed) if committed else 0.0,
            "top_abstention_reasons": reasons.most_common(3),
        }
    payload = {
        "schema_version": "video-skills/answer-chain-heldout-v1",
        "sample": len(chosen),
        "seed": args.seed,
        "top_k": args.top_k,
        "planner_model": args.planner_model,
        "note": "seeded subsample of heldout_test; the rest stays unread",
        "conditions": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Where Video-Holmes / CG-Bench answers are lost.

The benchmarks score QA accuracy, and an abstention counts as wrong, so the
end-to-end number is (completion rate) x (accuracy when answered).  Splitting it
that way says which half to work on.  On the cached GRPO rollouts the two halves
are very unequal, and the losing half is not the one the retrieval work targets.

The rollout cache is a biased sample -- those examples were selected for reward
variance during training and are not a test split -- so treat the rates as
diagnostic, never as a benchmark result.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
from typing import Any


def gold_answers(dataset: str, dataset_root: Path) -> dict[str, str]:
    from dataset_clip_wrapper.adapters import get_adapter

    answers: dict[str, str] = {}
    for split in ("train", "test"):
        adapter = get_adapter(dataset, dataset_root)
        adapter.split = split
        try:
            items = list(adapter.iter_items())
        except Exception:
            continue
        for item in items:
            question = getattr(item, "question", None) or {}
            example_id = getattr(item, "example_id", "")
            label = (question.get("answer") or {}).get("label")
            if example_id and label:
                answers[example_id] = str(label)
    return answers


def summarise(rollouts: list[dict[str, Any]], answers: dict[str, str]) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    examples: set[str] = set()
    for rollout in rollouts:
        example_id = str(rollout.get("example_id") or "")
        gold = answers.get(example_id)
        if not gold:
            counts["unmatched"] += 1
            continue
        examples.add(example_id)
        counts["total"] += 1
        label = (rollout.get("final_answer") or {}).get("label")
        if not label:
            counts["abstained"] += 1
            for reason in rollout.get("failure_reasons") or []:
                reasons[str(reason)[:60]] += 1
            continue
        counts["correct" if str(label).strip().upper() == gold.strip().upper() else "wrong"] += 1
    total = max(1, counts["total"])
    answered = counts["correct"] + counts["wrong"]
    return {
        "rollouts": counts["total"],
        "distinct_examples": len(examples),
        "completion_rate": 100.0 * answered / total,
        "abstention_rate": 100.0 * counts["abstained"] / total,
        "accuracy_end_to_end": 100.0 * counts["correct"] / total,
        "accuracy_when_answered": (100.0 * counts["correct"] / answered) if answered else 0.0,
        "top_abstention_reasons": reasons.most_common(5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-glob", required=True, help="glob over executor cache JSON files")
    parser.add_argument("--dataset", default="video_holmes")
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    rollouts = []
    for path in glob.glob(args.cache_glob):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("example_id") or "").startswith(args.dataset):
            rollouts.append(payload)
    report = {
        "schema_version": "video-skills/answer-chain-completion-v1",
        "dataset": args.dataset,
        "sample_is_biased": "rollout cache selected for reward variance; not a test split",
        **summarise(rollouts, gold_answers(args.dataset, args.dataset_root)),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

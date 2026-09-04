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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from atomic_skills.skill_executor import SkillExecutor
from atomic_skills.skill_model_client import SkillModelClient
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient
from trainer.grpo.live_rollout import _grpo_skill_backend_config, _GRPO_LLM_SKILLS
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.l2_dataset_rewards import temporal_hit
from dataset_clip_wrapper.training.lexical_retrieval_baseline import BM25  # noqa: F401  (shared tokeniser deps)
from trainer.build_l2_dataset_opd import load_dataset_reward_supervision, supervision_key
from trainer.grpo.l2_dataset_rewards import _text as clip_schema_text
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


DIRECT_SYSTEM = (
    "You answer multiple-choice questions about a video using only the clip "
    "descriptions provided. Reply with JSON only."
)


def direct_answer(
    client: OpenRouterClient,
    example: dict[str, Any],
    indices: list[int],
) -> dict[str, Any]:
    """Ask the same model the same question over the same clips, with no skill graph.

    This is the control for the atomic-skill decomposition: it holds the model,
    the evidence and the budget fixed and removes only the plan-and-execute
    structure, so any difference is attributable to the decomposition itself.
    """
    schemas, _ = retrieval_catalog(example)
    question = example.get("question") or {}
    clips = []
    for rank, index in enumerate(indices, start=1):
        if not (0 <= index < len(schemas)):
            continue
        schema = schemas[index] if isinstance(schemas[index], dict) else {}
        span = schema.get("time_span") or {}
        clips.append({
            "rank": rank,
            "time_span": span,
            "description": clip_schema_text(schema)[:1200],
        })
    payload = {
        "question": question.get("question_text") or "",
        "options": [
            {"label": o.get("label"), "text": o.get("text")}
            for o in question.get("options") or []
        ],
        "clips": clips,
        "answer_format": 'reply exactly {"label": "<option letter>"}',
    }
    text = client.chat([
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    label = None
    try:
        label = (json.loads(re.search(r"\{.*\}", text, re.S).group(0)) or {}).get("label")
    except Exception:
        match = re.search(r"\b([A-H])\b", text or "")
        label = match.group(1) if match else None
    return {"final_answer": {"label": label}, "failure_reasons": [] if label else ["no_label_parsed"]}


def degradation(rollout: dict[str, Any]) -> dict[str, Any]:
    """Markers for the two silent fallbacks: planner -> deterministic, skill -> rule.

    Both leave ok=True and errors=0.  The planner fallback shows only as
    llm_plan.fallback_reason; an LLM-executed skill shows only as
    metadata.answer_step_diagnostics[<step_id>].backend == "llm" (rollout nodes
    carry no backend field), and a skill that fell back to its rule carries a
    `*_fallback_to_rule` message there.  A rollout whose planner fell back is not
    a measurement of the system and is excluded from accuracy.
    """
    meta = rollout.get("metadata") or {}
    llm_plan = meta.get("llm_plan") or {}
    planner_fell_back = bool(llm_plan.get("fallback_reason") or llm_plan.get("planner_error"))
    diagnostics = meta.get("answer_step_diagnostics") or {}

    def step_backend(step: dict[str, Any]) -> str | None:
        if step.get("backend"):
            return str(step["backend"])
        for key in ("scored_hypothesis", "best_hypothesis", "verified_claim"):
            nested = step.get(key)
            if isinstance(nested, dict) and nested.get("backend"):
                return str(nested["backend"])
        return None

    llm_nodes = 0
    critical_on_llm = 0
    rule_fallbacks = 0
    for node in rollout.get("nodes") or []:
        step = diagnostics.get(str(node.get("step_id") or ""))
        if not isinstance(step, dict):
            continue
        backend = step_backend(step)
        messages = " ".join(str(m) for m in (step.get("messages") or []))
        if backend == "llm":
            llm_nodes += 1
            if node.get("skill_id") in _GRPO_LLM_SKILLS:
                critical_on_llm += 1
        if "fallback_to_rule" in messages:
            rule_fallbacks += 1
    return {
        "planner_fell_back": planner_fell_back,
        "llm_skill_nodes": llm_nodes,
        "critical_skills_on_llm": critical_on_llm,
        "rule_fallbacks": rule_fallbacks,
    }


def score(rollout: dict[str, Any], gold_label: str) -> dict[str, Any]:
    label = (rollout.get("final_answer") or {}).get("label")
    committed = bool(label)
    correct = committed and str(label).strip().upper() == str(gold_label).strip().upper()
    return {
        "committed": committed,
        "correct": bool(correct),
        "acceptance_status": rollout.get("acceptance_status"),
        "failure_reasons": list(rollout.get("failure_reasons") or [])[:3],
        **degradation(rollout),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-glob", required=True)
    parser.add_argument("--eval-report", type=Path, help="ranking source for the 'model' condition")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b",
                        help="LLM that executes answer-critical skills; mirrors the GRPO trainer's --skill-model.")
    parser.add_argument("--skill-timeout-s", type=int, default=90)
    parser.add_argument("--no-skill-executor", action="store_true",
                        help="Run skills as rules only (the degraded mode this script used to run in silently).")
    parser.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["model", "oracle", "direct"],
        help="'model'/'oracle' run the skill graph; 'direct' is the no-decomposition control.",
    )
    parser.add_argument(
        "--always-commit-mcq",
        action="store_true",
        help="Commit the best hypothesis even when verification fails (abstention scores as wrong on MCQ).",
    )
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

    # The trainer executes answer-critical skills with a separate LLM; without it
    # every skill degrades to a lexical rule and the chain scores below chance.
    executor = None
    if not args.no_skill_executor:
        executor = SkillExecutor(
            llm_client=SkillModelClient(
                model=args.skill_model,
                api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
                max_tokens=768,
                temperature=0.0,
                timeout_s=args.skill_timeout_s,
            ),
            vlm_client=None,
            config=_grpo_skill_backend_config(),
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
        try:
            if condition == "direct":
                rollout = direct_answer(client, example, indices)
            else:
                isolated, graph = filter_example_for_retrieval(example, indices)
                rollout = build_llm_reasoning_rollout(
                    isolated, graph, client=client, skill_executor=executor, motif_enabled=False,
                    commit_policy={"always_commit_mcq": True} if args.always_commit_mcq else None,
                )
        except Exception as exc:  # a transport failure is not an abstention
            return {"example_id": example_id, "condition": condition, "error": type(exc).__name__}
        return {"example_id": example_id, "condition": condition, **score(rollout, gold_label)}

    rows: list[dict[str, Any]] = []
    jobs = [(e, c) for e in chosen for c in args.conditions]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, e, c): (e, c) for e, c in jobs}
        for done in as_completed(futures):
            row = done.result()
            if row:
                rows.append(row)
            print(f"[{len(rows)}/{len(jobs)}]", flush=True)

    summary: dict[str, Any] = {}
    for condition in args.conditions:
        all_rows = [r for r in rows if r.get("condition") == condition and "error" not in r]
        degraded = [r for r in all_rows if r.get("planner_fell_back")]
        subset = [r for r in all_rows if not r.get("planner_fell_back")]
        n = max(1, len(subset))
        skill_rows = [r for r in subset if condition != "direct"]
        no_llm_skills = sum(1 for r in skill_rows if r.get("critical_skills_on_llm", 0) == 0)
        committed = sum(bool(r["committed"]) for r in subset)
        correct = sum(bool(r["correct"]) for r in subset)
        reasons: collections.Counter[str] = collections.Counter()
        for r in subset:
            if not r["committed"]:
                reasons.update(str(x)[:60] for x in r["failure_reasons"])
        summary[condition] = {
            "examples": len(subset),
            "excluded_planner_fallback": len(degraded),
            "rows_with_no_llm_critical_skill": no_llm_skills,
            "rows_with_rule_fallbacks": sum(1 for r in skill_rows if r.get("rule_fallbacks", 0) > 0),
            "errors": sum(1 for r in rows if r.get("condition") == condition and "error" in r),
            "completion_rate": 100.0 * committed / n,
            "accuracy_end_to_end": 100.0 * correct / n,
            "accuracy_when_answered": (100.0 * correct / committed) if committed else 0.0,
            "top_abstention_reasons": reasons.most_common(3),
        }
    invalid = [
        c for c, v in summary.items()
        if v["excluded_planner_fallback"] or (c != "direct" and not args.no_skill_executor and v["rows_with_no_llm_critical_skill"] > 0)
    ]
    if invalid:
        print(f"WARNING: degraded execution detected in conditions {invalid}; see excluded_planner_fallback / rows_with_no_llm_critical_skill", flush=True)
    payload = {
        "schema_version": "video-skills/answer-chain-heldout-v2",
        "valid": not invalid,
        "skill_model": None if args.no_skill_executor else args.skill_model,
        "sample": len(chosen),
        "seed": args.seed,
        "top_k": args.top_k,
        "planner_model": args.planner_model,
        "always_commit_mcq": bool(args.always_commit_mcq),
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

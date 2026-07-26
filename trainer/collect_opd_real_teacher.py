#!/usr/bin/env python3
"""P1: Collect OPD distill rows with real OpenRouter teacher + order shuffle.

Tries soft letter-logprob teacher first; if shuffle calibration gates fail
(or ``--teacher-mode ranking``), falls back to multi-order ranking →
Borda / Bradley–Terry aggregation (default ranking model: DeepSeek V4 Pro).
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import (  # noqa: E402
    build_llm_reasoning_rollout,
)
from dataset_clip_wrapper.perception.openrouter_client import (  # noqa: E402
    OpenRouterClient,
    load_openrouter_api_key,
)
from trainer.candidate_action_builder import (  # noqa: E402
    CandidateActionSet,
    build_l2_candidate_actions,
    gate_candidate_set,
)
from trainer.closed_loop_harness import ClosedLoopHarness, HarnessState, load_frozen_l1_examples  # noqa: E402
from trainer.exact_request_cache import ExactRequestCache  # noqa: E402
from trainer.opd_action_distill_adapter import OpdDistillRow, save_opd_rows  # noqa: E402
from trainer.posttraining_manifest import (  # noqa: E402
    build_posttraining_manifest,
    save_posttraining_manifest,
)
from trainer.reward import REWARD_SPEC_VERSION  # noqa: E402
from trainer.split_filter import assert_role_exclusive, filter_examples_by_role, load_split_manifest  # noqa: E402
from trainer.teacher_action_query import (  # noqa: E402
    TeacherActionDistribution,
    make_openrouter_letter_teacher,
    make_openrouter_ranking_teacher,
    order_shuffle_stability,
    query_teacher_averaged,
    query_teacher_ranking_averaged,
    soft_calibration_gates,
    soft_calibration_passed,
)
from trainer.train_opd_kl import run_opd_smoke  # noqa: E402

PreparedState = tuple[HarnessState, CandidateActionSet, dict[str, Any]]
TeacherQueryFn = Callable[
    [CandidateActionSet, dict[str, Any]],
    tuple[dict[str, float], list[TeacherActionDistribution]],
]


def _distill_prepared(
    prepared: list[PreparedState],
    *,
    query_fn: TeacherQueryFn,
    seeds: list[int],
    student_checkpoint: str,
    teacher_mode_label: str,
) -> tuple[list[OpdDistillRow], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[OpdDistillRow] = []
    shuffle_rows: list[dict[str, Any]] = []
    teacher_errors: list[dict[str, Any]] = []

    for state, action_set, precheck in prepared:
        try:
            avg_probs, dists = query_fn(action_set, state.to_dict())
        except Exception as exc:
            teacher_errors.append(
                {
                    "state_id": state.state_id,
                    "error": str(exc),
                    "teacher_mode": teacher_mode_label,
                }
            )
            print(f"  teacher_error[{teacher_mode_label}]: {exc}", flush=True)
            continue
        if len(dists) < 2:
            teacher_errors.append(
                {
                    "state_id": state.state_id,
                    "error": "need >=2 order seeds for shuffle stability",
                    "teacher_mode": teacher_mode_label,
                }
            )
            continue

        dist_a, dist_b = dists[0], dists[1]
        stab = order_shuffle_stability(dist_a.action_probs, dist_b.action_probs)
        top_ids = [max(d.action_probs, key=d.action_probs.get) for d in dists if d.action_probs]
        majority_top = max(set(top_ids), key=top_ids.count) if top_ids else None
        majority_frac = (top_ids.count(majority_top) / len(top_ids)) if top_ids else 0.0
        avg_top = max(avg_probs, key=avg_probs.get) if avg_probs else None
        shuffle_rows.append(
            {
                "state_id": state.state_id,
                "example_id": state.example_id,
                "teacher_mode": teacher_mode_label,
                "stability": stab,
                "majority_top": majority_top,
                "majority_frac": majority_frac,
                "avg_top": avg_top,
                "avg_probs": avg_probs,
                "seed_tops": top_ids,
                "top_a": dist_a.action_probs,
                "top_b": dist_b.action_probs,
                "letter_a": dist_a.letter_to_action_id,
                "letter_b": dist_b.letter_to_action_id,
                "ranked_a": list(dist_a.ranked_action_ids),
                "ranked_b": list(dist_b.ranked_action_ids),
            }
        )
        print(
            f"  [{teacher_mode_label}] top1_match={stab.get('top1_match')} "
            f"l1={stab.get('l1')} majority_frac={majority_frac:.2f} "
            f"avg_top={avg_top} cache_hit_a={dist_a.cache_hit}",
            flush=True,
        )
        teacher_for_row = dist_a
        teacher_for_row.action_probs = dict(avg_probs)
        teacher_for_row.probs = {
            letter: float(avg_probs.get(aid, 0.0))
            for letter, aid in dist_a.letter_to_action_id.items()
        }
        teacher_for_row.teacher_mode = teacher_mode_label
        rows.append(
            OpdDistillRow.from_parts(
                state=state,
                action_set=action_set,
                teacher=teacher_for_row,
                precheck={
                    **precheck,
                    "order_shuffle": stab,
                    "majority_frac": majority_frac,
                    "avg_top": avg_top,
                    "order_seeds": seeds,
                    "teacher_mode": teacher_mode_label,
                },
                student_checkpoint=student_checkpoint,
            )
        )
    return rows, shuffle_rows, teacher_errors


def _summarize_shuffle(shuffle_rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    top1_rate = (
        sum(1 for r in shuffle_rows if (r.get("stability") or {}).get("top1_match"))
        / max(len(shuffle_rows), 1)
    )
    mean_l1 = (
        sum(float((r.get("stability") or {}).get("l1") or 0.0) for r in shuffle_rows)
        / max(len(shuffle_rows), 1)
    )
    mean_majority = (
        sum(float(r.get("majority_frac") or 0.0) for r in shuffle_rows)
        / max(len(shuffle_rows), 1)
    )
    return {
        "n": len(shuffle_rows),
        "top1_match_rate": top1_rate,
        "mean_l1": mean_l1,
        "mean_majority_frac": mean_majority,
        "seeds": seeds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", required=True)
    parser.add_argument("--motif-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-manifest",
        default="",
        help="If set, keep only opd_pool examples from the split manifest",
    )
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--teacher-model", default="openai/gpt-4.1-mini")
    parser.add_argument(
        "--ranking-model",
        default="deepseek/deepseek-v4-pro",
        help="OpenRouter model for structured ranking fallback",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("soft", "ranking", "auto"),
        default="auto",
        help="soft=letter-logprob only; ranking=Borda/BT only; auto=soft then ranking on gate fail",
    )
    parser.add_argument(
        "--ranking-method",
        choices=("borda", "bt"),
        default="borda",
        help="Aggregation for multi-order rankings (borda or bradley-terry)",
    )
    parser.add_argument(
        "--rank-temperature",
        type=float,
        default=1.0,
        help="Fixed temperature for rank-score → probability softmax",
    )
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--shuffle-seeds", default="7,99,13,42")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in sorted(glob(args.frozen_l1_glob, recursive=True))]
    if not paths and "/**/" in args.frozen_l1_glob:
        root_s, _, suffix = args.frozen_l1_glob.partition("/**/")
        paths = sorted(Path(root_s).rglob(suffix))
    if not paths:
        raise SystemExit(f"No frozen L1 matched: {args.frozen_l1_glob}")
    examples = load_frozen_l1_examples(paths)
    if args.split_manifest:
        manifest = load_split_manifest(args.split_manifest)
        examples = filter_examples_by_role(examples, manifest=manifest, role="opd_pool", strict=False)
        if not examples:
            raise SystemExit("No opd_pool examples after split filter")
        assert_role_exclusive(examples, manifest=manifest, allowed_roles=("opd_pool",))
    examples = examples[: int(args.limit)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_openrouter_api_key(keys_py_path=args.keys_py)
    client = OpenRouterClient(
        model=args.planner_model,
        api_key=api_key,
        max_tokens=1800,
        reasoning={"effort": "minimal", "exclude": True},
        timeout_s=180,
    )
    soft_teacher_fn = make_openrouter_letter_teacher(api_key=api_key, model=args.teacher_model)
    ranking_teacher_fn = make_openrouter_ranking_teacher(
        api_key=api_key, model=args.ranking_model
    )
    seeds = [int(x) for x in str(args.shuffle_seeds).split(",") if str(x).strip()]
    if len(seeds) < 2:
        seeds = [7, 99]
    ranking_mode_label = f"ranking_{args.ranking_method}"

    def rollout_fn(example: dict, clue: dict) -> dict:
        meta = dict(example.get("metadata") or {})
        meta["motif_enabled"] = True
        meta["motif_bank_path"] = args.motif_bank
        example = {**example, "metadata": meta}
        return build_llm_reasoning_rollout(
            example,
            clue,
            client=client,
            skill_executor=None,
            motif_enabled=True,
            motif_bank_path=args.motif_bank,
        )

    harness = ClosedLoopHarness(rollout_fn=rollout_fn, motif_enabled=True)
    soft_cache = ExactRequestCache(
        out_dir / "teacher_cache_soft.json",
        {"teacher": args.teacher_model, "mode": "soft_logprob", "v": 1},
    )
    ranking_cache = ExactRequestCache(
        out_dir / "teacher_cache_ranking.json",
        {
            "teacher": args.ranking_model,
            "mode": ranking_mode_label,
            "temperature": float(args.rank_temperature),
            "v": 1,
        },
    )

    prepared: list[PreparedState] = []
    gate_failures: list[dict] = []

    for i, example in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] {example.get('example_id')}", flush=True)
        state = harness.run_example(example)
        print(
            f"  motif={state.motif_online.get('selected_motif_id')} "
            f"expand={state.motif_online.get('expansion_valid')}",
            flush=True,
        )
        oracle = state.student_action or {
            "schema_version": "video-skills/l2-specialist-action-v0.1",
            "tool_name": "choose_best_coarse_candidate",
            "arguments": {"coarse_index": 1},
        }
        action_set = build_l2_candidate_actions(
            state_id=state.state_id,
            student_action=state.student_action,
            oracle_action=oracle,
            coarse_indices=[0, 1, 2],
        )
        precheck = gate_candidate_set(action_set)
        if not precheck["passed"]:
            gate_failures.append({"state_id": state.state_id, "precheck": precheck})
            continue
        prepared.append((state, action_set, precheck))

    def soft_query(
        action_set: CandidateActionSet, state_dict: dict[str, Any]
    ) -> tuple[dict[str, float], list[TeacherActionDistribution]]:
        return query_teacher_averaged(
            action_set,
            state=state_dict,
            teacher_fn=soft_teacher_fn,
            order_seeds=seeds,
            cache=soft_cache,
        )

    def ranking_query(
        action_set: CandidateActionSet, state_dict: dict[str, Any]
    ) -> tuple[dict[str, float], list[TeacherActionDistribution]]:
        return query_teacher_ranking_averaged(
            action_set,
            state=state_dict,
            teacher_fn=ranking_teacher_fn,
            order_seeds=seeds,
            cache=ranking_cache,
            method=args.ranking_method,
            temperature=float(args.rank_temperature),
        )

    soft_order_shuffle: dict[str, Any] | None = None
    soft_gates: dict[str, bool] | None = None
    soft_teacher_errors: list[dict[str, Any]] = []
    teacher_mode_used = ranking_mode_label
    fallback_reason: str | None = None

    rows: list[OpdDistillRow] = []
    shuffle_rows: list[dict[str, Any]] = []
    teacher_errors: list[dict[str, Any]] = []

    if args.teacher_mode in ("soft", "auto"):
        print(f"Collecting soft letter-logprob teacher ({args.teacher_model})...", flush=True)
        soft_rows, soft_shuffle, soft_teacher_errors = _distill_prepared(
            prepared,
            query_fn=soft_query,
            seeds=seeds,
            student_checkpoint="frozen_l1_motif_online",
            teacher_mode_label="soft_logprob",
        )
        soft_order_shuffle = _summarize_shuffle(soft_shuffle, seeds)
        soft_gates = soft_calibration_gates(
            top1_match_rate=float(soft_order_shuffle["top1_match_rate"]),
            mean_l1=float(soft_order_shuffle["mean_l1"]),
            n_rows=len(soft_rows),
        )
        soft_ok = soft_calibration_passed(soft_gates) and not soft_teacher_errors
        if args.teacher_mode == "soft" or soft_ok:
            rows, shuffle_rows, teacher_errors = soft_rows, soft_shuffle, soft_teacher_errors
            teacher_mode_used = "soft_logprob"
        else:
            failed = [k for k, v in (soft_gates or {}).items() if not v]
            fallback_reason = (
                f"soft_calibration_failed:{','.join(failed) or 'teacher_errors'}"
            )
            print(
                f"Soft calibration failed ({fallback_reason}); "
                f"falling back to {ranking_mode_label} via {args.ranking_model}",
                flush=True,
            )

    if args.teacher_mode == "ranking" or (
        args.teacher_mode == "auto" and teacher_mode_used != "soft_logprob"
    ):
        print(
            f"Collecting ranking teacher ({args.ranking_model}, method={args.ranking_method})...",
            flush=True,
        )
        rows, shuffle_rows, teacher_errors = _distill_prepared(
            prepared,
            query_fn=ranking_query,
            seeds=seeds,
            student_checkpoint="frozen_l1_motif_online",
            teacher_mode_label=ranking_mode_label,
        )
        teacher_mode_used = ranking_mode_label

    distill_path = out_dir / "opd_distill_real_teacher.jsonl"
    save_opd_rows(distill_path, rows)
    kl = run_opd_smoke(distill_path, output_path=out_dir / "opd_kl_smoke.json") if rows else {}
    order_shuffle = _summarize_shuffle(shuffle_rows, seeds)
    calibration = soft_calibration_gates(
        top1_match_rate=float(order_shuffle["top1_match_rate"]),
        mean_l1=float(order_shuffle["mean_l1"]),
        n_rows=len(rows),
    )
    summary = {
        "n_examples": len(examples),
        "n_prepared": len(prepared),
        "n_distill_rows": len(rows),
        "gate_failures": gate_failures,
        "teacher_errors": teacher_errors,
        "soft_teacher_errors": soft_teacher_errors,
        "teacher_model": args.teacher_model,
        "ranking_model": args.ranking_model,
        "teacher_mode_requested": args.teacher_mode,
        "teacher_mode_used": teacher_mode_used,
        "ranking_method": args.ranking_method,
        "rank_temperature": float(args.rank_temperature),
        "fallback_reason": fallback_reason,
        "split_manifest": args.split_manifest or None,
        "order_shuffle": order_shuffle,
        "soft_order_shuffle": soft_order_shuffle,
        "soft_calibration_gates": soft_gates,
        "kl_smoke": {k: kl.get(k) for k in ("n_rows", "n_precheck_passed", "mean_kl", "mean_jsd")},
        "distill_path": str(distill_path),
        "shuffle_rows": shuffle_rows,
        "calibration_gates": calibration,
    }
    (out_dir / "collect_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.split_manifest:
        manifest = build_posttraining_manifest(
            stage="opd",
            split_manifest_path=args.split_manifest,
            reward_spec_version=REWARD_SPEC_VERSION,
            teacher_model=(
                args.teacher_model
                if teacher_mode_used == "soft_logprob"
                else args.ranking_model
            ),
            motif_bank_path=args.motif_bank,
            candidate_order_seeds=seeds,
            extras={
                "planner_model": args.planner_model,
                "n_distill_rows": len(rows),
                "order_shuffle": summary["order_shuffle"],
                "calibration_gates": summary["calibration_gates"],
                "teacher_mode_requested": args.teacher_mode,
                "teacher_mode_used": teacher_mode_used,
                "soft_teacher_model": args.teacher_model,
                "ranking_model": args.ranking_model,
                "ranking_method": args.ranking_method,
                "rank_temperature": float(args.rank_temperature),
                "fallback_reason": fallback_reason,
                "soft_calibration_gates": soft_gates,
                "soft_order_shuffle": soft_order_shuffle,
            },
        )
        save_posttraining_manifest(out_dir / "posttraining_run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "n_distill_rows": summary["n_distill_rows"],
                "teacher_mode_requested": args.teacher_mode,
                "teacher_mode_used": teacher_mode_used,
                "fallback_reason": fallback_reason,
                "order_shuffle": summary["order_shuffle"],
                "soft_calibration_gates": soft_gates,
                "n_teacher_errors": len(teacher_errors),
                "n_gate_failures": len(gate_failures),
            },
            indent=2,
        )
    )
    return 0 if rows and not teacher_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

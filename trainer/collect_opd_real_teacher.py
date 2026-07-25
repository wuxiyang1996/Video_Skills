#!/usr/bin/env python3
"""P1: Collect OPD distill rows with real OpenRouter letter-logprob teacher + order shuffle."""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

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
    build_l2_candidate_actions,
    gate_candidate_set,
)
from trainer.closed_loop_harness import ClosedLoopHarness  # noqa: E402
from trainer.exact_request_cache import ExactRequestCache  # noqa: E402
from trainer.opd_action_distill_adapter import OpdDistillRow, save_opd_rows  # noqa: E402
from trainer.teacher_action_query import (  # noqa: E402
    make_openrouter_letter_teacher,
    order_shuffle_stability,
    query_teacher_action_distribution,
    query_teacher_averaged,
)
from trainer.train_opd_kl import run_opd_smoke  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", required=True)
    parser.add_argument("--motif-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--teacher-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--shuffle-seeds", default="7,99,13,42")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in sorted(glob(args.frozen_l1_glob, recursive=True))]
    if not paths and "/**/" in args.frozen_l1_glob:
        root_s, _, suffix = args.frozen_l1_glob.partition("/**/")
        paths = sorted(Path(root_s).rglob(suffix))
    paths = paths[: int(args.limit)]
    if not paths:
        raise SystemExit(f"No frozen L1 matched: {args.frozen_l1_glob}")

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
    teacher_fn = make_openrouter_letter_teacher(api_key=api_key, model=args.teacher_model)
    seeds = [int(x) for x in str(args.shuffle_seeds).split(",") if str(x).strip()]
    if len(seeds) < 2:
        seeds = [7, 99]

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
    cache = ExactRequestCache(
        out_dir / "teacher_cache.json",
        {"teacher": args.teacher_model, "v": 1},
    )

    rows: list[OpdDistillRow] = []
    shuffle_rows: list[dict] = []
    gate_failures: list[dict] = []
    teacher_errors: list[dict] = []

    for i, path in enumerate(paths):
        example = json.loads(path.read_text(encoding="utf-8"))
        print(f"[{i+1}/{len(paths)}] {example.get('example_id')}", flush=True)
        state = harness.run_example(example)
        print(
            f"  motif={state.motif_online.get('selected_motif_id')} "
            f"expand={state.motif_online.get('expansion_valid')}",
            flush=True,
        )
        oracle = {
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
        try:
            avg_probs, dists = query_teacher_averaged(
                action_set,
                state=state.to_dict(),
                teacher_fn=teacher_fn,
                order_seeds=seeds,
                cache=cache,
            )
        except Exception as exc:
            teacher_errors.append({"state_id": state.state_id, "error": str(exc)})
            print(f"  teacher_error: {exc}", flush=True)
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
            }
        )
        print(
            f"  teacher top1_match={stab.get('top1_match')} l1={stab.get('l1')} "
            f"majority_frac={majority_frac:.2f} avg_top={avg_top} "
            f"cache_hit_a={dist_a.cache_hit}",
            flush=True,
        )
        # Distill uses seed-0 structure but averaged probs (order-robust).
        teacher_for_row = dist_a
        teacher_for_row.action_probs = dict(avg_probs)
        teacher_for_row.probs = {
            letter: float(avg_probs.get(aid, 0.0))
            for letter, aid in dist_a.letter_to_action_id.items()
        }
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
                },
                student_checkpoint="frozen_l1_motif_online",
            )
        )

    distill_path = out_dir / "opd_distill_real_teacher.jsonl"
    save_opd_rows(distill_path, rows)
    kl = run_opd_smoke(distill_path, output_path=out_dir / "opd_kl_smoke.json") if rows else {}
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
    summary = {
        "n_examples": len(paths),
        "n_distill_rows": len(rows),
        "gate_failures": gate_failures,
        "teacher_errors": teacher_errors,
        "teacher_model": args.teacher_model,
        "order_shuffle": {
            "n": len(shuffle_rows),
            "top1_match_rate": top1_rate,
            "mean_l1": mean_l1,
            "mean_majority_frac": mean_majority,
            "seeds": seeds,
        },
        "kl_smoke": {k: kl.get(k) for k in ("n_rows", "n_precheck_passed", "mean_kl", "mean_jsd")},
        "distill_path": str(distill_path),
        "shuffle_rows": shuffle_rows,
    }
    (out_dir / "collect_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_distill_rows": summary["n_distill_rows"],
                "order_shuffle": summary["order_shuffle"],
                "n_teacher_errors": len(teacher_errors),
                "n_gate_failures": len(gate_failures),
            },
            indent=2,
        )
    )
    return 0 if rows and not teacher_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

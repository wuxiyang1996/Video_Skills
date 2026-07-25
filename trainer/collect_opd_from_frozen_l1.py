#!/usr/bin/env python3
"""Collect OPD distill rows from Motif-gated rollouts on frozen L1 caches."""

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
    mock_teacher_preferring_oracle,
    query_teacher_action_distribution,
)
from trainer.train_opd_kl import run_opd_smoke  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", required=True)
    parser.add_argument("--motif-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument(
        "--teacher",
        choices=["mock", "skip"],
        default="mock",
        help="Real letter-logprob teacher comes later; mock validates the full collect path.",
    )
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
        model=args.model,
        api_key=api_key,
        max_tokens=1800,
        reasoning={"effort": "minimal", "exclude": True},
        timeout_s=180,
    )

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
    examples = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    states = []
    for i, example in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] harness {example.get('example_id')}", flush=True)
        state = harness.run_example(example)
        states.append(state)
        print(
            f"  motif={state.motif_online.get('selected_motif_id')} "
            f"expand={state.motif_online.get('expansion_valid')} "
            f"attempt={state.motif_online.get('motif_retrieval_attempted')}",
            flush=True,
        )

    cache = ExactRequestCache(out_dir / "teacher_cache.json", {"teacher": args.teacher, "v": 1})
    rows: list[OpdDistillRow] = []
    gate_failures = []
    for state in states:
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
        if args.teacher == "mock":
            teacher = query_teacher_action_distribution(
                action_set,
                state=state.to_dict(),
                teacher_fn=mock_teacher_preferring_oracle,
                order_seed=7,
                cache=cache,
            )
        else:
            continue
        rows.append(
            OpdDistillRow.from_parts(
                state=state,
                action_set=action_set,
                teacher=teacher,
                precheck=precheck,
                student_checkpoint="frozen_l1_motif_online",
            )
        )

    distill_path = out_dir / "opd_distill.jsonl"
    save_opd_rows(distill_path, rows)
    kl = run_opd_smoke(distill_path, output_path=out_dir / "opd_kl_smoke.json") if rows else {}
    summary = {
        "n_examples": len(examples),
        "n_states": len(states),
        "n_distill_rows": len(rows),
        "motif_attempt_rate": sum(
            1 for s in states if s.motif_online.get("motif_retrieval_attempted")
        )
        / max(len(states), 1),
        "expansion_valid_rate": sum(1 for s in states if s.motif_online.get("expansion_valid"))
        / max(len(states), 1),
        "gate_failures": gate_failures,
        "kl_smoke": {k: kl.get(k) for k in ("n_rows", "n_precheck_passed", "mean_kl", "mean_jsd")},
        "distill_path": str(distill_path),
    }
    (out_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not gate_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

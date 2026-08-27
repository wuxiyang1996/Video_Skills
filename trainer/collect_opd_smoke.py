#!/usr/bin/env python3
"""Collect a tiny OPD distill JSONL from harness states (mock teacher)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.candidate_action_builder import build_l2_candidate_actions, gate_candidate_set  # noqa: E402
from trainer.closed_loop_harness import ClosedLoopHarness, HarnessState  # noqa: E402
from trainer.exact_request_cache import ExactRequestCache  # noqa: E402
from trainer.opd_action_distill_adapter import OpdDistillRow, save_opd_rows  # noqa: E402
from trainer.teacher_action_query import mock_teacher_preferring_oracle, query_teacher_action_distribution  # noqa: E402
from trainer.train_opd_kl import run_opd_smoke  # noqa: E402


def _synthetic_rollout(example: dict, clue: dict) -> dict:
    del clue
    meta = example.get("metadata") or {}
    enabled = bool(meta.get("motif_enabled"))
    return {
        "acceptance_status": "accepted_strong",
        "final_answer": {"label": "A"},
        "metadata": {
            "runtime_verifier": {"passed": True},
            "llm_plan": {
                "reasoning_plan": [
                    {
                        "step_id": "m1",
                        "skill_id": "select_next_coarse_clip",
                        "args": {"coarse_index": 1},
                    }
                ]
            },
            "executed_skill_ids": ["select_next_coarse_clip"],
            "motif_online": {
                "motif_retrieval_attempted": enabled,
                "candidate_ids": ["motif_demo"] if enabled else [],
                "selected_motif_id": "motif_demo" if enabled else None,
                "bank_version": "demo",
                "expansion_valid": enabled,
                "fallback_reason": None if enabled else "motif_disabled",
                "downstream_verified_success": True,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for i in range(int(args.n)):
        examples.append(
            {
                "example_id": f"smoke:{i}",
                "dataset": "cg_bench",
                "task_family": "causal",
                "question": {"question_text": f"Why did event {i} happen?", "options": ["A", "B"]},
                "metadata": {
                    "motif_enabled": True,
                    "clue_memory_graph": {
                        "graph_id": f"g{i}",
                        "video_id": f"v{i}",
                        "nodes": [{"node_id": "n1", "node_type": "observation"}],
                        "edges": [],
                    },
                },
            }
        )

    harness = ClosedLoopHarness(rollout_fn=_synthetic_rollout, motif_enabled=True)
    states = harness.run_many(examples)
    cache = ExactRequestCache(out_dir / "teacher_cache.json", {"teacher": "mock", "v": 1})

    rows: list[OpdDistillRow] = []
    for state in states:
        student = state.student_action
        oracle = {
            "schema_version": "video-skills/l2-specialist-action-v0.1",
            "tool_name": "choose_best_coarse_candidate",
            "arguments": {"coarse_index": 2},
        }
        action_set = build_l2_candidate_actions(
            state_id=state.state_id,
            student_action=student,
            oracle_action=oracle,
            coarse_indices=[1, 2, 3],
        )
        precheck = gate_candidate_set(action_set)
        if not precheck["passed"]:
            raise RuntimeError(f"candidate gate failed: {precheck}")
        teacher = query_teacher_action_distribution(
            action_set,
            state=state.to_dict(),
            teacher_fn=mock_teacher_preferring_oracle,
            order_seed=17,
            cache=cache,
        )
        rows.append(
            OpdDistillRow.from_parts(
                state=state,
                action_set=action_set,
                teacher=teacher,
                precheck=precheck,
                student_checkpoint="smoke",
            )
        )

    distill_path = out_dir / "opd_distill.jsonl"
    save_opd_rows(distill_path, rows)
    report = run_opd_smoke(distill_path, output_path=out_dir / "opd_kl_smoke.json")
    summary = {
        "n_states": len(states),
        "distill_path": str(distill_path),
        "mean_kl": report.get("mean_kl"),
        "n_precheck_passed": report.get("n_precheck_passed"),
    }
    (out_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

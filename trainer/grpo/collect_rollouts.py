"""Collect K-sample GRPO rollout groups from ``grpo_pool`` (plan §7)."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any, Callable, Mapping

from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.advantages import assign_group_advantages
from trainer.grpo.isolation import assert_rollout_isolation, deep_isolate, fingerprint_example
from trainer.grpo.quality import (
    filter_groups_for_training,
    summarize_group_quality,
)
from trainer.grpo.types import (
    DEFAULT_UPDATE_MODULES,
    MODE_JOINT_L1,
    MODE_L2_REPAIR,
    GrpoGroup,
    GrpoRollout,
    GrpoTrainConfig,
)
from trainer.posttraining_manifest import build_posttraining_manifest, save_posttraining_manifest
from trainer.reward import (
    JUDGE_RUBRIC_VERSION,
    REWARD_SPEC_VERSION,
    policy_safe_rollout_view,
    score_rollout_trace,
)
from trainer.reward.semantic_judge import JudgeFn, mock_semantic_judge
from trainer.split_filter import assert_role_exclusive, example_video_key, filter_examples_by_role, load_split_manifest

RolloutFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _gold_from_example(example: Mapping[str, Any]) -> Any:
    q = example.get("question") or {}
    return q.get("answer") or example.get("gold_answer")


def collect_grpo_group(
    example: Mapping[str, Any],
    *,
    rollout_fn: RolloutFn,
    k: int,
    base_seed: int,
    mode: str = MODE_L2_REPAIR,
    judge_fn: JudgeFn | None = None,
    split_role: str = "grpo_pool",
    require_motif_attempt: bool = True,
) -> GrpoGroup:
    """Sample K isolated rollouts for one prompt and score with verified reward."""
    if k < 2:
        raise ValueError("GRPO requires K >= 2 rollouts per group")
    if mode not in DEFAULT_UPDATE_MODULES:
        raise ValueError(f"unsupported mode: {mode}")

    example_id = str(example.get("example_id") or (example.get("metadata") or {}).get("example_id") or "unknown")
    group_id = f"grpo:{example_id}"
    video_key = example_video_key(example)
    update_modules = DEFAULT_UPDATE_MODULES[mode]
    gold = _gold_from_example(example)
    base_fp = fingerprint_example(example)

    raw_rollouts: list[dict[str, Any]] = []
    scored: list[GrpoRollout] = []

    for i in range(int(k)):
        seed = int(base_seed) + i
        isolated = deep_isolate(example)
        # Ensure no accidental shared mutation of the parent example.
        if fingerprint_example(example) != base_fp:
            raise RuntimeError("parent example mutated during GRPO sampling")

        meta = dict(isolated.get("metadata") or {})
        meta["motif_enabled"] = True
        meta["grpo_sample_index"] = i
        meta["grpo_seed"] = seed
        meta["grpo_mode"] = mode
        isolated["metadata"] = meta
        clue = meta.get("clue_memory_graph") or {}
        print(f"[grpo-collect]   sample {i+1}/{k} seed={seed}", flush=True)
        rollout = rollout_fn(isolated, clue if isinstance(clue, dict) else {})
        motif_online = (rollout.get("metadata") or {}).get("motif_online") or {}
        if require_motif_attempt and not motif_online.get("motif_retrieval_attempted"):
            raise RuntimeError(
                f"motif_retrieval_attempted must be true for GRPO rollout {example_id}#{i}"
            )

        reward = score_rollout_trace(
            rollout,
            clue_graph=clue if isinstance(clue, dict) else {},
            gold_answer=gold,
            judge_fn=judge_fn,
        )
        fa = rollout.get("final_answer")
        label = fa.get("label") if isinstance(fa, dict) else fa
        print(
            f"[grpo-collect]   sample {i+1}/{k} done "
            f"label={label} term={reward.terminal_success} "
            f"progress={reward.progress_total} motif={motif_online.get('selected_motif_id')}",
            flush=True,
        )
        policy_view = policy_safe_rollout_view(rollout)
        raw_rollouts.append(rollout)
        scored.append(
            GrpoRollout(
                group_id=group_id,
                rollout_id=f"{group_id}:k{i}",
                example_id=example_id,
                sample_index=i,
                seed=seed,
                policy_view=policy_view,
                motif_online=dict(motif_online),
                reward=reward,
                update_modules=update_modules,
                extras={
                    "acceptance_status": rollout.get("acceptance_status"),
                    "final_answer": policy_view.get("final_answer"),
                },
            )
        )

    assert_rollout_isolation(raw_rollouts)
    advantages = assign_group_advantages([r.reward for r in scored])
    for rollout, adv in zip(scored, advantages):
        rollout.advantage = float(adv)

    return GrpoGroup(
        group_id=group_id,
        example_id=example_id,
        video_key=video_key,
        split_role=split_role,
        mode=mode,
        rollouts=scored,
    )


def save_grpo_groups(path: str | Path, groups: list[GrpoGroup]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")
    return out


def load_grpo_groups(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _default_mock_rollout(example: dict[str, Any], clue: dict[str, Any]) -> dict[str, Any]:
    """Deterministic smoke rollout (no LLM)."""
    seed = int((example.get("metadata") or {}).get("grpo_seed") or 0)
    q = example.get("question") or {}
    gold = q.get("answer") or {}
    success = (seed % 2) == 0
    graph = clue if isinstance(clue, dict) and clue.get("graph_id") else {
        "graph_id": f"clue:{example.get('video_id') or example.get('example_id') or 'smoke'}",
        "layer": "clue_memory",
        "nodes": [],
        "edges": [],
    }
    return {
        "layer": "reasoning",
        "rollout_id": f"smoke-{seed}",
        "clue_memory_ref": {"graph_id": graph.get("graph_id")},
        "acceptance_status": "accepted_strong" if success else "accepted_weak",
        "final_answer": gold if success else {"label": "Z", "text": "wrong"},
        "metadata": {
            "motif_online": {
                "motif_retrieval_attempted": True,
                "motif_phase": "accelerate",
                "selected_motif_id": "smoke_motif",
                "expansion_valid": True,
                "candidate_mined": False,
            },
            "executed_skill_ids": ["parse_question_target", "retrieve_by_event", "commit_answer"],
            "costs": {"clip_reads": 1 + (seed % 3), "tool_calls": 3, "tokens": 100, "repair_rounds": 0},
            "milestone_events": [
                {
                    "kind": "retrieval",
                    "key": f"ref_smoke_{seed}",
                    "step_index": 1,
                    "grounded": True,
                }
            ]
            if success
            else [],
            "final_used_milestone_keys": [f"retrieval:ref_smoke_{seed}"] if success else [],
            "clue_memory_graph": graph,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--mode", choices=[MODE_L2_REPAIR, MODE_JOINT_L1], default=MODE_L2_REPAIR)
    parser.add_argument("--motif-bank", default="motif/output/pilot_online_motif_bank.jsonl")
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--smoke-mock-rollout", action="store_true")
    parser.add_argument("--live", action="store_true", help="Motif-gated live planner rollouts")
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b")
    parser.add_argument(
        "--skill-temperature",
        type=float,
        default=0.7,
        help="LLM skill sampling temperature (needed for K-sample diversity)",
    )
    parser.add_argument(
        "--with-skill-executor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LLM SkillExecutor for answer-critical skills (default on for --live)",
    )
    parser.add_argument(
        "--rotate-motifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate forced Motif id by grpo_seed across ACTIVE bank",
    )
    parser.add_argument(
        "--force-explore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate among top-k scored options by grpo_seed (cold-start diversity)",
    )
    parser.add_argument(
        "--explore-top-k",
        type=int,
        default=2,
        help="Top-k pool size for force_explore (peaked scores may bump to 3)",
    )
    parser.add_argument(
        "--drop-dirty-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop label=None / progress=0 samples before writing train groups",
    )
    parser.add_argument("--min-k-after-filter", type=int, default=2)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--judge-mock", action="store_true", help="Use mock semantic judge")
    parser.add_argument("--l2-stable", action="store_true", help="Required for joint_l1 mode")
    args = parser.parse_args(argv)

    if args.mode == MODE_JOINT_L1:
        GrpoTrainConfig(mode=MODE_JOINT_L1, l2_stable_flag=bool(args.l2_stable)).update_modules()
    if not args.smoke_mock_rollout and not args.live:
        raise SystemExit("Pass --live for Motif-gated collection or --smoke-mock-rollout for offline smoke")

    paths = [Path(p) for p in sorted(glob(args.frozen_l1_glob, recursive=True))]
    if not paths and "/**/" in args.frozen_l1_glob:
        root_s, _, suffix = args.frozen_l1_glob.partition("/**/")
        paths = sorted(Path(root_s).rglob(suffix))
    examples = load_frozen_l1_examples(paths)
    manifest = load_split_manifest(args.split_manifest)
    pool = filter_examples_by_role(examples, manifest=manifest, role="grpo_pool", strict=False)
    if not pool:
        if args.smoke_mock_rollout:
            pool = list(examples)
        else:
            raise SystemExit("No grpo_pool examples after split filter")
    if args.shard_count is not None:
        shard_count = int(args.shard_count)
        shard_id = int(args.shard_id if args.shard_id is not None else 0)
        if shard_count < 1:
            raise SystemExit("--shard-count must be >= 1")
        if not (0 <= shard_id < shard_count):
            raise SystemExit(f"--shard-id must be in [0, {shard_count})")
        pool = [ex for i, ex in enumerate(pool) if i % shard_count == shard_id]
    pool = pool[: int(args.limit)]
    if not args.smoke_mock_rollout:
        assert_role_exclusive(pool, manifest=manifest, allowed_roles=("grpo_pool",))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.live:
        from trainer.grpo.live_rollout import make_motif_gated_rollout_fn

        if not args.motif_bank:
            raise SystemExit("--motif-bank is required for --live")
        rollout_fn = make_motif_gated_rollout_fn(
            motif_bank_path=args.motif_bank,
            planner_model=args.planner_model,
            skill_model=args.skill_model,
            keys_py=args.keys_py,
            skill_temperature=float(args.skill_temperature),
            with_skill_executor=bool(args.with_skill_executor),
            rotate_motifs=bool(args.rotate_motifs),
            force_explore=bool(args.force_explore),
            explore_top_k=int(args.explore_top_k),
            motif_candidate_sink_path=out_dir / "motif_candidates.jsonl",
        )
        # Semantic judge: mock keeps offline/live collect fail-closed until OpenRouter judge is wired.
        judge_fn = mock_semantic_judge
        collect_mode = "live_motif_llm_skills" if args.with_skill_executor else "live_motif"
    else:
        rollout_fn = _default_mock_rollout
        judge_fn = mock_semantic_judge
        collect_mode = "smoke_mock"

    groups: list[GrpoGroup] = []
    for idx, example in enumerate(pool):
        eid = example.get("example_id") or (example.get("metadata") or {}).get("example_id")
        print(f"[grpo-collect] {idx+1}/{len(pool)} example={eid} k={args.k} mode={collect_mode}", flush=True)
        groups.append(
            collect_grpo_group(
                example,
                rollout_fn=rollout_fn,
                k=int(args.k),
                base_seed=int(args.base_seed),
                mode=args.mode,
                judge_fn=judge_fn,
            )
        )
        term = sum(1 for r in groups[-1].rollouts if r.reward.terminal_success)
        print(
            f"[grpo-collect] done example={eid} "
            f"terminal_success={term}/{len(groups[-1].rollouts)}",
            flush=True,
        )

    raw_quality = summarize_group_quality(groups)
    raw_path = save_grpo_groups(out_dir / "grpo_groups_raw.jsonl", groups)
    train_groups = (
        filter_groups_for_training(
            groups,
            drop_dirty=bool(args.drop_dirty_samples),
            min_k=int(args.min_k_after_filter),
        )
        if args.drop_dirty_samples
        else list(groups)
    )
    train_quality = summarize_group_quality(train_groups)
    groups_path = save_grpo_groups(out_dir / "grpo_groups.jsonl", train_groups)
    run_manifest = build_posttraining_manifest(
        stage="grpo_collect",
        split_manifest_path=args.split_manifest,
        reward_spec_version=REWARD_SPEC_VERSION,
        grpo_mode=args.mode,
        update_modules=list(DEFAULT_UPDATE_MODULES[args.mode]),
        judge_model="mock_semantic_judge",
        judge_rubric_version=JUDGE_RUBRIC_VERSION,
        policy_checkpoint=args.policy_checkpoint,
        motif_bank_path=args.motif_bank,
        k_samples=int(args.k),
        extras={
            "collect_mode": collect_mode,
            "planner_model": args.planner_model if args.live else None,
            "n_groups_raw": len(groups),
            "n_groups_train": len(train_groups),
            "shard_id": args.shard_id,
            "shard_count": args.shard_count,
            "force_explore": bool(args.force_explore),
            "explore_top_k": int(args.explore_top_k),
            "drop_dirty_samples": bool(args.drop_dirty_samples),
            "framework": "hf_peft_custom_grpo",
            "verl": False,
            "ms_swift": False,
            "quality_raw": raw_quality,
            "quality_train": train_quality,
        },
    )
    save_posttraining_manifest(out_dir / "posttraining_run_manifest.json", run_manifest)

    summary = {
        "n_groups_raw": len(groups),
        "n_groups": len(train_groups),
        "k": int(args.k),
        "mode": args.mode,
        "collect_mode": collect_mode,
        "groups_path": str(groups_path),
        "raw_groups_path": str(raw_path),
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
        "mean_terminal_success": train_quality["mean_terminal_success"],
        "quality_raw": raw_quality,
        "quality_train": train_quality,
    }
    (out_dir / "collect_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

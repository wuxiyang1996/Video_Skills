#!/usr/bin/env python3
"""Replay only train-pool rollouts affected by the pre-v5 typed-plan binding bug."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.l2_dataset_rewards import load_dataset_reward_supervision, supervision_key
from trainer.grpo.train_l2_terminal_on_policy import (
    EXECUTOR_CACHE_VERSION,
    EXECUTOR_FALLBACK_VERSION,
    cached_executor_rollout,
    compact_rollout_diagnostic,
    executor_backend_for_dataset,
    executor_cache_key,
    filter_example_for_retrieval,
    retrieval_catalog,
    terminal_reward,
)
from trainer.split_filter import assert_role_exclusive, filter_examples_by_role, load_split_manifest


AFFECTED_FAILURES = {
    ("localize_clue", "invalid_skill_args"),
    ("assign_evidence_role", "missing_evidence_ref"),
    ("compose_evidence_chain", "missing_role_labeled_evidence"),
}


def needs_typedplan_replay(row: dict[str, Any]) -> bool:
    if row.get("dataset") != "cg_bench" or not row.get("process_supported"):
        return False
    failures = {
        (str(item.get("skill_id") or ""), str(item.get("failure_code") or ""))
        for item in ((row.get("rollout_diagnostic") or {}).get("failed_skill_codes") or [])
    }
    return bool(failures & AFFECTED_FAILURES)


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("dataset") or ""), int(row.get("group") or 0), int(row.get("sample") or 0)


def replace_affected_rows(
    original: list[dict[str, Any]], replacements: dict[tuple[str, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    keys = [row_key(row) for row in original]
    if len(keys) != len(set(keys)):
        raise ValueError("original sample log contains duplicate dataset/group/sample keys")
    unknown = set(replacements) - set(keys)
    if unknown:
        raise ValueError(f"replacement keys absent from original log: {sorted(unknown)[:5]}")
    return [replacements.get(row_key(row), row) for row in original]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset") or ""), str(row.get("example_id") or ""),
            int(row.get("repeat_index") or 0),
        )
        grouped.setdefault(key, []).append(row)
    result = {}
    for dataset in ("cg_bench", "video_holmes"):
        groups = [values for key, values in grouped.items() if key[0] == dataset and len(values) == 8]
        trainable = sum(
            any(bool(row.get("terminal_success")) for row in values)
            and len({round(float(row.get("reward") or 0.0), 8) for row in values}) > 1
            for values in groups
        )
        result[dataset] = {
            "complete_groups": len(groups),
            "trainable_groups": trainable,
            "trainable_group_rate": trainable / max(1, len(groups)),
            "terminal_successes": sum(
                bool(row.get("terminal_success")) for values in groups for row in values
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", required=True, help="SEED|SAMPLES_JSONL")
    parser.add_argument("--frozen-l1-glob", action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--skill-model", default="openai/gpt-oss-120b")
    parser.add_argument("--planner-timeout-s", type=int, default=180)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows-per-seed", type=int, default=1456)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.frozen_l1_glob:
        paths.extend(Path(value) for value in sorted(glob.glob(pattern, recursive=True)))
    examples = load_frozen_l1_examples(paths)
    manifest = load_split_manifest(args.split_manifest)
    examples = filter_examples_by_role(examples, manifest=manifest, role="grpo_pool", strict=False)
    assert_role_exclusive(examples, manifest=manifest, allowed_roles=("grpo_pool",))
    example_by_id = {str(row.get("example_id") or ""): row for row in examples}
    supervision = load_dataset_reward_supervision(args.dataset_root)
    api_key = load_openrouter_api_key(keys_py_path=args.keys_py)
    planner = OpenRouterClient(
        model=args.planner_model, api_key=api_key, max_tokens=1800, temperature=0.0,
        reasoning={"effort": "minimal", "exclude": True}, timeout_s=args.planner_timeout_s,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    for raw in args.seed:
        seed_text, path_text = raw.split("|", 1)
        seed = int(seed_text)
        source_path = Path(path_text)
        original = _read_jsonl(source_path)
        affected = [row for row in original if needs_typedplan_replay(row)]
        replacements: dict[tuple[str, int, int], dict[str, Any]] = {}
        cache_hits = 0
        for old in affected:
            example_id = str(old.get("example_id") or "")
            example = example_by_id.get(example_id)
            if example is None:
                raise ValueError(f"affected example missing from frozen grpo_pool: {example_id}")
            indices = [int(value) for value in (old.get("selected_indices") or [])]
            isolated, graph = filter_example_for_retrieval(example, indices)
            source_catalog, _ = retrieval_catalog(example)
            selected_entries = [source_catalog[index] for index in indices]

            def build_rollout() -> dict[str, Any]:
                return build_llm_reasoning_rollout(
                    isolated, graph, client=planner, skill_executor=None, motif_enabled=False
                )

            cache_key = executor_cache_key(
                example=isolated, indices=indices, graph=graph,
                planner_model=args.planner_model,
                skill_model=executor_backend_for_dataset("cg_bench", args.skill_model),
            )
            rollout, cache_hit = cached_executor_rollout(
                cache_dir=args.cache_dir, key=cache_key, build=build_rollout
            )
            cache_hits += int(cache_hit)
            outcome = terminal_reward(
                rollout, (example.get("question") or {}).get("answer") or {},
                dataset="cg_bench", selected_entries=selected_entries,
                supervision=supervision.get(supervision_key(example)),
                question_type=str((example.get("question") or {}).get("question_type") or ""),
            )
            outcome["executor_cache_hit"] = cache_hit
            outcome["executor_cache_key"] = cache_key
            outcome["rollout_diagnostic"] = compact_rollout_diagnostic(rollout)
            replacements[row_key(old)] = {**old, **outcome, "reward_dataset": "cg_bench"}

        corrected = replace_affected_rows(original, replacements)
        corrected_path = args.output_root / f"seed{seed}_terminal_samples.corrected.jsonl"
        corrected_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corrected),
            encoding="utf-8",
        )
        seed_reports.append({
            "seed": seed,
            "source_samples": str(source_path),
            "source_sha256": _sha256(source_path),
            "corrected_samples": str(corrected_path),
            "corrected_sha256": _sha256(corrected_path),
            "rows": len(original),
            "affected_rows_replayed": len(affected),
            "unique_affected_actions": len({
                (row.get("example_id"), tuple(row.get("selected_indices") or [])) for row in affected
            }),
            "replay_cache_hits": cache_hits,
            "old_group_stats": _group_stats(original),
            "corrected_group_stats": _group_stats(corrected),
        })

    report = {
        "schema_version": "video-skills/l2-typedplan-affected-replay-v1",
        "passed": len(seed_reports) == 3 and all(
            row["affected_rows_replayed"] > 0
            and row["rows"] == args.expected_rows_per_seed
            and row["old_group_stats"]["cg_bench"]["complete_groups"] == 91
            and row["old_group_stats"]["video_holmes"]["complete_groups"] == 91
            for row in seed_reports
        ),
        "expected_rows_per_seed": args.expected_rows_per_seed,
        "selection_uses_training_pool_only": True,
        "affected_failure_contract": sorted([list(value) for value in AFFECTED_FAILURES]),
        "executor_fallback_contract": EXECUTOR_FALLBACK_VERSION,
        "executor_cache_contract": EXECUTOR_CACHE_VERSION,
        "executor_cache_dir": str(args.cache_dir),
        "seeds": seed_reports,
    }
    report_path = args.output_root / "replay_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

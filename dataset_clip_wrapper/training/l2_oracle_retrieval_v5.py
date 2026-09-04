#!/usr/bin/env python3
"""Build leakage-safe L2 retrieval SFT from annotated evidence intervals.

The evidence intervals are used only to construct/evaluate the assistant target.
They are never serialized into the model-visible state.  Catalog rendering is
label-independent: selected rows receive exactly the same field budgets as all
other rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .l2_retrieval_sft_adapter import _catalog
from .l2_specialist_sft_adapter import _chat, _positive_expansions
from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_jsonl, write_json, write_jsonl


TRAIN_ROLES = {"sft_seed", "dev_tune"}
FAMILY_BUDGETS = {
    "select_set": 0.50,
    "atomic_select": 0.20,
    "ranking": 0.20,
    "stop_continue": 0.10,
}


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def select_oracle_windows(
    coarse_schemas: list[dict[str, Any]],
    clue_intervals: list[list[float]],
    *,
    topk: int = 2,
) -> tuple[list[int], float]:
    """Choose the top-k coarse windows by annotated-evidence overlap."""
    clues = [(float(row[0]), float(row[1])) for row in clue_intervals if len(row) >= 2]
    scored: list[tuple[float, int]] = []
    spans: dict[int, tuple[float, float]] = {}
    for index, schema in enumerate(coarse_schemas):
        span = schema.get("time_span") if isinstance(schema.get("time_span"), dict) else {}
        candidate = (float(span.get("start_s", 0.0)), float(span.get("end_s", 0.0)))
        spans[index] = candidate
        score = sum(interval_overlap(candidate, clue) for clue in clues)
        if score > 0.0:
            scored.append((score, index))
    chosen = sorted(index for _, index in sorted(scored, key=lambda row: (-row[0], row[1]))[:topk])
    clue_length = _merged_length(clues)
    covered_parts: list[tuple[float, float]] = []
    for clue in clues:
        for index in chosen:
            start = max(clue[0], spans[index][0])
            end = min(clue[1], spans[index][1])
            if end > start:
                covered_parts.append((start, end))
    coverage = _merged_length(covered_parts) / max(clue_length, 1e-12)
    return chosen, min(1.0, coverage)


def _role_index(manifest: dict[str, Any]) -> dict[tuple[str, str], str]:
    return {
        (str(row["dataset"]), str(row["video_id"])): str(row["role"])
        for row in manifest.get("videos") or []
    }


def _question_index(cg_bench: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["qid"]): row for row in cg_bench if row.get("qid") is not None}


def _visible_rationale(catalog: list[dict[str, Any]], selected: list[int]) -> str:
    by_index = {int(row["coarse_index"]): row for row in catalog}
    cues = []
    for index in selected:
        row = by_index.get(index) or {}
        cue = str(row.get("scene_description") or "").strip()
        if cue:
            cues.append(cue[:100])
    return ("Visible evidence candidates: " + " | ".join(cues))[:300]


def policy_catalog(coarse_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render a compact, label-independent catalog that fits long videos in 16K."""
    rows = _catalog(coarse_schemas)
    bounded = [
        {
            "coarse_index": row["coarse_index"],
            "time_span": row.get("time_span"),
            "scene_description": str(row.get("scene_description") or "")[:64],
            "observable_facts": [str(value)[:48] for value in (row.get("observable_facts") or [])[:1]],
            "events": [str(value)[:48] for value in (row.get("events") or [])[:1]],
            "searchable_phrases": [str(value)[:40] for value in (row.get("searchable_phrases") or [])[:1]],
        }
        for row in rows
    ]
    return [_minimal_catalog_row(row) for row in bounded]


def _minimal_catalog_row(row: dict[str, Any]) -> dict[str, Any]:
    cues = [str(row.get("scene_description") or "")]
    for key in ("observable_facts", "events", "searchable_phrases"):
        values = row.get(key) if isinstance(row.get(key), list) else []
        if values:
            cues.append(str(values[0]))
    summary = " | ".join(value.strip() for value in cues if value.strip())[:96]
    return {
        "coarse_index": int(row.get("coarse_index", -1)),
        "time_span": row.get("time_span") if isinstance(row.get("time_span"), dict) else {},
        "scene_description": summary,
    }


def _compact_transition_catalogs(transition: dict[str, Any]) -> None:
    state = transition.get("state_t") if isinstance(transition.get("state_t"), dict) else {}
    for key in ("l1_coarse_summary_catalog", "candidate_coarse_summaries"):
        rows = state.get(key) if isinstance(state.get(key), list) else None
        if rows is not None:
            state[key] = [_minimal_catalog_row(row) for row in rows if isinstance(row, dict)]


def build_oracle_package(
    rollout_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    cg_bench: list[dict[str, Any]],
    *,
    topk: int = 2,
    min_clue_coverage: float = 0.5,
    hard_negatives_per_selected: int = 6,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    roles = _role_index(manifest)
    questions = _question_index(cg_bench)
    excluded: Counter[str] = Counter()
    core_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in TRAIN_ROLES}
    audit_by_role: Counter[str] = Counter()
    coverage_by_role: dict[str, list[float]] = {}

    seen: set[str] = set()
    for row in rollout_rows:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen:
            continue
        seen.add(example_id)
        dataset = str(row.get("dataset") or "")
        video = row.get("video") if isinstance(row.get("video"), dict) else {}
        role = roles.get((dataset, str(video.get("video_id") or "")), "unknown")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        coarse = metadata.get("coarse_clip_schemas") if isinstance(metadata.get("coarse_clip_schemas"), list) else []
        if dataset != "cg_bench":
            excluded["dataset_without_interval_labels"] += 1
            continue
        if not coarse:
            excluded[f"{role}:missing_coarse"] += 1
            continue
        item = questions.get(example_id.rsplit(":", 1)[-1])
        clues = (item or {}).get("clue_intervals") or []
        if not clues:
            excluded[f"{role}:missing_clue_intervals"] += 1
            continue
        selected, coverage = select_oracle_windows(coarse, clues, topk=topk)
        if not selected:
            excluded[f"{role}:no_overlapping_window"] += 1
            continue
        audit_by_role[role] += 1
        coverage_by_role.setdefault(role, []).append(coverage)
        if role not in TRAIN_ROLES:
            continue
        if coverage + 1e-12 < min_clue_coverage:
            excluded[f"{role}:low_clue_coverage"] += 1
            continue

        # Critical: do not pass selected indices to _catalog.  Its legacy
        # selected-row enrichment is useful for reports but would leak labels
        # through asymmetric prompt detail in policy training.
        catalog = policy_catalog(coarse)
        state = {
            "schema_version": "video-skills/l2-retrieval-state-v0.3",
            "process_model": "mdp_style_l2_retrieval_controller",
            "dataset": dataset,
            "example_id": example_id,
            "question": compact_visibility(row.get("question") or {}),
            "l1_coarse_summary_catalog": catalog,
            "partial_l1_summary": {
                "coarse_summary_count": len(coarse),
                "fine_observation_count": 0,
            },
            "budget_state": {"topk": topk, "retrieval_round": 0},
        }
        if contains_forbidden_prompt_key(state):
            excluded[f"{role}:forbidden_prompt_key"] += 1
            continue
        core_by_role[role].append({
            "schema_version": "video-skills/l2-retrieval-transition-v0.2",
            "transition_id": f"{example_id}::l2_oracle_retrieval::0",
            "controller": "l2_retrieval",
            "state_t": state,
            "action_t": {
                "schema_version": "video-skills/l2-retrieval-action-v0.1",
                "tool_name": "select_coarse_clips",
                "arguments": {
                    "selected_coarse_indices": selected,
                    "rationale_short": _visible_rationale(catalog, selected),
                },
            },
            "source_example_id": example_id,
            "augmentation_family": "oracle_temporal_select",
            "is_core": True,
            "teacher_metadata": {
                "teacher": "deterministic_cg_bench_clue_interval_mapper",
                "target_only_supervision": True,
                "clue_coverage": coverage,
            },
            "split_role": role,
            "split_group_id": f"{dataset}:video:{video.get('video_id')}",
            "video_id": str(video.get("video_id") or ""),
        })

    package: dict[str, list[dict[str, Any]]] = {}
    for role, core in core_by_role.items():
        expanded = _positive_expansions(core, hard_negatives_per_selected)
        for transition in expanded:
            _compact_transition_catalogs(transition)
        family_counts = Counter(str(row.get("augmentation_family")) for row in expanded)
        by_source_family = Counter(
            (str(row.get("source_example_id")), str(row.get("augmentation_family")))
            for row in expanded
        )
        source_index = {str(row["source_example_id"]): row for row in core}
        chats = []
        for transition in expanded:
            family = str(transition.get("augmentation_family"))
            key = (str(transition.get("source_example_id")), family)
            transition["source_family_weight"] = FAMILY_BUDGETS[family] / by_source_family[key]
            chat = _chat(transition, str(transition["task"]))
            source = source_index[str(transition["source_example_id"])]
            chat["split_group_id"] = source["split_group_id"]
            chat["specialist"] = "l2"
            chat["metadata"].update({
                "split_role": role,
                "video_id": source["video_id"],
                "video_key": source["split_group_id"].replace(":video:", ":"),
                "teacher": "deterministic_cg_bench_clue_interval_mapper",
                "causal_gate": "annotated_temporal_overlap",
            })
            chats.append(chat)
        package[role] = chats

    report = {
        "schema_version": "video-skills/l2-oracle-retrieval-v5",
        "teacher": {
            "target_generator": "deterministic_cg_bench_clue_interval_mapper",
            "prompt_uses_hidden_supervision": False,
            "catalog_is_label_independent": True,
            "temporal_gate": "annotated clue overlap",
            "downstream_executor_gate": "pending",
        },
        "topk": topk,
        "min_clue_coverage": min_clue_coverage,
        "auditable_core_by_role": dict(audit_by_role),
        "exported_core_by_role": {role: len(rows) for role, rows in core_by_role.items()},
        "exported_rows_by_role": {role: len(rows) for role, rows in package.items()},
        "mean_clue_coverage_by_role": {
            role: sum(values) / len(values) for role, values in coverage_by_role.items() if values
        },
        "excluded": dict(excluded),
        "family_loss_budgets": FAMILY_BUDGETS,
        "prompt_forbidden_key_hits": sum(
            contains_forbidden_prompt_key(json.loads(row["messages"][1]["content"]))
            for rows in package.values() for row in rows
        ),
    }
    return package, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cg-bench", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--min-clue-coverage", type=float, default=0.5)
    parser.add_argument("--hard-negatives-per-selected", type=int, default=6)
    parser.add_argument(
        "--frozen-dev-jsonl", type=Path,
        help="Optional previously frozen dev chats; keeps the evaluation core unchanged during train expansion.",
    )
    args = parser.parse_args(argv)
    package, report = build_oracle_package(
        read_jsonl(args.rollouts),
        json.loads(args.split_manifest.read_text(encoding="utf-8")),
        json.loads(args.cg_bench.read_text(encoding="utf-8")),
        topk=args.topk,
        min_clue_coverage=args.min_clue_coverage,
        hard_negatives_per_selected=args.hard_negatives_per_selected,
    )
    if args.frozen_dev_jsonl is not None:
        frozen_dev = read_jsonl(args.frozen_dev_jsonl)
        package["dev_tune"] = frozen_dev
        frozen_core_examples = len({
            str((row.get("metadata") or {}).get("source_example_id") or "")
            for row in frozen_dev
            if (row.get("metadata") or {}).get("task") == "select_coarse_set"
            and (row.get("metadata") or {}).get("is_core") is True
        })
        report["frozen_dev"] = {
            "path": str(args.frozen_dev_jsonl),
            "rows": len(frozen_dev),
            "core_examples": frozen_core_examples,
        }
        report["exported_core_by_role"]["dev_tune"] = frozen_core_examples
        report["exported_rows_by_role"]["dev_tune"] = len(frozen_dev)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", package.get("sft_seed", []))
    write_jsonl(args.output_dir / "dev.jsonl", package.get("dev_tune", []))
    write_json(args.output_dir / "oracle_build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["prompt_forbidden_key_hits"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

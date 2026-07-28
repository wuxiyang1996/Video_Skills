#!/usr/bin/env python3
"""Collapse question-prefix M3 work into causal completed-clip snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("ovo_bench", "videomme", "streaming_bench")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_id(dataset: str, video_path: str) -> str:
    path = Path(video_path)
    digest = hashlib.sha256(f"{dataset}\0{path.resolve()}".encode()).hexdigest()[:16]
    return f"{dataset}__{path.stem}__source__{digest}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--clip-duration-s", type=float, default=30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_rows = _read_jsonl(args.input_dir / "memorization_inputs.jsonl")
    old_by_id = {row["id"]: row for row in old_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        grouped[(row["dataset"], row["video_path"])].append(row)

    sources: list[dict[str, Any]] = []
    source_by_old_id: dict[str, dict[str, Any]] = {}
    for (dataset, video_path), rows in sorted(grouped.items()):
        source_id = _source_id(dataset, video_path)
        root = args.artifact_root.resolve() / dataset / source_id
        cutoffs = [
            float(row["observation_end_s"])
            for row in rows
            if isinstance(row.get("observation_end_s"), (int, float))
        ]
        snapshots: dict[str, str] = {}
        if dataset == "streaming_bench":
            required = sorted({math.floor(cutoff / args.clip_duration_s) - 1 for cutoff in cutoffs})
            snapshots = {
                str(clip_id): str(root / "snapshots" / f"through_clip_{clip_id}.pkl")
                for clip_id in required
            }
            max_complete_end = max(0.0, math.floor(max(cutoffs) / args.clip_duration_s) * args.clip_duration_s)
            observation_end_s: float | None = max_complete_end
        else:
            observation_end_s = None

        source = {
            "id": source_id,
            "video_path": video_path,
            "clip_path": str(root / "clips"),
            "mem_path": str(root / "memory_graph.pkl"),
            "intermediate_outputs": str(root / "intermediate_outputs"),
            "dataset": dataset,
            "observation_end_s": observation_end_s,
            "clip_duration_s": args.clip_duration_s,
            "snapshot_paths": snapshots,
            "alignment_class": "official_model_adapted_benchmarks",
            "causal_policy": "completed_30s_graph_snapshots",
        }
        sources.append(source)
        for row in rows:
            source_by_old_id[row["id"]] = source

    controls: dict[str, OrderedDict[str, dict[str, Any]]] = {
        dataset: OrderedDict() for dataset in DATASETS
    }
    for dataset in DATASETS:
        old_control = json.loads((args.input_dir / f"control_{dataset}.json").read_text(encoding="utf-8"))
        for old_graph_id, graph in old_control.items():
            source = source_by_old_id[old_graph_id]
            for qa in graph["qa_list"]:
                qa = dict(qa)
                cutoff = qa.get("timestamp_s")
                if dataset == "streaming_bench" and isinstance(cutoff, (int, float)):
                    clip_id = math.floor(float(cutoff) / args.clip_duration_s) - 1
                    mem_path = source["snapshot_paths"][str(clip_id)]
                    control_id = f"{source['id']}__through_{clip_id}"
                    qa["completed_before_clip"] = clip_id
                    qa["discarded_visible_tail_s"] = float(cutoff) - max(
                        0.0, (clip_id + 1) * args.clip_duration_s
                    )
                    qa["causal_graph_is_question_prefix"] = False
                    qa["causal_policy"] = "completed_30s_graph_snapshot"
                else:
                    mem_path = source["mem_path"]
                    control_id = source["id"]
                target = controls[dataset].setdefault(
                    control_id,
                    {
                        "video_path": source["video_path"],
                        "mem_path": mem_path,
                        "qa_list": [],
                    },
                )
                target["qa_list"].append(qa)

    _write_jsonl(args.output_dir / "memorization_inputs.jsonl", sources)
    _write_jsonl(args.output_dir / "clip_plan.jsonl", sources)
    for dataset, payload in controls.items():
        (args.output_dir / f"control_{dataset}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "alignment_class": "official_model_adapted_benchmarks",
        "causal_policy": "completed_30s_graph_snapshots",
        "source_graphs": len(sources),
        "source_graphs_by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in sources) for dataset in DATASETS
        },
        "questions": {
            dataset: sum(len(graph["qa_list"]) for graph in controls[dataset].values())
            for dataset in DATASETS
        },
        "original_prefix_graphs": len(old_rows),
        "avoided_graphs": len(old_rows) - len(sources),
        "tradeoff": (
            "Only complete 30-second clips visible at each question are used; "
            "the remaining visible tail is conservatively discarded."
        ),
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export three local benchmarks into official M3-Agent input schemas.

This prepares an adapted-benchmark run. It uses official M3-Agent memorization
and Control schemas, but does not claim M3-Bench or official-paper dataset
reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ("ovo_bench", "videomme", "streaming_bench")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/m3_agent_adapted/manifests"
)


def ensure_repo_on_path(repo_root: Path) -> None:
    value = str(repo_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_graph_id(dataset: str, video_path: Path, observation_end_s: float | None) -> str:
    cutoff = "full" if observation_end_s is None else f"{observation_end_s:.6f}"
    identity = f"{dataset}\0{video_path.resolve()}\0{cutoff}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{dataset}__{video_path.stem}__{cutoff.replace('.', '_')}__{digest}"


def _question_options(question: dict[str, Any]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for index, option in enumerate(question.get("options") or []):
        if not isinstance(option, dict):
            continue
        label = str(option.get("label") or chr(ord("A") + index)).strip().upper()
        text = str(option.get("text") or "").strip()
        if label and text:
            options.append({"label": label, "text": text})
    return options


def _control_question(question: dict[str, Any]) -> str:
    lines = [str(question.get("question_text") or "").strip()]
    options = _question_options(question)
    if options:
        lines.append("Options:")
        lines.extend(f"{option['label']}. {option['text']}" for option in options)
        lines.append("Respond with only the letter of the correct answer.")
    return "\n".join(line for line in lines if line)


def _control_answer(question: dict[str, Any]) -> str:
    answer = question.get("answer") or {}
    label = str(answer.get("label") or "").strip().upper()
    text = str(answer.get("text") or "").strip()
    if label and text:
        return f"{label}. {text}"
    return label or text


def _observation_end_s(dataset: str, item: Any) -> float | None:
    if dataset == "videomme":
        return None
    value = item.question.get("time_anchor_s")
    return float(value) if isinstance(value, (int, float)) else None


def _assert_output_location(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path("/home/xwu") or Path("/home/xwu") in resolved.parents:
        raise ValueError("M3 manifests and generated media must be stored under /mnt, not /home")


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
    from dataset_clip_wrapper.adapters import get_adapter

    output_dir = args.output_dir.resolve()
    _assert_output_location(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    controls: dict[str, OrderedDict[str, dict[str, Any]]] = {
        dataset: OrderedDict() for dataset in args.datasets
    }
    skipped: list[dict[str, str]] = []

    for dataset in args.datasets:
        adapter = get_adapter(dataset, args.dataset_root.resolve(), split=args.split)
        for item in adapter.iter_items(limit=args.limit_per_dataset):
            if item.video_path is None or not item.video_path.is_file():
                skipped.append(
                    {
                        "dataset": dataset,
                        "example_id": item.example_id,
                        "reason": "video path is missing",
                    }
                )
                continue
            answer = _control_answer(item.question)
            question = _control_question(item.question)
            if not question or not answer:
                skipped.append(
                    {
                        "dataset": dataset,
                        "example_id": item.example_id,
                        "reason": "question or answer is missing",
                    }
                )
                continue

            observation_end_s = _observation_end_s(dataset, item)
            if dataset != "videomme" and observation_end_s is None:
                skipped.append(
                    {
                        "dataset": dataset,
                        "example_id": item.example_id,
                        "reason": "streaming time anchor is missing",
                    }
                )
                continue

            graph_id = _stable_graph_id(dataset, item.video_path, observation_end_s)
            graph_root = args.artifact_root.resolve() / dataset / graph_id
            mem_path = graph_root / "memory_graph.pkl"
            clip_path = graph_root / "clips"
            intermediate_path = graph_root / "intermediate_outputs"

            memory_rows.setdefault(
                graph_id,
                {
                    "id": graph_id,
                    "video_path": str(item.video_path.resolve()),
                    "clip_path": str(clip_path),
                    "mem_path": str(mem_path),
                    # Official scripts currently read this spelling.
                    "intermediate_outputs": str(intermediate_path),
                    "dataset": dataset,
                    "observation_end_s": observation_end_s,
                    "clip_duration_s": args.clip_duration_s,
                    "alignment_class": "official_model_adapted_benchmarks",
                },
            )

            group = controls[dataset].setdefault(
                graph_id,
                {
                    "video_path": str(item.video_path.resolve()),
                    "mem_path": str(mem_path),
                    "qa_list": [],
                },
            )
            options = _question_options(item.question)
            group["qa_list"].append(
                {
                    "question_id": str(item.question.get("question_id") or item.example_id),
                    "question": question,
                    "answer": answer,
                    "type": [str(item.question.get("question_type") or item.task_family)],
                    "timestamp_s": observation_end_s,
                    "source_example_id": item.example_id,
                    "answer_format": item.question.get("answer_format"),
                    "gold_label": (item.question.get("answer") or {}).get("label"),
                    "options": options,
                    # The graph itself is materialized only to this causal cutoff.
                    # Do not additionally truncate it by a 30-second clip index.
                    "causal_graph_is_question_prefix": observation_end_s is not None,
                }
            )

    memory_list = list(memory_rows.values())
    _write_jsonl(output_dir / "memorization_inputs.jsonl", memory_list)
    _write_jsonl(output_dir / "clip_plan.jsonl", memory_list)
    for dataset, records in controls.items():
        (output_dir / f"control_{dataset}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "alignment_class": "official_model_adapted_benchmarks",
        "datasets": list(args.datasets),
        "graphs": len(memory_list),
        "questions": {
            dataset: sum(len(record["qa_list"]) for record in records.values())
            for dataset, records in controls.items()
        },
        "skipped": len(skipped),
        "output_dir": str(output_dir),
        "artifact_root": str(args.artifact_root.resolve()),
        "causal_policy": (
            "VideoMME uses one full-video graph; OVO-Bench and StreamingBench "
            "use one question-prefix graph per distinct video/cutoff."
        ),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "skipped.jsonl", skipped)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/home/xwu/atomic_skills_for_video"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/is_data/xwu/video_skills/data/datasets"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit-per-dataset", type=int, default=1)
    parser.add_argument("--clip-duration-s", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/m3_agent_adapted/artifacts"
        ),
    )
    args = parser.parse_args()

    if args.limit_per_dataset is not None and args.limit_per_dataset < 0:
        args.limit_per_dataset = None
    if args.clip_duration_s <= 0:
        parser.error("--clip-duration-s must be positive")
    ensure_repo_on_path(args.repo_root)
    summary = build_manifests(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

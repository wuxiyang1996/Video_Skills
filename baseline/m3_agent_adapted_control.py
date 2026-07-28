#!/usr/bin/env python3
"""Run official M3-Agent Control logic on adapted local benchmarks.

The official Control checkpoint, prompt, decoding, retrieval rounds, and graph
search are retained. Hosted embedding and GPT judging are replaced by local
embedding plus deterministic benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .m3_local_backends import install_into_m3


SYSTEM_PROMPT = (
    "You are given a question and some relevant knowledge. Your task is to reason about "
    "whether the provided knowledge is sufficient to answer the question. If it is "
    "sufficient, output [Answer] followed by the answer. If it is not sufficient, output "
    "[Search] and generate a query that will be encoded into embeddings for a vector "
    "similarity search. The query will help retrieve additional information from a memory "
    "bank.\n\nQuestion: {question}"
)
INSTRUCTION = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query that would help retrieve the missing information. The search {content} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{i}>" or "What is the character id of {name}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {content} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{i}>."""
ACTION_PATTERN = re.compile(r"Action:\s*\[(.*?)\].*?Content:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _dump_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _parse_label(response: str, options: list[dict[str, str]]) -> str | None:
    labels = {str(option["label"]).upper() for option in options}
    stripped = response.strip().upper()
    if stripped in labels:
        return stripped
    for pattern in (
        r"\b(?:ANSWER|OPTION)\s*[:=]?\s*([A-Z])\b",
        r"^\s*([A-Z])[\).:\s]",
        r"\b([A-Z])[\).]",
    ):
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match and match.group(1).upper() in labels:
            return match.group(1).upper()
    lowered = response.lower()
    matches = [
        str(option["label"]).upper()
        for option in options
        if str(option.get("text") or "").strip().lower() in lowered
    ]
    return matches[0] if len(matches) == 1 else None


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.lower(), flags=re.UNICODE))


def _load_examples(
    path: Path,
    shard_index: int,
    num_shards: int,
    *,
    selected_graph_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for graph_id, graph in payload.items():
        if selected_graph_id is not None and graph_id != selected_graph_id:
            continue
        for qa in graph["qa_list"]:
            if ordinal % num_shards == shard_index:
                rows.append(
                    {
                        "id": qa["question_id"],
                        "graph_id": graph_id,
                        "mem_path": graph["mem_path"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "gold_label": qa.get("gold_label"),
                        "options": qa.get("options") or [],
                        "answer_format": qa.get("answer_format"),
                        "source_example_id": qa.get("source_example_id"),
                    }
                )
            ordinal += 1
    return rows[:limit] if limit is not None else rows


def _existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line).get("id")))
    return ids


def _summarize(records_path: Path) -> dict[str, Any]:
    rows = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    ok = [row for row in rows if row.get("ok")]
    mcq = [row for row in rows if row.get("metric_type") == "multiple_choice"]
    parsed = [row for row in mcq if row.get("prediction_label")]
    correct = [row for row in mcq if row.get("correct") is True]
    text = [row for row in rows if row.get("metric_type") == "normalized_exact_text"]
    text_correct = [row for row in text if row.get("correct") is True]
    return {
        "total": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "multiple_choice": {
            "total": len(mcq),
            "parsed": len(parsed),
            "correct": len(correct),
            "accuracy": len(correct) / len(mcq) if mcq else None,
            "accuracy_on_parsed": len(correct) / len(parsed) if parsed else None,
        },
        "oracle_cutoff_open_text": {
            "total": len(text),
            "exact_correct": len(text_correct),
            "normalized_exact_match": len(text_correct) / len(text) if text else None,
            "official_streamingbench_trigger_metric": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-repo", type=Path, default=Path("/mnt/is_data/xwu/video_skills/code/m3-agent"))
    parser.add_argument("--speakerlab-repo", type=Path, default=Path("/mnt/is_data/xwu/video_skills/code/3D-Speaker"))
    parser.add_argument("--model", type=Path, default=Path("/mnt/is_data/xwu/video_skills/models/M3-Agent-Control"))
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--total-rounds", type=int, default=5)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--retrieval-threshold", type=float, default=0.5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--graph-id")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo = args.m3_repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(args.speakerlab_repo.resolve()))
    os.environ["M3_LIGHTWEIGHT_PACKAGE"] = "1"
    backend = install_into_m3()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import mmagent.videograph
    from mmagent.retrieve import search
    from mmagent.utils.general import load_video_graph

    sys.modules["videograph"] = mmagent.videograph
    examples = _load_examples(
        args.data_file.resolve(),
        args.shard_index,
        args.num_shards,
        selected_graph_id=args.graph_id,
        limit=args.limit,
    )
    records_path = args.output_dir / f"records_{args.shard_index:04d}.jsonl"
    metrics_path = args.output_dir / f"metrics_{args.shard_index:04d}.json"
    done = _existing_ids(records_path)
    examples = [row for row in examples if str(row["id"]) not in done]

    run_config = {
        "alignment_class": "official_model_adapted_benchmarks",
        "upstream_logic": "m3_agent/control.py",
        "model": str(args.model.resolve()),
        "data_file": str(args.data_file.resolve()),
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "graph_id_filter": args.graph_id,
        "limit": args.limit,
        "official_control": {
            "total_rounds": args.total_rounds,
            "topk": args.topk,
            "retrieval_threshold": args.retrieval_threshold,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 1024,
        },
        "deviations": {
            "embedding": backend,
            "judge": "deterministic MCQ label / normalized exact text; no GPT-4o",
            "streamingbench_open_text": "oracle-cutoff QA only; not proactive trigger evaluation",
        },
    }
    (args.output_dir / f"run_config_{args.shard_index:04d}.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024)
    model = LLM(
        model=str(args.model.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    mode = "a" if records_path.exists() else "w"
    with records_path.open(mode, encoding="utf-8") as handle:
        for batch_start in range(0, len(examples), args.batch_size):
            batch = examples[batch_start : batch_start + args.batch_size]
            states = [
                {
                    **row,
                    "conversations": [
                        {"role": "system", "content": SYSTEM_PROMPT.format(question=row["question"])},
                        {"role": "user", "content": "Searched knowledge: {}"},
                    ],
                    "finish": False,
                    "current_clips": [],
                    "trace": [],
                }
                for row in batch
            ]
            started = time.perf_counter()
            try:
                for round_index in range(args.total_rounds):
                    active = [state for state in states if not state["finish"]]
                    prompts = []
                    for state in active:
                        content = state["conversations"][-1]["content"] + INSTRUCTION
                        if round_index == args.total_rounds - 1:
                            content += (
                                "\n(The Action of this round must be [Answer]. If there is "
                                "insufficient information, you can make reasonable guesses.)"
                            )
                        state["conversations"][-1]["content"] = content
                        token_ids = tokenizer.apply_chat_template(
                            state["conversations"],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        prompts.append({"prompt_token_ids": token_ids})
                    if not prompts:
                        break
                    outputs = model.generate(prompts=prompts, sampling_params=sampling, use_tqdm=False)
                    for state, output in zip(active, outputs, strict=True):
                        generated = output.outputs[0].text
                        state["conversations"].append({"role": "assistant", "content": generated})
                        match = ACTION_PATTERN.search(generated.split("</think>")[-1])
                        action = match.group(1).strip().lower() if match else "search"
                        content = match.group(2).strip() if match else ""
                        trace = {"round": round_index, "action": action, "content": content}
                        if action == "answer":
                            state["response"] = content
                            state["finish"] = True
                        else:
                            memories = {}
                            if content:
                                graph = load_video_graph(state["mem_path"])
                                graph.refresh_equivalences()
                                if "character id" in content.lower():
                                    memories, _, _ = search(
                                        graph, content, [], mem_wise=True, topk=20
                                    )
                                else:
                                    memories, current_clips, _ = search(
                                        graph,
                                        content,
                                        state["current_clips"],
                                        threshold=args.retrieval_threshold,
                                        topk=args.topk,
                                    )
                                    state["current_clips"] = current_clips
                            trace["retrieved"] = len(memories)
                            state["trace"].append(trace)
                            result = "Searched knowledge: " + json.dumps(memories, ensure_ascii=False)
                            if not memories:
                                result += (
                                    "\n(The search result is empty. Please try searching from "
                                    "another perspective.)"
                                )
                            state["conversations"].append({"role": "user", "content": result})

                elapsed = time.perf_counter() - started
                for state in states:
                    response = str(state.get("response") or "")
                    options = state["options"]
                    prediction = _parse_label(response, options) if options else None
                    if options:
                        correct = bool(prediction and state["gold_label"] and prediction == state["gold_label"])
                        metric_type = "multiple_choice"
                    else:
                        correct = _normalize_text(response) == _normalize_text(state["answer"])
                        metric_type = "normalized_exact_text"
                    _dump_jsonl(
                        handle,
                        {
                            "id": state["id"],
                            "graph_id": state["graph_id"],
                            "source_example_id": state["source_example_id"],
                            "ok": bool(response),
                            "response": response,
                            "gold": state["answer"],
                            "gold_label": state["gold_label"],
                            "prediction_label": prediction,
                            "correct": correct,
                            "metric_type": metric_type,
                            "trace": state["trace"],
                            "batch_elapsed_s": elapsed,
                        },
                    )
            except Exception as exc:
                for state in states:
                    _dump_jsonl(
                        handle,
                        {
                            "id": state["id"],
                            "graph_id": state["graph_id"],
                            "source_example_id": state["source_example_id"],
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "metric_type": (
                                "multiple_choice" if state["options"] else "normalized_exact_text"
                            ),
                        },
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = _summarize(records_path)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

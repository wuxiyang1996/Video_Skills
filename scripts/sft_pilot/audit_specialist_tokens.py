#!/usr/bin/env python3
"""Audit five specialist SFT datasets with the actual Qwen tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from dataset_clip_wrapper.training.sft_common import read_jsonl, write_json
from dataset_clip_wrapper.training.train_lora_sft import _encode_chat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = {}
    for name in ("l1", "l2", "verifier", "repair", "motif"):
        rows = read_jsonl(args.data_root / name / "all_sft.jsonl")
        lengths = []
        failures = []
        for row in rows:
            try:
                lengths.append(len(_encode_chat(tokenizer, row, args.max_length)["input_ids"]))
            except Exception as exc:  # report every incompatible row together
                failures.append({
                    "id": row.get("transition_id") or row.get("demo_id"),
                    "error": str(exc),
                })
        ordered = sorted(lengths)
        report[name] = {
            "rows": len(rows),
            "encoded": len(lengths),
            "failures": failures,
            "token_min": min(lengths) if lengths else 0,
            "token_p50": ordered[len(ordered) // 2] if ordered else 0,
            "token_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0,
            "token_max": max(lengths) if lengths else 0,
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    report["all_pass_16k"] = all(
        not item["failures"] and item["token_max"] <= args.max_length
        for item in report.values()
        if isinstance(item, dict)
    )
    write_json(args.data_root / "token_audit.json", report)
    print(json.dumps({"all_pass_16k": report["all_pass_16k"]}), flush=True)
    return 0 if report["all_pass_16k"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

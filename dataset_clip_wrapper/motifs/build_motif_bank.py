#!/usr/bin/env python3
"""CLI for mining accepted L1/L2 rollouts into a motif bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import MotifAgent, MotifAgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable motif bank from accepted L1/L2 artifacts.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Final acceptance JSON or expert-demo JSONL files.")
    parser.add_argument("--output-bank", required=True, help="Output motif bank JSONL.")
    parser.add_argument("--summary-output", help="Optional summary JSON path.")
    parser.add_argument(
        "--agent-mode",
        choices=("hybrid", "llm", "deterministic"),
        default="hybrid",
        help="hybrid uses Qwen/GPT-OSS when available and deterministic seed fallback otherwise.",
    )
    parser.add_argument("--extractor-model", default="qwen/qwen3.5", help="OpenRouter model for motif proposal.")
    parser.add_argument("--curator-model", default="openai/gpt-oss-120b", help="OpenRouter model for motif curation.")
    parser.add_argument("--keys-py-path", help="Optional Python file containing OPENROUTER_API_KEY.")
    parser.add_argument("--llm-timeout-s", type=int, default=180)
    parser.add_argument("--max-rows", type=int, help="Optional cap for debugging LLM runs.")
    parser.add_argument("--min-support-count", type=int, default=2)
    parser.add_argument("--min-verifier-pass-rate", type=float, default=0.8)
    parser.add_argument("--min-dataset-coverage", type=int, default=1)
    args = parser.parse_args()

    summary = MotifAgent(
        MotifAgentConfig(
            input_paths=tuple(Path(path) for path in args.inputs),
            output_bank=Path(args.output_bank),
            summary_output=Path(args.summary_output) if args.summary_output else None,
            agent_mode=args.agent_mode,
            extractor_model=args.extractor_model,
            curator_model=args.curator_model,
            keys_py_path=args.keys_py_path,
            llm_timeout_s=args.llm_timeout_s,
            max_rows=args.max_rows,
            min_support_count=args.min_support_count,
            min_verifier_pass_rate=args.min_verifier_pass_rate,
            min_dataset_coverage=args.min_dataset_coverage,
        )
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

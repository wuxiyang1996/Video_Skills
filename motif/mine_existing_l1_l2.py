"""CLI for mining motif candidates from existing L1/L2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .miner import mine_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON/JSONL L1/L2 output files")
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=2)
    args = parser.parse_args()

    result = mine_paths(args.inputs, min_support=args.min_support)
    result.bank.save_jsonl(args.output_bank)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

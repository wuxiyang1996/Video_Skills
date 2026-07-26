"""Check soft-logit teacher calibration gates from OPD collect_summary.json.

When collect used ranking fallback (``teacher_mode_used`` starts with
``ranking_``), soft shuffle L1/top1 gates do not apply; the run passes if
distill rows were produced. Soft attempt metrics remain in
``soft_calibration_gates`` / ``soft_order_shuffle`` for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def evaluate_gates(summary: dict) -> dict:
    shuffle = summary.get("order_shuffle") or {}
    top1 = float(shuffle.get("top1_match_rate") or 0.0)
    mean_l1 = float(shuffle.get("mean_l1") or 1.0)
    n_rows = int(summary.get("n_distill_rows") or 0)
    mode_used = str(summary.get("teacher_mode_used") or "soft_logprob")
    ranking_fallback = mode_used.startswith("ranking_")
    soft_gates = {
        "n_rows_ge_1": n_rows >= 1,
        "top1_match_ge_0_90": top1 >= 0.90,
        "mean_l1_le_0_15": mean_l1 <= 0.15,
    }
    if ranking_fallback:
        gates = {"n_rows_ge_1": n_rows >= 1, "ranking_fallback_accepted": True}
        passed = n_rows >= 1
    else:
        gates = soft_gates
        passed = all(gates.values())
    return {
        "passed": passed,
        "gates": gates,
        "soft_logit_gates": soft_gates,
        "teacher_mode_used": mode_used,
        "fallback_reason": summary.get("fallback_reason"),
        "metrics": {
            "n_distill_rows": n_rows,
            "top1_match_rate": top1,
            "mean_l1": mean_l1,
            "soft_order_shuffle": summary.get("soft_order_shuffle"),
            "soft_calibration_gates": summary.get("soft_calibration_gates"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    # Allow `python -m scripts.opd.gate_teacher_calibration` and direct path.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    result = evaluate_gates(summary)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    # Ensure repo root import when run as a file.
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    raise SystemExit(main())

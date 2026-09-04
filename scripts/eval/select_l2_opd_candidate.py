#!/usr/bin/env python3
"""Select the lowest-strength OPD adapter that passes the frozen dev gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.artifact_hash import adapter_weight_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Comma-separated alpha,adapter_dir,gate_json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for value in args.candidate:
        alpha_text, adapter_text, gate_text = value.split(",", 2)
        adapter, gate_path = Path(adapter_text), Path(gate_text)
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        candidates.append({
            "alpha": float(alpha_text),
            "adapter": str(adapter),
            "adapter_weight_sha256": adapter_weight_sha256(adapter),
            "gate": str(gate_path),
            "passed": bool(gate.get("passed")),
            "checks": gate.get("checks") or {},
            "gains": gate.get("gains") or {},
        })
    passed = sorted(
        (row for row in candidates if row["passed"]), key=lambda row: row["alpha"]
    )
    report = {
        "schema_version": "video-skills/l2-opd-candidate-selection-v0.1",
        "selection_rule": "minimum_alpha_among_frozen_dev_gate_passes",
        "passed": bool(passed),
        "selected": passed[0] if passed else None,
        "candidates": sorted(candidates, key=lambda row: row["alpha"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

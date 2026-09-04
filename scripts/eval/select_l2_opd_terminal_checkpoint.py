#!/usr/bin/env python3
"""Select the smallest pointwise-qualified OPD checkpoint that passes terminal gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def select_terminal_checkpoint(
    pointwise: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    pointwise_by_name = {
        str(row["name"]): row for row in pointwise.get("candidates", [])
    }
    audited: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: float(row["alpha"])):
        name = str(candidate["name"])
        pw = pointwise_by_name.get(name)
        terminal = candidate["terminal"]
        cg_gate = candidate["cg_gate"]
        vh_gate = candidate["vh_gate"]
        expected_hash = None if pw is None else pw.get("adapter_weight_sha256")
        checks = {
            "present_in_pointwise_selection": pw is not None,
            "pointwise_passed": bool(pw and pw.get("passed") is True),
            "alpha_matches_pointwise": bool(
                pw and float(pw.get("alpha")) == float(candidate["alpha"])
            ),
            "adapter_matches_pointwise": bool(
                pw and str(pw.get("adapter")) == str(candidate["adapter"])
            ),
            "terminal_source_hash_matches": bool(
                expected_hash
                and terminal.get("source_adapter_weight_sha256") == expected_hash
            ),
            "terminal_source_adapter_matches": (
                terminal.get("source_adapter") == str(candidate["adapter"])
            ),
            "cg_gate_dataset": cg_gate.get("dataset") == "cg_bench",
            "vh_gate_dataset": vh_gate.get("dataset") == "video_holmes",
            "cg_terminal_passed": cg_gate.get("passed") is True,
            "vh_terminal_passed": vh_gate.get("passed") is True,
        }
        audited.append(
            {
                "name": name,
                "alpha": float(candidate["alpha"]),
                "adapter": str(candidate["adapter"]),
                "adapter_weight_sha256": expected_hash,
                "terminal_report": candidate["terminal_report"],
                "cg_gate_report": candidate["cg_gate_report"],
                "vh_gate_report": candidate["vh_gate_report"],
                "terminal_success_rate": terminal.get("terminal_success_rate"),
                "terminal_reward_contract": terminal.get("terminal_reward_contract"),
                "dataset_metrics": terminal.get("dataset_metrics"),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passing = [row for row in audited if row["passed"]]
    return {
        "schema_version": "video-skills/l2-opd-terminal-checkpoint-selection-v1",
        "passed": bool(passing),
        "selection_rule": "smallest-alpha-passing-pointwise-and-both-terminal-gates",
        "pointwise_selection_passed": pointwise.get("passed") is True,
        "candidates": audited,
        "selected": passing[0] if passing else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointwise-selection", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="name|alpha|adapter|terminal_report|cg_gate|vh_gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for raw in args.candidate:
        parts = raw.split("|", 5)
        if len(parts) != 6:
            parser.error(f"invalid --candidate: {raw!r}")
        name, alpha, adapter, terminal_path, cg_path, vh_path = parts
        rows.append(
            {
                "name": name,
                "alpha": float(alpha),
                "adapter": adapter,
                "terminal_report": terminal_path,
                "terminal": _load(Path(terminal_path)),
                "cg_gate_report": cg_path,
                "cg_gate": _load(Path(cg_path)),
                "vh_gate_report": vh_path,
                "vh_gate": _load(Path(vh_path)),
            }
        )
    report = select_terminal_checkpoint(_load(args.pointwise_selection), rows)
    report["pointwise_selection_report"] = str(args.pointwise_selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

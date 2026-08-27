#!/usr/bin/env python3
"""Post-gate artifact + policy verification for five-LoRA SFT pilots.

Checks adapters/reports exist, re-evaluates warm-up gates, and emits next-step
recommendations (e.g. L1 full-data substrate before freeze/OPD).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate_lora_sft_gates import evaluate_five_lora_sft_gates
from .sft_common import read_json, write_json

SPECIALISTS = ("l1", "l2", "repair", "verifier", "motif")


def _n_gen_examples(report: dict[str, Any]) -> int:
    examples = report.get("examples")
    if isinstance(examples, list):
        return len(examples)
    if isinstance(examples, int):
        return examples
    return int(report.get("n_examples") or 0)


def verify_pilot_artifacts(
    *,
    pipe_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    pilot_root = pipe_root / "pilot"
    gates_report_path = pipe_root / "gates" / "lora_sft_gates_report.json"
    failures: list[str] = []
    warnings: list[str] = []
    specialists: dict[str, Any] = {}

    for name in SPECIALISTS:
        base = pilot_root / name / "pilot"
        adapter_cfg = base / "adapter" / "adapter_config.json"
        train_report_path = base / "training_report.json"
        gen_report_path = base / "generation_report.json"
        entry: dict[str, Any] = {
            "adapter_present": adapter_cfg.exists(),
            "training_report_present": train_report_path.exists(),
            "generation_report_present": gen_report_path.exists(),
        }
        if not entry["adapter_present"]:
            failures.append(f"{name}: missing adapter")
        if not entry["training_report_present"]:
            failures.append(f"{name}: missing training_report.json")
        if not entry["generation_report_present"]:
            failures.append(f"{name}: missing generation_report.json")

        if train_report_path.exists():
            train = read_json(train_report_path)
            entry["total_steps"] = train.get("total_steps")
            entry["train_rows"] = train.get("train_rows")
            entry["source_train_rows"] = train.get("source_train_rows")
            entry["epochs_completed"] = train.get("epochs_completed")
            entry["json_valid_rate"] = train.get("json_valid_rate")
            entry["action_match_rate"] = train.get("action_match_rate")
            last_loss = train.get("last_train_loss")
            if last_loss is None or not isinstance(last_loss, (int, float)):
                failures.append(f"{name}: invalid last_train_loss")
            elif not (last_loss == last_loss):  # NaN
                failures.append(f"{name}: NaN last_train_loss")
            else:
                entry["last_train_loss"] = float(last_loss)

        if gen_report_path.exists():
            gen = read_json(gen_report_path)
            n_ex = _n_gen_examples(gen)
            entry["n_generation_examples"] = n_ex
            if n_ex and n_ex < 8:
                warnings.append(
                    f"{name}: thin generation sample n={n_ex}; treat gate pass as warm-up only"
                )

        specialists[name] = entry

    # Re-run gates from collected reports if present; otherwise from pilot copies.
    reports_root = pipe_root / "gates" / "reports"
    if not reports_root.exists() or not any(reports_root.iterdir()):
        reports_root.mkdir(parents=True, exist_ok=True)
        for name in SPECIALISTS:
            dest = reports_root / name
            dest.mkdir(parents=True, exist_ok=True)
            for src_name, dst_name in (
                ("generation_report.json", "generation_report.json"),
                ("training_report.json", "train_metrics.json"),
            ):
                src = pilot_root / name / "pilot" / src_name
                if src.exists():
                    (dest / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            base_src = pipe_root / "baselines" / name / "base_generation_report.json"
            if base_src.exists():
                (dest / "base_generation_report.json").write_text(
                    base_src.read_text(encoding="utf-8"), encoding="utf-8"
                )

    gates = evaluate_five_lora_sft_gates(reports_root=reports_root, package_root=package_root)
    if not gates.get("passed"):
        failures.append(f"sft_gates_failed:{','.join(gates.get('failed_specialists') or [])}")
    for item in gates.get("specialists") or []:
        for w in item.get("warnings") or []:
            warnings.append(f"{item.get('specialist')}: {w}")

    l1 = specialists.get("l1") or {}
    source_rows = int(l1.get("source_train_rows") or 0)
    train_rows = int(l1.get("train_rows") or 0)
    l1_ready = bool(l1.get("adapter_present") and l1.get("training_report_present"))
    l1_capped = l1_ready and source_rows > 0 and train_rows > 0 and train_rows < source_rows
    next_actions: list[str] = []
    if not l1_ready:
        next_actions.append("Wait for L1 pilot to finish before substrate / freeze decisions")
    elif l1_capped:
        warnings.append(
            f"l1: pilot used capped data ({train_rows}/{source_rows}); "
            "not yet a freeze-quality substrate"
        )
        next_actions.append(
            "Submit L1 substrate: L1_FULL=1 EPOCHS=1 SPECIALISTS=l1 "
            "(full data, 1 epoch) before freezing L1 for OPD/RL"
        )
    else:
        next_actions.append("L1 looks full-data; freeze adapter as substrate after gate pass")

    next_actions.extend(
        [
            "Do not start OPD until Motif online retrieve/expand/fallback is wired",
            "OPD covers L2/Repair/Verifier/Motif only; L1 stays frozen this round",
            "Functional metrics (repair conversion, motif utility, terminal success) are OPD/RL gates",
        ]
    )

    report = {
        "schema_version": "video-skills/sft-pilot-verify-v1",
        "pipe_root": str(pipe_root),
        "package_root": str(package_root),
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "specialists": specialists,
        "gates": {
            "passed": gates.get("passed"),
            "failed_specialists": gates.get("failed_specialists"),
            "path": str(gates_report_path),
        },
        "l1_capped_pilot": l1_capped,
        "next_actions": next_actions,
    }
    if gates_report_path.exists():
        report["gates"]["on_disk_passed"] = bool(read_json(gates_report_path).get("passed"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = verify_pilot_artifacts(pipe_root=args.pipe_root, package_root=args.package_root)
    write_json(args.output, report)
    print(json.dumps({"event": "verify_complete", "passed": report["passed"], "output": str(args.output)}, ensure_ascii=False))
    for action in report.get("next_actions") or []:
        print(json.dumps({"event": "next_action", "action": action}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

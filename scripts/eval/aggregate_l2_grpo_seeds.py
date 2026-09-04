#!/usr/bin/env python3
"""Aggregate three routed L2 GRPO seeds with fail-closed paper gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.eval.select_l2_opd_checkpoint import metrics as pointwise_metrics


DATASETS = ("cg_bench", "video_holmes")
REQUIRED_TRAINING_CONTRACTS = (
    "controller_action_contract",
    "sampling_protocol",
    "relationship_support_contract",
    "reference_runtime_contract",
    "terminal_reward_contract",
    "training_data_contract",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "values": values,
    }


def aggregate_seed_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audited = []
    metric_values: dict[str, list[float]] = {}
    contracts = []
    for row in sorted(rows, key=lambda value: int(value["seed"])):
        train, terminal = row["train"], row["terminal"]
        trained = (train.get("trained_adapter_outputs") or {}).get("default") or {}
        trained_hash = trained.get("adapter_weight_sha256")
        cg_train = (train.get("dataset_metrics") or {}).get("cg_bench") or {}
        vh_train = (train.get("dataset_metrics") or {}).get("video_holmes") or {}
        cg_groups_seen = int(cg_train.get("groups_seen") or 0)
        vh_groups_seen = int(vh_train.get("groups_seen") or 0)
        cg_dev = pointwise_metrics(row["cg_dev"], dataset="cg_bench")
        vh_dev = pointwise_metrics(row["vh_dev"], dataset="video_holmes")
        terminal_metrics = terminal.get("dataset_metrics") or {}
        checks = {
            "trained_artifact": train.get("artifact_status") == "trained" and bool(trained_hash),
            "saw_balanced_50_to_100_groups_per_dataset": (
                50 <= cg_groups_seen <= 100
                and cg_groups_seen == vh_groups_seen
                and int(train.get("groups_seen") or 0) == cg_groups_seen + vh_groups_seen
            ),
            "cg_trainable_group_rate_at_least_25pct": float(
                cg_train.get("trainable_group_rate") or 0.0
            ) >= 0.25,
            "vh_trainable_group_rate_at_least_25pct": float(
                vh_train.get("trainable_group_rate") or 0.0
            ) >= 0.25,
            "optimizer_updated_both_datasets": (
                int(cg_train.get("groups_trained") or 0) > 0
                and int(vh_train.get("groups_trained") or 0) > 0
            ),
            "pointwise_preservation_gate": row["pointwise_gate"].get("passed") is True,
            "cg_terminal_gate": row["cg_gate"].get("passed") is True,
            "vh_terminal_gate": row["vh_gate"].get("passed") is True,
            "cg_dev_hash_matches_trained": bool(trained_hash)
            and row["cg_dev"].get("adapter_weight_sha256") == trained_hash,
            "vh_dev_hash_matches_trained": bool(trained_hash)
            and row["vh_dev"].get("adapter_weight_sha256") == trained_hash,
            "terminal_hash_matches_trained": bool(trained_hash)
            and terminal.get("source_adapter_weight_sha256") == trained_hash,
        }
        values = {
            "cg_pointwise_recall_at_2": cg_dev.get("mean_recall", 0.0),
            "cg_pointwise_hit_rate": cg_dev.get("hit_rate", 0.0),
            "vh_segment_recall": vh_dev.get("segment_recall", 0.0),
            "vh_inference_shot_recall": vh_dev.get("inference_shot_recall", 0.0),
            "vh_relationship_support": vh_dev.get("relationship_support", 0.0),
            "cg_terminal_success_rate": float(
                (terminal_metrics.get("cg_bench") or {}).get("terminal_success_rate") or 0.0
            ),
            "vh_terminal_success_rate": float(
                (terminal_metrics.get("video_holmes") or {}).get("terminal_success_rate") or 0.0
            ),
            "cg_trainable_group_rate": float(cg_train.get("trainable_group_rate") or 0.0),
            "vh_trainable_group_rate": float(vh_train.get("trainable_group_rate") or 0.0),
        }
        for name, value in values.items():
            metric_values.setdefault(name, []).append(float(value))
        contract = {
            "controller_action_contract": train.get("controller_action_contract"),
            "sampling_protocol": train.get("sampling_protocol"),
            "relationship_support_contract": train.get("relationship_support_contract"),
            "reference_runtime_contract": train.get("reference_runtime_contract"),
            "terminal_reward_contract": train.get("terminal_reward_contract"),
            "training_data_contract": {
                "split_role": train.get("split_role"),
                "split_manifest_sha256": train.get("split_manifest_sha256"),
                "allowlist_sha256": (train.get("pool_filters") or {}).get(
                    "example_id_allowlist_sha256"
                ),
                "exact_mined_group_allowlist": (train.get("pool_filters") or {}).get(
                    "exact_mined_group_allowlist"
                ),
                "preserve_allowlist_order": (train.get("pool_filters") or {}).get(
                    "preserve_allowlist_order"
                ),
                "dataset_balanced_sampling": (train.get("pool_filters") or {}).get(
                    "dataset_balanced_sampling"
                ),
            },
        }
        contracts.append(contract)
        audited.append(
            {
                "seed": int(row["seed"]),
                "train_report": row["train_report"],
                "terminal_report": row["terminal_report"],
                "trained_adapter": trained,
                "training_dataset_metrics": {"cg_bench": cg_train, "video_holmes": vh_train},
                "dev_metrics": values,
                "contracts": contract,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    complete_contracts = all(
        all(contract.get(name) is not None for name in REQUIRED_TRAINING_CONTRACTS)
        and all((contract.get("training_data_contract") or {}).values())
        for contract in contracts
    )
    same_contracts = (
        bool(contracts)
        and complete_contracts
        and all(value == contracts[0] for value in contracts[1:])
    )
    return {
        "schema_version": "video-skills/l2-grpo-three-seed-aggregate-v1",
        "passed": len(audited) == 3 and all(row["passed"] for row in audited) and same_contracts,
        "seed_count": len(audited),
        "same_training_contracts": same_contracts,
        "seeds": audited,
        "metrics": {name: _mean_std(values) for name, values in sorted(metric_values.items())},
    }


def _load_or_missing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        required=True,
        help="SEED|TRAIN|CG_DEV|VH_DEV|TERMINAL|POINTWISE_GATE|CG_GATE|VH_GATE",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for raw in args.seed:
        parts = raw.split("|", 7)
        if len(parts) != 8:
            parser.error(f"invalid --seed: {raw!r}")
        seed, train, cg_dev, vh_dev, terminal, pointwise_gate, cg_gate, vh_gate = parts
        paths = {name: Path(value) for name, value in {
            "train": train, "cg_dev": cg_dev, "vh_dev": vh_dev,
            "terminal": terminal, "pointwise_gate": pointwise_gate,
            "cg_gate": cg_gate, "vh_gate": vh_gate,
        }.items()}
        rows.append({
            "seed": int(seed),
            **{name: _load_or_missing(path) for name, path in paths.items()},
            "train_report": train,
            "terminal_report": terminal,
        })
    report = aggregate_seed_runs(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

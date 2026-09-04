#!/usr/bin/env python3
"""Gate an OPD terminal executor report against the matched SFT baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def gate_terminal_reports(
    sft: dict[str, Any],
    opd: dict[str, Any],
    *,
    dataset: str,
    min_samples: int = 10,
    min_success_rate: float = 0.10,
    allowed_rate_regression: float = 0.05,
) -> dict[str, Any]:
    base = sft["dataset_metrics"][dataset]
    new = opd["dataset_metrics"][dataset]

    def rate(metrics: dict[str, Any], name: str, fallback: float = 0.0) -> float:
        return float(metrics.get(name, fallback))

    def same_required(name: str) -> bool:
        """Compare protocol fields fail-closed instead of accepting two missing values."""
        return name in sft and name in opd and sft[name] == opd[name]

    def normalized_pool_filters(report: dict[str, Any]) -> dict[str, Any] | None:
        filters = report.get("pool_filters")
        if not isinstance(filters, dict):
            return None
        # This field was added after the frozen SFT baseline was produced.  Its
        # absent legacy value and an explicit false both mean the ordinary
        # example allowlist contract; treating them as different makes a schema
        # migration look like a population mismatch.
        normalized = dict(filters)
        normalized.setdefault("exact_mined_group_allowlist", False)
        return normalized

    def normalized_eval_sampling_protocol(
        report: dict[str, Any],
    ) -> dict[str, Any] | None:
        protocol = report.get("sampling_protocol")
        if not isinstance(protocol, dict):
            return None
        # These fields describe only how gradients are computed after rollout
        # collection.  They were added after the frozen eval baseline was
        # produced and cannot change an eval-only rollout.  Keep every other
        # field fail-closed so a new generation/sampling option cannot be
        # silently ignored by the matched-protocol gate.
        training_only_fields = {
            "pointwise_gradient_contract",
            "pointwise_train_batch_size",
        }
        return {
            key: value
            for key, value in protocol.items()
            if key not in training_only_fields
        }

    group_protocol_fields = (
        "groups_seen",
        "unique_pool_examples_before_repeats",
        "repeats_per_example",
        "repeat_start_index",
    )

    checks = {
        # A metric comparison is meaningful only when both reports were produced
        # from the same frozen dev population and rollout protocol.  Previously,
        # two reports with different allowlists or balancing settings could pass.
        "same_split_role": same_required("split_role"),
        "same_pool_filters": (
            normalized_pool_filters(sft) is not None
            and normalized_pool_filters(sft) == normalized_pool_filters(opd)
        ),
        "same_group_protocol": all(
            same_required(name) for name in group_protocol_fields
        ),
        "same_boundary_anchor_contract": same_required("boundary_anchor_index0"),
        "same_eval_mode": (
            sft.get("eval_only") is True
            and opd.get("eval_only") is True
            and same_required("retrieval_only")
            and same_required("terminal_on_process_hit")
        ),
        "same_remote_execution_policy": (
            sft.get("mock_semantic_judge") is False
            and opd.get("mock_semantic_judge") is False
            and sft.get("remote_rollout_policy") is False
            and opd.get("remote_rollout_policy") is False
            and sft.get("fixed_remote_environment_executor") is True
            and opd.get("fixed_remote_environment_executor") is True
        ),
        "same_controller_action_contract": (
            sft.get("controller_action_contract") == opd.get("controller_action_contract")
        ),
        "same_sampling_protocol": (
            normalized_eval_sampling_protocol(sft) is not None
            and normalized_eval_sampling_protocol(sft)
            == normalized_eval_sampling_protocol(opd)
        ),
        "same_relationship_support_contract": (
            sft.get("relationship_support_contract")
            == opd.get("relationship_support_contract")
        ),
        "same_terminal_reward_contract": same_required("terminal_reward_contract"),
        "same_executor_isolation_contract": (
            sft.get("executor_isolation_contract") == opd.get("executor_isolation_contract")
        ),
        "same_executor_fallback_contract": (
            sft.get("executor_fallback_contract") == opd.get("executor_fallback_contract")
        ),
        "same_dataset_executor_backends": (
            sft.get("dataset_executor_backends") == opd.get("dataset_executor_backends")
        ),
        "same_executor_cache_contract": (
            sft.get("executor_cache_contract") == opd.get("executor_cache_contract")
        ),
        "same_executor_cache_dir": sft.get("executor_cache_dir") == opd.get("executor_cache_dir"),
        "sft_has_min_samples": int(base.get("samples") or 0) >= min_samples,
        "opd_has_min_samples": int(new.get("samples") or 0) >= min_samples,
        "same_dataset_sample_count": int(base.get("samples") or 0) == int(new.get("samples") or 0),
        "opd_terminal_not_below_sft": rate(new, "terminal_success_rate") >= rate(base, "terminal_success_rate"),
        "opd_terminal_reaches_unlock_rate": rate(new, "terminal_success_rate") >= min_success_rate,
        "valid_action_not_regressed": rate(new, "valid_retrieval_action_rate") + allowed_rate_regression >= rate(base, "valid_retrieval_action_rate"),
        "verifier_not_regressed": rate(new, "verifier_pass_rate") + allowed_rate_regression >= rate(base, "verifier_pass_rate"),
    }
    if "format_compliance_rate" in base and "format_compliance_rate" in new:
        checks["format_not_regressed"] = (
            rate(new, "format_compliance_rate") + allowed_rate_regression
            >= rate(base, "format_compliance_rate")
        )
    return {
        "schema_version": "video-skills/l2-terminal-gate-v0.1",
        "dataset": dataset,
        "passed": all(checks.values()),
        "checks": checks,
        "sft": base,
        "opd": new,
        "thresholds": {
            "min_samples": min_samples,
            "min_success_rate": min_success_rate,
            "allowed_rate_regression": allowed_rate_regression,
        },
        "controller_action_contract": {
            "sft": sft.get("controller_action_contract"),
            "opd": opd.get("controller_action_contract"),
        },
        "executor_isolation_contract": {
            "sft": sft.get("executor_isolation_contract"),
            "opd": opd.get("executor_isolation_contract"),
        },
        "executor_fallback_contract": {
            "sft": sft.get("executor_fallback_contract"),
            "opd": opd.get("executor_fallback_contract"),
        },
        "dataset_executor_backends": {
            "sft": sft.get("dataset_executor_backends"),
            "opd": opd.get("dataset_executor_backends"),
        },
        "executor_cache_contract": {
            "sft": sft.get("executor_cache_contract"),
            "opd": opd.get("executor_cache_contract"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--opd", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-success-rate", type=float, default=0.10)
    parser.add_argument("--allowed-rate-regression", type=float, default=0.05)
    args = parser.parse_args()
    report = gate_terminal_reports(
        json.loads(args.sft.read_text()), json.loads(args.opd.read_text()),
        dataset=args.dataset, min_samples=args.min_samples,
        min_success_rate=args.min_success_rate,
        allowed_rate_regression=args.allowed_rate_regression,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

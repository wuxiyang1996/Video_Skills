"""Promotion gates for mined motif candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import MotifBank, MotifRecord


@dataclass(frozen=True)
class PromotionConfig:
    min_support_count: int = 2
    min_verifier_pass_rate: float = 0.8
    min_dataset_coverage: int = 1
    require_expansion_template: bool = True


def promotion_failures(record: MotifRecord, config: PromotionConfig) -> list[str]:
    failures: list[str] = []
    if record.support_count < config.min_support_count:
        failures.append("support_count_below_threshold")
    if record.verifier_pass_rate < config.min_verifier_pass_rate:
        failures.append("verifier_pass_rate_below_threshold")
    if len(record.datasets_seen) < config.min_dataset_coverage:
        failures.append("dataset_coverage_below_threshold")
    if config.require_expansion_template and not record.expansion_template:
        failures.append("missing_expansion_template")
    return failures


def apply_promotion_gates(bank: MotifBank, config: PromotionConfig) -> dict[str, int]:
    counts = {"promoted": 0, "candidate": 0}
    for record in bank.records:
        failures = promotion_failures(record, config)
        if failures:
            record.status = "candidate"
            record.notes = [f"promotion_blocked:{failure}" for failure in failures]
            counts["candidate"] += 1
        else:
            record.status = "promoted"
            record.notes = ["promotion_passed"]
            counts["promoted"] += 1
    return counts

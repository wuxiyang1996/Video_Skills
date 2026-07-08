"""Transfer evaluation adapter for motif promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .schemas import MotifRecord, MotifTransferReport


@dataclass(frozen=True)
class MotifTransferExample:
    dataset: str
    example_id: str
    task_family: str
    payload: dict


@dataclass(frozen=True)
class MotifEvalResult:
    answer_correct: bool
    verifier_passed: bool
    evidence_valid: bool
    no_hidden_leakage: bool

    @property
    def success(self) -> bool:
        return (
            self.answer_correct
            and self.verifier_passed
            and self.evidence_valid
            and self.no_hidden_leakage
        )


RunFn = Callable[[MotifTransferExample, MotifRecord | None], MotifEvalResult]


class MotifTransferAdapter:
    """Compare L1/L2-only runs against L1/L2 plus a motif prior."""

    def __init__(self, run_fn: RunFn) -> None:
        self.run_fn = run_fn

    def evaluate(
        self,
        motif: MotifRecord,
        examples: Iterable[MotifTransferExample],
    ) -> MotifTransferReport:
        items = list(examples)
        if not items:
            return MotifTransferReport(
                target_dataset="",
                target_task_family="",
                notes=["no_examples"],
            )

        baseline_successes = 0
        motif_successes = 0
        verifier_successes = 0
        evidence_successes = 0
        leakage_successes = 0

        for example in items:
            baseline = self.run_fn(example, None)
            with_motif = self.run_fn(example, motif)
            baseline_successes += int(baseline.success)
            motif_successes += int(with_motif.success)
            verifier_successes += int(with_motif.verifier_passed)
            evidence_successes += int(with_motif.evidence_valid)
            leakage_successes += int(with_motif.no_hidden_leakage)

        n_total = len(items)
        return MotifTransferReport(
            target_dataset=items[0].dataset,
            target_task_family=items[0].task_family,
            n_total=n_total,
            n_success=motif_successes,
            baseline_success_rate=baseline_successes / n_total,
            motif_success_rate=motif_successes / n_total,
            verifier_pass_rate=verifier_successes / n_total,
            evidence_valid_rate=evidence_successes / n_total,
            no_leakage_rate=leakage_successes / n_total,
        )

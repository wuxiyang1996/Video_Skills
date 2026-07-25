"""Lexicographic verified reward for GRPO (plan §6).

Dictionary order only:

1. hard feasible
2. terminal success
3. verified atomic progress (milestone vector)
4. evidence tie-break
5. cost tie-break (lower better)

Never mix into a free scalar. Motif lifecycle / teacher preference / skill-call
counts are not reward channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .milestone_ledger import MilestoneLedger

REWARD_SPEC_VERSION = "video-skills/verified-reward-v2"

_STRONG_PREFIXES = ("accepted_strong", "resolved_strong")


@dataclass(frozen=True)
class VerifiedRewardBreakdown:
    spec_version: str
    hard_feasible: bool
    terminal_success: bool
    verified_atomic_progress: tuple[int, ...]
    progress_total: int
    evidence_checks: int
    cost_total: int
    rank_key: tuple
    hard_failures: tuple[str, ...] = ()
    blocked_strong_commit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "hard_feasible": self.hard_feasible,
            "terminal_success": self.terminal_success,
            "verified_atomic_progress": list(self.verified_atomic_progress),
            "progress_total": self.progress_total,
            "evidence_checks": self.evidence_checks,
            "cost_total": self.cost_total,
            "rank_key": list(self.rank_key),
            "hard_failures": list(self.hard_failures),
            "blocked_strong_commit": self.blocked_strong_commit,
        }


def _is_strong(acceptance_status: str) -> bool:
    status = (acceptance_status or "").strip()
    return any(status.startswith(p) for p in _STRONG_PREFIXES)


def _normalize_progress(
    progress: Sequence[int] | Mapping[str, int] | MilestoneLedger | None,
) -> tuple[int, ...]:
    if progress is None:
        return (0, 0, 0, 0, 0)
    if isinstance(progress, MilestoneLedger):
        return progress.progress_vector()
    if isinstance(progress, Mapping):
        from .milestone_ledger import MILESTONE_KINDS

        return tuple(int(progress.get(k, 0)) for k in MILESTONE_KINDS)
    values = tuple(max(int(x), 0) for x in progress)
    if len(values) < 5:
        values = values + (0,) * (5 - len(values))
    return values[:5]


def score_verified_rollout(
    *,
    answer_correct: bool,
    acceptance_status: str,
    schema_valid: bool = True,
    skill_allowed: bool = True,
    refs_exist: bool = True,
    no_hidden_leakage: bool = True,
    streaming_visibility_ok: bool = True,
    within_hard_budget: bool = True,
    unanswerable: bool = False,
    abstained: bool = False,
    commit_evidence_ok: bool = False,
    non_diagnostic_visual_ok: bool = False,
    claim_support_hard_ok: bool = False,
    clip_reads: int = 0,
    tool_calls: int = 0,
    tokens: int = 0,
    repair_rounds: int = 0,
    verified_atomic_progress: Sequence[int] | Mapping[str, int] | MilestoneLedger | None = None,
    blocked_strong_commit: bool = False,
) -> VerifiedRewardBreakdown:
    hard_flags = {
        "schema_valid": schema_valid,
        "skill_allowed": skill_allowed,
        "refs_exist": refs_exist,
        "no_hidden_leakage": no_hidden_leakage,
        "streaming_visibility_ok": streaming_visibility_ok,
        "within_hard_budget": within_hard_budget,
    }
    hard_failures = tuple(name for name, ok in hard_flags.items() if not ok)
    feasible = not hard_failures

    strong = _is_strong(acceptance_status) and not blocked_strong_commit
    if not feasible:
        success = False
    elif unanswerable:
        success = bool(abstained and strong)
    elif abstained:
        success = False
    else:
        success = bool(answer_correct and strong)

    progress_vec = _normalize_progress(verified_atomic_progress) if feasible else (0, 0, 0, 0, 0)
    progress_total = int(sum(progress_vec))

    if feasible:
        evidence_checks = (
            int(commit_evidence_ok)
            + int(non_diagnostic_visual_ok)
            + int(claim_support_hard_ok)
        )
    else:
        evidence_checks = 0

    cost_total = (
        max(int(clip_reads), 0)
        + max(int(tool_calls), 0)
        + max(int(tokens), 0)
        + max(int(repair_rounds), 0)
    )

    rank_key = (
        int(feasible),
        int(success),
        progress_vec,
        int(evidence_checks),
        -int(cost_total),
    )
    return VerifiedRewardBreakdown(
        spec_version=REWARD_SPEC_VERSION,
        hard_feasible=feasible,
        terminal_success=success,
        verified_atomic_progress=progress_vec,
        progress_total=progress_total,
        evidence_checks=evidence_checks,
        cost_total=cost_total,
        rank_key=rank_key,
        hard_failures=hard_failures,
        blocked_strong_commit=blocked_strong_commit,
    )


def group_rank_advantages(scores: Sequence[VerifiedRewardBreakdown]) -> list[float]:
    """Dense rank advantage in [-1, 1] from ``rank_key`` only."""
    if not scores:
        return []
    if len(scores) == 1:
        return [0.0]
    order = sorted(range(len(scores)), key=lambda i: scores[i].rank_key)
    n = len(scores)
    advantages = [0.0] * n
    for rank, idx in enumerate(order):
        advantages[idx] = (rank / (n - 1)) * 2.0 - 1.0
    return advantages

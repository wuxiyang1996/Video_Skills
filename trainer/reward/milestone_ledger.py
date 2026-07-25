"""Verified atomic progress ledger (plan §6).

Partial credit is issued only for first-time, grounded state deltas.
Skill invocation alone never scores. Unused / contradicted milestones are revoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

MILESTONE_KINDS = (
    "retrieval",
    "inference",
    "compose",
    "verify",
    "repair",
)

# Per-episode caps prevent milestone farming.
DEFAULT_CAPS: dict[str, int] = {
    "retrieval": 4,
    "inference": 4,
    "compose": 3,
    "verify": 3,
    "repair": 2,
}

REWARD_MILESTONE_SPEC = "video-skills/verified-atomic-progress-v1"


@dataclass
class Milestone:
    kind: str
    key: str
    step_index: int
    detail: dict[str, Any] = field(default_factory=dict)
    used_by_final: bool = False
    revoked: bool = False
    revoke_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "step_index": self.step_index,
            "detail": self.detail,
            "used_by_final": self.used_by_final,
            "revoked": self.revoked,
            "revoke_reason": self.revoke_reason,
        }


class MilestoneLedger:
    """Episode-scoped ledger with dedup, caps, and end-of-episode revocation."""

    def __init__(self, *, caps: Mapping[str, int] | None = None) -> None:
        self.caps = {k: int(caps.get(k, DEFAULT_CAPS[k])) for k in MILESTONE_KINDS} if caps else dict(DEFAULT_CAPS)
        self._by_key: dict[str, Milestone] = {}
        self.spec_version = REWARD_MILESTONE_SPEC

    def _active_count(self, kind: str) -> int:
        return sum(1 for m in self._by_key.values() if m.kind == kind and not m.revoked)

    def try_add(
        self,
        *,
        kind: str,
        key: str,
        step_index: int,
        detail: Mapping[str, Any] | None = None,
        grounded: bool = True,
    ) -> Milestone | None:
        """Add a milestone if grounded, novel, and under cap. Returns None if rejected."""
        if kind not in MILESTONE_KINDS:
            raise ValueError(f"unknown milestone kind: {kind}")
        if not grounded:
            return None
        dedup_key = f"{kind}:{key}"
        if dedup_key in self._by_key:
            return None
        if self._active_count(kind) >= self.caps[kind]:
            return None
        milestone = Milestone(
            kind=kind,
            key=dedup_key,
            step_index=int(step_index),
            detail=dict(detail or {}),
        )
        self._by_key[dedup_key] = milestone
        return milestone

    def mark_used(self, keys: Iterable[str]) -> None:
        for raw in keys:
            key = str(raw)
            # Accept either full "kind:..." keys or bare suffixes.
            matched = [self._by_key[k] for k in self._by_key if k == key or k.endswith(f":{key}")]
            for m in matched:
                if not m.revoked:
                    m.used_by_final = True

    def revoke(self, key: str, *, reason: str) -> bool:
        m = self._by_key.get(key)
        if m is None:
            # Try suffix match.
            for full, cand in self._by_key.items():
                if full.endswith(f":{key}"):
                    m = cand
                    break
        if m is None or m.revoked:
            return False
        m.revoked = True
        m.revoke_reason = reason
        m.used_by_final = False
        return True

    def finalize(
        self,
        *,
        final_used_keys: Sequence[str] | None = None,
        contradicted_keys: Sequence[str] | None = None,
        revoke_unused: bool = True,
    ) -> None:
        if contradicted_keys:
            for key in contradicted_keys:
                self.revoke(key, reason="contradicted")
        if final_used_keys is not None:
            self.mark_used(final_used_keys)
        if revoke_unused:
            for m in self._by_key.values():
                if not m.revoked and not m.used_by_final:
                    m.revoked = True
                    m.revoke_reason = "unused_at_episode_end"

    def active_milestones(self) -> list[Milestone]:
        return [m for m in self._by_key.values() if not m.revoked]

    def progress_counts(self) -> dict[str, int]:
        counts = {k: 0 for k in MILESTONE_KINDS}
        for m in self.active_milestones():
            counts[m.kind] += 1
        return counts

    def progress_vector(self) -> tuple[int, ...]:
        counts = self.progress_counts()
        return tuple(counts[k] for k in MILESTONE_KINDS)

    def progress_total(self) -> int:
        return int(sum(self.progress_vector()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "caps": dict(self.caps),
            "milestones": [m.to_dict() for m in self._by_key.values()],
            "progress_vector": list(self.progress_vector()),
            "progress_total": self.progress_total(),
        }


def ledger_from_events(
    events: Sequence[Mapping[str, Any]],
    *,
    final_used_keys: Sequence[str] | None = None,
    contradicted_keys: Sequence[str] | None = None,
) -> MilestoneLedger:
    """Build a ledger from structured rollout events.

    Each event may contain:
      kind, key, step_index, grounded (default True), detail
    """
    ledger = MilestoneLedger()
    for i, event in enumerate(events):
        kind = str(event.get("kind") or "")
        key = str(event.get("key") or f"event_{i}")
        if kind not in MILESTONE_KINDS:
            continue
        ledger.try_add(
            kind=kind,
            key=key,
            step_index=int(event.get("step_index", i)),
            detail=event.get("detail") if isinstance(event.get("detail"), Mapping) else {},
            grounded=bool(event.get("grounded", True)),
        )
    ledger.finalize(
        final_used_keys=final_used_keys,
        contradicted_keys=contradicted_keys,
        revoke_unused=True,
    )
    return ledger

"""
Step 6 — Persistent Skill Bank for effects-only contracts (MVP).

Stores ``SkillEffectsContract`` and ``VerificationReport`` per skill with
versioning and JSONL persistence.  Provides a compact summary for debugging
and downstream integration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from skill_agents.stage3_mvp.schemas import (
    SkillEffectsContract,
    VerificationReport,
)


def _effects_compat_score(
    contract: SkillEffectsContract,
    predicates_start: Optional[dict],
    predicates_end: Optional[dict],
    p_thresh: float = 0.5,
    missing_penalty: float = -0.5,
    contradiction_penalty: float = -1.0,
) -> float:
    """Compute effects-based compatibility between a contract and observed predicates.

    This is the core closed-loop signal from Stage 3 → Stage 2.  It rewards
    segments whose observed predicate changes match the skill's learned effects
    and penalises mismatches.

    Parameters
    ----------
    contract : SkillEffectsContract
        The skill's effects contract (eff_add, eff_del, eff_event).
    predicates_start, predicates_end : dict or None
        Predicate dicts at segment start/end.  Values are floats in [0, 1]
        (probabilities) or bools.
    p_thresh : float
        Threshold above which a predicate is considered "true".
    missing_penalty : float
        Score contribution when an expected effect literal is absent.
    contradiction_penalty : float
        Score contribution when an effect is directly contradicted.

    Returns
    -------
    float
        Normalised score in approx. [-1, +1].  0.0 when contract is empty.
    """
    n_clauses = len(contract.eff_add or set()) + len(contract.eff_del or set()) + len(contract.eff_event or set())
    if n_clauses == 0:
        return 0.0

    P_end = predicates_end or {}
    # Convert bools to floats
    P_end_f = {k: (float(v) if not isinstance(v, (float, int)) else float(v))
               for k, v in P_end.items()}

    score = 0.0

    # eff_add: predicate should be true at end
    for lit in (contract.eff_add or set()):
        val = P_end_f.get(lit)
        if val is not None and val >= p_thresh:
            score += 1.0  # match
        elif val is not None and val < p_thresh:
            score += contradiction_penalty  # contradicted
        else:
            score += missing_penalty  # not observed

    # eff_del: predicate should be false at end
    for lit in (contract.eff_del or set()):
        val = P_end_f.get(lit)
        if val is not None and val < p_thresh:
            score += 1.0  # match (deleted = now false)
        elif val is not None and val >= p_thresh:
            score += contradiction_penalty  # still true = contradicted
        else:
            score += missing_penalty  # not observed

    # eff_event: event should have occurred (check if present in end predicates)
    for lit in (contract.eff_event or set()):
        val = P_end_f.get(lit)
        if val is not None and val >= p_thresh:
            score += 1.0
        else:
            score += missing_penalty

    return score / n_clauses


class SkillBankMVP:
    """Persistent skill bank for effects-only contracts.

    Each skill maps to a ``SkillEffectsContract`` and an optional
    ``VerificationReport``.  All mutations are logged for reproducibility.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._contracts: Dict[str, SkillEffectsContract] = {}
        self._reports: Dict[str, VerificationReport] = {}
        self._history: List[dict] = []
        self._path = path

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def skill_ids(self) -> List[str]:
        return list(self._contracts.keys())

    def get_contract(self, skill_id: str) -> Optional[SkillEffectsContract]:
        return self._contracts.get(skill_id)

    def get_report(self, skill_id: str) -> Optional[VerificationReport]:
        return self._reports.get(skill_id)

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._contracts

    def __len__(self) -> int:
        return len(self._contracts)

    # ── Mutations ────────────────────────────────────────────────────

    def add_or_update(
        self,
        contract: SkillEffectsContract,
        report: Optional[VerificationReport] = None,
    ) -> None:
        """Store or update a skill's contract and report."""
        self._contracts[contract.skill_id] = contract
        if report is not None:
            self._reports[contract.skill_id] = report
        self._log("add_or_update", contract.skill_id, contract.version)

    def remove(self, skill_id: str) -> None:
        self._contracts.pop(skill_id, None)
        self._reports.pop(skill_id, None)
        self._log("remove", skill_id, -1)

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, filepath: Optional[str] = None) -> None:
        """Save bank to JSONL (one contract per line)."""
        path = Path(filepath or self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for skill_id, contract in self._contracts.items():
                entry = {
                    "contract": contract.to_dict(),
                    "report": (
                        self._reports[skill_id].to_dict()
                        if skill_id in self._reports
                        else None
                    ),
                }
                f.write(json.dumps(entry, default=str) + "\n")

    def load(self, filepath: Optional[str] = None) -> None:
        """Load bank from JSONL."""
        path = Path(filepath or self._path)
        if not path.exists():
            return
        self._contracts.clear()
        self._reports.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                contract = SkillEffectsContract.from_dict(entry["contract"])
                self._contracts[contract.skill_id] = contract
                if entry.get("report"):
                    self._reports[contract.skill_id] = VerificationReport.from_dict(
                        entry["report"]
                    )

    # ── Stage 2 integration ─────────────────────────────────────────

    def compat_fn(
        self,
        skill: str,
        predicates_start: Optional[dict],
        predicates_end: Optional[dict],
    ) -> float:
        """Effects-based contract compatibility scorer for ``SegmentScorer``.

        Plug into Stage 2::

            scorer = SegmentScorer(..., compat_fn=bank.compat_fn)

        Scoring logic (effects-only, easy to extend to preconditions later):
          +1 for each expected eff_add literal observed true at segment end
          +1 for each expected eff_del literal observed false at segment end
          -0.5 for each expected eff_add literal *missing* at segment end
          -1.0 for each *contradictory* effect (eff_add false, eff_del true)

        The raw score is normalized to [-1, +1] by the number of contract
        literals.  Returns 0.0 when the skill has no contract or no effects.
        """
        contract = self._contracts.get(skill)
        if contract is None:
            return 0.0
        return _effects_compat_score(contract, predicates_start, predicates_end)

    def get_skill_names(self) -> List[str]:
        """Active skill names for Stage 2 pipeline input."""
        return list(self._contracts.keys())

    # ── Summary / debug ─────────────────────────────────────────────

    def summary(self) -> Dict[str, dict]:
        """Compact per-skill summary for logging / inspection."""
        result: Dict[str, dict] = {}
        for skill_id, contract in self._contracts.items():
            info: dict = {
                "version": contract.version,
                "n_eff_add": len(contract.eff_add),
                "n_eff_del": len(contract.eff_del),
                "n_eff_event": len(contract.eff_event),
                "total_literals": contract.total_literals,
                "n_instances": contract.n_instances,
            }
            if skill_id in self._reports:
                info["pass_rate"] = self._reports[skill_id].overall_pass_rate
            result[skill_id] = info
        return result

    # ── History ──────────────────────────────────────────────────────

    def _log(self, event: str, skill_id: str, version: int) -> None:
        self._history.append({
            "event": event,
            "skill_id": skill_id,
            "version": version,
            "timestamp": time.time(),
        })

    @property
    def history(self) -> List[dict]:
        return list(self._history)

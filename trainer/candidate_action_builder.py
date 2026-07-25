"""Build complete-action candidate sets for OPD (schema-valid JSON actions)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


L2_SCHEMA = "video-skills/l2-specialist-action-v0.1"
MOTIF_SCHEMA = "video-skills/motif-online-action-v0.1"


@dataclass
class CandidateAction:
    action_id: str
    family: str
    action: dict[str, Any]
    is_stop: bool = False
    is_abstain: bool = False
    is_fallback: bool = False
    is_hard_negative: bool = False
    schema_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "family": self.family,
            "action": self.action,
            "is_stop": self.is_stop,
            "is_abstain": self.is_abstain,
            "is_fallback": self.is_fallback,
            "is_hard_negative": self.is_hard_negative,
            "schema_ok": self.schema_ok,
            "notes": self.notes,
        }


@dataclass
class CandidateActionSet:
    state_id: str
    candidates: list[CandidateAction]
    oracle_action_id: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_recall(self) -> float | None:
        if self.oracle_action_id is None:
            return None
        ids = {c.action_id for c in self.candidates}
        return 1.0 if self.oracle_action_id in ids else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "oracle_action_id": self.oracle_action_id,
            "candidate_recall": self.candidate_recall,
            "coverage": self.coverage,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _l2_action(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": L2_SCHEMA,
        "tool_name": tool_name,
        "arguments": arguments,
    }


def _validate_l2(action: dict[str, Any]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if action.get("schema_version") != L2_SCHEMA:
        notes.append("bad_schema_version")
    tool = action.get("tool_name")
    if not isinstance(tool, str) or not tool.strip():
        notes.append("missing_tool_name")
    args = action.get("arguments")
    if not isinstance(args, dict):
        notes.append("arguments_not_object")
    return (len(notes) == 0), notes


def build_l2_candidate_actions(
    *,
    state_id: str,
    student_action: dict[str, Any] | None = None,
    oracle_action: dict[str, Any] | None = None,
    coarse_indices: Sequence[int] | None = None,
    include_motif_fallback: bool = True,
    max_candidates: int = 8,
) -> CandidateActionSet:
    """Construct 4–8 complete L2/Motif actions including STOP/abstain/fallback + hard neg."""
    candidates: list[CandidateAction] = []
    indices = list(coarse_indices or [0, 1, 2])

    def _add(
        action_id: str,
        family: str,
        action: dict[str, Any],
        **flags: Any,
    ) -> None:
        ok, notes = _validate_l2(action) if action.get("schema_version") == L2_SCHEMA else (True, [])
        if action.get("schema_version") == MOTIF_SCHEMA:
            if not action.get("tool_name"):
                ok, notes = False, ["missing_motif_tool"]
        candidates.append(
            CandidateAction(
                action_id=action_id,
                family=family,
                action=action,
                schema_ok=ok,
                notes=notes,
                **flags,
            )
        )

    if student_action:
        _add("student", str(student_action.get("tool_name") or "student"), student_action)

    if oracle_action and oracle_action != student_action:
        _add("oracle", str(oracle_action.get("tool_name") or "oracle"), oracle_action)

    if indices:
        _add(
            "select_next",
            "select_next_coarse_clip",
            _l2_action("select_next_coarse_clip", {"coarse_index": int(indices[0])}),
        )
        _add(
            "choose_best",
            "choose_best_coarse_candidate",
            _l2_action("choose_best_coarse_candidate", {"coarse_index": int(indices[min(1, len(indices) - 1)])}),
        )

    _add(
        "stop",
        "stop_coarse_retrieval",
        _l2_action("stop_coarse_retrieval", {"reason": "budget_or_enough_evidence"}),
        is_stop=True,
    )
    _add(
        "abstain",
        "reject_commit_and_retrieve_more",
        _l2_action("reject_commit_and_retrieve_more", {"reason": "insufficient_evidence"}),
        is_abstain=True,
    )
    if include_motif_fallback:
        _add(
            "motif_fallback",
            "motif_fallback_to_l2",
            {
                "schema_version": MOTIF_SCHEMA,
                "tool_name": "motif_fallback_to_l2",
                "arguments": {"reason": "expansion_invalid_or_no_candidates"},
            },
            is_fallback=True,
        )

    # Hard negative: valid schema but wrong/absurd coarse index.
    bad_index = 10_000
    _add(
        "hard_neg",
        "select_next_coarse_clip",
        _l2_action("select_next_coarse_clip", {"coarse_index": bad_index}),
        is_hard_negative=True,
    )

    # Deduplicate by JSON dumps of action body, keep first.
    seen: set[str] = set()
    unique: list[CandidateAction] = []
    for cand in candidates:
        key = json.dumps(cand.action, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
        if len(unique) >= max_candidates:
            break

    oracle_id = None
    if oracle_action is not None:
        oracle_key = json.dumps(oracle_action, sort_keys=True, ensure_ascii=False)
        for cand in unique:
            if json.dumps(cand.action, sort_keys=True, ensure_ascii=False) == oracle_key:
                oracle_id = cand.action_id
                break
        if oracle_id is None:
            # Ensure oracle is present for candidate_recall gate.
            ok, notes = _validate_l2(oracle_action)
            unique.insert(
                0,
                CandidateAction(
                    action_id="oracle",
                    family=str(oracle_action.get("tool_name") or "oracle"),
                    action=oracle_action,
                    schema_ok=ok,
                    notes=notes,
                ),
            )
            oracle_id = "oracle"
            unique = unique[:max_candidates]

    families = sorted({c.family for c in unique})
    coverage = {
        "n_candidates": len(unique),
        "n_schema_ok": sum(1 for c in unique if c.schema_ok),
        "has_stop": any(c.is_stop for c in unique),
        "has_abstain": any(c.is_abstain for c in unique),
        "has_fallback": any(c.is_fallback for c in unique),
        "has_hard_negative": any(c.is_hard_negative for c in unique),
        "families": families,
        "family_coverage_ok": len(families) >= 3,
        "min_candidates_ok": 4 <= len(unique) <= max_candidates,
    }
    return CandidateActionSet(
        state_id=state_id,
        candidates=unique,
        oracle_action_id=oracle_id,
        coverage=coverage,
    )


def gate_candidate_set(action_set: CandidateActionSet) -> dict[str, Any]:
    """OPD preflight: candidate_recall + coverage before spending teacher calls."""
    failures: list[str] = []
    cov = action_set.coverage or {}
    if not cov.get("min_candidates_ok"):
        failures.append("candidate_count_out_of_range")
    if not cov.get("has_stop"):
        failures.append("missing_stop")
    if not cov.get("has_abstain"):
        failures.append("missing_abstain")
    if not cov.get("has_hard_negative"):
        failures.append("missing_hard_negative")
    if not cov.get("family_coverage_ok"):
        failures.append("family_coverage_insufficient")
    if action_set.oracle_action_id is not None and action_set.candidate_recall != 1.0:
        failures.append("oracle_not_in_candidates")
    if any(not c.schema_ok for c in action_set.candidates):
        failures.append("schema_invalid_candidate_present")
    return {
        "passed": not failures,
        "failures": failures,
        "candidate_recall": action_set.candidate_recall,
        "coverage": cov,
    }

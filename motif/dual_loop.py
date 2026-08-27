"""Dual Motif loop: accelerate (VERIFIED/ACTIVE) + failure→repair→mine CANDIDATE.

Accelerate path never consumes candidate/shadow for planner skip.
Repair path may retrieve shadow/candidate as priors.
Mining writes CANDIDATE only after verified terminal success with repair contribution;
never auto-promotes (GRPO/OPD reward must not depend on lifecycle labels).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .bank import MotifBank
from .lifecycle import MotifLifecycleManager
from .online_expand import expand_motif_record
from .retrieval import MotifQueryEngine
from .schemas import MotifEvidenceRef, MotifLifecycleStatus, MotifRecord


def empty_dual_loop_meta() -> dict[str, Any]:
    return {
        "motif_phase": "none",
        "repair_retrieval_attempted": False,
        "repair_candidate_ids": [],
        "repair_selected_motif_id": None,
        "repair_expansion_valid": False,
        "repair_fallback_reason": None,
        "candidate_mined": False,
        "mined_motif_id": None,
        "mined_skill_sequence": [],
    }


@dataclass
class RepairMotifPhaseResult:
    meta_updates: dict[str, Any] = field(default_factory=dict)
    reasoning_plan: list[dict[str, Any]] = field(default_factory=list)
    skill_sequence: list[str] = field(default_factory=list)
    selected_motif_id: str | None = None
    used_repair_motif: bool = False


@dataclass
class MineCandidateResult:
    mined: bool
    motif_id: str | None = None
    record: MotifRecord | None = None
    sink_path: str | None = None
    reason: str | None = None


def _fault_query_suffix(faults: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for fault in faults[:4]:
        if not isinstance(fault, dict):
            continue
        parts.append(str(fault.get("fault_type") or ""))
        parts.append(str(fault.get("failure_code") or ""))
        parts.append(str(fault.get("skill_id") or ""))
        parts.append(str(fault.get("repair_strategy") or ""))
    return " ".join(p for p in parts if p)


def select_repair_motif(
    *,
    bank_path: str | Path,
    question: dict[str, Any] | None,
    task_family: str,
    dataset: str,
    faults: Sequence[dict[str, Any]],
    exclude_motif_ids: Sequence[str] | None = None,
    top_k: int = 3,
) -> RepairMotifPhaseResult:
    """Retrieve+expand a repair-phase motif (may include shadow/candidate)."""
    meta = {
        "motif_phase": "repair",
        "repair_retrieval_attempted": True,
        "repair_candidate_ids": [],
        "repair_selected_motif_id": None,
        "repair_expansion_valid": False,
        "repair_fallback_reason": None,
    }
    path = Path(bank_path)
    if not path.exists():
        meta["repair_fallback_reason"] = "motif_bank_missing"
        return RepairMotifPhaseResult(meta_updates=meta)

    bank = MotifBank.load_jsonl(path)
    query = " ".join(
        part
        for part in (
            str((question or {}).get("question_text") or ""),
            task_family,
            dataset,
            "repair",
            _fault_query_suffix(faults),
        )
        if part
    ).strip()
    engine = MotifQueryEngine(bank)
    selections = engine.select(
        query=query or "repair video_qa",
        task_family=task_family,
        dataset=dataset,
        phase="repair",
        top_k=max(1, int(top_k)),
        exclude_motif_ids=set(exclude_motif_ids or ()),
    )
    meta["repair_candidate_ids"] = [item.motif_id for item in selections]
    if not selections:
        meta["repair_fallback_reason"] = "no_repair_motif_candidates"
        return RepairMotifPhaseResult(meta_updates=meta)

    selected_id = selections[0].motif_id
    meta["repair_selected_motif_id"] = selected_id
    record = bank.get(selected_id)
    if record is None:
        meta["repair_fallback_reason"] = "repair_motif_missing_in_bank"
        return RepairMotifPhaseResult(meta_updates=meta)

    expansion = expand_motif_record(record)
    meta["repair_expansion_valid"] = bool(expansion.expansion_valid)
    if not expansion.expansion_valid or not expansion.reasoning_plan:
        meta["repair_fallback_reason"] = expansion.fallback_reason or "repair_expansion_invalid"
        return RepairMotifPhaseResult(meta_updates=meta)

    return RepairMotifPhaseResult(
        meta_updates=meta,
        reasoning_plan=list(expansion.reasoning_plan),
        skill_sequence=list(expansion.skill_sequence),
        selected_motif_id=selected_id,
        used_repair_motif=True,
    )


def _stable_motif_id(skill_sequence: Sequence[str], *, prefix: str = "l2_repaired") -> str:
    digest = hashlib.sha1(
        json.dumps(list(skill_sequence), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"{prefix}:{digest}"


def build_candidate_from_repaired_sequence(
    *,
    skill_sequence: Sequence[str],
    example: dict[str, Any],
    faults: Sequence[dict[str, Any]] | None = None,
    repair_motif_id: str | None = None,
    source_path: str = "",
) -> MotifRecord:
    """Build a CANDIDATE MotifRecord from a repaired verified skill sequence."""
    seq = [str(s).strip() for s in skill_sequence if str(s).strip()]
    motif_id = _stable_motif_id(seq)
    fault_types = sorted(
        {
            str(f.get("fault_type") or "")
            for f in (faults or [])
            if isinstance(f, dict) and f.get("fault_type")
        }
    )
    record = MotifRecord(
        motif_id=motif_id,
        name=f"Repaired L2 sequence ({len(seq)} steps)",
        description=(
            "Candidate mined after failure→repair→verified success. "
            f"source_repair_motif={repair_motif_id or 'none'}; "
            f"faults={','.join(fault_types) or 'unspecified'}"
        ),
        motif_type="l2_skill_sequence",
        status=MotifLifecycleStatus.CANDIDATE,
        trigger_signature={
            "task_family": str(example.get("task_family") or ""),
            "answer_format": str(
                ((example.get("question") or {}).get("answer_format") or "multiple_choice")
            ),
            "mined_from": "dual_loop_repair",
            "fault_types": fault_types,
        },
        l2_template={
            "skill_sequence": seq,
            "compressed_skill_sequence": seq,
        },
        proposal_source="dual_loop_repair_mine",
        notes=[
            "mined_after_verified_success",
            "not_auto_promoted",
            f"repair_motif_id={repair_motif_id or ''}",
        ],
    )
    record.add_evidence(
        MotifEvidenceRef(
            dataset=str(example.get("dataset") or ""),
            example_id=str(example.get("example_id") or ""),
            task_family=str(example.get("task_family") or ""),
            source_path=source_path,
            verifier_passed=True,
            evidence_valid=True,
            no_hidden_leakage=True,
            final_answer_correct=None,
        )
    )
    return record


def persist_candidate(
    record: MotifRecord,
    *,
    sink_path: str | Path,
) -> MineCandidateResult:
    """Append/upsert CANDIDATE into sink bank. Never promotes."""
    path = Path(sink_path)
    bank = MotifBank.load_jsonl(path) if path.exists() else MotifBank()
    manager = MotifLifecycleManager(bank)
    if record.motif_id in bank:
        existing = bank.require(record.motif_id)
        for ref in record.evidence_refs:
            existing.add_evidence(ref)
        existing.notes.append("dual_loop_support_increment")
        bank.add(existing)
        bank.save_jsonl(path)
        return MineCandidateResult(
            mined=True,
            motif_id=existing.motif_id,
            record=existing,
            sink_path=str(path),
            reason="support_increment",
        )
    manager.add_candidate(record)
    bank.save_jsonl(path)
    return MineCandidateResult(
        mined=True,
        motif_id=record.motif_id,
        record=record,
        sink_path=str(path),
        reason="added_candidate",
    )


def maybe_mine_candidate_after_verified(
    *,
    downstream_verified_success: bool,
    repair_contributed: bool,
    skill_sequence: Sequence[str],
    example: dict[str, Any],
    faults: Sequence[dict[str, Any]] | None = None,
    repair_motif_id: str | None = None,
    candidate_sink_path: str | Path | None = None,
    source_path: str = "",
) -> MineCandidateResult:
    """Mine CANDIDATE only after verified success with repair contribution."""
    if not downstream_verified_success:
        return MineCandidateResult(mined=False, reason="not_verified_success")
    if not repair_contributed:
        return MineCandidateResult(mined=False, reason="repair_did_not_contribute")
    seq = [str(s).strip() for s in skill_sequence if str(s).strip()]
    if not seq:
        return MineCandidateResult(mined=False, reason="empty_skill_sequence")
    record = build_candidate_from_repaired_sequence(
        skill_sequence=seq,
        example=example,
        faults=faults,
        repair_motif_id=repair_motif_id,
        source_path=source_path,
    )
    if candidate_sink_path:
        return persist_candidate(record, sink_path=candidate_sink_path)
    return MineCandidateResult(
        mined=True,
        motif_id=record.motif_id,
        record=record,
        reason="built_not_persisted",
    )

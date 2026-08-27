"""Expand a motif L2 template into an executable reasoning_plan.

Motifs are never executed as black boxes. This helper only instantiates
ordinary reasoning skill steps from ``skill_sequence`` /
``compressed_skill_sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Keep aligned with reasoning_planner.REASONING_SKILL_IDS without importing that
# heavy module at package import time.
ALLOWED_REASONING_SKILL_IDS = frozenset(
    {
        "parse_question_target",
        "propose_evidence_roles",
        "retrieve_by_event",
        "retrieve_by_entity",
        "retrieve_by_time",
        "retrieve_by_relation",
        "localize_clue",
        "extract_claim",
        "assign_evidence_role",
        "generate_answer_hypotheses",
        "retrieve_evidence_for_hypothesis",
        "score_hypothesis_support",
        "compare_hypotheses",
        "bridge_evidence_hops",
        "verify_temporal_social_consistency",
        "compose_evidence_chain",
        "detect_missing_role",
        "search_counterevidence",
        "infer_temporal_relation",
        "infer_state_change",
        "infer_causal_relation",
        "infer_intention_or_motive",
        "infer_social_contradiction",
        "verify_claim_support",
        "commit_answer",
    }
)


def _ref(step_map: dict[str, str], skill_id: str, field: str | None = None) -> str | None:
    step_id = step_map.get(skill_id)
    if not step_id:
        return None
    return f"$step.{step_id}.{field}" if field else f"$step.{step_id}"


def _args_for_skill(skill_id: str, step_map: dict[str, str], prev_step_id: str | None) -> dict[str, Any]:
    """Arg templates keyed by earlier skill steps (not only immediate prev)."""
    parse = _ref(step_map, "parse_question_target", "parsed_target")
    hyps = _ref(step_map, "generate_answer_hypotheses", "hypotheses")
    retrieve = _ref(step_map, "retrieve_evidence_for_hypothesis")
    scored = _ref(step_map, "score_hypothesis_support", "scored_hypotheses")
    best = _ref(step_map, "compare_hypotheses", "best_hypothesis")
    bridge = _ref(step_map, "bridge_evidence_hops", "multi_hop_chain")
    verify = step_map.get("verify_claim_support")

    if skill_id == "parse_question_target":
        return {"question_text": "$bindings.question_text", "options": "$bindings.options"}
    if skill_id == "propose_evidence_roles":
        return {
            "question_text": "$bindings.question_text",
            "parsed_target": parse or "$bindings.question_text",
            "task_family": "$bindings.task_family",
        }
    if skill_id == "generate_answer_hypotheses":
        return {
            "question_text": "$bindings.question_text",
            "options": "$bindings.options",
            "parsed_target": parse,
        }
    if skill_id == "retrieve_evidence_for_hypothesis":
        return {"hypothesis": hyps or [], "max_refs": 6}
    if skill_id == "score_hypothesis_support":
        return {
            "hypothesis": hyps or [],
            "support_evidence": retrieve or (f"$step.{prev_step_id}" if prev_step_id else []),
            "counterevidence": [],
        }
    if skill_id == "compare_hypotheses":
        return {"scored_hypotheses": scored or []}
    if skill_id == "bridge_evidence_hops":
        return {
            "source_evidence": f"{best}.support_refs" if best else [],
            "target_hypothesis": best or {},
            "max_hops": 2,
        }
    if skill_id == "verify_temporal_social_consistency":
        return {
            "evidence_chain": bridge or {"evidence_refs": []},
            "hypothesis": best or {},
        }
    if skill_id == "verify_claim_support":
        chain = bridge or {"evidence_refs": []}
        return {
            "claim": best or {},
            "evidence_chain": chain,
            "support_policy": {"min_evidence_refs": 1},
        }
    if skill_id == "commit_answer":
        return {
            "verified_claim": f"$step.{verify}.verified_claim" if verify else {},
            "options": "$bindings.options",
            "answer_format": "multiple_choice",
            "support_chain": f"$step.{verify}.evidence_chain" if verify else {"evidence_refs": []},
        }
    if skill_id.startswith("retrieve_by_"):
        return {"query": "$bindings.question_text", "max_refs": 6}
    if skill_id == "extract_claim":
        return {"text": "$bindings.question_text", "option_label": None}
    if skill_id == "compose_evidence_chain":
        return {
            "evidence_refs": _ref(step_map, "retrieve_evidence_for_hypothesis", "evidence_refs") or [],
            "items": [],
        }
    return {"question_text": "$bindings.question_text"}


def _depends_for_skill(skill_id: str, step_map: dict[str, str], prev_step_id: str | None) -> list[str]:
    deps: list[str] = []
    need = {
        "propose_evidence_roles": ["parse_question_target"],
        "generate_answer_hypotheses": ["parse_question_target"],
        "retrieve_evidence_for_hypothesis": ["generate_answer_hypotheses"],
        "score_hypothesis_support": ["generate_answer_hypotheses", "retrieve_evidence_for_hypothesis"],
        "compare_hypotheses": ["score_hypothesis_support"],
        "bridge_evidence_hops": ["compare_hypotheses"],
        "verify_temporal_social_consistency": ["bridge_evidence_hops", "compare_hypotheses"],
        "verify_claim_support": ["compare_hypotheses"],
        "commit_answer": ["verify_claim_support"],
        "compose_evidence_chain": ["retrieve_evidence_for_hypothesis"],
    }.get(skill_id, [])
    for skill in need:
        if skill in step_map:
            deps.append(step_map[skill])
    if not deps and prev_step_id:
        deps = [prev_step_id]
    # unique preserve order
    return list(dict.fromkeys(deps))


@dataclass
class MotifExpansionResult:
    expansion_valid: bool
    reasoning_plan: list[dict[str, Any]] = field(default_factory=list)
    skill_sequence: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expansion_valid": self.expansion_valid,
            "reasoning_plan": self.reasoning_plan,
            "skill_sequence": self.skill_sequence,
            "fallback_reason": self.fallback_reason,
            "notes": self.notes,
        }


def extract_skill_sequence(l2_template: dict[str, Any] | None) -> list[str]:
    if not isinstance(l2_template, dict):
        return []
    for key in ("skill_sequence", "compressed_skill_sequence"):
        raw = l2_template.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            seq = [str(x).strip() for x in raw if str(x).strip()]
            if seq:
                return seq
    return []


def expand_skill_sequence_to_plan(
    skill_ids: Iterable[str],
    *,
    allowed_skill_ids: frozenset[str] | None = None,
    max_steps: int = 24,
    truncate_after_first_commit: bool = True,
) -> MotifExpansionResult:
    allowed = allowed_skill_ids or ALLOWED_REASONING_SKILL_IDS
    sequence = [str(s).strip() for s in skill_ids if str(s).strip()]
    if truncate_after_first_commit and "commit_answer" in sequence:
        cut = sequence.index("commit_answer") + 1
        sequence = sequence[:cut]
    if not sequence:
        return MotifExpansionResult(
            expansion_valid=False,
            fallback_reason="empty_skill_sequence",
            notes=["motif l2_template has no skill_sequence"],
        )
    if len(sequence) > max_steps:
        return MotifExpansionResult(
            expansion_valid=False,
            skill_sequence=sequence,
            fallback_reason="skill_sequence_too_long",
            notes=[f"sequence length {len(sequence)} > max_steps {max_steps}"],
        )

    unknown = [s for s in sequence if s not in allowed]
    if unknown:
        return MotifExpansionResult(
            expansion_valid=False,
            skill_sequence=sequence,
            fallback_reason="unknown_skill_id",
            notes=[f"unknown skills: {unknown[:8]}"],
        )

    plan: list[dict[str, Any]] = []
    step_map: dict[str, str] = {}
    prev_step_id: str | None = None
    for index, skill_id in enumerate(sequence, start=1):
        step_id = f"m{index}"
        # Register current skill before building later deps that may need it? 
        # Args reference earlier skills only; register after computing args from prior map.
        args = _args_for_skill(skill_id, step_map, prev_step_id)
        depends_on = _depends_for_skill(skill_id, step_map, prev_step_id)
        plan.append(
            {
                "step_id": step_id,
                "skill_id": skill_id,
                "args": args,
                "depends_on": depends_on,
                "from_motif": True,
            }
        )
        step_map[skill_id] = step_id
        prev_step_id = step_id

    return MotifExpansionResult(
        expansion_valid=True,
        reasoning_plan=plan,
        skill_sequence=sequence,
        notes=["expanded_from_skill_sequence"],
    )


def expand_motif_record(record: Any) -> MotifExpansionResult:
    """Expand a MotifRecord (or dict-like) into a reasoning_plan."""
    if record is None:
        return MotifExpansionResult(expansion_valid=False, fallback_reason="no_motif")
    if hasattr(record, "l2_template"):
        l2_template = getattr(record, "l2_template") or {}
        motif_id = getattr(record, "motif_id", "")
    elif isinstance(record, dict):
        l2_template = record.get("l2_template") or {}
        motif_id = record.get("motif_id") or ""
    else:
        return MotifExpansionResult(
            expansion_valid=False,
            fallback_reason="unsupported_motif_type",
            notes=[type(record).__name__],
        )

    sequence = extract_skill_sequence(l2_template if isinstance(l2_template, dict) else {})
    result = expand_skill_sequence_to_plan(sequence)
    if motif_id:
        result.notes.append(f"motif_id={motif_id}")
    return result

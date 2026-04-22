"""Controller — the 8B orchestrator (v0: rule-based stub).

Implements the controller's four required behaviors from
``infra_plans/03_controller/actors_reasoning_model.md`` §0.3:

- ``analyze_question(q)`` → :class:`QuestionAnalysis`
- ``next_hop(trace)`` → :class:`HopGoal`
- ``select_skill(hop_goal, bank)`` → ``skill_id``
- ``compose_answer(trace)`` → ``str``

V1 is intentionally **rule-based**:

- Question analysis uses keyword heuristics over the question text.
- Hop planning consumes the analysis's ``decomposition`` field plus a per-
  question-type template.
- Skill routing maps ``HopGoal.target_claim_type`` → an ordered preference
  list of skills present in the bank.
- Answer composition is a deterministic templated read of the final claim.

Why a rule-based v0: the harness, retriever, verifier, and bank must be
exercised end-to-end **before** the LLM-mediated controller is plugged in.
Swapping this module for an 8B-backed controller in a later phase requires
only that the new implementation produce the same canonical objects.

Per the project plan, training (SFT → GRPO) consumes the
:class:`ReasoningTrace` objects this controller (and its successor) produce;
the trace shape is invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .contracts import (
    AbstainDecision,
    HopGoal,
    QuestionAnalysis,
    ReasoningTrace,
    RetrievalQuery,
    new_id,
)
from .skills.bank import ReasoningSkillBank


# ---------------------------------------------------------------------------
# Question analysis heuristics
# ---------------------------------------------------------------------------


_TYPE_KEYWORDS: Dict[str, Sequence[str]] = {
    "ordering": ("before", "after", "first", "earlier", "later", "then", "preceded", "followed"),
    "belief": ("know", "knew", "believe", "thought", "aware", "realized", "expect"),
    "causal": ("why", "because", "caused", "made", "led to", "resulted"),
    "presence": ("is in", "present", "appear", "saw", "see", "visible"),
    "state": ("status", "state", "current", "now"),
}

_YES_NO_LEADERS = ("did", "does", "is", "was", "were", "will", "can", "could", "would", "has", "have", "had")


@dataclass
class ControllerConfig:
    max_hops_per_question: int = 6
    min_evidence_for_answer: int = 1
    answer_template: str = "{claim}"


class Controller:
    """Rule-based v0 controller. LLM-backed v1 swaps this whole class."""

    def __init__(self, bank: ReasoningSkillBank, config: Optional[ControllerConfig] = None) -> None:
        self.bank = bank
        self.config = config or ControllerConfig()
        self._blacklist_per_trace: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # Question analysis
    # ------------------------------------------------------------------

    def analyze_question(
        self,
        question_text: str,
        *,
        question_id: Optional[str] = None,
        target_entities: Optional[List[str]] = None,
        time_anchor: Optional[Sequence[float]] = None,
        perspective_anchor: Optional[str] = None,
    ) -> QuestionAnalysis:
        qid = question_id or new_id("q")
        qtype = _classify_question(question_text)
        expected = "yes_no" if _is_yes_no(question_text) else (
            "entity" if qtype == "presence" else "free_text"
        )
        decomp = _decompose(question_text, qtype, target_entities or [])
        return QuestionAnalysis(
            question_id=qid,
            question_text=question_text,
            question_type=qtype,
            target_entities=list(target_entities or []),
            time_anchor=tuple(time_anchor) if time_anchor else None,  # type: ignore[arg-type]
            perspective_anchor=perspective_anchor,
            expected_answer_type=expected,
            decomposition=decomp,
        )

    # ------------------------------------------------------------------
    # Hop planning
    # ------------------------------------------------------------------

    def next_hop(self, trace: ReasoningTrace) -> Optional[HopGoal]:
        qa = trace.question_analysis
        n_done = len(trace.hops)
        if n_done >= self.config.max_hops_per_question:
            return None
        if n_done >= len(qa.decomposition):
            return None
        sub_goal_text = qa.decomposition[n_done]
        return HopGoal(
            hop_id=new_id("hop"),
            parent_question_id=qa.question_id,
            goal_text=sub_goal_text,
            target_claim_type=_target_claim_type_for_hop(qa, n_done),
            required_entities=list(qa.target_entities),
            required_time_scope=qa.time_anchor,  # type: ignore[arg-type]
            perspective_anchor=qa.perspective_anchor,
            retrieval_hints=[
                RetrievalQuery(
                    query_id=new_id("rq"),
                    text=sub_goal_text,
                    entity_filter=list(qa.target_entities),
                    time_filter=qa.time_anchor,  # type: ignore[arg-type]
                    perspective=qa.perspective_anchor,
                    store_filter="any",
                )
            ],
            success_predicate=f"hop_{n_done + 1}_resolves:{qa.question_type}",
            max_atomic_steps=4,
        )

    # ------------------------------------------------------------------
    # Skill routing
    # ------------------------------------------------------------------

    _ROUTING_PREFERENCES: Dict[str, List[str]] = {
        "ordering": [
            "ground_event_span",
            "order_two_events",
            "decide_answer_or_abstain",
        ],
        "belief": [
            "ground_entity_reference",
            "infer_observation_access",
            "update_belief_state",
            "decide_answer_or_abstain",
        ],
        "causal": [
            "ground_event_span",
            "check_causal_support",
            "decide_answer_or_abstain",
        ],
        "presence": [
            "ground_entity_reference",
            "retrieve_relevant_episode",
            "decide_answer_or_abstain",
        ],
        "state": [
            "retrieve_relevant_episode",
            "check_state_change",
            "decide_answer_or_abstain",
        ],
        "free": [
            "identify_question_target",
            "retrieve_relevant_episode",
            "check_evidence_sufficiency",
            "decide_answer_or_abstain",
        ],
    }

    def select_skill(
        self,
        hop_goal: HopGoal,
        bank: ReasoningSkillBank,
        *,
        trace: Optional[ReasoningTrace] = None,
    ) -> str:
        # Routing is keyed on the question_type (stable across hops), with the
        # hop index advancing through the preference list. target_claim_type on
        # the hop_goal is consumed by the verifier, not the router.
        question_type = (
            trace.question_analysis.question_type
            if trace is not None
            else "free"
        )
        prefs = self._ROUTING_PREFERENCES.get(
            question_type, self._ROUTING_PREFERENCES["free"]
        )
        blacklist = (
            self._blacklist_per_trace.get(trace.trace_id, set())
            if trace is not None
            else set()
        )
        hop_idx = len(trace.hops) if trace is not None else 0
        if hop_idx < len(prefs):
            candidate_name = prefs[hop_idx]
            if bank.has(candidate_name) and candidate_name not in blacklist:
                return bank.get_by_name(candidate_name).skill_id
        # Fall back: any skill from the preference list not blacklisted
        for name in prefs:
            if bank.has(name) and name not in blacklist:
                return bank.get_by_name(name).skill_id
        # Last resort: identify_question_target
        return bank.get_by_name("identify_question_target").skill_id

    def blacklist(self, skill_id: str, *, trace: Optional[ReasoningTrace] = None) -> None:
        if trace is None:
            return
        self._blacklist_per_trace.setdefault(trace.trace_id, set()).add(skill_id)
        skill = self.bank.get(skill_id) if self.bank.has(skill_id) else None
        if skill is not None:
            self._blacklist_per_trace[trace.trace_id].add(skill.name)

    # ------------------------------------------------------------------
    # Answer composition
    # ------------------------------------------------------------------

    def compose_answer(self, trace: ReasoningTrace) -> str:
        # Walk the hops backward; use the last decision step if any
        for hop in reversed(trace.hops):
            for step in reversed(hop.steps):
                if step.output_type == "decision":
                    return self._answer_from_decision(step.output, trace)
        # Else, summarize the last claim-shaped step
        for hop in reversed(trace.hops):
            for step in reversed(hop.steps):
                if step.is_claim() and step.verification.passed:
                    return self._answer_from_claim(step.output, trace)
        return "I cannot determine the answer from the available evidence."

    def _answer_from_decision(self, out: Dict[str, Any], trace: ReasoningTrace) -> str:
        if out.get("decision") == "answer":
            # Use the last passing claim's content
            for hop in reversed(trace.hops):
                for step in reversed(hop.steps):
                    if step.is_claim() and step.verification.passed:
                        return self._answer_from_claim(step.output, trace)
            return f"Answer (score={out.get('score', 0):.2f})."
        return f"Abstain (score={out.get('score', 0):.2f}, threshold={out.get('threshold', 0):.2f})."

    def _answer_from_claim(self, out: Dict[str, Any], trace: ReasoningTrace) -> str:
        qa = trace.question_analysis
        if qa.expected_answer_type == "yes_no":
            for k, v in out.items():
                if isinstance(v, bool):
                    return "Yes." if v else "No."
                if k == "order":
                    return f"Order: {v}."
        # Generic templated answer
        keys = ("entity_id", "canonical_name", "order", "supported", "access", "decision")
        parts = [f"{k}={out[k]}" for k in keys if k in out]
        return "; ".join(parts) if parts else str(out)

    # ------------------------------------------------------------------
    # Final abstain (delegated to the verifier; controller may override)
    # ------------------------------------------------------------------

    def maybe_override_abstain(
        self,
        trace: ReasoningTrace,
        decision: AbstainDecision,
    ) -> AbstainDecision:
        """V1: do not override; later phases may inject controller-level rules."""
        return decision


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _classify_question(text: str) -> str:
    t = text.lower()
    for qtype, keywords in _TYPE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return qtype
    if t.startswith(("who", "which", "what", "when", "where")):
        return "presence" if t.startswith(("who", "which")) else "free"
    if t.startswith("how"):
        return "free"
    return "free"


def _is_yes_no(text: str) -> bool:
    t = text.strip().lower().split(maxsplit=1)
    if not t:
        return False
    return t[0] in _YES_NO_LEADERS


def _decompose(text: str, qtype: str, targets: List[str]) -> List[str]:
    if qtype == "ordering" and len(targets) >= 2:
        return [
            f"ground event span for {targets[0]}",
            f"determine ordering of {targets[0]} and {targets[1]}",
        ]
    if qtype == "belief":
        holder = targets[0] if targets else "agent"
        return [
            f"ground entity reference for {holder}",
            f"infer observation access for {holder}",
            f"update belief state for {holder}",
        ]
    if qtype == "causal":
        return [
            f"ground candidate cause and effect spans",
            f"check causal support",
        ]
    if qtype == "presence":
        return [
            f"ground entity reference for {targets[0] if targets else 'subject'}",
            f"retrieve episodes featuring the subject",
        ]
    if qtype == "state":
        return [
            f"retrieve relevant episode for the queried subject",
            f"check whether the queried state changed",
        ]
    return [
        "identify question target",
        "retrieve relevant episode",
        "check evidence sufficiency and decide",
    ]


def _target_claim_type_for_hop(qa: QuestionAnalysis, hop_idx: int) -> str:
    """Map (question_type, hop_idx) → target_claim_type for that hop."""
    qt = qa.question_type
    if qt == "ordering":
        return "span" if hop_idx == 0 else "ordering"
    if qt == "belief":
        return ("entity_ref", "claim", "belief")[min(hop_idx, 2)]
    if qt == "causal":
        return "span" if hop_idx == 0 else "claim"
    if qt == "presence":
        return "entity_ref" if hop_idx == 0 else "presence"
    if qt == "state":
        return "evidence_set" if hop_idx == 0 else "claim"
    return "claim"

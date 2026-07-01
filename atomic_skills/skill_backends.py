"""Skill backend dispatch: rule vs LLM/VLM depending on context.

Each atomic skill can run in two modes:
- rule: fast, deterministic, no API calls. Good for testing, schema validation,
  and generating skeleton trajectories.
- llm: calls an LLM/VLM for real semantic reasoning. Required for quality
  expert demos and actual QA.

The choice is made per-invocation via SkillBackendConfig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SkillBackendMode(str, Enum):
    RULE = "rule"
    LLM = "llm"
    VLM = "vlm"


@dataclass
class SkillBackendConfig:
    """Controls whether skills use rule-based or LLM-backed execution."""

    default_mode: SkillBackendMode = SkillBackendMode.RULE

    llm_skills: set[str] = field(default_factory=lambda: {
        "infer_causal_relation",
        "infer_temporal_relation",
        "infer_state_change",
        "infer_intention_or_motive",
        "infer_social_contradiction",
        "verify_claim_support",
        "verify_temporal_social_consistency",
        "score_hypothesis_support",
        "compare_hypotheses",
        "localize_clue",
        "extract_claim",
        "assign_evidence_role",
        "bridge_evidence_hops",
    })

    rule_only_skills: set[str] = field(default_factory=lambda: {
        "parse_question_target",
        "propose_evidence_roles",
        "generate_answer_hypotheses",
        "compose_evidence_chain",
        "detect_missing_role",
        "commit_answer",
        "segment_video_or_select_clip",
        "link_graph_relation",
    })

    retrieval_skills: set[str] = field(default_factory=lambda: {
        "retrieve_by_event",
        "retrieve_by_entity",
        "retrieve_by_time",
        "retrieve_by_relation",
        "retrieve_evidence_for_hypothesis",
        "search_counterevidence",
    })

    vlm_skills: set[str] = field(default_factory=lambda: {
        "extract_observation",
        "extract_dialogue_span",
        "detect_entity_mention",
    })

    def mode_for(self, skill_id: str) -> SkillBackendMode:
        if skill_id in self.rule_only_skills:
            return SkillBackendMode.RULE
        if skill_id in self.vlm_skills:
            return SkillBackendMode.VLM
        return self.default_mode


class LLMClient(Protocol):
    """Minimal interface for LLM calls within skills."""

    def reason(self, prompt: str, *, max_tokens: int = 512) -> str: ...


_SKILL_LLM_PROMPTS: dict[str, str] = {
    "infer_causal_relation": (
        "Given the following evidence, determine whether '{cause}' plausibly caused '{effect}'.\n"
        "Evidence: {evidence}\n"
        "Answer with a JSON object: {{\"causal\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "infer_temporal_relation": (
        "Given these events with timestamps, determine their temporal order.\n"
        "Events: {events}\n"
        "Answer with JSON: {{\"relation\": \"before|after|overlap|simultaneous\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "infer_state_change": (
        "Given the evidence, did '{entity}' change state '{predicate}' between the before and after observations?\n"
        "Before: {before}\nAfter: {after}\n"
        "Answer with JSON: {{\"changed\": true/false, \"before_state\": \"...\", \"after_state\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "infer_intention_or_motive": (
        "Given that '{agent}' performed actions: {actions}\n"
        "And context evidence: {context}\n"
        "What is the most likely intention or motive?\n"
        "Answer with JSON: {{\"intention\": \"...\", \"confidence\": 0.0-1.0, \"alternatives\": [...]}}"
    ),
    "infer_social_contradiction": (
        "Given the claim: '{claim}'\n"
        "And evidence chain: {evidence}\n"
        "And counterevidence: {counter}\n"
        "Is there a social contradiction (statement vs action, alibi vs evidence)?\n"
        "Answer with JSON: {{\"contradicted\": true/false, \"contradiction_claim\": \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "verify_claim_support": (
        "Does the following evidence ENTAIL or SUPPORT the claim?\n"
        "Claim: '{claim}'\n"
        "Evidence: {evidence}\n"
        "Answer with JSON: {{\"supported\": true/false, \"score\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "score_hypothesis_support": (
        "How well does the evidence support this hypothesis?\n"
        "Hypothesis: '{hypothesis}'\n"
        "Supporting evidence: {support}\n"
        "Counterevidence: {counter}\n"
        "Answer with JSON: {{\"support_score\": 0.0-1.0, \"contradiction_score\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "compare_hypotheses": (
        "Compare these scored hypotheses and select the best-supported one.\n"
        "Hypotheses: {hypotheses}\n"
        "Answer with JSON: {{\"best_label\": \"...\", \"margin\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "localize_clue": (
        "Which of these evidence candidates best serves the role '{role}' for the question '{question}'?\n"
        "Candidates: {candidates}\n"
        "Answer with JSON: {{\"best_refs\": [...], \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "extract_claim": (
        "Extract the main factual claim from this evidence text.\n"
        "Text: '{text}'\n"
        "Query context: '{query}'\n"
        "Answer with JSON: {{\"claim_text\": \"...\", \"speaker\": null or \"...\", \"confidence\": 0.0-1.0}}"
    ),
    "assign_evidence_role": (
        "Does this evidence fit the semantic role '{role}' for the question '{question}'?\n"
        "Evidence: '{text}'\n"
        "Answer with JSON: {{\"fits_role\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    ),
    "verify_temporal_social_consistency": (
        "Check whether this evidence chain is temporally ordered and socially plausible.\n"
        "Hypothesis: '{hypothesis}'\n"
        "Evidence chain: {chain}\n"
        "Answer with JSON: {{\"temporal_ok\": true/false, \"social_ok\": true/false, \"conflicts\": [...]}}"
    ),
    "bridge_evidence_hops": (
        "Can you find a reasoning bridge from source evidence to the target hypothesis?\n"
        "Source refs: {sources}\n"
        "Target hypothesis: '{target}'\n"
        "Available intermediate nodes: {intermediates}\n"
        "Answer with JSON: {{\"bridge_path\": [...], \"confidence\": 0.0-1.0}}"
    ),
}


def get_llm_prompt(skill_id: str) -> str | None:
    """Return the LLM prompt template for a skill, or None if rule-only."""
    return _SKILL_LLM_PROMPTS.get(skill_id)


def format_skill_prompt(skill_id: str, **kwargs: Any) -> str | None:
    """Format an LLM prompt for a skill with given arguments."""
    template = _SKILL_LLM_PROMPTS.get(skill_id)
    if not template:
        return None
    try:
        return template.format(**{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for k, v in kwargs.items()})
    except KeyError:
        return template

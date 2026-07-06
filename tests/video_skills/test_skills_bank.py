"""Tests for the Reasoning Skill Bank + starter inventory."""
from __future__ import annotations

import pytest

from video_skills.skills import (
    AtomicSkill,
    ReasoningSkillBank,
    SkillRecord,
    build_starter_bank,
)


STARTER_NAMES = (
    "identify_question_target",
    "decompose_into_subgoals",
    "retrieve_relevant_episode",
    "ground_entity_reference",
    "ground_event_span",
    "infer_observation_access",
    "order_two_events",
    "check_state_change",
    "check_causal_support",
    "update_belief_state",
    "check_evidence_sufficiency",
    "decide_answer_or_abstain",
    "check_alternative_hypothesis",
    "locate_counterevidence",
)


def test_starter_bank_registers_all_required_atomics():
    bank = build_starter_bank()
    names = {s.name for s in bank.all()}
    for n in STARTER_NAMES:
        assert n in names, f"missing required starter skill {n!r}"


def test_every_skill_has_required_record_fields():
    bank = build_starter_bank()
    for skill in bank.all():
        r = skill.record
        assert r.skill_id and r.name and r.family and r.output_type
        assert r.type in ("atomic", "composite")
        assert isinstance(r.input_schema, dict)
        assert isinstance(r.output_schema, dict)
        assert r.verification_rule, f"{r.name} missing verification_rule"
        assert isinstance(r.failure_modes, list)
        assert r.usage_stats is not None
        assert r.version is not None


def test_each_skill_has_at_least_one_canonical_check():
    canonical = {
        "claim_evidence_alignment",
        "evidence_sufficiency",
        "temporal_consistency",
        "perspective_consistency",
        "entity_consistency",
        "counterevidence",
    }
    bank = build_starter_bank()
    for skill in bank.all():
        check_names = {c.name for c in skill.record.verification_rule}
        assert check_names & canonical, (
            f"{skill.name} has no canonical verification check (got {check_names})"
        )


def test_bank_lookup_by_name_and_id():
    bank = build_starter_bank()
    s = bank.get_by_name("order_two_events")
    assert s.skill_id == "atom.order_two_events"
    assert bank.get(s.skill_id) is s


def test_bank_rejects_duplicate_registration():
    bank = ReasoningSkillBank()
    sk = build_starter_bank().get_by_name("order_two_events")
    bank.register(sk)
    with pytest.raises(ValueError):
        bank.register(sk)


def test_by_family_and_by_output_type_indices():
    bank = build_starter_bank()
    temporal = {s.name for s in bank.by_family("temporal")}
    assert {"order_two_events", "check_state_change"} <= temporal
    decisions = {s.name for s in bank.by_output_type("decision")}
    assert "decide_answer_or_abstain" in decisions

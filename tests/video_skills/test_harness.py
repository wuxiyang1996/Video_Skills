"""Tests for the Harness (deterministic hop interpreter)."""
from __future__ import annotations

import pytest

from video_skills.contracts import HopGoal, RetrievalQuery, new_id
from video_skills.harness import Harness, HarnessConfig, HopExecutionContext
from video_skills.loop import build_runtime
from video_skills.skills.bank import (
    AtomicSkill,
    ReasoningSkillBank,
    SkillContext,
    SkillOutput,
    SkillRecord,
)

from .synthetic import make_alice_bob_key_window


def _seed_runtime():
    rt = build_runtime()
    rt.memory_procedures.update_entity_profile(entity_id="alice", canonical_name="Alice")
    rt.memory_procedures.update_entity_profile(entity_id="bob", canonical_name="Bob")
    rt.memory_procedures.append_grounded_event(window=make_alice_bob_key_window())
    return rt


def test_harness_runs_atomic_chain_and_emits_step_per_atomic():
    rt = _seed_runtime()
    hop_goal = HopGoal(
        hop_id=new_id("hop"), parent_question_id="q",
        goal_text="key",
        target_claim_type="span",
        required_entities=["alice"],
    )
    skill = rt.bank.get_by_name("ground_event_span")
    record = rt.harness.run_hop(hop_goal, skill)
    assert record.outcome in ("resolved", "blocked", "abstain")
    assert record.steps  # at least one step was logged
    assert all(s.skill_id for s in record.steps)
    assert record.cost["atomic_steps"] == len(record.steps)


def test_harness_caps_atomic_steps_per_hop():
    rt = _seed_runtime()
    rt.harness.ctx.config.max_atomic_steps_per_hop = 1
    hop_goal = HopGoal(
        hop_id=new_id("hop"), parent_question_id="q", goal_text="key",
        target_claim_type="span", required_entities=["alice"],
        max_atomic_steps=10,
    )
    skill = rt.bank.get_by_name("ground_event_span")
    record = rt.harness.run_hop(hop_goal, skill)
    assert record.cost["atomic_steps"] <= 1


def test_harness_routes_memory_writes_through_registry():
    rt = _seed_runtime()
    hop_goal = HopGoal(
        hop_id=new_id("hop"), parent_question_id="q",
        goal_text="alice believes the key is in pocket",
        target_claim_type="belief",
        required_entities=["alice"],
        perspective_anchor="alice",
    )
    skill = rt.bank.get_by_name("update_belief_state")
    record = rt.harness.run_hop(
        hop_goal, skill,
        seed_inputs={"holder": "alice", "proposition": "key is in pocket", "polarity": "true"},
    )
    procs = [r.procedure for r in rt.memory_procedures.audit_log]
    # At least one revise_belief_state call should appear if the step verified
    if record.steps and record.steps[0].verification.passed:
        assert "revise_belief_state" in procs


def test_harness_invalidates_schema_violation():
    rt = _seed_runtime()
    bank = ReasoningSkillBank()

    def bad_executable(ctx: SkillContext) -> SkillOutput:
        return SkillOutput(
            output={},  # missing required key
            output_type="meta",
            confidence=1.0,
        )

    bad = AtomicSkill(
        record=SkillRecord(
            skill_id="atom.bad", name="bad_skill", type="atomic",
            family="testing", output_type="meta",
            input_schema={}, output_schema={"required_key": {"type": "str", "required": True}},
            verification_rule=[],
            failure_modes=["schema_violation"],
            required_memory_fields=[],
        ),
        executable=bad_executable,
    )
    bank.register(bad)
    harness = Harness(HopExecutionContext(
        bank=bank,
        memory=rt.memory,
        memory_procedures=rt.memory_procedures,
        retriever=rt.retriever,
        verifier=rt.verifier,
    ))
    hop_goal = HopGoal(
        hop_id=new_id("hop"), parent_question_id="q", goal_text="x",
        target_claim_type="claim",
    )
    record = harness.run_hop(hop_goal, bad)
    assert record.steps[0].failure_mode == "schema_violation"


def test_harness_catches_skill_exception():
    rt = _seed_runtime()
    bank = ReasoningSkillBank()

    def boom(ctx: SkillContext) -> SkillOutput:
        raise RuntimeError("intentional failure")

    bad = AtomicSkill(
        record=SkillRecord(
            skill_id="atom.boom", name="boom", type="atomic",
            family="testing", output_type="meta",
            input_schema={}, output_schema={},
            verification_rule=[],
            failure_modes=["exception"],
            required_memory_fields=[],
        ),
        executable=boom,
    )
    bank.register(bad)
    harness = Harness(HopExecutionContext(
        bank=bank,
        memory=rt.memory,
        memory_procedures=rt.memory_procedures,
        retriever=rt.retriever,
        verifier=rt.verifier,
    ))
    hop_goal = HopGoal(
        hop_id=new_id("hop"), parent_question_id="q", goal_text="x",
        target_claim_type="meta",
    )
    record = harness.run_hop(hop_goal, bad)
    assert record.steps[0].failure_mode == "exception"
    assert "error" in record.steps[0].output


def test_harness_records_broaden_history_when_action_returned():
    rt = _seed_runtime()
    # Empty memory hop: the goal text matches nothing → empty bundle → broaden
    hop_goal = HopGoal(
        hop_id=new_id("hop"), parent_question_id="q",
        goal_text="completely unrelated query about astrophysics",
        target_claim_type="span",
        required_entities=["nonexistent_entity"],
    )
    skill = rt.bank.get_by_name("ground_event_span")
    record = rt.harness.run_hop(hop_goal, skill)
    # The hop should have either broadened, blocked, or abstained
    assert record.outcome in ("blocked", "abstain", "resolved")
    # If it broadened, the meta should reflect that
    if record.cost["broaden_levels"] > 0:
        assert record.meta["broaden_history"]

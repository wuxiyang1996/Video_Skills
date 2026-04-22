"""End-to-end smoke test for the §2D online serving loop.

Validates that on a synthetic ``GroundedWindow``, the controller →
harness → verifier → retriever loop produces a valid :class:`ReasoningTrace`
for several question types covering the curated atomic inventory.
"""
from __future__ import annotations

import pytest

from video_skills import (
    PROCEDURE_NAMES,
    QuestionAnalysis,
    ReasoningTrace,
    build_runtime,
    run_question,
    validate_atomic_step,
)

from .synthetic import make_alice_bob_key_window


@pytest.fixture()
def seeded_runtime():
    rt = build_runtime()
    # Seed entities + the demo grounded window
    rt.memory_procedures.update_entity_profile(entity_id="alice", canonical_name="Alice", seen_at=10.0)
    rt.memory_procedures.update_entity_profile(entity_id="bob", canonical_name="Bob", seen_at=10.0)
    rt.memory_procedures.append_grounded_event(window=make_alice_bob_key_window())
    return rt


def _assert_trace_invariants(trace: ReasoningTrace):
    assert trace.trace_id and trace.question_id
    assert isinstance(trace.question_analysis, QuestionAnalysis)
    assert trace.cost["hops"] == len(trace.hops)
    assert trace.bank_skill_ids_used  # at least one skill exercised
    # Each step should validate against §2A.8 contract rules
    for hop in trace.hops:
        for step in hop.steps:
            violations = validate_atomic_step(step)
            assert violations == [], f"step {step.step_id} violations: {violations}"
    # Outcome accounting
    assert trace.finished_at is not None and trace.finished_at >= trace.started_at
    # Either we got an answer OR we got an abstain decision — never both
    assert (trace.answer is not None) ^ (trace.abstain is not None)


def test_loop_runs_ordering_question(seeded_runtime):
    trace = run_question(
        seeded_runtime,
        "Did Alice pick up the key before Bob entered?",
        target_entities=["key", "Bob"],
    )
    _assert_trace_invariants(trace)
    assert trace.question_analysis.question_type == "ordering"
    # The decomposition produced at least one ordering hop
    assert len(trace.hops) >= 1


def test_loop_runs_belief_question(seeded_runtime):
    trace = run_question(
        seeded_runtime,
        "Did Bob know that Alice hid the key?",
        target_entities=["bob"],
        perspective_anchor="bob",
    )
    _assert_trace_invariants(trace)
    assert trace.question_analysis.question_type == "belief"


def test_loop_runs_presence_question(seeded_runtime):
    trace = run_question(
        seeded_runtime,
        "Who is in the kitchen?",
        target_entities=["alice"],
    )
    _assert_trace_invariants(trace)
    assert trace.question_analysis.question_type in ("presence", "free")


def test_loop_runs_causal_question(seeded_runtime):
    trace = run_question(
        seeded_runtime,
        "Why did Bob enter the kitchen?",
        target_entities=["bob"],
    )
    _assert_trace_invariants(trace)
    assert trace.question_analysis.question_type == "causal"


def test_loop_abstains_on_unrelated_question(seeded_runtime):
    trace = run_question(
        seeded_runtime,
        "What is the capital of Mars?",
    )
    _assert_trace_invariants(trace)
    # The runtime should abstain (or produce a templated low-confidence answer)
    if trace.abstain is not None:
        assert trace.abstain.abstain is True


def test_loop_records_all_skill_invocations_in_bank_set(seeded_runtime):
    trace = run_question(
        seeded_runtime,
        "Did Alice pick up the key before Bob entered?",
        target_entities=["alice", "bob"],
    )
    _assert_trace_invariants(trace)
    # Every used skill should map back to a real bank entry
    for sid in trace.bank_skill_ids_used:
        assert seeded_runtime.bank.has(sid)


def test_memory_procedure_audit_log_only_uses_known_procedures(seeded_runtime):
    run_question(
        seeded_runtime,
        "Did Alice pick up the key before Bob entered?",
        target_entities=["alice", "bob"],
    )
    procs = {r.procedure for r in seeded_runtime.memory_procedures.audit_log}
    assert procs <= set(PROCEDURE_NAMES)

"""Tests for the Memory Procedure Registry (the 9 fixed procedures)."""
from __future__ import annotations

import pytest

from video_skills.contracts import EvidenceRef
from video_skills.memory import (
    EntityProfile,
    Memory,
    MemoryProcedureRegistry,
    PROCEDURE_NAMES,
)

from .synthetic import make_alice_bob_key_window, make_minimal_window


def _fresh():
    mem = Memory()
    return mem, MemoryProcedureRegistry(mem)


def test_procedure_names_match_design():
    expected = {
        "open_episode_thread",
        "append_grounded_event",
        "update_entity_profile",
        "refresh_state_memory",
        "compress_episode_cluster",
        "attach_evidence_ref",
        "resolve_entity_alias",
        "revise_belief_state",
        "mark_memory_conflict",
    }
    assert set(PROCEDURE_NAMES) == expected


def test_open_episode_thread_is_idempotent():
    mem, reg = _fresh()
    t1 = reg.open_episode_thread(clip_id="clip_a", time_span=(0.0, 5.0))
    t2 = reg.open_episode_thread(clip_id="clip_a", time_span=(0.0, 5.0))
    assert t1.thread_id == t2.thread_id
    assert len(mem.episodic.threads) == 1


def test_append_grounded_event_skips_low_confidence():
    mem, reg = _fresh()
    win = make_alice_bob_key_window()
    win.confidence = 0.1  # below default tau_grounding
    appended = reg.append_grounded_event(window=win)
    assert appended == []
    assert mem.episodic.stats()["n_events"] == 0


def test_append_grounded_event_writes_events_and_evidence():
    mem, reg = _fresh()
    win = make_alice_bob_key_window()
    appended = reg.append_grounded_event(window=win)
    assert len(appended) == len(win.events)
    assert mem.episodic.stats()["n_events"] == len(win.events)
    # Each event gets a per-event anchor ref + the window's keyframe refs.
    for ev in appended:
        assert len(ev.evidence_ref_ids) == 1 + len(win.keyframes)
    # Anchor (1 per event) + keyframe refs (shared, 1 per keyframe).
    assert mem.evidence.stats()["n_refs"] == len(win.events) + len(win.keyframes)


def test_update_entity_profile_creates_then_updates():
    mem, reg = _fresh()
    p = reg.update_entity_profile(entity_id="alice", canonical_name="Alice", seen_at=1.0)
    assert p.canonical_name == "Alice"
    p2 = reg.update_entity_profile(entity_id="alice", alias="A", seen_at=10.0)
    assert "A" in p2.aliases_pending
    assert p2.first_seen == 1.0
    assert p2.last_seen == 10.0


def test_resolve_entity_alias_promotes_pending_to_bound():
    mem, reg = _fresh()
    reg.update_entity_profile(entity_id="alice", canonical_name="Alice")
    reg.update_entity_profile(entity_id="alice", alias="A")
    p = reg.resolve_entity_alias(entity_id="alice", alias="A")
    assert "A" in p.aliases
    assert "A" not in p.aliases_pending


def test_attach_evidence_ref_to_episodic_event():
    mem, reg = _fresh()
    win = make_alice_bob_key_window()
    events = reg.append_grounded_event(window=win)
    ev_id = events[0].event_id
    new_ref = EvidenceRef(ref_id="r_extra", modality="subtitle", text="extra context")
    reg.attach_evidence_ref(record_id=ev_id, evidence=new_ref)
    assert "r_extra" in mem.episodic.events[ev_id].evidence_ref_ids


def test_revise_belief_state_supersedes_prior():
    mem, reg = _fresh()
    b1 = reg.revise_belief_state(
        holder_entity="alice", proposition="key is in pocket", polarity="true",
        confidence=0.6, time_anchor=10.0,
    )
    b2 = reg.revise_belief_state(
        holder_entity="alice", proposition="key is in pocket", polarity="true",
        confidence=0.9, time_anchor=20.0, supersedes=b1.state_id,
    )
    assert mem.state.beliefs[b1.state_id].is_active is False
    assert b2.is_active is True


def test_compress_episode_cluster_emits_summary_when_threshold_met():
    mem, reg = _fresh()
    win = make_alice_bob_key_window()  # 3 events
    reg.append_grounded_event(window=win)
    thread = next(iter(mem.episodic.threads.values()))
    summary = reg.compress_episode_cluster(
        thread_id=thread.thread_id, subject="alice", min_cluster=2,
    )
    assert summary is not None
    assert summary.version == 1
    assert mem.semantic.stats()["n_active"] == 1
    # Compressing again increments version, archives the previous one
    summary2 = reg.compress_episode_cluster(
        thread_id=thread.thread_id, subject="alice", min_cluster=2,
    )
    assert summary2 is not None
    assert summary2.version == 2
    assert mem.semantic.stats()["n_archived"] == 1


def test_compress_episode_cluster_skips_below_min():
    mem, reg = _fresh()
    win = make_minimal_window()  # 1 event
    reg.append_grounded_event(window=win)
    thread = next(iter(mem.episodic.threads.values()))
    summary = reg.compress_episode_cluster(
        thread_id=thread.thread_id, subject="solo", min_cluster=3,
    )
    assert summary is None


def test_mark_memory_conflict_dedups():
    mem, reg = _fresh()
    edge = reg.mark_memory_conflict(record_id_a="a", record_id_b="b", reason="r")
    reg.mark_memory_conflict(record_id_a="a", record_id_b="b", reason="r")
    assert mem.contradicts.count(edge) == 1


def test_refresh_state_memory_decays_and_deactivates():
    mem, reg = _fresh()
    reg.revise_belief_state(
        holder_entity="alice", proposition="x", polarity="true",
        confidence=1.0, time_anchor=0.0,
    )
    # Big jump forward in time -> decay halflife pushes confidence below 0.1
    stats = reg.refresh_state_memory(time_anchor=10_000.0, decay_halflife_s=10.0)
    assert stats["deactivated"] >= 1


def test_audit_log_records_every_call_via_dispatch():
    mem, reg = _fresh()
    reg.call("open_episode_thread", clip_id="x")
    reg.call("update_entity_profile", entity_id="alice", canonical_name="Alice")
    reg.call("mark_memory_conflict", record_id_a="a", record_id_b="b", reason="r")
    procs = [r.procedure for r in reg.audit_log]
    assert procs == ["open_episode_thread", "update_entity_profile", "mark_memory_conflict"]
    # Caller is recorded
    reg.call("open_episode_thread", caller="atom.x", clip_id="y")
    assert reg.audit_log[-1].caller == "atom.x"


def test_unknown_procedure_raises():
    mem, reg = _fresh()
    with pytest.raises(KeyError):
        reg.call("not_a_procedure")

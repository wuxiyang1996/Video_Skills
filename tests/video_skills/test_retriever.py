"""Tests for the Retriever subsystem (§2B)."""
from __future__ import annotations

import pytest

from video_skills.contracts import RetrievalQuery, new_id
from video_skills.memory import Memory, MemoryProcedureRegistry
from video_skills.retriever import Retriever

from .synthetic import make_alice_bob_key_window


def _seeded_runtime():
    mem = Memory()
    procs = MemoryProcedureRegistry(mem)
    win = make_alice_bob_key_window()
    procs.update_entity_profile(entity_id="alice", canonical_name="Alice")
    procs.update_entity_profile(entity_id="bob", canonical_name="Bob")
    procs.append_grounded_event(window=win)
    procs.revise_belief_state(
        holder_entity="alice", proposition="key is in pocket",
        polarity="true", confidence=0.9, time_anchor=24.0,
    )
    return mem, Retriever(mem)


def test_retrieve_finds_episodic_events_by_keyword():
    mem, retriever = _seeded_runtime()
    q = RetrievalQuery(query_id=new_id("rq"), text="key", store_filter="episodic")
    bundle = retriever.retrieve(q)
    assert not bundle.is_empty()
    # Should find both "Alice picks up the key" and "Alice hides the key"
    descriptions = " ".join(r.text or "" for r in bundle.refs).lower()
    assert "key" in descriptions


def test_retrieve_entity_filter_restricts_results():
    mem, retriever = _seeded_runtime()
    q = RetrievalQuery(
        query_id=new_id("rq"), text="enters",
        entity_filter=["bob"], store_filter="episodic",
    )
    bundle = retriever.retrieve(q)
    assert not bundle.is_empty()
    # Every returned ref's entities (where present) should include bob
    for r in bundle.refs:
        if r.entities:
            assert "bob" in r.entities


def test_retrieve_state_finds_belief():
    mem, retriever = _seeded_runtime()
    q = RetrievalQuery(query_id=new_id("rq"), text="key in pocket", store_filter="state")
    bundle = retriever.retrieve(q)
    assert not bundle.is_empty()
    assert any(r.modality == "state" for r in bundle.refs)


def test_retrieve_dedups_by_source_id():
    mem, retriever = _seeded_runtime()
    q = RetrievalQuery(query_id=new_id("rq"), text="key", store_filter="any", k=32)
    bundle = retriever.retrieve(q)
    source_ids = [r.source_id for r in bundle.refs if r.source_id]
    assert len(source_ids) == len(set(source_ids))


def test_broaden_advances_ladder_per_hop():
    mem, retriever = _seeded_runtime()
    base = RetrievalQuery(
        query_id=new_id("rq"), text="key",
        entity_filter=["alice", "bob"], store_filter="episodic", k=2,
    )
    b1 = retriever.broaden("hop1", base)
    b2 = retriever.broaden("hop1", base)
    assert b1.meta["broaden_level"] == 1
    assert b2.meta["broaden_level"] == 2
    assert b1.query.entity_filter != base.entity_filter or b1.query.store_filter != base.store_filter


def test_retrieve_counter_runs_negation_query():
    mem, retriever = _seeded_runtime()
    bundle = retriever.retrieve_counter({"claim": "alice picks up key"})
    assert bundle.meta.get("counter") is True


def test_fuse_merges_bundles_dedup():
    mem, retriever = _seeded_runtime()
    q1 = RetrievalQuery(query_id=new_id("rq"), text="key", store_filter="episodic")
    q2 = RetrievalQuery(query_id=new_id("rq"), text="alice", store_filter="episodic")
    fused = retriever.fuse([retriever.retrieve(q1), retriever.retrieve(q2)])
    source_ids = [r.source_id for r in fused.refs if r.source_id]
    assert len(source_ids) == len(set(source_ids))


def test_coverage_reports_entities():
    mem, retriever = _seeded_runtime()
    q = RetrievalQuery(query_id=new_id("rq"), text="alice key", store_filter="episodic")
    bundle = retriever.retrieve(q)
    assert "alice" in bundle.coverage["entities"]

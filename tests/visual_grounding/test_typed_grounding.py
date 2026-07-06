"""Unit tests for the typed grounding stack.

Covers, end to end without touching real videos / models:

- The new typed schemas (``EntityState``, ``EventSpan``,
  ``InteractionEdge``, ``TemporalRelation``, ``VisibilityState``,
  ``BeliefCandidate``, ``GroundedClip``, ``MemoryRecord``,
  ``RawObservation``, ``VideoSegment``, ``SubtitleSpan``).
- ``EntityTracker`` (alias resolution + segment-local id remap).
- ``EventGrounder`` + ``SocialStateGrounder`` + ``TemporalGrounder``.
- ``GroundingNormalizer`` (per-segment ``GroundedClip`` build).
- ``MemoryProjection`` (entity-thread merge across clips).
- ``GroundingRuntime`` (the §10 + §12 query API).

A separate ``TestBenchmarkAdapters`` block hits the
``benchmark_adapters`` driver against the existing Video-Holmes sample
video using the deterministic stub VLM, mirroring
``test_benchmarks_schema.py``. Tests skip when the sample is missing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest

from visual_grounding import (
    BeliefCandidate,
    BENCHMARK_CONFIGS,
    BenchmarkAdapter,
    EntityState,
    EntityTracker,
    EventGrounder,
    EventSpan,
    EvidenceRef,
    GroundedClip,
    GroundingNormalizer,
    GroundingRuntime,
    InteractionEdge,
    MemoryProjection,
    MemoryRecord,
    ObservationExtractor,
    RawObservation,
    SocialStateGrounder,
    SubtitleSpan,
    TemporalGrounder,
    TemporalRelation,
    VideoSegment,
    VisibilityState,
    new_grounding_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_segment(seg_id: str, t0: float, t1: float) -> VideoSegment:
    return VideoSegment(
        segment_id=seg_id,
        video_id="vid_test",
        start_time=t0,
        end_time=t1,
        metadata={"frame_times": [t0, (t0 + t1) / 2.0, t1]},
    )


def _ev_ref(ref_id: str, t0: float, t1: float, *,
            modality: str = "frame", text: str = "",
            clip_id: str = "seg_a") -> EvidenceRef:
    return EvidenceRef(
        ref_id=ref_id,
        modality=modality,
        timestamp=(t0, t1),
        text=text,
        clip_id=clip_id,
        video_id="vid_test",
        source_type="observed",
        confidence=1.0,
    )


def _stub_vlm(prompt: str, **_kwargs: Any) -> str:
    return json.dumps({
        "scene": "kitchen conversation",
        "entities": [
            {"id": "p1", "type": "person",
             "attributes": {"name": "Alice", "role": "host", "clothing": "red"}},
            {"id": "p2", "type": "person",
             "attributes": {"name": "Bob", "role": "guest", "clothing": "blue"}},
        ],
        "interactions": [
            {"src": "p1", "rel": "talking_to", "dst": "p2", "confidence": 0.84},
            {"src": "p2", "rel": "looking_at", "dst": "p1", "confidence": 0.7},
        ],
        "events": [
            {"type": "speaks", "agents": ["p1"], "confidence": 0.9,
             "description": "Alice greets Bob"},
            {"type": "enters_room", "agents": ["p2"], "confidence": 0.8,
             "description": "Bob walks in"},
        ],
        "social_hypotheses": [
            {"type": "intention", "target": "p1",
             "value": "wants to discuss schedule", "confidence": 0.55,
             "provenance": "inferred_from_dialogue",
             "supporting_evidence": []},
        ],
    })


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


class TestSchemaInvariants:
    def test_evidence_ref_extended_defaults(self):
        ref = EvidenceRef(ref_id="r1", modality="frame", timestamp=(0.0, 1.0))
        assert ref.source_type == "observed"
        assert ref.confidence == 1.0
        assert ref.video_id is None
        assert ref.time_range == (0.0, 1.0)

    def test_typed_objects_serialize(self):
        ref = _ev_ref("r1", 0.0, 1.0)
        ent = EntityState(
            entity_id="e1", canonical_name="Alice",
            entity_type="person", evidence_refs=[ref],
        )
        ev = EventSpan(
            event_id="ev1", event_type="speaks",
            participants=["e1"], start_time=0.0, end_time=1.0,
            evidence_refs=[ref],
        )
        edge = InteractionEdge(
            edge_id="i1", src_entity="e1", dst_entity="e2",
            interaction_type="talks_to", evidence_refs=[ref],
        )
        rel = TemporalRelation(
            relation_id="t1", lhs_event_id="ev1", rhs_event_id="ev2",
            relation_type="before", evidence_refs=[ref],
        )
        vis = VisibilityState(
            state_id="v1", holder_entity="e1",
            target_event_or_object="ev1", relation_type="saw",
            evidence_refs=[ref],
        )
        belief = BeliefCandidate(
            belief_id="b1", holder_entity="e1",
            proposition="suspects deception", evidence_refs=[ref],
        )
        clip = GroundedClip(
            clip_id="c1", video_id="vid_test",
            start_time=0.0, end_time=1.0,
            entities=[ent], events=[ev], interactions=[edge],
            temporal_relations=[rel], visibility_states=[vis],
            belief_candidates=[belief], evidence_refs=[ref],
        )
        blob = clip.to_dict()
        # JSON-serializable end to end.
        s = json.dumps(blob)
        assert "vid_test" in s
        assert blob["entities"][0]["source_type"] == "observed"
        assert blob["belief_candidates"][0]["source_type"] == "inferred"

    def test_inferred_provenance_defaults(self):
        belief = BeliefCandidate(
            belief_id="b1", holder_entity="e1", proposition="X",
        )
        rel = TemporalRelation(
            relation_id="t1", lhs_event_id="a", rhs_event_id="b",
            relation_type="enables",
        )
        vis = VisibilityState(
            state_id="v1", holder_entity="e1",
            target_event_or_object="ev1", relation_type="could_see",
        )
        # Inferred-by-default per the plan.
        assert belief.source_type == "inferred"
        assert rel.source_type == "inferred"
        assert vis.source_type == "inferred"


# ---------------------------------------------------------------------------
# ObservationExtractor
# ---------------------------------------------------------------------------


class TestObservationExtractor:
    def test_subtitle_echo_and_speaker_turn_observations(self):
        seg = _make_segment("seg_a", 0.0, 4.0)
        sub = SubtitleSpan(
            span_id="sub_1", text="Hi Bob.",
            start_time=0.5, end_time=1.0, speaker="Alice",
            evidence_ref=_ev_ref("sub_1", 0.5, 1.0,
                                 modality="subtitle", text="Hi Bob."),
        )
        ext = ObservationExtractor()  # no VLM
        obs = ext.extract(seg, frames=[], subtitles=[sub])
        kinds = sorted({o.observation_type for o in obs})
        assert "subtitle_echo" in kinds
        assert "speaker_turn" in kinds
        # Every observation must keep evidence + source_type.
        for o in obs:
            assert o.evidence_refs
            assert o.source_type in ("observed", "inferred",
                                     "retrieved", "distilled")

    def test_vlm_proposals_emit_typed_observations(self):
        seg = _make_segment("seg_a", 0.0, 4.0)
        ext = ObservationExtractor(vlm_fn=_stub_vlm)
        obs = ext.extract(seg, frames=[], subtitles=[])
        types = {o.observation_type for o in obs}
        assert "caption" in types
        assert "participant_mention" in types
        assert "interaction_proposal" in types
        assert "event_proposal" in types
        assert "social_hypothesis_proposal" in types
        # Hypotheses are inferred.
        for o in obs:
            if o.observation_type == "social_hypothesis_proposal":
                assert o.source_type == "inferred"


# ---------------------------------------------------------------------------
# EntityTracker
# ---------------------------------------------------------------------------


class TestEntityTracker:
    def _obs(self, segment_id: str, local_id: str, attrs: Dict[str, Any]):
        return RawObservation(
            obs_id=new_grounding_id("obs"),
            segment_id=segment_id,
            observation_type="participant_mention",
            payload={"id": local_id, "type": "person", "attributes": attrs},
            evidence_refs=[_ev_ref(f"r_{local_id}", 0.0, 1.0,
                                   clip_id=segment_id)],
            confidence=0.9,
        )

    def test_alias_resolution_merges_by_canonical_name(self):
        tracker = EntityTracker(match_threshold=0.99)  # disable signature merge
        tracker.update([
            self._obs("seg_a", "p1", {"name": "Alice", "clothing": "red"}),
        ])
        tracker.update([
            self._obs("seg_b", "p9", {"name": "Alice", "clothing": "red"}),
        ])
        # Two distinct ids before aliasing.
        states = tracker.snapshot()
        assert len(states) == 2
        merged = tracker.resolve_aliases()
        assert len(merged) == 1
        m = merged[0]
        assert m.canonical_name == "Alice"
        # Both segment-local ids now resolve to the merged global id.
        gid = m.entity_id
        assert tracker.map_local_id("seg_a", "p1") == gid
        assert tracker.map_local_id("seg_b", "p9") == gid

    def test_signature_jaccard_match(self):
        tracker = EntityTracker(match_threshold=0.5)
        tracker.update([
            self._obs("seg_a", "p1",
                      {"role": "host", "clothing": "red"}),
        ])
        tracker.update([
            self._obs("seg_b", "p1",
                      {"role": "host", "clothing": "red"}),
        ])
        # Same signature → same entity.
        states = tracker.snapshot()
        assert len(states) == 1


# ---------------------------------------------------------------------------
# EventGrounder + SocialStateGrounder + TemporalGrounder
# ---------------------------------------------------------------------------


def _build_pipeline_obs():
    """Synthesize observations + segment index for two clips."""
    seg_a = _make_segment("seg_a", 0.0, 5.0)
    seg_b = _make_segment("seg_b", 5.0, 10.0)
    obs: List[RawObservation] = []

    # Segment A: Alice + Bob speak; Alice talks_to Bob.
    obs.append(RawObservation(
        obs_id="o1", segment_id=seg_a.segment_id,
        observation_type="participant_mention",
        payload={"id": "p1", "type": "person",
                 "attributes": {"name": "Alice"}},
        evidence_refs=[_ev_ref("r1", 0.0, 5.0, clip_id=seg_a.segment_id)],
        confidence=0.9,
    ))
    obs.append(RawObservation(
        obs_id="o2", segment_id=seg_a.segment_id,
        observation_type="participant_mention",
        payload={"id": "p2", "type": "person",
                 "attributes": {"name": "Bob"}},
        evidence_refs=[_ev_ref("r2", 0.0, 5.0, clip_id=seg_a.segment_id)],
        confidence=0.9,
    ))
    obs.append(RawObservation(
        obs_id="o3", segment_id=seg_a.segment_id,
        observation_type="event_proposal",
        payload={"type": "speaks", "agents": ["p1"],
                 "description": "Alice greets"},
        evidence_refs=[_ev_ref("r3", 0.0, 4.0, clip_id=seg_a.segment_id)],
        confidence=0.85,
    ))
    obs.append(RawObservation(
        obs_id="o4", segment_id=seg_a.segment_id,
        observation_type="interaction_proposal",
        payload={"src": "p1", "rel": "talking_to", "dst": "p2"},
        evidence_refs=[_ev_ref("r4", 0.0, 4.0, clip_id=seg_a.segment_id)],
        confidence=0.8,
    ))

    # Segment B: Bob enters; Bob looks at Alice.
    obs.append(RawObservation(
        obs_id="o5", segment_id=seg_b.segment_id,
        observation_type="participant_mention",
        payload={"id": "p1", "type": "person",
                 "attributes": {"name": "Alice"}},
        evidence_refs=[_ev_ref("r5", 5.0, 10.0, clip_id=seg_b.segment_id)],
        confidence=0.9,
    ))
    obs.append(RawObservation(
        obs_id="o6", segment_id=seg_b.segment_id,
        observation_type="participant_mention",
        payload={"id": "p2", "type": "person",
                 "attributes": {"name": "Bob"}},
        evidence_refs=[_ev_ref("r6", 5.0, 10.0, clip_id=seg_b.segment_id)],
        confidence=0.9,
    ))
    obs.append(RawObservation(
        obs_id="o7", segment_id=seg_b.segment_id,
        observation_type="event_proposal",
        payload={"type": "enters_room", "agents": ["p2"],
                 "description": "Bob walks in"},
        evidence_refs=[_ev_ref("r7", 5.0, 9.0, clip_id=seg_b.segment_id)],
        confidence=0.8,
    ))
    obs.append(RawObservation(
        obs_id="o8", segment_id=seg_b.segment_id,
        observation_type="interaction_proposal",
        payload={"src": "p2", "rel": "looking_at", "dst": "p1"},
        evidence_refs=[_ev_ref("r8", 5.0, 9.0, clip_id=seg_b.segment_id)],
        confidence=0.8,
    ))
    return [seg_a, seg_b], obs


class TestEventAndSocialGrounding:
    def test_full_pipeline_builds_clips_and_runtime(self):
        segments, observations = _build_pipeline_obs()
        seg_index = {s.segment_id: (s.start_time, s.end_time) for s in segments}

        tracker = EntityTracker(match_threshold=0.6)
        entities = tracker.update(observations)
        entities = tracker.resolve_aliases(entities)

        # Two unique people after merge.
        assert len(entities) == 2

        events = EventGrounder().build_events(
            observations, entities,
            tracker=tracker, segment_index=seg_index,
        )
        assert len(events) == 2
        speaks = [e for e in events if e.event_type == "speaks"][0]
        enters = [e for e in events if e.event_type == "enters_room"][0]
        assert speaks.source_type == "observed"
        assert speaks.start_time is not None
        assert speaks.end_time is not None
        assert speaks.participants  # remapped to global ids
        # Every event must keep evidence_refs.
        for e in events:
            assert e.evidence_refs

        social = SocialStateGrounder().build_social_states(
            observations, entities, events, tracker=tracker,
        )
        assert len(social["interactions"]) == 2
        for edge in social["interactions"]:
            # Interaction relations come from the observed list.
            assert edge.source_type == "observed"
        # Visibility includes both edge-derived ('saw') and event-derived
        # ('could_see') entries — always inferred.
        kinds = {v.relation_type for v in social["visibility_states"]}
        assert "could_see" in kinds
        for v in social["visibility_states"]:
            assert v.source_type == "inferred"

        temporal = TemporalGrounder().build_relations(events)
        rels = {r.relation_type for r in temporal}
        assert "before" in rels  # speaks (0-4) before enters_room (5-9)

        normalizer = GroundingNormalizer()
        clips = []
        for seg in segments:
            clips.append(normalizer.normalize(
                seg, observations, entities, events,
                social_outputs=social, temporal_relations=temporal,
            ))
        assert all(isinstance(c, GroundedClip) for c in clips)
        # Every clip must carry evidence aggregating its nested objects.
        for clip in clips:
            assert clip.evidence_refs

        # Memory projection → record types.
        memory = MemoryProjection().project_clips(clips)
        kinds_seen = {r.kind for r in memory}
        assert "episodic_event" in kinds_seen
        assert "entity_thread" in kinds_seen
        assert "relation" in kinds_seen
        assert "semantic_summary" in kinds_seen
        # One entity_thread per global entity (merged across clips).
        entity_threads = [r for r in memory if r.kind == "entity_thread"]
        assert len(entity_threads) == 2

        # GroundingRuntime API.
        runtime = GroundingRuntime(
            video_id="vid_test", clips=clips,
            memory_records=memory, mode="retrieval",
        )

        # get_local_grounded_context.
        local = runtime.get_local_grounded_context(time_range=(0.0, 4.0))
        assert len(local) >= 1

        alice_id = entities[0].entity_id if entities[0].canonical_name == "Alice" else entities[1].entity_id
        bob_id = entities[1].entity_id if entities[0].canonical_name == "Alice" else entities[0].entity_id

        # retrieve_by_entity (alias-friendly).
        hits = runtime.retrieve_by_entity("Alice")
        assert hits, "alias lookup must work via canonical_name"
        # retrieve_events_for_entity.
        bob_events = runtime.retrieve_events_for_entity("Bob")
        assert any(e.event_type == "enters_room" for e in bob_events)

        # retrieve_event_chain around timestamp 5.5.
        chain = runtime.retrieve_event_chain(5.5, window=10.0)
        assert any(e.event_type == "enters_room" for e in chain)

        # retrieve_supporting_evidence (keyword fallback).
        evid = runtime.retrieve_supporting_evidence("enters room", top_k=3)
        assert isinstance(evid, list)

        # retrieve_visibility — both holder filter and unfiltered.
        all_vis = runtime.retrieve_visibility()
        assert all_vis
        # Every visibility entry must carry evidence_refs + confidence.
        for v in all_vis:
            assert v.evidence_refs
            assert 0.0 <= v.confidence <= 1.0

        # stats sanity.
        stats = runtime.stats()
        assert stats["clips"] == len(clips)
        assert stats["memory_records"] == len(memory)


# ---------------------------------------------------------------------------
# BenchmarkAdapter (real video sample, stub VLM)
# ---------------------------------------------------------------------------


_REPO = "/fs/gamma-projects/vlm-robot"
SAMPLE_VIDEO = (
    f"{_REPO}/Video_Skills/dataset_examples/video_holmes/0at001QMutY.mp4"
)


@pytest.mark.skipif(
    not os.path.isfile(SAMPLE_VIDEO),
    reason="Video-Holmes sample not on disk; benchmark adapter test skipped.",
)
class TestBenchmarkAdapters:
    def test_video_holmes_adapter_returns_grounding_runtime(self):
        adapter = BenchmarkAdapter.for_benchmark(
            "video_holmes", vlm_fn=_stub_vlm,
        )
        adapter.config.max_frames_per_window = 1
        adapter.config.window_seconds = 30.0
        adapter.config.fps = 0.05
        adapter.config.include_scene_changes = False
        rt = adapter.build(SAMPLE_VIDEO)
        assert rt.mode == "direct"
        assert rt.clips, "adapter must yield at least one grounded clip"
        # No memory in direct mode.
        assert rt.memory_records == []
        # All clips reference the same video.
        assert all(c.video_id == os.path.basename(SAMPLE_VIDEO) for c in rt.clips)
        # Every clip's evidence list must be non-empty (frames + clip ref).
        for clip in rt.clips:
            assert clip.evidence_refs

    def test_known_benchmark_names(self):
        for name in BENCHMARK_CONFIGS:
            adapter = BenchmarkAdapter.for_benchmark(name)
            assert adapter.config.name == name

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            BenchmarkAdapter.for_benchmark("not_a_benchmark")

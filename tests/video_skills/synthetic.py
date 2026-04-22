"""Shared synthetic-data helpers for video_skills unit tests.

These factories build small, deterministic ``GroundedWindow`` objects so the
end-to-end loop has something to ground over without any video / VLM calls.
"""
from __future__ import annotations

from typing import List

from video_skills.contracts import (
    DialogueSpan,
    EntityRef,
    EventRef,
    FrameRef,
    GroundedWindow,
    new_id,
)


def make_alice_bob_key_window() -> GroundedWindow:
    """A small window where Alice picks up a key and Bob enters later.

    Designed to support several question types:
    - ordering: did Alice pick up the key before Bob entered?
    - belief:   did Bob see Alice pick up the key?
    - presence: was Alice in the room?
    """
    return GroundedWindow(
        window_id=new_id("win"),
        clip_id="clip_kitchen_001",
        time_span=(10.0, 30.0),
        entities=[
            EntityRef(entity_id="alice", canonical_name="Alice", aliases=["A"], role="person", confidence=0.95),
            EntityRef(entity_id="bob", canonical_name="Bob", aliases=["B"], role="person", confidence=0.92),
        ],
        events=[
            EventRef(
                event_id=new_id("evt"),
                event_type="action",
                description="Alice picks up the key from the table",
                participants=["alice"],
                time_span=(12.0, 14.0),
                confidence=0.9,
            ),
            EventRef(
                event_id=new_id("evt"),
                event_type="action",
                description="Bob enters the kitchen",
                participants=["bob"],
                time_span=(20.0, 21.0),
                confidence=0.88,
            ),
            EventRef(
                event_id=new_id("evt"),
                event_type="action",
                description="Alice hides the key in her pocket",
                participants=["alice"],
                time_span=(22.0, 24.0),
                confidence=0.85,
            ),
        ],
        dialogue=[
            DialogueSpan(
                span_id=new_id("ds"),
                text="Where did you put the key?",
                speaker="bob",
                time_span=(25.0, 26.5),
                confidence=0.94,
            ),
        ],
        spatial_state={
            "alice": {"location": "kitchen", "visibility": "visible"},
            "bob": {"location": "kitchen", "visibility": "visible"},
        },
        keyframes=[
            FrameRef(frame_id="frm_001", timestamp=12.5, locator={"path": "demo.mp4", "frame": 300}),
            FrameRef(frame_id="frm_002", timestamp=20.5, locator={"path": "demo.mp4", "frame": 492}),
        ],
        provenance={"detector": "synthetic", "model_version": "test-1"},
        confidence=0.9,
    )


def make_minimal_window() -> GroundedWindow:
    """A trivially small window with one event for the lifecycle threshold tests."""
    return GroundedWindow(
        window_id=new_id("win"),
        clip_id="clip_minimal",
        time_span=(0.0, 5.0),
        entities=[EntityRef(entity_id="solo", canonical_name="Solo", role="person")],
        events=[
            EventRef(
                event_id=new_id("evt"),
                event_type="state_change",
                description="Solo opens the door",
                participants=["solo"],
                time_span=(1.0, 2.0),
                confidence=0.7,
            ),
        ],
        keyframes=[FrameRef(frame_id="frm_solo", timestamp=1.5)],
        confidence=0.7,
    )


def all_demo_windows() -> List[GroundedWindow]:
    return [make_alice_bob_key_window(), make_minimal_window()]

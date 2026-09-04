"""One L1 example per question from per-video frozen L1.

The layer-1 catalog is question-agnostic, but the staged runner keyed its cache
by example_id and --unique-videos kept the first question per video, leaving 270
of 1,837 Video-Holmes questions (260 SR).
"""

from types import SimpleNamespace

from scripts.eval.derive_full_question_examples import derive_example, index_frozen_by_video


def _frozen(video, q):
    return {
        "example_id": f"video_holmes:test:{video}:q{q}",
        "dataset": "video_holmes",
        "video": {"video_id": video},
        "question": {"question_id": str(q), "question_text": "old?", "answer": {"label": "A"}},
        "metadata": {
            "clip_schemas": [
                {"clip_id": "c1", "scene_description": "generic"},
                {"clip_id": "c2", "scene_description": "repassed", "schema_attempt_context": "query_time_anchor_repass"},
            ],
            "anchor_repass": {"enabled": True},
            "clue_memory_graph": {"nodes": [{"node_id": "n1"}], "edges": []},
        },
    }


def test_derived_example_takes_new_identity_and_question_but_keeps_l1() -> None:
    frozen = _frozen("vid", 1)
    item = SimpleNamespace(example_id="video_holmes:test:vid:q7", video_id="vid",
                           question={"question_id": "7", "question_text": "new?", "answer": {"label": "C"}})
    out = derive_example(frozen, item)
    assert out["example_id"] == "video_holmes:test:vid:q7"
    assert out["question"]["question_text"] == "new?"
    assert out["question"]["answer"]["label"] == "C"
    assert out["metadata"]["clue_memory_graph"] == frozen["metadata"]["clue_memory_graph"]
    assert out["metadata"]["derived_from_example_id"] == "video_holmes:test:vid:q1"


def test_question_conditioned_repass_rows_are_dropped() -> None:
    out = derive_example(_frozen("vid", 1), SimpleNamespace(example_id="x:q2", video_id="vid", question={}))
    ids = [s["clip_id"] for s in out["metadata"]["clip_schemas"]]
    assert ids == ["c1"]
    assert "anchor_repass" not in out["metadata"]


def test_frozen_input_is_not_mutated() -> None:
    frozen = _frozen("vid", 1)
    derive_example(frozen, SimpleNamespace(example_id="x:q2", video_id="vid", question={"question_text": "new?"}))
    assert frozen["question"]["question_text"] == "old?"
    assert len(frozen["metadata"]["clip_schemas"]) == 2


def test_index_keeps_one_frozen_example_per_video(tmp_path) -> None:
    import json
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps(_frozen("vid", 1))); b.write_text(json.dumps(_frozen("vid", 2)))
    idx = index_frozen_by_video([a, b])
    assert list(idx) == ["vid"]
    assert idx["vid"]["example_id"].endswith(":q1")

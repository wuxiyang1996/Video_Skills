"""The human-annotation catalog: timed segment rows, optional inference shots, time-ordered."""
from scripts.eval.build_oracle_description_examples import annotation_clips


def _annotation():
    return {
        "Segment Description": [
            {"TimeRange": "00:24-01:34", "Description": "the elevator goes down"},
            {"TimeRange": "00:01-00:24", "Description": "she walks in"},
            {"TimeRange": "bad", "Description": "unparseable"},
            {"TimeRange": "01:40-01:50", "Description": ""},
        ],
        "Inference Shots": [
            {"Time": "01:28", "Clue": "the lift is empty", "Conclusion": "she was killed"},
        ],
    }


def test_clips_are_timed_sorted_and_carry_the_text() -> None:
    clips = annotation_clips(_annotation(), include_inference=True)
    assert [c["time_span"]["start_s"] for c in clips] == [1.0, 24.0, 88.0]
    assert clips[0]["scene_description"] == "she walks in"
    assert clips[-1]["granularity"] == "inference_shot"
    assert clips[-1]["scene_description"] == "the lift is empty she was killed"


def test_inference_shots_can_be_left_out() -> None:
    clips = annotation_clips(_annotation(), include_inference=False)
    assert len(clips) == 2 and all(c["granularity"] == "segment" for c in clips)


def test_empty_annotation_yields_no_clips() -> None:
    assert annotation_clips({}, include_inference=True) == []


def test_slim_drops_heavy_metadata_but_keeps_the_catalog() -> None:
    from scripts.eval.derive_full_question_examples import slim_example
    ex = {"metadata": {"clip_schemas": [1], "coarse_clip_schemas": [], "clue_memory_graph": {"big": 1}, "graph_compose": {}, "retrieval": {}}}
    slim_example(ex)
    assert set(ex["metadata"]) == {"clip_schemas", "coarse_clip_schemas", "retrieval"}

"""Shortlist selection for query-aware clip re-captioning.

First-pass captions are written without the question in context, so they record
a clip generically.  Measured on the heldout set, clips overlapping a gold
inference span share 11.1% of the gold wording against 8.9% for clips that do
not, which caps every reranker over that text at the same place.  The repass
fixes this, but its time-anchor trigger reached 0.5% of Video-Holmes candidates
and none on CG-Bench.
"""

from dataset_clip_wrapper.runners.run_staged_llm_pipeline import _retrieval_repass_spans


class _Span:
    def __init__(self, i):
        self.i = i
        self.start_s = float(i * 4)
        self.end_s = float(i * 4 + 4)

    def __repr__(self):
        return f"Span({self.i})"


def _fixture(descriptions):
    spans = [_Span(i) for i in range(len(descriptions))]
    derived = [{"clip_id": f"clip:v:fine:{i:04d}"} for i in range(len(descriptions))]
    schemas = [
        {"clip_id": f"clip:v:fine:{i:04d}", "scene_description": d}
        for i, d in enumerate(descriptions)
    ]
    return spans, derived, schemas


def _select(descriptions, question, top_n):
    spans, derived, schemas = _fixture(descriptions)
    picked, picked_derived = _retrieval_repass_spans(
        spans=spans, derived_clips=derived, clip_schemas=schemas,
        question_text=question, top_n=top_n,
    )
    assert len(picked) == len(picked_derived)
    return [s.i for s in picked]


def test_picks_the_clip_matching_the_question() -> None:
    got = _select(
        ["an empty street", "a man lights a cigarette", "a parked car"],
        "who lights a cigarette?", top_n=1,
    )
    assert got == [1]


def test_respects_the_shortlist_budget() -> None:
    got = _select(["cigarette smoke", "cigarette lighter", "an empty street"], "cigarette", top_n=2)
    assert len(got) == 2
    assert 2 not in got


def test_disabled_by_default_budget() -> None:
    assert _select(["a man lights a cigarette"], "cigarette", top_n=0) == []


def test_no_selection_without_a_usable_question() -> None:
    # Stopword-only questions carry no retrieval signal.
    assert _select(["a man lights a cigarette"], "what is it?", top_n=2) == []


def test_placeholder_captions_do_not_win() -> None:
    """Failed clips carry no text and must not displace a real match."""
    got = _select(
        ["clip schema generation failed", "a man lights a cigarette"],
        "cigarette", top_n=1,
    )
    assert got == [1]


def test_selection_is_deterministic_under_ties() -> None:
    descriptions = ["cigarette", "cigarette", "cigarette"]
    first = _select(descriptions, "cigarette", top_n=2)
    assert first == _select(descriptions, "cigarette", top_n=2)
    assert first == [0, 1]


def test_empty_input_is_handled() -> None:
    picked, derived = _retrieval_repass_spans(
        spans=[], derived_clips=[], clip_schemas=[], question_text="cigarette", top_n=4
    )
    assert picked == [] and derived == []

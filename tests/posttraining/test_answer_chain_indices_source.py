"""Retrieval-index sources for the answer-chain measurement.

The 'report' source takes rankings from a reranker eval report, which only
covers the examples that report was run on; derived sibling questions are
absent from it and were silently skipped, so a "full benchmark" run measured
263 of 1,837 questions.  BM25 over each example's own captions covers every
example with no learning.
"""

from scripts.eval.measure_answer_chain import bm25_indices, model_indices, retrieval_rank_indices


def _example(descs, question):
    return {
        "question": {"question_text": question, "options": []},
        "metadata": {"clip_schemas": [
            {"clip_id": f"c{i}", "scene_description": d, "retrieval_rank": len(descs) - i,
             "time_span": {"start_s": i * 4.0, "end_s": i * 4.0 + 4.0}}
            for i, d in enumerate(descs)
        ]},
    }


def test_report_source_is_empty_for_uncovered_examples() -> None:
    assert model_indices("video_holmes:test:v:q9", {"video_holmes:test:v:q1": [0, 1]}, 4) == []


def test_bm25_source_covers_every_example_and_prefers_matching_clips() -> None:
    ex = _example(["an empty street", "a man lights a cigarette", "a parked car"], "who lights a cigarette?")
    out = bm25_indices(ex, 2)
    assert out and out[0] == 1


def test_bm25_source_is_empty_only_when_there_is_nothing_to_match() -> None:
    assert bm25_indices(_example(["x"], "what is it?"), 2) == []


def test_retrieval_rank_source_follows_the_catalog_order() -> None:
    # retrieval_rank is 3,2,1 for clips 0,1,2 -> best rank is clip 2.
    assert retrieval_rank_indices(_example(["a", "b", "c"], "q"), 2) == [2, 1]

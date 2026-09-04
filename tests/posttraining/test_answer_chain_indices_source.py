"""Retrieval-index sources for the answer-chain measurement.

The 'report' source takes rankings from a reranker eval report, which only
covers the examples that report was run on; derived sibling questions are
absent from it and were silently skipped, so a "full benchmark" run measured
263 of 1,837 questions.  BM25 over each example's own captions covers every
example with no learning.
"""

from scripts.eval.measure_answer_chain import apply_temporal_nms, bm25_indices, model_indices, oracle_indices, retrieval_rank_indices


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


def test_oracle_source_picks_candidates_overlapping_gold_spans() -> None:
    ex = _example(["a", "b", "c", "d"], "q")          # clips at [0,4),[4,8),[8,12),[12,16)
    sup = {"segment_spans": [{"start_s": 9.0, "end_s": 11.0}]}
    assert oracle_indices(ex, sup, 4) == [2]


def test_temporal_nms_over_a_ranking_skips_overlapping_picks() -> None:
    ex = _example(["a", "b", "c", "d"], "q")   # clips [0,4),[4,8),[8,12),[12,16): touching, not overlapping
    ex["metadata"]["clip_schemas"][1]["time_span"] = {"start_s": 2.0, "end_s": 6.0}   # make clip 1 overlap clip 0
    assert apply_temporal_nms(ex, [0, 1, 2, 3], 3) == [0, 2, 3]


def test_bm25_returns_the_full_ranking_when_asked() -> None:
    ex = _example(["cigarette", "car", "street"], "cigarette")
    assert len(bm25_indices(ex, 0)) == 3

"""BM25 reference point for candidate reranking.

A learned reranker has to beat matching the question against the clip text
before its cost is justified, so this baseline is part of the evidence, not a
convenience.
"""

import pytest

from dataset_clip_wrapper.training.lexical_retrieval_baseline import (
    BM25,
    build_report,
    question_text,
    tokenize,
)


def test_tokenize_drops_stopwords_and_short_tokens() -> None:
    assert tokenize("What is the man holding in his hand?") == ["man", "holding", "hand"]


def test_tokenize_is_case_insensitive_and_keeps_digits() -> None:
    assert tokenize("Sausage 2 HOTDOG") == ["sausage", "hotdog"]


def test_bm25_prefers_the_document_containing_the_query_terms() -> None:
    docs = [tokenize("a man cooks a sausage"), tokenize("an empty parking lot")]
    bm25 = BM25(docs)
    query = tokenize("sausage")
    assert bm25.score(0, query) > bm25.score(1, query)


def test_bm25_scores_an_empty_document_zero() -> None:
    """Placeholder clips carry no text; they must not win by accident."""
    bm25 = BM25([tokenize("a man cooks a sausage"), []])
    assert bm25.score(1, tokenize("sausage")) == 0.0


def test_bm25_downweights_terms_common_to_every_candidate() -> None:
    common = BM25([tokenize("man walks"), tokenize("man runs"), tokenize("man sits")])
    rare = BM25([tokenize("man walks"), tokenize("dog runs"), tokenize("cat sits")])
    # "man" is in every document on the left and only one on the right.
    assert common.score(0, tokenize("man")) < rare.score(0, tokenize("man"))


def test_question_text_includes_the_options() -> None:
    state = {"question": {"question_text": "How many sausages?", "options": [{"text": "two roots"}]}}
    assert "sausages" in question_text(state)
    assert "roots" in question_text(state)


def _row(example, index, description, question="cooking a sausage"):
    import json
    return {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": json.dumps({"state_t": {"question": {"question_text": question, "options": []}}})},
        ],
        "metadata": {
            "source_example_id": example,
            "candidate_index": index,
            "retrieval_rank": index + 1,
            "gold_indices": [1],
            "candidate_entry": {"scene_description": description},
        },
    }


def test_report_shape_matches_the_pointwise_evaluator() -> None:
    report = build_report([
        _row("cg:1", 0, "an empty parking lot"),
        _row("cg:1", 1, "a man cooking a sausage"),
    ])
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["gold"] == [1]
    assert {entry["candidate_index"] for entry in result["ranking"]} == {0, 1}
    # The matching clip must outrank the unrelated one.
    ranked = sorted(result["ranking"], key=lambda entry: -entry["score"])
    assert ranked[0]["candidate_index"] == 1


def test_report_groups_candidates_by_example() -> None:
    report = build_report([_row("cg:1", 0, "x"), _row("cg:2", 0, "y")])
    assert [r["example_id"] for r in report["results"]] == ["cg:1", "cg:2"]

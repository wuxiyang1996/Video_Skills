"""Positive sampling in the OPD builder.

The builder kept one gold per example, which on Video-Holmes discarded roughly 35
of the ~36 positives each example already carries.  These pin the sampling and
the weight balance so raising the cap does not skew the teacher distribution.
"""

import pytest

from trainer.build_l2_pointwise_opd import build_opd_rows


def _fixture(n_candidates=6, gold=(1, 3, 5)):
    example = "cg_bench:1"
    chats = [
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
         "metadata": {"source_example_id": example, "candidate_index": i}}
        for i in range(n_candidates)
    ]
    report = {
        "adapter": "/tmp/adapter",
        "results": [{
            "example_id": example,
            "gold": list(gold),
            # Descending score, so index 0 is the student's top pick.
            "ranking": [{"candidate_index": i, "score": 1.0 - 0.1 * i} for i in range(n_candidates)],
        }],
    }
    return chats, report


def _indices(rows):
    return [r["state"]["candidate_index"] for r in rows]


def test_default_keeps_one_positive() -> None:
    rows, summary = build_opd_rows(*_fixture(), negatives_per_source=3)
    positives = [i for i in _indices(rows) if i in {1, 3, 5}]
    assert len(positives) == 1
    # The hardest positive is the lowest-scoring gold.
    assert positives == [5]
    assert summary["positives_per_source"] == 1


def test_raising_the_cap_keeps_more_golds_hardest_first() -> None:
    rows, _ = build_opd_rows(*_fixture(), negatives_per_source=3, positives_per_source=2)
    positives = [i for i in _indices(rows) if i in {1, 3, 5}]
    assert positives == [5, 3]


def test_non_positive_cap_keeps_every_gold() -> None:
    rows, _ = build_opd_rows(*_fixture(), negatives_per_source=3, positives_per_source=0)
    assert sorted(i for i in _indices(rows) if i in {1, 3, 5}) == [1, 3, 5]


def test_negatives_are_the_highest_scoring_non_golds() -> None:
    rows, _ = build_opd_rows(*_fixture(), negatives_per_source=2, positives_per_source=1)
    assert sorted(i for i in _indices(rows) if i not in {1, 3, 5}) == [0, 2]


def test_positive_and_negative_halves_stay_balanced() -> None:
    for cap in (1, 2, 0):
        rows, _ = build_opd_rows(*_fixture(), negatives_per_source=3, positives_per_source=cap)
        pos = sum(r["state"]["sample_weight"] for r in rows if r["state"]["candidate_index"] in {1, 3, 5})
        neg = sum(r["state"]["sample_weight"] for r in rows if r["state"]["candidate_index"] not in {1, 3, 5})
        assert pos == pytest.approx(0.5), cap
        assert neg == pytest.approx(0.5), cap


def test_examples_whose_gold_escapes_the_candidate_pool_are_dropped() -> None:
    chats, report = _fixture()
    report["results"][0]["gold"] = [1, 99]
    rows, summary = build_opd_rows(chats, report)
    assert rows == []
    assert summary["excluded_gold_outside_candidate_pool"] == 1

"""Greedy top-k with temporal non-max suppression.

A reranker can spend its whole top-k on one region and cover one gold segment
four times.  This rule has no free parameter: any candidate overlapping an
already-chosen pick is skipped.
"""

from dataset_clip_wrapper.training.evaluate_l2_pointwise_adapter import _topk_temporal_nms


def _sp(a, b):
    return {"start_s": float(a), "end_s": float(b)}


def test_overlapping_candidates_are_skipped_in_favour_of_later_ranks() -> None:
    spans = {0: _sp(0, 4), 1: _sp(2, 6), 2: _sp(10, 14), 3: _sp(3, 5), 4: _sp(20, 24)}
    # rank order 0,1,2,3,4 -- 1 and 3 overlap 0; 2 and 4 do not.
    assert _topk_temporal_nms([0, 1, 2, 3, 4], spans, top_k=3) == [0, 2, 4]


def test_no_overlap_reduces_to_plain_topk() -> None:
    spans = {i: _sp(i * 10, i * 10 + 4) for i in range(5)}
    assert _topk_temporal_nms([3, 1, 4, 0, 2], spans, top_k=3) == [3, 1, 4]


def test_touching_spans_do_not_count_as_overlap() -> None:
    spans = {0: _sp(0, 4), 1: _sp(4, 8)}
    assert _topk_temporal_nms([0, 1], spans, top_k=2) == [0, 1]


def test_candidates_without_spans_are_kept_in_rank_order() -> None:
    spans = {0: _sp(0, 4)}
    assert _topk_temporal_nms([0, 7, 8], spans, top_k=3) == [0, 7, 8]


def test_returns_fewer_than_k_when_everything_overlaps() -> None:
    spans = {i: _sp(0, 4) for i in range(4)}
    assert _topk_temporal_nms([0, 1, 2, 3], spans, top_k=4) == [0]


def test_stops_at_k() -> None:
    spans = {i: _sp(i * 10, i * 10 + 4) for i in range(6)}
    assert len(_topk_temporal_nms(list(range(6)), spans, top_k=2)) == 2

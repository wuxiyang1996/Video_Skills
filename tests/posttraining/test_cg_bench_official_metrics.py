"""CG-Bench official grounding metrics, checked against the reference semantics.

Reference: CG-Bench/CG-Bench ``run/utils.py`` -- merge both interval lists, then
score a set-IoU over the timeline.  These cases pin the behaviours that differ
from a per-gold best-match IoU.
"""

import pytest

from dataset_clip_wrapper.training.cg_bench_official_metrics import (
    MAX_OFFICIAL_INTERVALS,
    intervals_iou,
    merge_intervals,
    score_questions,
)


def test_merge_overlapping_intervals() -> None:
    assert merge_intervals([[10, 20], [15, 25], [40, 50]]) == [[10, 25], [40, 50]]


def test_merge_treats_touching_intervals_as_contiguous() -> None:
    assert merge_intervals([[0, 10], [10, 20]]) == [[0, 20]]


def test_exact_match_is_one() -> None:
    assert intervals_iou([[60, 70]], [[60, 70]]) == pytest.approx(1.0)


def test_disjoint_is_zero() -> None:
    assert intervals_iou([[0, 30]], [[60, 70]]) == 0.0


def test_coarse_clip_covering_a_clue_is_capped_by_its_own_duration() -> None:
    """A 30s clip containing a 9s clue scores 9/30 -- the granularity ceiling."""
    assert intervals_iou([[60, 90]], [[60, 69]]) == pytest.approx(9 / 30)


def test_over_prediction_is_penalised() -> None:
    """Union grows with each extra interval, so more predictions score worse.

    This inverts the usual 'larger budget is better' intuition that holds for
    recall-style metrics.
    """
    one = intervals_iou([[60, 90]], [[60, 69]])
    five = intervals_iou([[0, 30], [30, 60], [60, 90], [90, 120], [120, 150]], [[60, 69]])
    assert one == pytest.approx(9 / 30)
    assert five == pytest.approx(9 / 150)
    assert five < one


def test_multiple_gold_intervals_use_set_semantics() -> None:
    # Intersection 10+10=20; union = 60 + 20 - 20 = 60.
    assert intervals_iou([[0, 30], [50, 80]], [[20, 30], [50, 60]]) == pytest.approx(20 / 60)


def test_score_questions_reports_miou_and_threshold_recalls() -> None:
    out = score_questions([
        ([[60, 70]], [[60, 70]]),        # IoU 1.0
        ([[0, 30]], [[60, 70]]),         # IoU 0.0
    ])
    assert out["questions"] == 2
    assert out["miou"] == pytest.approx(50.0)
    # One of two questions clears every threshold.
    assert out["rec@0.1"] == pytest.approx(50.0)
    assert out["rec@0.5"] == pytest.approx(50.0)
    assert out["rec@IoU"] == pytest.approx(50.0)


def test_predictions_beyond_the_protocol_limit_are_dropped() -> None:
    six = [[i * 10, i * 10 + 5] for i in range(6)]
    out = score_questions([(six, [[0, 5]])])
    assert out["predictions_truncated_to_limit"] == 1
    # Scored as if only the first five were submitted.
    assert out["miou"] == pytest.approx(
        100.0 * intervals_iou(six[:MAX_OFFICIAL_INTERVALS], [[0, 5]])
    )

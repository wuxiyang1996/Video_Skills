"""Middle-band supervision in the dataset OPD builder.

The teacher labels a candidate positive at >=0.50 and negative at <=0.05 on
Video-Holmes, and everything between was dropped.  Measured over the OPD pool
that band holds 72% of all candidates (median 59 per example) against the 3+3
actually trained on, and it is where the top-k decision is made.
"""

import pytest

from trainer.build_l2_dataset_opd import _opd_row


def _row(score, relevant):
    return _opd_row(
        {"example_id": "video_holmes:test:v:q1", "dataset": "video_holmes", "question": {}},
        {"clip_id": "c", "time_span": {"start_s": 0.0, "end_s": 4.0}, "scene_description": "x"},
        candidate_index=0,
        score=score,
        relevant=relevant,
        sample_weight=0.1,
    )


def _p_true(row):
    return row["teacher"]["action_probs"]["relevant_true"]


def test_middle_band_uses_the_teacher_score_as_the_probability() -> None:
    assert _p_true(_row(0.30, None)) == pytest.approx(0.30)
    assert _p_true(_row(0.45, None)) == pytest.approx(0.45)


def test_middle_band_probability_stays_off_the_hard_ends() -> None:
    # Never fully committed either way: these are the ambiguous candidates.
    assert _p_true(_row(0.0, None)) == pytest.approx(0.05)
    assert _p_true(_row(1.0, None)) == pytest.approx(0.95)


def test_action_probs_remain_a_distribution() -> None:
    for score, relevant in [(0.3, None), (0.9, True), (0.01, False)]:
        probs = _row(score, relevant)["teacher"]["action_probs"]
        assert probs["relevant_true"] + probs["relevant_false"] == pytest.approx(1.0)


def test_hard_labels_are_unchanged_by_the_middle_band_path() -> None:
    # Positive branch: 0.50 + 0.48*score, clipped to [0.55, 0.98].
    assert _p_true(_row(0.75, True)) == pytest.approx(0.86)
    # Negative branch: 0.50*score, clipped to [0.02, 0.45].
    assert _p_true(_row(0.04, False)) == pytest.approx(0.02)


def test_middle_band_is_ordered_between_the_two_hard_branches() -> None:
    """A middle candidate must not look more relevant than a true positive."""
    assert _p_true(_row(0.04, False)) < _p_true(_row(0.30, None)) < _p_true(_row(0.75, True))


def test_annotation_score_is_recorded_for_audit() -> None:
    assert _row(0.31, None)["teacher"]["annotation_score"] == pytest.approx(0.31)

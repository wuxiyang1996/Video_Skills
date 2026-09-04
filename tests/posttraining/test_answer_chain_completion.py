"""End-to-end accuracy splits into completion x accuracy-when-answered.

An abstention scores as wrong on both benchmarks, so the split says which half
is losing the points.
"""

import pytest

from scripts.eval.analyze_answer_chain_completion import summarise


def _rollout(example, label, reasons=()):
    return {
        "example_id": example,
        "final_answer": {"label": label},
        "failure_reasons": list(reasons),
    }


GOLD = {"video_holmes:test:v:q1": "A", "video_holmes:test:v:q2": "B"}


def test_abstention_counts_as_wrong_end_to_end() -> None:
    out = summarise([_rollout("video_holmes:test:v:q1", "A"),
                     _rollout("video_holmes:test:v:q2", None)], GOLD)
    assert out["accuracy_end_to_end"] == pytest.approx(50.0)
    # But the model was right every time it committed.
    assert out["accuracy_when_answered"] == pytest.approx(100.0)


def test_completion_and_abstention_are_complements() -> None:
    out = summarise([_rollout("video_holmes:test:v:q1", "A"),
                     _rollout("video_holmes:test:v:q2", None)], GOLD)
    assert out["completion_rate"] + out["abstention_rate"] == pytest.approx(100.0)


def test_wrong_answers_are_separated_from_abstentions() -> None:
    out = summarise([_rollout("video_holmes:test:v:q1", "C")], GOLD)
    assert out["accuracy_end_to_end"] == 0.0
    assert out["abstention_rate"] == 0.0
    assert out["completion_rate"] == pytest.approx(100.0)


def test_label_comparison_ignores_case_and_padding() -> None:
    out = summarise([_rollout("video_holmes:test:v:q1", " a ")], GOLD)
    assert out["accuracy_end_to_end"] == pytest.approx(100.0)


def test_rollouts_without_a_gold_answer_are_excluded() -> None:
    out = summarise([_rollout("video_holmes:test:v:unknown", "A")], GOLD)
    assert out["rollouts"] == 0


def test_abstention_reasons_are_reported() -> None:
    out = summarise([_rollout("video_holmes:test:v:q1", None, ["no_final_answer"])], GOLD)
    assert out["top_abstention_reasons"][0][0] == "no_final_answer"


def test_distinct_examples_are_counted_not_rollouts() -> None:
    out = summarise([_rollout("video_holmes:test:v:q1", "A")] * 5, GOLD)
    assert out["rollouts"] == 5
    assert out["distinct_examples"] == 1

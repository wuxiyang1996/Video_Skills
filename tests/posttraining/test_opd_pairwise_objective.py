"""Ranking objective for OPD.

The pointwise KL loss calibrates each candidate independently, but the reported
metric is a top-k ranking over an example's candidates, so nothing in that loss
compares candidates to one another.  These pin the pairing and the hinge.
"""

import pytest

from trainer.train_l2_pointwise_opd import build_ranking_pairs, pairwise_margin_loss

torch = pytest.importorskip("torch")


def _rows(spec):
    """spec: list of (example_id, teacher_score)."""
    return [
        {"source_example_id": e, "teacher_score": s, "weight": 1.0}
        for e, s in spec
    ]


def test_pairs_only_form_within_an_example() -> None:
    rows = _rows([("a", 0.9), ("a", 0.1), ("b", 0.8), ("b", 0.2)])
    pairs = build_ranking_pairs(rows, max_pairs_per_example=0)
    assert sorted(pairs) == [(0, 1), (2, 3)]


def test_pairs_are_ordered_better_first() -> None:
    rows = _rows([("a", 0.1), ("a", 0.9)])
    assert build_ranking_pairs(rows, max_pairs_per_example=0) == [(1, 0)]


def test_ties_produce_no_pair() -> None:
    rows = _rows([("a", 0.4), ("a", 0.4)])
    assert build_ranking_pairs(rows, max_pairs_per_example=0) == []


def test_budget_keeps_the_widest_teacher_gaps() -> None:
    rows = _rows([("a", 1.0), ("a", 0.5), ("a", 0.0)])
    # Gaps: (0,2)=1.0, (0,1)=0.5, (1,2)=0.5 -- the widest survives a budget of one.
    assert build_ranking_pairs(rows, max_pairs_per_example=1) == [(0, 2)]


def test_middle_band_candidates_participate_in_ranking() -> None:
    """A graded middle candidate should be ordered against both hard ends."""
    rows = _rows([("a", 0.9), ("a", 0.3), ("a", 0.02)])
    assert sorted(build_ranking_pairs(rows, max_pairs_per_example=0)) == [(0, 1), (0, 2), (1, 2)]


def test_hinge_is_zero_once_the_gap_clears_the_margin() -> None:
    loss = pairwise_margin_loss(torch.tensor(3.0), torch.tensor(1.0), margin=1.0)
    assert float(loss) == pytest.approx(0.0)


def test_hinge_penalises_an_insufficient_gap() -> None:
    loss = pairwise_margin_loss(torch.tensor(1.2), torch.tensor(1.0), margin=1.0)
    assert float(loss) == pytest.approx(0.8)


def test_hinge_penalises_an_inverted_pair_most() -> None:
    inverted = pairwise_margin_loss(torch.tensor(0.0), torch.tensor(2.0), margin=1.0)
    assert float(inverted) == pytest.approx(3.0)


def test_hinge_is_differentiable_while_active() -> None:
    high = torch.tensor(1.0, requires_grad=True)
    low = torch.tensor(1.0, requires_grad=True)
    pairwise_margin_loss(high, low, margin=1.0).backward()
    assert float(high.grad) == pytest.approx(-1.0)
    assert float(low.grad) == pytest.approx(1.0)

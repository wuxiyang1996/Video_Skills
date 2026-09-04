"""Scoring-mode contract for the L2 pointwise reranker.

The legacy ``sequence_logprob`` estimator differences two long, nearly identical
sequence log-likelihoods whose per-token logits are bf16-quantised, which
collapses candidate scores onto multiples of ~0.125 and creates top-k ties.
``decision_logit`` reads the log-odds off the single divergent token instead.
"""

import pytest

from dataset_clip_wrapper.training.sft_common import decision_position as _decision_position


def test_decision_position_finds_the_single_divergent_token() -> None:
    true_ids = [10, 11, 12, 900, 13, 14]
    false_ids = [10, 11, 12, 901, 13, 14]
    prefix, true_token, false_token = _decision_position(true_ids, false_ids)
    assert prefix == 3
    assert (true_token, false_token) == (900, 901)
    # The scored prefix must be shared by both variants, so a single forward pass
    # over it is valid for both.
    assert true_ids[:prefix] == false_ids[:prefix]


def test_decision_position_handles_differing_completion_lengths() -> None:
    # "false" may tokenise to more tokens than "true"; divergence is still single.
    prefix, true_token, false_token = _decision_position([10, 900, 13], [10, 901, 902, 13])
    assert prefix == 1
    assert (true_token, false_token) == (900, 901)


@pytest.mark.parametrize("true_ids,false_ids", [([1, 2], [1, 2]), ([1, 2], [9, 2]), ([], [1])])
def test_decision_position_rejects_non_diverging_completions(true_ids, false_ids) -> None:
    # Identical, immediately-divergent, or empty completions mean the prompt/JSON
    # contract changed; failing loudly beats silently scoring the wrong position.
    with pytest.raises(ValueError):
        _decision_position(true_ids, false_ids)


def test_logit_gap_equals_log_odds() -> None:
    """log_softmax shares one logsumexp, so it cancels in the difference."""
    torch = pytest.importorskip("torch")
    logits = torch.tensor([2.5, -1.25, 4.0, 0.5], dtype=torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)
    assert float(logits[0] - logits[1]) == pytest.approx(float(log_probs[0] - log_probs[1]), abs=1e-6)


def test_bf16_logits_quantise_but_fp32_does_not() -> None:
    """Reproduces the mechanism behind the observed 0.125 score grid."""
    torch = pytest.importorskip("torch")
    # Logit magnitudes in this model sit around 10-30, where bf16 has ~0.125 ULP.
    exact = torch.tensor([16.03, 16.05], dtype=torch.float32)
    quantised = exact.to(torch.bfloat16).float()
    assert float(quantised[0]) == float(quantised[1])  # a spurious exact tie
    assert float(exact[0]) != float(exact[1])
    # bf16 steps by 0.125 in this range, which is the score grid seen in the reports.
    assert float(torch.tensor(16.125, dtype=torch.bfloat16)) - float(torch.tensor(16.0, dtype=torch.bfloat16)) == 0.125

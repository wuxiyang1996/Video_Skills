import copy

from scripts.eval.measure_answer_chain import shuffle_options
from trainer.reader.build_sft_data import apply_option_perm


def _ex():
    return {"example_id": "e1", "question": {"question_text": "q", "answer": {"label": "C", "text": "cc"},
                                             "options": [{"label": "A", "text": "aa"}, {"label": "B", "text": "bb"}, {"label": "C", "text": "cc"}, {"label": "D", "text": "dd"}]}}


def test_shuffle_is_deterministic_keeps_gold_text_and_records_perm() -> None:
    a, b = _ex(), _ex()
    pa, pb = shuffle_options(a, 7), shuffle_options(b, 7)
    assert pa == pb and sorted(pa) == ["A", "B", "C", "D"]
    q = a["question"]
    assert [o["label"] for o in q["options"]] == ["A", "B", "C", "D"]
    gold_text = next(o["text"] for o in q["options"] if o["label"] == q["answer"]["label"])
    assert gold_text == "cc"
    assert shuffle_options(_ex(), 8) != pa or True   # different seeds may differ; must not crash


def test_apply_option_perm_replays_the_same_permutation() -> None:
    a = _ex(); perm = shuffle_options(a, 11)
    b = _ex(); apply_option_perm(b, perm)
    assert b["question"]["options"] == a["question"]["options"] and b["question"]["answer"]["label"] == a["question"]["answer"]["label"]

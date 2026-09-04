from trainer.build_l2_pointwise_opd import build_opd_rows


def test_build_opd_rows_uses_hardest_positive_and_top_negative() -> None:
    chats = []
    for index in (1, 2, 3):
        chats.append({
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
            "metadata": {"source_example_id": "x", "candidate_index": index},
        })
    report = {
        "adapter": "student",
        "results": [{
            "example_id": "x", "gold": [1, 2],
            "ranking": [
                {"candidate_index": 3, "score": 4.0},
                {"candidate_index": 1, "score": 2.0},
                {"candidate_index": 2, "score": -1.0},
            ],
        }],
    }
    rows, summary = build_opd_rows(chats, report, negatives_per_source=1)
    assert [row["state"]["candidate_index"] for row in rows] == [2, 3]
    assert rows[0]["teacher"]["action_probs"]["relevant_true"] == 0.98
    assert rows[1]["teacher"]["action_probs"]["relevant_false"] == 0.98
    assert summary["positive_rows"] == summary["negative_rows"] == 1


def test_build_opd_rows_excludes_gold_outside_label_independent_pool() -> None:
    chats = [{
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "metadata": {"source_example_id": "x", "candidate_index": 1},
    }]
    report = {"results": [{
        "example_id": "x", "gold": [9],
        "ranking": [{"candidate_index": 1, "score": 1.0}],
    }]}
    rows, summary = build_opd_rows(chats, report)
    assert rows == []
    assert summary["excluded_gold_outside_candidate_pool"] == 1

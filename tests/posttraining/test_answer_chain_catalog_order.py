from scripts.eval.measure_answer_chain import order_catalog_by_time


def test_rows_are_interleaved_by_start_time_and_untimed_rows_go_last() -> None:
    ex = {"metadata": {"clip_schemas": [
        {"clip_id": "narrative:1", "time_span": {"start_s": 0.0, "end_s": 30.0}},
        {"clip_id": "narrative:2", "time_span": {"start_s": 30.0, "end_s": 60.0}},
        {"clip_id": "meta"},
        {"clip_id": "c1", "time_span": {"start_s": 0.0, "end_s": 4.0}},
        {"clip_id": "c2", "time_span": {"start_s": 33.0, "end_s": 37.0}},
    ]}}
    order_catalog_by_time(ex)
    assert [r["clip_id"] for r in ex["metadata"]["clip_schemas"]] == ["narrative:1", "c1", "narrative:2", "c2", "meta"]

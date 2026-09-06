from scripts.eval.build_asr_rows_catalog import dialogue_rows


def test_dialogue_rows_group_by_window_and_keep_order() -> None:
    segs = [{"start_s": 0.0, "end_s": 2.0, "text": "a"}, {"start_s": 20.0, "end_s": 25.0, "text": "b"},
            {"start_s": 31.0, "end_s": 33.0, "text": "c"}, {"start_s": 70.0, "end_s": 71.0, "text": "d"}]
    rows = dialogue_rows(segs, window_s=30.0)
    assert [r["time_span"] for r in rows] == [{"start_s": 0.0, "end_s": 25.0}, {"start_s": 31.0, "end_s": 33.0},
                                             {"start_s": 70.0, "end_s": 71.0}]
    assert rows[0]["scene_description"].endswith("a b") and rows[0]["clip_id"] == "dialogue:1"
    assert dialogue_rows([]) == []

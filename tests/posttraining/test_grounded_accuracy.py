from scripts.eval.grounded_accuracy import prose_spans, ranks_in_prose


def test_prose_clip_references_are_parsed_as_ranks() -> None:
    assert ranks_in_prose("Clip 12 shows X; clips 3-5 and clip 7 confirm it") == [3, 4, 5, 7, 12]
    assert ranks_in_prose("no citations here") == []
    assert ranks_in_prose("clips 9 to 4") == [9]          # inverted range -> single


def test_prose_spans_map_ranks_through_indices() -> None:
    spans = [{"start_s": 0.0, "end_s": 4.0}, {"start_s": 4.0, "end_s": 8.0}, {"start_s": 8.0, "end_s": 12.0}]
    rollout = {"thinking": "see clip 2 and clip 9"}
    assert prose_spans(rollout, [2, 0], spans) == [spans[0]]   # rank 2 -> catalog index 0; rank 9 out of range

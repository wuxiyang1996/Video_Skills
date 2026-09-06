from scripts.eval.count_events_skill import choose_option, merge_instances, option_number


def test_option_numbers_are_parsed_from_words_and_digits() -> None:
    assert option_number("There were three drinking scenes in all.") == 3
    assert option_number("twice") == 2 and option_number("once") == 1 and option_number("There are 4 shots") == 4
    assert option_number("A single major event.") == 1
    assert option_number("The hero visits the market") is None


def test_choose_option_prefers_exact_then_nearest() -> None:
    opts = {"A": "two times", "B": "three times", "C": "five times", "D": "once"}
    assert choose_option(opts, 3) == ("B", "exact")
    assert choose_option(opts, 4) == ("B", "nearest")          # tie 3 vs 5 -> lower label wins deterministically
    assert choose_option({"A": "the market", "B": "the beach"}, 2) == (None, "no_numeric_options")


def test_merge_dedupes_by_link_and_by_time_adjacency_of_same_identity() -> None:
    inst = [{"id": "i1", "start_s": 10, "end_s": 20, "identity": "man drinks"},
            {"id": "i2", "start_s": 25, "end_s": 30, "identity": "man drinks"},            # within 15 s of i1 -> same
            {"id": "i3", "start_s": 200, "end_s": 210, "identity": "man drinks"},          # far -> new
            {"id": "i4", "start_s": 300, "end_s": 305, "identity": "woman drinks", "same_as": "i3"}]  # linked -> same as i3
    merged = merge_instances(inst)
    assert [m["id"] for m in merged] == ["i1", "i3"] and merged[0]["end_s"] == 30 and merged[1]["end_s"] == 305

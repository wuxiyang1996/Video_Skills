import json

from scripts.eval.measure_answer_chain import merge_probe_targets, probe_window, window_grounder


def test_probe_window_is_centred_and_clipped_to_the_video() -> None:
    assert probe_window({"start_s": 10.0, "end_s": 14.0}, 30.0, 100.0) == {"start_s": 0.0, "end_s": 27.0}
    assert probe_window({"start_s": 90.0, "end_s": 94.0}, 30.0, 100.0) == {"start_s": 77.0, "end_s": 100.0}


def test_overlapping_windows_merge_and_keep_their_subquestions() -> None:
    merged = merge_probe_targets([({"start_s": 40.0, "end_s": 70.0}, "who enters?"), ({"start_s": 0.0, "end_s": 30.0}, None),
                                  ({"start_s": 60.0, "end_s": 90.0}, "what is said?")])
    assert merged == [({"start_s": 0.0, "end_s": 30.0}, []), ({"start_s": 40.0, "end_s": 90.0}, ["who enters?", "what is said?"])]


def test_window_grounder_sends_frames_and_dialogue_and_returns_a_row() -> None:
    class _C:
        def chat(self, messages):
            payload = json.loads(messages[1]["content"][0]["text"])
            assert payload["dialogue"] == [{"start_s": 1.0, "end_s": 2.0, "text": "hi"}] and payload["questions"] == ["q"]
            assert sum(1 for p in messages[1]["content"] if p["type"] == "image_url") == 3
            return "The man in the hat says 'hi' and leaves."
    row = window_grounder(_C(), {"start_s": 0.0, "end_s": 30.0}, ["a", "b", "c"], [{"start_s": 1.0, "end_s": 2.0, "text": "hi"}], ["q"])
    assert row["observation"].startswith("The man") and row["frames_seen"] == 3 and row["dialogue_lines"] == 1
    assert window_grounder(_C(), {"start_s": 0.0, "end_s": 30.0}, [], [], ["q"]) is None

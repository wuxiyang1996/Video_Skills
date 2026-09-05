"""Timeline skill: parse numbered events and permutation options, locate each
event's clip, assemble the order from clip start times deterministically."""
import json

from scripts.eval.measure_answer_chain import localize_events, order_by_time, timeline_events


def _example():
    return {
        "question": {"question_text": "Please arrange the events in order:\n① wakes up ② drinks water ③ goes out",
                     "options": [{"label": "A", "text": "①②③"}, {"label": "B", "text": "②①③"}, {"label": "C", "text": "③②①"}]},
        "metadata": {"clip_schemas": [
            {"clip_id": f"c{i}", "scene_description": f"scene {i}", "time_span": {"start_s": i * 10.0, "end_s": i * 10.0 + 10.0}}
            for i in range(5)
        ]},
    }


def test_timeline_events_parses_events_and_permutation_options() -> None:
    events, options = timeline_events(_example()["question"])
    assert events == [("①", "wakes up"), ("②", "drinks water"), ("③", "goes out")]
    assert options == {"A": "①②③", "B": "②①③", "C": "③②①"}


def test_timeline_events_is_empty_for_ordinary_questions() -> None:
    q = {"question_text": "why?", "options": [{"label": "A", "text": "fear"}, {"label": "B", "text": "anger"}]}
    assert timeline_events(q) == ([], {})


def test_localize_and_order_pick_the_option_matching_clip_start_times() -> None:
    class _C:
        def chat(self, messages):
            self.payload = json.loads(messages[-1]["content"])
            return '{"events": {"①": 4, "②": 2, "③": 5}}'      # ranks -> clips 3, 1, 4 -> starts 30, 10, 40
    client = _C(); ex = _example(); indices = [0, 1, 2, 3, 4]
    events, options = timeline_events(ex["question"])
    located = localize_events(client, ex, indices, events)
    assert located == {"①": 4, "②": 2, "③": 5}
    assert [c["id"] for c in client.payload["events"]] == ["①", "②", "③"] and len(client.payload["clips"]) == 5
    assert order_by_time(located, indices, ex, options) == "B"        # ② (10s) < ① (30s) < ③ (40s)


def test_order_by_time_falls_back_when_an_event_is_unlocated_or_nothing_matches() -> None:
    ex = _example(); indices = [0, 1, 2, 3, 4]; _, options = timeline_events(ex["question"])
    assert order_by_time({"①": 1, "②": None, "③": 3}, indices, ex, options) is None
    assert order_by_time({"①": 1, "②": 3, "③": 2}, indices, ex, options) is None   # ①③② is not an option
    assert order_by_time({"①": 99, "②": 1, "③": 2}, indices, ex, options) is None  # rank out of range

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


def test_concordance_commits_the_uniquely_best_option_and_tolerates_one_unlocated_event() -> None:
    from scripts.eval.measure_answer_chain import order_by_concordance
    ex = {
        "question": {"question_text": "order:\n① a ② b ③ c ④ d",
                     "options": [{"label": "A", "text": "①②③④"}, {"label": "B", "text": "②①③④"}, {"label": "C", "text": "④③②①"}]},
        "metadata": {"clip_schemas": [
            {"clip_id": f"c{i}", "scene_description": "s", "time_span": {"start_s": i * 10.0, "end_s": i * 10.0 + 10.0}} for i in range(6)
        ]},
    }
    indices = [0, 1, 2, 3, 4, 5]
    options = {"A": "①②③④", "B": "②①③④", "C": "④③②①"}
    # ② at 10s, ① at 20s, ③ at 40s; ④ unlocated -> B orders all three located pairs correctly, A gets 2/3, C 0/3
    assert order_by_concordance({"①": 3, "②": 2, "③": 5, "④": None}, indices, ex, options) == "B"
    # too few located events -> None
    assert order_by_concordance({"①": 3, "②": 2, "③": None, "④": None}, indices, ex, options) is None
    # a tie between two options -> None
    assert order_by_concordance({"①": 3, "②": 3, "③": 5, "④": 6}, indices, ex, {"A": "①②③④", "B": "②①③④"}) is None

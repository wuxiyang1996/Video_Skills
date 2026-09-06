import json

from scripts.eval.mine_failure_subtrajectories import ONTOLOGY, fit_one, summarize


class _Client:
    def chat(self, messages):
        return json.dumps({"failure_code": "wrong_temporal_order",
                           "subtrajectory": [{"skill": "localize_clue", "needs": "text_reread", "what": "find both events"},
                                             {"skill": "order_events_by_time", "needs": "deterministic", "what": "sort by clip start"},
                                             {"skill": "made_up_skill", "needs": "magic", "what": "x"}],
                           "pattern": "locate events, order by time, pick matching option"})


def test_fit_maps_unknown_skills_and_needs_to_safe_values() -> None:
    out = fit_one(_Client(), {"question": "q", "options": {}, "answer": "A", "explanation": "e"}, {"label": "B", "thinking": "t"})
    assert out["failure_code"] == "wrong_temporal_order"
    assert [s["skill"] for s in out["subtrajectory"]] == ["localize_clue", "order_events_by_time", "other"]
    assert [s["needs"] for s in out["subtrajectory"]] == ["text_reread", "deterministic", "text_reread"]
    assert all(s in ONTOLOGY for s in [x["skill"] for x in out["subtrajectory"]])


def test_summary_counts_non_text_and_deterministic_failures() -> None:
    recs = [{"fit": {"failure_code": "wrong_temporal_order", "subtrajectory": [{"skill": "a", "needs": "deterministic"}]}},
            {"fit": {"failure_code": "other", "subtrajectory": [{"skill": "a", "needs": "text_reread"}]}},
            {"fit": {"failure_code": "dialogue_needed", "subtrajectory": [{"skill": "b", "needs": "listen_dialogue"}, {"skill": "a", "needs": "text_reread"}]}}]
    s = summarize(recs)
    assert s["failures"] == 3 and s["failures_with_any_non_text_step"] == 2 and s["failures_with_deterministic_step"] == 1
    assert s["distinct_subtrajectories"] == 2 and s["top5_subtrajectory_coverage"] == 3

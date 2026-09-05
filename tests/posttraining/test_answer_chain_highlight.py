"""--highlight-from: the whole catalog stays in the prompt, the retriever's picks
are flagged as ranks inside it, and the flag is refused for graph conditions."""
import json

import pytest

from scripts.eval.measure_answer_chain import direct_answer, main


class _FakeClient:
    def __init__(self):
        self.payload = None

    def chat(self, messages):
        self.payload = json.loads(messages[-1]["content"])
        return '{"label": "B"}'


def _example():
    return {
        "question": {"question_text": "q", "options": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"}]},
        "metadata": {"clip_schemas": [
            {"clip_id": f"c{i}", "scene_description": f"scene {i}", "time_span": {"start_s": i * 4.0, "end_s": i * 4.0 + 4.0}}
            for i in range(5)
        ]},
    }


def test_highlight_marks_ranks_of_the_retriever_picks_over_the_whole_catalog() -> None:
    client = _FakeClient()
    out = direct_answer(client, _example(), indices=[0, 1, 2, 3, 4], highlight=[3, 1, 9])
    assert out["final_answer"]["label"] == "B"
    assert len(client.payload["clips"]) == 5
    assert client.payload["likely_key_clips"] == [4, 2]   # 1-based ranks; 9 is not in the catalog
    assert "may be wrong" in client.payload["likely_key_clips_note"]


def test_no_highlight_leaves_the_payload_unchanged() -> None:
    client = _FakeClient()
    direct_answer(client, _example(), indices=[1, 2])
    assert "likely_key_clips" not in client.payload
    assert [c["rank"] for c in client.payload["clips"]] == [1, 2]


def test_highlight_is_refused_for_graph_conditions(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--l1-glob", "x", "--output", "o.json", "--indices-from", "all", "--highlight-from", "bm25", "--conditions", "direct", "model"])
    assert "only applies to the direct, hybrid, probe and graph2 conditions" in capsys.readouterr().err


def test_answer_model_budget_grows_with_reasoning_effort() -> None:
    from scripts.eval.measure_answer_chain import answer_model_budget
    assert answer_model_budget("minimal", None) == 1800
    assert answer_model_budget("high", None) == 8000
    assert answer_model_budget("high", 3000) == 3000


def test_dump_rollout_appends_one_json_line_per_record(tmp_path) -> None:
    import threading
    from scripts.eval.measure_answer_chain import dump_rollout
    path = tmp_path / "r.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        lock = threading.Lock()
        dump_rollout(handle, lock, {"example_id": "e1", "rollout": {"x": 1}})
        dump_rollout(handle, lock, {"example_id": "e2", "rollout": {"y": object()}})   # non-JSON values are stringified
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["example_id"] for l in lines] == ["e1", "e2"]
    assert lines[0]["rollout"] == {"x": 1}


def _rollout():
    return {
        "final_answer": {"label": "B", "confidence": 0.2},
        "acceptance_status": "rejected",
        "metadata": {"answer_step_diagnostics": {
            "r3": {"hypotheses": [{"option_label": "A"}], "backend": "llm"},
            "r9": {"scored_hypothesis": {"option_label": "A", "support_score": 0.6, "contradiction_score": 0.0,
                                         "llm_reasoning": "dim lighting suggests A"}, "backend": "llm"},
            "r10": {"scored_hypothesis": {"option_label": "B", "support_score": 0.65, "contradiction_score": 0.1,
                                          "llm_reasoning": "wire suggests B"}, "backend": "llm"},
            "r14": {"best_hypothesis": {"option_label": "B", "llm_reasoning": "wire suggests B"}, "backend": "llm"},
        }},
    }


def test_extract_findings_collects_per_option_reasoning_and_the_graph_vote() -> None:
    from scripts.eval.measure_answer_chain import extract_findings
    out = extract_findings(_rollout())
    assert [n["option_label"] for n in out["notes"]] == ["A", "B", "B"]
    assert out["notes"][0]["note"] == "dim lighting suggests A"
    assert out["vote"] == {"label": "B", "confidence": 0.2, "acceptance_status": "rejected"}


def test_extract_findings_respects_the_character_budget() -> None:
    from scripts.eval.measure_answer_chain import extract_findings
    assert len(extract_findings(_rollout(), max_chars=200)["notes"]) < 3


def test_direct_answer_with_findings_keeps_all_clips_and_adds_the_notes() -> None:
    from scripts.eval.measure_answer_chain import extract_findings
    client = _FakeClient()
    direct_answer(client, _example(), indices=[0, 1, 2, 3, 4], findings=extract_findings(_rollout()))
    assert len(client.payload["clips"]) == 5
    assert len(client.payload["skill_findings"]) == 3
    assert client.payload["graph_vote"]["label"] == "B"
    assert "may be wrong" in client.payload["skill_findings_note"]

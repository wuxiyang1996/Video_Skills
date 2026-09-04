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
    assert "only applies to the direct condition" in capsys.readouterr().err


def test_answer_model_budget_grows_with_reasoning_effort() -> None:
    from scripts.eval.measure_answer_chain import answer_model_budget
    assert answer_model_budget("minimal", None) == 1800
    assert answer_model_budget("high", None) == 8000
    assert answer_model_budget("high", 3000) == 3000

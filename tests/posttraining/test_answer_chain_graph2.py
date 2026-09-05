"""The repaired decomposition: one comparative ranking over shared evidence, with
citations, and a look-again step when the top-two margin is small."""
import json

from scripts.eval.measure_answer_chain import (
    disputed_clip_indices, rank_hypotheses, ranking_margin,
)


class _FakeClient:
    def __init__(self, reply):
        self.reply, self.payload = reply, None

    def chat(self, messages):
        self.payload = json.loads(messages[-1]["content"])
        return self.reply


def _example():
    return {
        "question": {"question_text": "why did he leave?",
                     "options": [{"label": "A", "text": "fear"}, {"label": "B", "text": "anger"}]},
        "metadata": {"clip_schemas": [
            {"clip_id": f"c{i}", "scene_description": f"scene {i}", "time_span": {"start_s": i * 4.0, "end_s": i * 4.0 + 4.0}}
            for i in range(6)
        ]},
    }


def test_ranking_sees_every_option_and_every_given_clip_in_one_call() -> None:
    client = _FakeClient('{"ranking": [{"label": "B", "score": 0.7, "clip_ranks": [2, 4], "reason": "he shouts"}, '
                         '{"label": "A", "score": 0.3, "clip_ranks": [1], "reason": "no fear cue"}]}')
    out = rank_hypotheses(client, _example(), indices=[0, 1, 2, 3, 4, 5])
    assert len(client.payload["clips"]) == 6                  # shared evidence, not per-hypothesis
    assert [o["label"] for o in client.payload["options"]] == ["A", "B"]
    assert [r["label"] for r in out["ranking"]] == ["B", "A"]
    assert out["ranking"][0]["clip_ranks"] == [2, 4]          # the evidence chain the graph exists to emit


def test_ranking_survives_an_unparseable_reply() -> None:
    assert rank_hypotheses(_FakeClient("sorry"), _example(), indices=[0, 1])["ranking"] == []


def test_margin_is_the_gap_to_the_runner_up() -> None:
    assert abs(ranking_margin([{"score": 0.7}, {"score": 0.3}]) - 0.4) < 1e-9
    assert ranking_margin([{"score": 0.5}]) == 1.0
    assert ranking_margin([]) == 0.0


def test_disputed_clips_are_the_ones_the_top_two_cite() -> None:
    ranking = [{"label": "B", "clip_ranks": [2, 4]}, {"label": "A", "clip_ranks": [1, 2]}, {"label": "C", "clip_ranks": [6]}]
    indices = [10, 11, 12, 13, 14, 15]
    assert disputed_clip_indices(ranking, indices, limit=4) == [11, 13, 10]   # ranks 2,4 then 1; C is ignored
    assert disputed_clip_indices(ranking, indices, limit=2) == [11, 13]
    assert disputed_clip_indices([], indices, limit=4) == []


def test_visual_observations_ride_along_on_the_second_ranking() -> None:
    client = _FakeClient('{"ranking": [{"label": "A", "score": 0.9, "clip_ranks": [1], "reason": "he trembles"}]}')
    rank_hypotheses(client, _example(), indices=[0, 1], observations=[{"observation": "he trembles"}])
    assert client.payload["visual_observations"] == [{"observation": "he trembles"}]


def test_decompose_dispute_yields_factual_subquestions_on_the_cited_clips() -> None:
    from scripts.eval.measure_answer_chain import decompose_dispute
    client = _FakeClient('{"subquestions": [{"clip_rank": 2, "question": "is she holding an inhaler?"}, '
                         '{"clip_rank": "x", "question": "bad"}, {"clip_rank": 4, "question": "does he shout?"}]}')
    ranking = [{"label": "B", "clip_ranks": [2, 4], "reason": "he shouts"}, {"label": "A", "clip_ranks": [1], "reason": "fear"}]
    out = decompose_dispute(client, _example(), indices=[0, 1, 2, 3, 4, 5], ranking=ranking)
    assert out == [{"clip_rank": 2, "question": "is she holding an inhaler?"}, {"clip_rank": 4, "question": "does he shout?"}]
    assert client.payload["option_1"]["label"] == "B" and client.payload["option_2"]["text"] == "fear"
    assert [c["rank"] for c in client.payload["cited_clips"]] == [2, 4, 1]


def test_decompose_dispute_needs_two_options_and_survives_bad_json() -> None:
    from scripts.eval.measure_answer_chain import decompose_dispute
    assert decompose_dispute(_FakeClient("{}"), _example(), [0, 1], [{"label": "A"}]) == []
    assert decompose_dispute(_FakeClient("nope"), _example(), [0, 1], [{"label": "A", "clip_ranks": [1]}, {"label": "B"}]) == []


def test_visual_probe_uses_the_subquestion_when_given() -> None:
    from scripts.eval.measure_answer_chain import visual_probe

    class _C:
        def chat(self, messages):
            self.payload = json.loads(messages[-1]["content"][0]["text"]); return "yes, an inhaler"
    client = _C()
    out = visual_probe(client, _example(), index=1, frames=["AAAA"], probe_question="is she holding an inhaler?")
    assert client.payload["question"] == "is she holding an inhaler?"
    assert out["question"] == "is she holding an inhaler?" and out["observation"] == "yes, an inhaler"


def test_probability_mode_switches_the_ranking_prompt() -> None:
    from scripts.eval.measure_answer_chain import RANK_SYSTEM_PROB

    class _C:
        def chat(self, messages):
            self.system = messages[0]["content"]; return '{"ranking": [{"label": "A", "score": 0.6}, {"label": "B", "score": 0.4}]}'
    client = _C()
    out = rank_hypotheses(client, _example(), indices=[0, 1], probabilities=True)
    assert client.system == RANK_SYSTEM_PROB and "sum to 1" in client.system
    assert abs(ranking_margin(out["ranking"]) - 0.2) < 1e-9

import json

from trainer.reader.prompting import cited_ranks, parse_reader_output, reader_messages, reader_target
from trainer.reader.rewards import reader_reward


def _example():
    return {"question": {"question_text": "Who lied?", "options": [{"label": "A", "text": "the man"}, {"label": "B", "text": "the woman"}]},
            "metadata": {"clip_schemas": [{"time_span": {"start_s": 0, "end_s": 4}, "scene_description": "a man speaks"},
                                          {"time_span": {"start_s": 4, "end_s": 8}, "scene_description": "a woman hides a key"},
                                          {"time_span": {"start_s": 8, "end_s": 12}, "scene_description": "door"}],
                         "process_supervision": {}},
            "hidden_supervision": {}}


def test_messages_match_the_evaluator_format_and_target_round_trips() -> None:
    msgs = reader_messages(_example())
    assert msgs[0]["role"] == "system" and "reasoning" in msgs[0]["content"]
    payload = json.loads(msgs[1]["content"])
    assert [c["rank"] for c in payload["clips"]] == [1, 2, 3] and "likely_key_clips" not in payload
    t = reader_target("clip 2 shows the key", "B")
    assert parse_reader_output(t) == ("B", "clip 2 shows the key")
    assert cited_ranks("see clips 1-2 and clip 3") == [1, 2, 3]


def test_reward_pays_process_only_when_correct(monkeypatch) -> None:
    import trainer.reader.rewards as rw
    monkeypatch.setattr(rw, "oracle_gold_spans", lambda ex: [{"start_s": 4.0, "end_s": 8.0}])
    ex = _example()
    good = rw.reader_reward(ex, reader_target("clip 2 hides the key so B", "B"), "B")
    assert good["correct"] and good["citation_precision"] == 1.0 and abs(good["reward"] - 1.6) < 1e-6
    wrong = rw.reader_reward(ex, reader_target("clip 2 hides the key so A", "A"), "B")
    assert not wrong["correct"] and abs(wrong["reward"] - 0.1) < 1e-6           # citations do not pay when wrong
    off = rw.reader_reward(ex, reader_target("clip 1 and clip 3", "B"), "B")
    assert off["citation_precision"] == 0.0 and abs(off["reward"] - 1.1) < 1e-6
    assert rw.reader_reward(ex, "garbage", "B")["reward"] == 0.0

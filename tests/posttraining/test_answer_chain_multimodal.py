"""Multimodal answer step: frames of the watched clips ride along with the JSON
payload; voting takes the majority label; frame sampling reads real video."""
import json

import pytest

from scripts.eval.measure_answer_chain import (
    clip_frames_for_answer, direct_answer, majority_label, multimodal_user_content, sample_clip_frames,
)


class _FakeClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return '{"label": "C"}'


def _example(video_path=""):
    return {
        "video": {"primary_path": video_path},
        "question": {"question_text": "q", "options": [{"label": "A", "text": "a"}, {"label": "C", "text": "c"}]},
        "metadata": {"clip_schemas": [
            {"clip_id": f"c{i}", "scene_description": f"scene {i}", "time_span": {"start_s": i * 1.0, "end_s": i * 1.0 + 1.0}}
            for i in range(4)
        ]},
    }


def test_text_only_payload_is_unchanged_without_frames() -> None:
    assert isinstance(multimodal_user_content({"q": 1}, [0, 1], {}), str)


def test_frames_become_image_parts_after_the_json_and_name_the_clip_rank() -> None:
    parts = multimodal_user_content({"q": 1}, [3, 1], {1: ["AAAA", "BBBB"]})
    assert parts[0]["type"] == "text" and json.loads(parts[0]["text"]) == {"q": 1}
    assert "clip rank 2" in parts[1]["text"]
    assert [p["type"] for p in parts[2:]] == ["image_url", "image_url"]
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,AAAA")


def test_direct_answer_with_frames_switches_to_the_multimodal_prompt_and_counts_frames() -> None:
    client = _FakeClient()
    out = direct_answer(client, _example(), indices=[0, 1], frames={0: ["AAAA"]})
    assert out["final_answer"]["label"] == "C" and out["frames_attached"] == 1
    assert "frames" in client.messages[0]["content"]
    assert isinstance(client.messages[1]["content"], list)


def test_majority_label_breaks_ties_toward_the_earliest_sample() -> None:
    assert majority_label(["A", "B", "B"]) == "B"
    assert majority_label(["A", "B", None]) == "A"
    assert majority_label([None, None]) is None


def test_no_frames_when_disabled_or_video_missing() -> None:
    assert clip_frames_for_answer(_example(), [0, 1], None, per_clip=0, max_clips=4) == {}
    assert clip_frames_for_answer(_example(""), [0, 1], None, per_clip=2, max_clips=4) == {}


def test_sample_clip_frames_reads_a_real_video(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(40):   # 4 seconds
        writer.write(np.full((48, 64, 3), i * 6, dtype=np.uint8))
    writer.release()
    frames = sample_clip_frames(path, {"start_s": 1.0, "end_s": 3.0}, 3, width=32)
    assert len(frames) == 3 and all(isinstance(f, str) and len(f) > 100 for f in frames)
    watched = clip_frames_for_answer(_example(str(path)), [0, 1, 2, 3], [2, 9], per_clip=2, max_clips=4)
    assert list(watched) == [2] and len(watched[2]) == 2


def test_strip_verdicts_keeps_observations_and_drops_scores_and_vote() -> None:
    from scripts.eval.measure_answer_chain import strip_verdicts
    findings = {"notes": [{"option_label": "A", "support_score": 0.6, "note": "dim lighting"},
                          {"option_label": "B", "support_score": 0.9, "note": ""}],
                "vote": {"label": "B"}}
    out = strip_verdicts(findings)
    assert out == {"notes": [{"observation": "dim lighting"}], "vote": None}


def test_direct_answer_omits_the_graph_vote_when_it_was_stripped() -> None:
    from scripts.eval.measure_answer_chain import strip_verdicts
    client = _FakeClient()
    direct_answer(client, _example(), indices=[0, 1],
                  findings=strip_verdicts({"notes": [{"note": "a man leaves"}], "vote": {"label": "B"}}))
    payload = json.loads(client.messages[1]["content"])
    assert payload["skill_findings"] == [{"observation": "a man leaves"}]
    assert "graph_vote" not in payload


def test_visual_probe_asks_one_clip_with_the_question_and_returns_the_observation() -> None:
    from scripts.eval.measure_answer_chain import visual_probe
    client = _FakeClient()
    client.chat = lambda messages: (setattr(client, "messages", messages), "a woman looks terrified")[1]
    out = visual_probe(client, _example(), index=2, frames=["AAAA", "BBBB"])
    assert out == {"time_span": {"start_s": 2.0, "end_s": 3.0}, "question": None, "observation": "a woman looks terrified"}
    payload = json.loads(client.messages[1]["content"][0]["text"])
    assert payload["question"] == "q" and payload["time_span"] == {"start_s": 2.0, "end_s": 3.0}
    assert [p["type"] for p in client.messages[1]["content"][1:]] == ["image_url", "image_url"]


def test_visual_probe_returns_nothing_without_frames_or_reply() -> None:
    from scripts.eval.measure_answer_chain import visual_probe
    client = _FakeClient()
    assert visual_probe(client, _example(), index=0, frames=[]) is None
    client.chat = lambda messages: "   "
    assert visual_probe(client, _example(), index=0, frames=["AAAA"]) is None


def test_probe_condition_requires_frames() -> None:
    import pytest
    from scripts.eval.measure_answer_chain import main
    with pytest.raises(SystemExit):
        main(["--l1-glob", "x", "--output", "o.json", "--indices-from", "all", "--conditions", "probe"])


def test_rationale_mode_keeps_the_think_block_and_parses_the_label_after_it() -> None:
    class _C:
        def chat(self, messages):
            self.system = messages[0]["content"]
            return '<think>clip 3 shows the knife, so B is wrong; A fits.</think>\n{"label": "A"}'
    from scripts.eval.measure_answer_chain import DIRECT_SYSTEM_RATIONALE
    client = _C()
    out = direct_answer(client, _example(), indices=[0, 1], rationale=True)
    assert client.system == DIRECT_SYSTEM_RATIONALE
    assert out["final_answer"]["label"] == "A"
    assert out["thinking"] == "clip 3 shows the knife, so B is wrong; A fits."
    assert out["raw_text"].startswith("<think>")


def test_rationale_mode_reads_the_json_reasoning_field_when_there_are_no_tags() -> None:
    class _C:
        def chat(self, messages):
            return '{"reasoning": "clip 2 shows her inhaler; E fits", "label": "E"}'
    out = direct_answer(_C(), _example(), indices=[0, 1], rationale=True)
    assert out["final_answer"]["label"] == "E" and out["thinking"] == "clip 2 shows her inhaler; E fits"


def test_rationale_mode_falls_back_to_prose_before_the_json() -> None:
    class _C:
        def chat(self, messages):
            return 'She holds an inhaler in clip 2, so E.\n{"label": "E"}'
    out = direct_answer(_C(), _example(), indices=[0, 1], rationale=True)
    assert out["thinking"] == "She holds an inhaler in clip 2, so E."


def test_plain_mode_has_no_thinking_field() -> None:
    out = direct_answer(_FakeClient(), _example(), indices=[0, 1])
    assert "thinking" not in out

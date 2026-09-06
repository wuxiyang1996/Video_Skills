"""Narrative synthesis: time windows over the clips, sequential paragraphs with a persistent cast."""
import json

from scripts.eval.build_narrative_catalog import narrate_video, window_clips


def _schemas(n=12, step=4.0):
    return [{"clip_id": f"c{i}", "scene_description": f"scene {i}", "time_span": {"start_s": i * step, "end_s": i * step + step}}
            for i in range(n)]


def test_windows_cover_every_clip_in_order_and_respect_the_minimum() -> None:
    windows = window_clips(_schemas(12), target_s=20.0, min_windows=3)   # 48 s -> round(2.4)=2 -> min 3
    assert len(windows) == 3
    assert [i for w in windows for i in w] == list(range(12))


def test_windows_scale_with_duration() -> None:
    assert len(window_clips(_schemas(60), target_s=45.0)) == 5         # 240 s / 45 -> 5
    assert window_clips([], target_s=45.0) == []


class _FakeClient:
    def __init__(self):
        self.payloads = []

    def chat(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(payload)
        k = payload["window"]
        return json.dumps({"narrative": f"paragraph {k} about the woman in blue", "cast": ["the woman in blue", f"person {k}"]})


def test_sequential_narration_passes_previous_paragraph_and_cast_forward() -> None:
    client = _FakeClient()
    schemas = _schemas(12)
    rows = narrate_video(client, schemas, window_clips(schemas, target_s=20.0, min_windows=3))
    assert [r["clip_id"] for r in rows] == ["narrative:1", "narrative:2", "narrative:3"]
    assert rows[0]["time_span"] == {"start_s": 0.0, "end_s": 16.0}
    assert client.payloads[0]["previous_narrative"] == ""
    assert client.payloads[1]["previous_narrative"] == "paragraph 1 about the woman in blue"
    assert client.payloads[2]["cast_so_far"] == ["the woman in blue", "person 1", "person 2"]
    assert rows[2]["source_clip_count"] == 4


def test_a_bad_reply_never_drops_a_window() -> None:
    class _Broken:
        def chat(self, messages):
            return "not json at all"
    rows = narrate_video(_Broken(), _schemas(6), window_clips(_schemas(6), target_s=10.0, min_windows=2))
    assert len(rows) == 2 and all(r["scene_description"] for r in rows)


def test_asr_segments_are_selected_by_overlap_with_padding() -> None:
    from scripts.eval.build_narrative_catalog import asr_in_span
    segs = [{"start_s": 0.0, "end_s": 3.0, "text": "a"}, {"start_s": 10.5, "end_s": 12.0, "text": "b"},
            {"start_s": 30.0, "end_s": 31.0, "text": "c"}]
    picked = asr_in_span(segs, {"start_s": 4.0, "end_s": 10.0})      # 1 s pad reaches 'b', not 'a' or 'c'
    assert [p["text"] for p in picked] == ["b"]


def test_looking_mode_sends_frames_and_dialogue_and_keeps_clip_text_optional(monkeypatch) -> None:
    import scripts.eval.build_narrative_catalog as mod
    monkeypatch.setattr(mod, "sample_clip_frames", lambda path, span, count, width=448: ["AAA="] * count)

    class _Client:
        def __init__(self):
            self.messages = []

        def chat(self, messages):
            self.messages.append(messages)
            return json.dumps({"narrative": "the man with the backpack hears footsteps", "cast": ["the man with the backpack"]})

    client = _Client()
    schemas = _schemas(6)
    asr = [{"start_s": 1.0, "end_s": 2.0, "text": "**footsteps**"}]
    rows = mod.narrate_video(client, schemas, mod.window_clips(schemas, target_s=12.0, min_windows=2),
                             video_path="/nonexistent.mp4", frames_per_window=4, asr_segments=asr, use_clip_text=False)
    system, user = client.messages[0]
    assert system["content"] == mod.ANNOTATE_SYSTEM
    parts = user["content"]
    assert isinstance(parts, list) and sum(1 for p in parts if p["type"] == "image_url") == 4
    payload = json.loads(parts[0]["text"])
    assert "clips" not in payload and payload["dialogue"] == asr
    assert rows[0]["frames_seen"] == 4 and rows[0]["dialogue_lines"] == 1
    # the second window has no dialogue and still carries the previous paragraph
    payload2 = json.loads(client.messages[1][1]["content"][0]["text"])
    assert payload2["dialogue"] == [] and payload2["previous_narrative"].startswith("the man with the backpack")

import sys
from types import SimpleNamespace

from dataset_clip_wrapper.perception.clip_schema import QwenClipSchemaProducer
from dataset_clip_wrapper.schemas import ClipSchemaConfig, ClipSpan


class _ThreeFailuresThenMinimalClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_response_metadata = {"max_tokens": 1600}

    def chat_json(self, messages, *, response_format=None):
        self.calls += 1
        if self.calls <= 3:
            raise ValueError("truncated JSON")
        return {
            "scene_description": "A person is visible.",
            "observable_facts": [{"text": "A person is visible.", "modality": "visual"}],
            "dialogue_spans": [],
            "entity_mentions": [],
            "state_assertions": [],
            "salient_objects": [],
            "place": {"description": "", "searchable_phrases": []},
            "events": [],
            "visual_social_cues": [],
            "cross_clip_cues": [],
            "searchable_phrases": ["visible person"],
            "uncertainty": "",
        }


def test_clip_schema_uses_bounded_minimal_fourth_attempt() -> None:
    client = _ThreeFailuresThenMinimalClient()
    producer = QwenClipSchemaProducer(
        ClipSchemaConfig(model="Qwen/Qwen3.5-9B", request_frames=4, max_tokens=1600),
        client,
    )

    row = producer.build_clip_schema(
        clip_id="clip:test:fine:0001",
        clip=ClipSpan(start_s=0.0, end_s=4.0, granularity="fine", clip_index=1),
        video_path=None,
    )

    assert client.calls == 4
    assert row.get("model_error") is None
    assert row["schema_attempt"] == "minimal_json_object_retry"
    assert row["llm_usage"]["compact_retry_count"] == 3


def test_frame_sampler_recovers_from_exact_endpoint_decode_failure(tmp_path, monkeypatch) -> None:
    class _Buffer:
        def tobytes(self):
            return b"jpeg"

    class _Capture:
        def __init__(self, _path):
            self.position_ms = 0.0
            self.positions = []

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 1:
                return 30.0
            if prop == 2:
                return 121
            return 0

        def set(self, _prop, value):
            self.position_ms = value
            self.positions.append(value)

        def read(self):
            if self.position_ms >= 4000.0:
                return False, None
            return True, object()

        def release(self):
            return None

    captures = []

    def _video_capture(path):
        capture = _Capture(path)
        captures.append(capture)
        return capture

    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_MSEC=0,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        VideoCapture=_video_capture,
        imencode=lambda _extension, _frame: (True, _Buffer()),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"stub")
    producer = QwenClipSchemaProducer(
        ClipSchemaConfig(model="Qwen/Qwen3.5-9B", request_frames=6, max_tokens=1600),
        _ThreeFailuresThenMinimalClient(),
    )

    frames = producer._sample_frame_jpegs(
        video_path,
        ClipSpan(start_s=0.0, end_s=4.0, granularity="fine", clip_index=1),
    )

    assert len(frames) == 6
    assert 4000.0 in captures[0].positions
    assert captures[0].positions[-1] < 4000.0

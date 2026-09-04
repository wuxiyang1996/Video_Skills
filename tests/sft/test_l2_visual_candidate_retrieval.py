from __future__ import annotations

import numpy as np

from dataset_clip_wrapper.training.evaluate_l2_visual_candidate_retrieval import (
    fine_spans,
    question_text,
    reduce_fine_scores,
    sample_frames,
)


class _Capture:
    def __init__(self, failures: int = 0) -> None:
        self.timestamps = []
        self.failures = failures

    def set(self, _key, value):
        self.timestamps.append(value)

    def read(self):
        if self.failures:
            self.failures -= 1
            return False, None
        return True, np.zeros((20, 40, 3), dtype=np.uint8)


def test_visual_query_and_uniform_frame_sampling():
    query = question_text({"question_text": "What happens?", "options": [{"text": "A"}, {"text": "B"}]})
    assert "What happens?" in query
    assert "A | B" in query

    capture = _Capture()
    frames = sample_frames(capture, {"start_s": 10, "end_s": 20}, num_frames=3, max_side=10)
    assert frames.shape == (3, 5, 10, 3)
    assert capture.timestamps == [11000.0, 15000.0, 19000.0]

    retry_capture = _Capture(failures=1)
    retry_frames = sample_frames(retry_capture, {"start_s": 10, "end_s": 20}, num_frames=1, max_side=10)
    assert retry_frames.shape == (1, 5, 10, 3)
    assert retry_capture.timestamps == [11000.0, 10750.0]


def test_fine_spans_cover_tail_and_reduce_to_coarse_max():
    assert fine_spans({"start_s": 0, "end_s": 30}, window_sec=8, stride_sec=4) == [
        {"start_s": 0.0, "end_s": 8.0},
        {"start_s": 4.0, "end_s": 12.0},
        {"start_s": 8.0, "end_s": 16.0},
        {"start_s": 12.0, "end_s": 20.0},
        {"start_s": 16.0, "end_s": 24.0},
        {"start_s": 20.0, "end_s": 28.0},
        {"start_s": 22.0, "end_s": 30.0},
    ]
    assert reduce_fine_scores([0.1, 0.7, 0.4], [0, 0, 1], 2) == [0.7, 0.4]

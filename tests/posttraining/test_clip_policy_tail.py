from dataset_clip_wrapper.perception.clip_policy import segment_video
from dataset_clip_wrapper.schemas import ClipPolicyConfig


def test_fixed_windows_do_not_emit_redundant_millisecond_tail():
    spans = segment_video(
        150.01668335001668,
        ClipPolicyConfig(strategy="fixed_window", window_s=4.0, overlap_s=1.0),
    )

    assert len(spans) == 50
    assert spans[-1].start_s == 147.0
    assert spans[-1].end_s == 150.01668335001668
    assert min(span.end_s - span.start_s for span in spans) > 3.0

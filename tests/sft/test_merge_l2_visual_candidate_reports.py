from dataset_clip_wrapper.training.merge_l2_visual_candidate_reports import merge_reports


def _payload(example_id: str, hit: bool) -> dict:
    metrics = {str(k): {"hit": hit, "recall": float(hit)} for k in (4, 8, 16, 24, 32)}
    return {
        "model": "m",
        "num_frames_per_coarse_window": 4,
        "max_frame_side": 448,
        "fine_window_sec": 8,
        "fine_stride_sec": 4,
        "results": [{
            "example_id": example_id,
            "catalog_size": 40,
            "gold": [0],
            "metrics": metrics,
            "boundary_hybrid_at_32": {"hit": hit, "recall": float(hit)},
        }],
    }


def test_merge_disjoint_candidate_reports():
    merged = merge_reports([_payload("b", False), _payload("a", True)])
    assert merged["summary"]["examples"] == 2
    assert merged["summary"]["hit_at_32"] == 0.5
    assert merged["boundary_hybrid_summary"]["hit_at_32"] == 0.5
    assert [row["example_id"] for row in merged["results"]] == ["a", "b"]

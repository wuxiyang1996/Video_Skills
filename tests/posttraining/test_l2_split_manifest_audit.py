from scripts.eval.audit_l2_split_manifest import audit


def test_split_audit_rejects_cross_role_and_test_leakage() -> None:
    payload = {
        "schema_version": "video-skills/split-manifest-v1",
        "salt": "x",
        "assignment": {},
        "manifest_hash": "wrong",
        "summary": {"n_videos": 2, "role_video_counts": {"sft_seed": 1, "dev_tune": 1}},
        "videos": [
            {"key": "video_holmes:v", "dataset": "video_holmes", "role": "sft_seed", "official_split": "test", "n_questions": 1},
            {"key": "video_holmes:v", "dataset": "video_holmes", "role": "dev_tune", "official_split": "train", "n_questions": 1},
        ],
    }
    report = audit(payload)
    assert not report["passed"]
    assert not report["checks"]["video_keys_unique"]
    assert not report["checks"]["one_role_per_video"]
    assert not report["checks"]["vh_official_test_heldout_only"]

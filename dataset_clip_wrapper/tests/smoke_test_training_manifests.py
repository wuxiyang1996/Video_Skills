"""Compatibility shim: real test lives under tests/training/."""
from dataset_clip_wrapper.tests.training.smoke_test_training_manifests import *  # noqa: F401,F403

if __name__ == "__main__":
    test_split_groups_are_video_isolated()
    test_manifest_row_strips_gold_question_fields()
    print("training manifest smoke test passed")

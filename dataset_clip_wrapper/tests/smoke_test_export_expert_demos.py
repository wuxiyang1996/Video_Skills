"""Compatibility shim: real test lives under tests/training/."""
from dataset_clip_wrapper.tests.training.smoke_test_export_expert_demos import *  # noqa: F401,F403

if __name__ == "__main__":
    test_export_expert_demo_boundaries()
    print("expert demo export smoke test passed")

"""Compatibility shim: real test lives under tests/l2/."""
from dataset_clip_wrapper.tests.l2.smoke_test_l2_recursive_trace import *  # noqa: F401,F403

if __name__ == "__main__":
    test_initial_l2_trajectory()
    test_repair_artifacts_to_trajectory()
    print("l2 recursive trace smoke test passed")

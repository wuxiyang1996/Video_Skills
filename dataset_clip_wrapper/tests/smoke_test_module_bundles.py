"""Compatibility shim: real test lives under tests/core/."""
from dataset_clip_wrapper.tests.core.smoke_test_module_bundles import *  # noqa: F401,F403

if __name__ == "__main__":
    test_module_bundle_coverage()
    print("module bundle smoke test passed")

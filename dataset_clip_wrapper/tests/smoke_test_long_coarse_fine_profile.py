"""Compatibility shim: real test lives under tests/l1/."""
from dataset_clip_wrapper.tests.l1.smoke_test_long_coarse_fine_profile import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.l1.smoke_test_long_coarse_fine_profile import main

if __name__ == "__main__":
    raise SystemExit(main())

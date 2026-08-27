"""Compatibility shim: real test lives under tests/l2/."""
from dataset_clip_wrapper.tests.l2.smoke_test_reasoning_rollout import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.l2.smoke_test_reasoning_rollout import main

if __name__ == "__main__":
    raise SystemExit(main())

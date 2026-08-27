"""Compatibility shim: real test lives under tests/perception/."""
from dataset_clip_wrapper.tests.perception.smoke_test_video_only_takein import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.perception.smoke_test_video_only_takein import main

if __name__ == "__main__":
    raise SystemExit(main())

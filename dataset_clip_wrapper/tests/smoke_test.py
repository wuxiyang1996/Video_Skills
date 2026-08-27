"""Compatibility shim: real test lives under tests/core/."""
from dataset_clip_wrapper.tests.core.smoke_test import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.core.smoke_test import main

if __name__ == "__main__":
    raise SystemExit(main())

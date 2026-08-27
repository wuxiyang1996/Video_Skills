"""Compatibility shim: real test lives under tests/core/."""
from dataset_clip_wrapper.tests.core.smoke_test_two_layer_schema import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.core.smoke_test_two_layer_schema import main

if __name__ == "__main__":
    raise SystemExit(main())

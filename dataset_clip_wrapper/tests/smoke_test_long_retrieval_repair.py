"""Compatibility shim: real test lives under tests/verification/."""
from dataset_clip_wrapper.tests.verification.smoke_test_long_retrieval_repair import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.verification.smoke_test_long_retrieval_repair import main

if __name__ == "__main__":
    raise SystemExit(main())

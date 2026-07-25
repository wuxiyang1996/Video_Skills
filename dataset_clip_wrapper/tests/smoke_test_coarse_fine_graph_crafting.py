"""Compatibility shim: real test lives under tests/l1/."""
from dataset_clip_wrapper.tests.l1.smoke_test_coarse_fine_graph_crafting import *  # noqa: F401,F403
from dataset_clip_wrapper.tests.l1.smoke_test_coarse_fine_graph_crafting import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Compatibility entrypoint for `dataset_clip_wrapper.training.trace_adapter`."""

from .training.trace_adapter import *  # noqa: F401,F403
from .training.trace_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Compatibility entrypoint for `dataset_clip_wrapper.training.stepwise_sft_adapter`."""

from .training.stepwise_sft_adapter import *  # noqa: F401,F403
from .training.stepwise_sft_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())

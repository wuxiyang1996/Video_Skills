"""Compatibility entrypoint for atomic L1 builder SFT export."""

from .training.l1_builder_sft_adapter import *  # noqa: F401,F403
from .training.l1_builder_sft_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())

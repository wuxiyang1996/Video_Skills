"""Compatibility entrypoint for atomic L2 retrieval SFT export."""

from .training.l2_retrieval_sft_adapter import *  # noqa: F401,F403
from .training.l2_retrieval_sft_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())

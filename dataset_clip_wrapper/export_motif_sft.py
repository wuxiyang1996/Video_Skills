"""Compatibility entrypoint for motif SFT export."""

from .training.motif_sft_adapter import *  # noqa: F401,F403
from .training.motif_sft_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())

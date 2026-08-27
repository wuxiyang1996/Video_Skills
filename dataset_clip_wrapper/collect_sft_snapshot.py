"""Compatibility entrypoint for SFT snapshot collection."""

from .training.collect_sft_snapshot import *  # noqa: F401,F403
from .training.collect_sft_snapshot import main


if __name__ == "__main__":
    raise SystemExit(main())

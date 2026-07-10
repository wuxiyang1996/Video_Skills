"""Compatibility entrypoint for verifier SFT export."""

from .training.verifier_sft_adapter import *  # noqa: F401,F403
from .training.verifier_sft_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())

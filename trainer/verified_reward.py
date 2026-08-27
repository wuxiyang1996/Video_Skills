"""Back-compat shim — implementation lives in ``trainer.reward.verified_reward``."""

from trainer.reward.verified_reward import (  # noqa: F401
    REWARD_SPEC_VERSION,
    VerifiedRewardBreakdown,
    group_rank_advantages,
    score_verified_rollout,
)

__all__ = [
    "REWARD_SPEC_VERSION",
    "VerifiedRewardBreakdown",
    "group_rank_advantages",
    "score_verified_rollout",
]

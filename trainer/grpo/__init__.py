"""GRPO / RLVR collection and training (L2+Repair first, optional joint L1).

Import collection/training helpers from submodules:

  from trainer.grpo.collect_rollouts import collect_grpo_group
  from trainer.grpo.train_verified import run_grpo_smoke

CLI:
  python -m trainer.grpo.collect_rollouts
  python -m trainer.grpo.train_verified
"""

from .advantages import assign_group_advantages, grpo_surrogate_loss, return_to_go_credits
from .attn_utils import flash_attn_available, resolve_attn_implementation
from .isolation import assert_rollout_isolation, deep_isolate
from .types import (
    MODE_JOINT_L1,
    MODE_L2_REPAIR,
    GrpoGroup,
    GrpoRollout,
    GrpoTrainConfig,
)

__all__ = [
    "MODE_JOINT_L1",
    "MODE_L2_REPAIR",
    "GrpoGroup",
    "GrpoRollout",
    "GrpoTrainConfig",
    "assert_rollout_isolation",
    "assign_group_advantages",
    "deep_isolate",
    "flash_attn_available",
    "grpo_surrogate_loss",
    "resolve_attn_implementation",
    "return_to_go_credits",
]

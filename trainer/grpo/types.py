"""Shared GRPO dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trainer.reward.verified_reward import VerifiedRewardBreakdown


# Module update modes.
MODE_L2_REPAIR = "l2_repair"
MODE_JOINT_L1 = "joint_l1"
GRPO_MODES = (MODE_L2_REPAIR, MODE_JOINT_L1)

DEFAULT_UPDATE_MODULES = {
    MODE_L2_REPAIR: ("l2", "repair"),
    MODE_JOINT_L1: ("l1", "l2", "repair"),
}


@dataclass
class GrpoRollout:
    group_id: str
    rollout_id: str
    example_id: str
    sample_index: int
    seed: int
    policy_view: dict[str, Any]
    motif_online: dict[str, Any]
    reward: VerifiedRewardBreakdown
    advantage: float = 0.0
    logprob: float | None = None
    ref_logprob: float | None = None
    update_modules: tuple[str, ...] = ("l2", "repair")
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "example_id": self.example_id,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "policy_view": self.policy_view,
            "motif_online": self.motif_online,
            "reward": self.reward.to_dict(),
            "advantage": self.advantage,
            "logprob": self.logprob,
            "ref_logprob": self.ref_logprob,
            "update_modules": list(self.update_modules),
            "extras": self.extras,
        }


@dataclass
class GrpoGroup:
    group_id: str
    example_id: str
    video_key: str
    split_role: str
    mode: str
    rollouts: list[GrpoRollout] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "example_id": self.example_id,
            "video_key": self.video_key,
            "split_role": self.split_role,
            "mode": self.mode,
            "rollouts": [r.to_dict() for r in self.rollouts],
        }


@dataclass(frozen=True)
class GrpoTrainConfig:
    mode: str = MODE_L2_REPAIR
    kl_coef: float = 0.05
    l1_lr_scale: float = 0.1
    require_l2_stable_for_l1: bool = True
    l2_stable_flag: bool = False
    max_grad_norm: float = 1.0

    def update_modules(self) -> tuple[str, ...]:
        if self.mode not in GRPO_MODES:
            raise ValueError(f"unknown GRPO mode: {self.mode}")
        if self.mode == MODE_JOINT_L1:
            if self.require_l2_stable_for_l1 and not self.l2_stable_flag:
                raise RuntimeError(
                    "joint_l1 GRPO requires l2_stable_flag=True "
                    "(L2+Repair RLVR must be stable first)"
                )
        return DEFAULT_UPDATE_MODULES[self.mode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "kl_coef": self.kl_coef,
            "l1_lr_scale": self.l1_lr_scale,
            "require_l2_stable_for_l1": self.require_l2_stable_for_l1,
            "l2_stable_flag": self.l2_stable_flag,
            "max_grad_norm": self.max_grad_norm,
            "update_modules": list(self.update_modules()),
        }

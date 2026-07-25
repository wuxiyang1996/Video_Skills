"""Post-SFT training stack: Motif-gated harness, OPD, verified reward, GRPO.

Layout
------
- OPD / closed-loop (flat modules): ``closed_loop_harness``, ``candidate_action_builder``,
  ``teacher_action_query``, ``opd_action_distill_adapter``, ``train_opd_kl``
- Reward package: ``trainer.reward`` (milestones, semantic judge, lexicographic score, bridge)
- GRPO package: ``trainer.grpo`` (collect K-rollouts, train L2+Repair / joint L1)
- Split / run contracts: ``split_filter``, ``posttraining_manifest``

SFT LoRA trainers remain under ``dataset_clip_wrapper/training/``.
"""

from .candidate_action_builder import CandidateActionSet, build_l2_candidate_actions
from .closed_loop_harness import ClosedLoopHarness, HarnessState
from .exact_request_cache import ExactRequestCache
from .opd_action_distill_adapter import OpdDistillRow, load_opd_rows, save_opd_rows
from .teacher_action_query import TeacherActionDistribution, query_teacher_action_distribution
from .verified_reward import VerifiedRewardBreakdown, score_verified_rollout

__all__ = [
    "CandidateActionSet",
    "ClosedLoopHarness",
    "ExactRequestCache",
    "HarnessState",
    "OpdDistillRow",
    "TeacherActionDistribution",
    "VerifiedRewardBreakdown",
    "build_l2_candidate_actions",
    "load_opd_rows",
    "query_teacher_action_distribution",
    "save_opd_rows",
    "score_verified_rollout",
]

"""Training infrastructure for Video_Skills.

This package currently contains two stacks:

- original COS-PLAY co-evolution / SFT under ``coevolution/``, ``common/``, and ``SFT/``
- L1/L2 Motif-gated post-SFT OPD/GRPO under the modules exported below

SFT LoRA trainers for the five-specialist L1/L2 path remain under
``dataset_clip_wrapper/training/``.
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

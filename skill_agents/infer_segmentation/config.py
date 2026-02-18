"""
Configuration for Stage 2 InferSegmentation.

Dataclasses for scorer weights, decoder parameters, the NEW-skill channel,
LLM preference-teacher settings, and preference-learning hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScorerWeights:
    """Relative weights for each term in the segment score decomposition."""

    behavior_fit: float = 1.0
    duration_prior: float = 0.3
    transition_prior: float = 1.0
    contract_compat: float = 0.0  # folded into behavior_fit via LLM ranking


@dataclass
class DurationPriorConfig:
    """Parameters for the per-skill duration prior log p(l | k)."""

    default_mean: float = 20.0
    default_std: float = 10.0
    min_length: int = 2
    max_length: int = 200


@dataclass
class NewSkillConfig:
    """Parameters for the special NEW-skill channel."""

    enabled: bool = True
    penalty: float = 5.0  # alpha: subtracted from background score
    background_log_prob: float = -3.0  # fallback log-prob per step


@dataclass
class LLMTeacherConfig:
    """Settings for the LLM preference teacher (provides rankings, not scores)."""

    model: Optional[str] = None  # None = use ask_model default (gpt-4o)
    temperature: float = 0.3
    max_tokens: int = 1000


@dataclass
class PreferenceLearningConfig:
    """Hyperparameters for the preference-learning loop."""

    num_iterations: int = 3
    margin_threshold: float = 1.0  # query segments with margin below this
    max_queries_per_iter: int = 5
    training_epochs: int = 20
    learning_rate: float = 0.1
    collect_transitions: bool = True  # also collect transition preferences


@dataclass
class DecoderConfig:
    """Parameters shared by Viterbi DP and beam decoders."""

    top_m_skills: Optional[int] = None  # per segment, only score top-M skills (None = all)
    top_r_transitions: Optional[int] = None  # per skill, only consider top-R next skills

    # Beam-specific
    beam_width: int = 16
    beam_max_segments: Optional[int] = None  # early-stop after this many segments

    # Diagnostics
    top_k_diagnostics: int = 3  # how many skill candidates to return per segment


@dataclass
class SegmentationConfig:
    """Top-level configuration for InferSegmentation (Stage 2)."""

    weights: ScorerWeights = field(default_factory=ScorerWeights)
    duration: DurationPriorConfig = field(default_factory=DurationPriorConfig)
    new_skill: NewSkillConfig = field(default_factory=NewSkillConfig)
    llm_teacher: LLMTeacherConfig = field(default_factory=LLMTeacherConfig)
    preference: PreferenceLearningConfig = field(default_factory=PreferenceLearningConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    method: str = "dp"  # "dp" (Viterbi) or "beam"

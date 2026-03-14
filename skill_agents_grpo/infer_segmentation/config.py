"""
Configuration for Stage 2 InferSegmentation.

Dataclasses for scorer weights, decoder parameters, the NEW-skill channel,
LLM preference-teacher settings, and preference-learning hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContractFeedbackConfig:
    """Controls the Stage 3 → Stage 2 contract feedback loop.

    Three modes selected by ``mode``:
      - ``"off"``:    no contract feedback (``contract_compat`` weight = 0).
      - ``"weak"``:   light bias from contracts (default ``strength`` = 0.3).
      - ``"strong"``: heavy reliance on contracts (default ``strength`` = 1.0).

    When ``mode != "off"``, the ``contract_compat`` weight in ``ScorerWeights``
    is automatically set to ``strength`` unless manually overridden.
    """

    mode: str = "off"  # "off" | "weak" | "strong"
    strength: float = 0.3  # effective weight when mode != "off"
    p_thresh: float = 0.5
    missing_penalty: float = -0.5
    contradiction_penalty: float = -1.0


@dataclass
class ScorerWeights:
    """Relative weights for each term in the segment score decomposition."""

    behavior_fit: float = 1.0
    duration_prior: float = 0.3
    transition_prior: float = 1.0
    contract_compat: float = 0.0  # set by ContractFeedbackConfig or manually
    boundary_preference: float = 0.5  # Phase 3: "cut here" vs "do not cut here"


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
    # Number of worker threads (None or 1 = sequential).
    max_workers: Optional[int] = 8
    # Cap on concurrent LLM calls (e.g. GPU inference). None = no cap.
    # Set to 1 for local GPU to avoid OOM; keep parallel task flow.
    max_concurrent_llm_calls: Optional[int] = None


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
class UncertainLabelConfig:
    """Phase 3: uncertain-label path thresholds.

    Instead of forcing every segment into a known skill, three outcomes:
      - confident known skill (margin >= confident_margin)
      - low-confidence known skill (margin < confident_margin but >= uncertain_margin)
      - NEW / unknown (assigned __NEW__)

    Low-confidence segments go to an uncertain pool for reconsideration.
    """

    confident_margin: float = 2.0
    uncertain_margin: float = 1.0
    reconsider_after_n_updates: int = 3


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
    contract_feedback: ContractFeedbackConfig = field(default_factory=ContractFeedbackConfig)
    uncertain_label: UncertainLabelConfig = field(default_factory=UncertainLabelConfig)

    method: str = "dp"  # "dp" (Viterbi) or "beam"

    def __post_init__(self) -> None:
        """Apply contract feedback mode to scorer weights if not manually set."""
        cf = self.contract_feedback
        if cf.mode == "off":
            pass  # keep contract_compat at whatever the user set (default 0.0)
        elif cf.mode == "weak":
            if self.weights.contract_compat == 0.0:
                self.weights.contract_compat = cf.strength
        elif cf.mode == "strong":
            if self.weights.contract_compat == 0.0:
                self.weights.contract_compat = max(cf.strength, 1.0)

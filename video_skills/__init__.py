"""Video_Skills — evidence-grounded multi-hop video reasoning runtime.

Public API entry points:

- :class:`Runtime` and :func:`build_runtime` — wire all subsystems.
- :func:`run_question` — execute the §2D online serving loop on one question
  and return a :class:`ReasoningTrace`.
- Canonical typed objects from :mod:`video_skills.contracts` — the only
  allowed wire format between major runtime components.
- :class:`Memory` and :class:`MemoryProcedureRegistry` — the fixed
  three-store memory substrate plus its 9 named procedures.
- :class:`ReasoningSkillBank` and :func:`build_starter_bank` — the v1
  curated atomic reasoning inventory.
- :class:`Controller`, :class:`Retriever`, :class:`Verifier`,
  :class:`Harness` — the four runtime subsystems.

Read the design plans under ``infra_plans/`` for the normative spec.
"""
from .contracts import (
    SCHEMA_VERSION,
    AbstainDecision,
    AtomicStepResult,
    DialogueSpan,
    EntityRef,
    EventRef,
    EvidenceBundle,
    EvidenceRef,
    FrameRef,
    GroundedWindow,
    HopGoal,
    HopRecord,
    QuestionAnalysis,
    ReasoningTrace,
    RetrievalQuery,
    TriggerSpec,
    VerificationCheck,
    VerificationCheckSpec,
    VerificationResult,
    most_severe_next_action,
    new_id,
    now_ts,
    validate_atomic_step,
)
from .controller import Controller, ControllerConfig
from .harness import Harness, HarnessConfig, HopExecutionContext
from .loop import Runtime, build_runtime, run_question
from .memory import (
    BeliefState,
    EntityProfile,
    EntityProfileRegistry,
    EpisodicEvent,
    EpisodicStore,
    EpisodicThread,
    EvidenceStore,
    Memory,
    MemoryProcedureRegistry,
    PROCEDURE_NAMES,
    SemanticStore,
    SemanticSummary,
    SpatialState,
    StateStore,
)
from .retriever import Retriever, RetrieverConfig
from .skills import (
    AtomicSkill,
    ReasoningSkillBank,
    SkillRecord,
    SkillUsage,
    SkillVersion,
    build_starter_bank,
    register_starter_skills,
)
from .verifier import CHECK_NAMES, Verifier, VerifierConfig

__all__ = [
    # Contracts
    "SCHEMA_VERSION",
    "AbstainDecision",
    "AtomicStepResult",
    "DialogueSpan",
    "EntityRef",
    "EventRef",
    "EvidenceBundle",
    "EvidenceRef",
    "FrameRef",
    "GroundedWindow",
    "HopGoal",
    "HopRecord",
    "QuestionAnalysis",
    "ReasoningTrace",
    "RetrievalQuery",
    "TriggerSpec",
    "VerificationCheck",
    "VerificationCheckSpec",
    "VerificationResult",
    "most_severe_next_action",
    "new_id",
    "now_ts",
    "validate_atomic_step",
    # Memory
    "BeliefState",
    "EntityProfile",
    "EntityProfileRegistry",
    "EpisodicEvent",
    "EpisodicStore",
    "EpisodicThread",
    "EvidenceStore",
    "Memory",
    "MemoryProcedureRegistry",
    "PROCEDURE_NAMES",
    "SemanticStore",
    "SemanticSummary",
    "SpatialState",
    "StateStore",
    # Skills
    "AtomicSkill",
    "ReasoningSkillBank",
    "SkillRecord",
    "SkillUsage",
    "SkillVersion",
    "build_starter_bank",
    "register_starter_skills",
    # Subsystems
    "CHECK_NAMES",
    "Controller",
    "ControllerConfig",
    "Harness",
    "HarnessConfig",
    "HopExecutionContext",
    "Retriever",
    "RetrieverConfig",
    "Verifier",
    "VerifierConfig",
    # Runtime
    "Runtime",
    "build_runtime",
    "run_question",
]

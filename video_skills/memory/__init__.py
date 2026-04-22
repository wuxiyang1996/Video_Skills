"""Memory subsystem for Video_Skills.

Implements the three-store memory + evidence layer + entity-profile registry
described in ``infra_plans/02_memory/agentic_memory_design.md``, and the
``MemoryProcedureRegistry`` of fixed memory-management operations.

Per the design principle (stable memory, evolving reasoning), this subsystem
is intentionally **not** trainable in v1. All updates flow through the nine
named procedures in :mod:`video_skills.memory.procedures`.
"""

from .procedures import (
    MemoryProcedureRegistry,
    PROCEDURE_NAMES,
)
from .stores import (
    BeliefState,
    EntityProfile,
    EntityProfileRegistry,
    EpisodicEvent,
    EpisodicStore,
    EpisodicThread,
    EvidenceStore,
    Memory,
    SemanticStore,
    SemanticSummary,
    SpatialState,
    StateStore,
)

__all__ = [
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
]

"""Reasoning Skill Bank for Video_Skills.

Implements the bank described in
``infra_plans/05_skills/skill_extraction_bank.md``:

- :class:`SkillRecord` — canonical, serializable, versioned bank entry.
- :class:`ReasoningSkillBank` — registry + lookup helpers.
- :mod:`.atomics` — the curated v1 starter inventory of atomic skills
  (12 skills covering question parsing, retrieval+grounding, temporal,
  causal, social/belief, verification, and decision families).

Per the design principle ("stable memory, evolving reasoning"), the bank
holds **reasoning skills only**. Memory writes go through
:mod:`video_skills.memory.procedures` instead.
"""

from .atomics import build_starter_bank, register_starter_skills
from .bank import (
    AtomicSkill,
    ReasoningSkillBank,
    SkillRecord,
    SkillUsage,
    SkillVersion,
)

__all__ = [
    "AtomicSkill",
    "ReasoningSkillBank",
    "SkillRecord",
    "SkillUsage",
    "SkillVersion",
    "build_starter_bank",
    "register_starter_skills",
]

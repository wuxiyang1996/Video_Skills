"""
Central function identifier for LoRA adapter routing.

Each skill-bank function maps to exactly one LoRA adapter.
Use ``SkillFunction`` as the routing key everywhere.
"""

from __future__ import annotations

from enum import Enum


class SkillFunction(str, Enum):
    """Skill-bank functions, each backed by a dedicated LoRA adapter."""

    BOUNDARY = "boundary"
    SEGMENT = "segment"
    CONTRACT = "contract"
    RETRIEVAL = "retrieval"
    CURATOR = "curator"

    @classmethod
    def from_str(cls, value: str) -> "SkillFunction":
        """Case-insensitive lookup, e.g. ``SkillFunction.from_str("Boundary")``."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown skill function {value!r}. Choose from: {valid}")

    @property
    def adapter_name(self) -> str:
        """Name used when registering / selecting a PEFT adapter."""
        return self.value

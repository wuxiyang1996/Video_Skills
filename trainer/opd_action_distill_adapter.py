"""Persist OPD distill rows: state, candidates, teacher distribution, seeds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .candidate_action_builder import CandidateActionSet
from .closed_loop_harness import HarnessState
from .teacher_action_query import TeacherActionDistribution


@dataclass
class OpdDistillRow:
    schema_version: str = "video-skills/opd-distill-v1"
    state: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, Any] = field(default_factory=dict)
    teacher: dict[str, Any] = field(default_factory=dict)
    student_checkpoint: str | None = None
    precheck: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "candidates": self.candidates,
            "teacher": self.teacher,
            "student_checkpoint": self.student_checkpoint,
            "precheck": self.precheck,
        }

    @classmethod
    def from_parts(
        cls,
        *,
        state: HarnessState,
        action_set: CandidateActionSet,
        teacher: TeacherActionDistribution,
        precheck: dict[str, Any],
        student_checkpoint: str | None = None,
    ) -> "OpdDistillRow":
        return cls(
            state=state.to_dict(),
            candidates=action_set.to_dict(),
            teacher=teacher.to_dict(),
            student_checkpoint=student_checkpoint,
            precheck=precheck,
        )


def save_opd_rows(path: str | Path, rows: Iterable[OpdDistillRow]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_opd_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

"""Closed-loop harness: frozen L1 + Motif-gated student rollouts → on-policy states."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


RolloutFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass
class HarnessState:
    """One student-induced OPD/GRPO state after Motif retrieve/expand attempt."""

    state_id: str
    example_id: str
    dataset: str
    task_family: str
    question: dict[str, Any]
    l1_graph_summary: dict[str, Any]
    motif_online: dict[str, Any]
    student_action: dict[str, Any] | None = None
    rollout_acceptance: str | None = None
    verifier_passed: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "example_id": self.example_id,
            "dataset": self.dataset,
            "task_family": self.task_family,
            "question": self.question,
            "l1_graph_summary": self.l1_graph_summary,
            "motif_online": self.motif_online,
            "student_action": self.student_action,
            "rollout_acceptance": self.rollout_acceptance,
            "verifier_passed": self.verifier_passed,
            "extras": self.extras,
        }


def summarize_l1_graph(clue_memory_graph: dict[str, Any]) -> dict[str, Any]:
    nodes = clue_memory_graph.get("nodes") or []
    edges = clue_memory_graph.get("edges") or []
    return {
        "graph_id": clue_memory_graph.get("graph_id"),
        "video_id": clue_memory_graph.get("video_id"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": sorted(
            {
                str(n.get("node_type") or "")
                for n in nodes
                if isinstance(n, dict) and n.get("node_type")
            }
        ),
    }


def extract_student_action_from_rollout(rollout: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort first complete action from motif plan or executed skills."""
    meta = rollout.get("metadata") or {}
    plan = (meta.get("llm_plan") or {}).get("reasoning_plan") or []
    if plan and isinstance(plan[0], dict):
        step = plan[0]
        return {
            "schema_version": "video-skills/l2-specialist-action-v0.1",
            "tool_name": step.get("skill_id") or "unknown",
            "arguments": step.get("args") or {},
        }
    skills = meta.get("executed_skill_ids") or []
    if skills:
        return {
            "schema_version": "video-skills/l2-specialist-action-v0.1",
            "tool_name": str(skills[0]),
            "arguments": {},
        }
    return None


class ClosedLoopHarness:
    """Run Motif-gated L2 rollouts over frozen L1 examples and emit OPD states."""

    def __init__(
        self,
        *,
        rollout_fn: RolloutFn,
        motif_enabled: bool = True,
        require_motif_attempt: bool = True,
    ) -> None:
        self.rollout_fn = rollout_fn
        self.motif_enabled = motif_enabled
        self.require_motif_attempt = require_motif_attempt

    def run_example(self, example: dict[str, Any]) -> HarnessState:
        meta = dict(example.get("metadata") or {})
        meta["motif_enabled"] = self.motif_enabled
        example = {**example, "metadata": meta}
        clue = meta.get("clue_memory_graph") or {}
        rollout = self.rollout_fn(example, clue)
        motif_online = (rollout.get("metadata") or {}).get("motif_online") or {}
        if self.require_motif_attempt and self.motif_enabled:
            if not motif_online.get("motif_retrieval_attempted"):
                raise RuntimeError("motif_retrieval_attempted must be true when motif_enabled")

        example_id = str(example.get("example_id") or meta.get("example_id") or "unknown")
        state = HarnessState(
            state_id=f"opd:{example_id}",
            example_id=example_id,
            dataset=str(example.get("dataset") or ""),
            task_family=str(example.get("task_family") or ""),
            question=dict(example.get("question") or {}),
            l1_graph_summary=summarize_l1_graph(clue if isinstance(clue, dict) else {}),
            motif_online=dict(motif_online),
            student_action=extract_student_action_from_rollout(rollout),
            rollout_acceptance=str(rollout.get("acceptance_status") or "") or None,
            verifier_passed=bool(((rollout.get("metadata") or {}).get("runtime_verifier") or {}).get("passed")),
            extras={
                "final_answer": rollout.get("final_answer"),
                "failure_reasons": rollout.get("failure_reasons") or [],
            },
        )
        return state

    def run_many(self, examples: Iterable[dict[str, Any]]) -> list[HarnessState]:
        return [self.run_example(example) for example in examples]


def load_frozen_l1_examples(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load staged ``04_l1_example.json`` (or any JSON with clue_memory_graph)."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_harness_states(path: str | Path, states: Iterable[HarnessState]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state.to_dict(), ensure_ascii=False) + "\n")

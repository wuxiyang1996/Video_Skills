"""Harness — the deterministic hop interpreter.

Implements the harness contract from
``infra_plans/04_harness/atomic_skills_hop_refactor_execution_plan.md``:

- Receives a ``HopGoal`` + a chosen skill (atomic OR composite).
- Builds the skill's input dict, calls the executable, validates the output
  against ``output_schema``, wraps it in an :class:`AtomicStepResult`, and
  routes through the :class:`Verifier`.
- Honors the verifier's ``next_action`` (continue / retry / broaden /
  switch_skill / abstain).
- Emits a :class:`HopRecord` with cost accounting and trace logging.
- Routes any ``requested_memory_writes`` from the skill through the
  :class:`MemoryProcedureRegistry`.

Per the design split: the harness **does not learn**, **does not select
skills**, **does not abstain on its own**. It executes plans and emits
canonical objects.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    AtomicStepResult,
    EvidenceBundle,
    HopGoal,
    HopRecord,
    RetrievalQuery,
    VerificationResult,
    new_id,
    validate_atomic_step,
)
from .skills.bank import (
    AtomicSkill,
    ReasoningSkillBank,
    SkillContext,
    SkillOutput,
)


@dataclass
class HarnessConfig:
    """Operational caps from the harness spec."""

    max_atomic_steps_per_hop: int = 6
    max_verifier_retries_per_step: int = 2
    max_broaden_per_hop: int = 2
    delta_hop_progress: float = 0.05
    atomic_latency_budget_s: float = 5.0


@dataclass
class HopExecutionContext:
    """Aggregates all the runtime handles a hop needs."""

    bank: ReasoningSkillBank
    memory: Any
    memory_procedures: Any
    retriever: Any
    verifier: Any
    config: HarnessConfig = field(default_factory=HarnessConfig)


class Harness:
    """Deterministic hop interpreter."""

    def __init__(self, ctx: HopExecutionContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------
    # Composite expansion
    # ------------------------------------------------------------------

    def expand(self, skill: AtomicSkill) -> List[AtomicSkill]:
        """Resolve a composite to its ordered atomic chain.

        Composites never execute as opaque blocks (§HarnessCompositeExpansion).
        Atomic skills expand to a 1-element list.
        """
        if skill.record.type == "atomic":
            return [skill]
        chain: List[AtomicSkill] = []
        for child_id in skill.record.child_links:
            child = self.ctx.bank.get(child_id) if self.ctx.bank.has(child_id) else None
            if child is None:
                # Try by name as a convenience
                if self.ctx.bank.has(child_id):
                    child = self.ctx.bank.get_by_name(child_id)
            if child is None:
                raise KeyError(
                    f"composite {skill.skill_id} references missing child {child_id!r}"
                )
            chain.extend(self.expand(child))
        return chain

    # ------------------------------------------------------------------
    # Single-hop execution
    # ------------------------------------------------------------------

    def run_hop(
        self,
        hop_goal: HopGoal,
        skill: AtomicSkill,
        *,
        seed_inputs: Optional[Dict[str, Any]] = None,
    ) -> HopRecord:
        """Execute one hop end-to-end."""
        atomic_chain = self.expand(skill)
        record = HopRecord(
            hop_goal=hop_goal,
            steps=[],
            outcome="in_progress",
            cost={
                "atomic_steps": 0,
                "retrieval_calls": 0,
                "broaden_levels": 0,
                "latency_ms": 0,
            },
            meta={
                "expanded_from_skill_id": skill.skill_id,
                "expanded_chain": [a.skill_id for a in atomic_chain],
                "broaden_history": [],
                "skill_switch_history": [],
            },
        )
        self.ctx.retriever.reset_broaden(hop_goal.hop_id)
        last_evidence: Optional[EvidenceBundle] = None
        broaden_count = 0
        consecutive_failures = 0
        chain_idx = 0

        # Start time for hop-level latency
        hop_start = time.perf_counter()

        while chain_idx < len(atomic_chain):
            if record.cost["atomic_steps"] >= min(
                hop_goal.max_atomic_steps,
                self.ctx.config.max_atomic_steps_per_hop,
            ):
                # Cap reached -> blocked
                record.outcome = "blocked"
                record.meta["block_reason"] = "max_atomic_steps_reached"
                break

            atomic = atomic_chain[chain_idx]
            inputs = self._build_inputs(
                atomic=atomic,
                hop_goal=hop_goal,
                steps_so_far=record.steps,
                seed_inputs=seed_inputs,
            )
            step_evidence = last_evidence
            # Auto-issue retrieval if the skill requires it and no upstream evidence
            if (
                step_evidence is None
                and "search_memory" in atomic.record.required_primitives
            ):
                step_evidence = self._issue_default_retrieval(atomic, hop_goal, record)

            step, output = self._invoke_atomic(
                atomic=atomic,
                hop_goal=hop_goal,
                inputs=inputs,
                evidence=step_evidence,
                steps_so_far=record.steps,
            )
            record.steps.append(step)
            record.cost["atomic_steps"] += 1
            record.cost["latency_ms"] += step.latency_ms

            # Apply any memory-write requests via the procedure registry
            if output is not None and step.verification.passed:
                for req in output.requested_memory_writes:
                    proc = req.get("procedure")
                    args = req.get("args", {})
                    if proc:
                        self.ctx.memory_procedures.call(
                            proc, caller=atomic.skill_id, **args
                        )

            # Decide what to do next
            action = step.verification.next_action
            if action == "continue":
                consecutive_failures = 0
                # Forward this step's evidence to the next atomic if it
                # produced an evidence_set.
                if step.evidence is not None:
                    last_evidence = step.evidence
                chain_idx += 1
            elif action == "retry":
                # Retry counts as a step against the cap (per harness spec).
                consecutive_failures += 1
                if consecutive_failures > self.ctx.config.max_verifier_retries_per_step:
                    record.outcome = "blocked"
                    record.meta["block_reason"] = "max_retries_exceeded"
                    break
                # No chain_idx advance — same atomic re-runs next iteration
            elif action == "broaden":
                if broaden_count >= self.ctx.config.max_broaden_per_hop:
                    record.outcome = "blocked"
                    record.meta["block_reason"] = "broaden_exhausted"
                    break
                base_query = self._last_query_for_hop(record)
                widened = self.ctx.retriever.broaden(hop_goal.hop_id, base_query)
                broaden_count += 1
                record.cost["broaden_levels"] = broaden_count
                record.cost["retrieval_calls"] += 1
                record.meta["broaden_history"].append({
                    "step_id": step.step_id,
                    "level": broaden_count,
                    "query_id": widened.query.query_id,
                })
                last_evidence = widened
                # Re-run same atomic with widened evidence
            elif action == "switch_skill":
                record.outcome = "blocked"
                record.meta["block_reason"] = "switch_skill_requested"
                record.meta["skill_switch_history"].append(atomic.skill_id)
                break
            elif action == "abstain":
                record.outcome = "abstain"
                break
            else:
                record.outcome = "blocked"
                record.meta["block_reason"] = f"unknown_action:{action}"
                break

            # Fail-fast: two consecutive failed verifications block the hop
            if not step.verification.passed:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    record.outcome = "blocked"
                    record.meta["block_reason"] = "two_consecutive_step_failures"
                    break
            else:
                consecutive_failures = 0

        if record.outcome == "in_progress":
            # Reached end of chain without explicit termination
            record.outcome = "resolved"

        # Hop-level verification (the verifier reads all steps)
        record.hop_verification = self.ctx.verifier.verify_hop(record)
        record.cost["latency_ms"] = max(
            record.cost["latency_ms"],
            int((time.perf_counter() - hop_start) * 1000),
        )
        return record

    # ------------------------------------------------------------------
    # Atomic invocation + I/O contract
    # ------------------------------------------------------------------

    def _invoke_atomic(
        self,
        *,
        atomic: AtomicSkill,
        hop_goal: HopGoal,
        inputs: Dict[str, Any],
        evidence: Optional[EvidenceBundle],
        steps_so_far: List[AtomicStepResult],
    ) -> Tuple[AtomicStepResult, Optional[SkillOutput]]:
        skill_ctx = SkillContext(
            hop_goal=hop_goal,
            inputs=inputs,
            evidence=evidence,
            memory=self.ctx.memory,
            memory_procedures=self.ctx.memory_procedures,
            retriever=self.ctx.retriever,
            trace_so_far=list(steps_so_far),
        )
        t0 = time.perf_counter()
        failure_mode: Optional[str] = None
        output: Optional[SkillOutput] = None
        try:
            output = atomic.executable(skill_ctx)
        except Exception as exc:  # noqa: BLE001
            failure_mode = "exception"
            output = SkillOutput(
                output={"error": str(exc), "type": type(exc).__name__},
                output_type="meta",
                confidence=0.0,
                inferred=True,
                failure_mode="exception",
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Schema validation
        if output is not None and failure_mode is None:
            schema_ok = self._validate_output_schema(atomic, output.output)
            if not schema_ok:
                failure_mode = "schema_violation"
                output = SkillOutput(
                    output=output.output,
                    output_type=output.output_type,
                    confidence=output.confidence,
                    inferred=output.inferred,
                    failure_mode="schema_violation",
                    used_evidence=output.used_evidence,
                )

        used_evidence = (
            output.used_evidence if output is not None else None
        ) or evidence

        # Build the AtomicStepResult shell
        step = AtomicStepResult(
            step_id=new_id("step"),
            hop_id=hop_goal.hop_id,
            skill_id=atomic.skill_id,
            inputs=_jsonify(inputs),
            output=output.output if output is not None else {},
            output_type=output.output_type if output is not None else "meta",
            evidence=used_evidence,
            verification=VerificationResult(passed=False),  # filled below
            confidence=output.confidence if output is not None else 0.0,
            inferred=output.inferred if output is not None else True,
            failure_mode=output.failure_mode if output is not None else failure_mode,
            latency_ms=latency_ms,
            meta={
                "atomic_skill_id": atomic.skill_id,
                **({"requested_writes": [r["procedure"] for r in output.requested_memory_writes]} if output and output.requested_memory_writes else {}),
            },
        )

        # Per-step verification through the global Verifier (cheaper than
        # re-implementing the same checks inside every skill).
        step.verification = self.ctx.verifier.verify_step(step)
        if not step.verification.passed and step.failure_mode is None:
            step.failure_mode = "verification_failed"

        # Final contract validation (Rule 3)
        violations = validate_atomic_step(step)
        if violations:
            step.failure_mode = step.failure_mode or "schema_violation"
            step.verification.passed = False
            step.verification.reasons.extend(violations)
        return step, output

    def _validate_output_schema(
        self,
        atomic: AtomicSkill,
        out: Dict[str, Any],
    ) -> bool:
        for key, spec in atomic.record.output_schema.items():
            if isinstance(spec, dict) and spec.get("required") and key not in out:
                return False
        return True

    # ------------------------------------------------------------------
    # Input building
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        *,
        atomic: AtomicSkill,
        hop_goal: HopGoal,
        steps_so_far: List[AtomicStepResult],
        seed_inputs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        if seed_inputs:
            inputs.update(seed_inputs)
        # Pull common slots from the hop goal
        if "question_text" in atomic.record.input_schema:
            inputs.setdefault("question_text", hop_goal.goal_text)
        if "target_entities" in atomic.record.input_schema:
            inputs.setdefault("target_entities", list(hop_goal.required_entities))
        if "holder" in atomic.record.input_schema and hop_goal.perspective_anchor:
            inputs.setdefault("holder", hop_goal.perspective_anchor)

        # Forward useful outputs from prior steps in this hop
        for prior in steps_so_far:
            if prior.output_type == "entity_ref" and prior.output.get("entity_id"):
                inputs.setdefault("mention", prior.output.get("mention"))
            if prior.output_type == "span" and prior.output.get("event_ids"):
                ids = prior.output["event_ids"]
                if "event_a_id" in atomic.record.input_schema and "event_a_id" not in inputs:
                    inputs["event_a_id"] = ids[0]
                if "event_b_id" in atomic.record.input_schema and "event_b_id" not in inputs and len(ids) >= 2:
                    inputs["event_b_id"] = ids[1]
                if "event_ids" in atomic.record.input_schema:
                    inputs.setdefault("event_ids", list(ids))
        return inputs

    def _issue_default_retrieval(
        self,
        atomic: AtomicSkill,
        hop_goal: HopGoal,
        record: HopRecord,
    ) -> EvidenceBundle:
        # Use skill-declared hint or a default rewritten query
        if atomic.record.retrieval_hints:
            query = atomic.record.retrieval_hints[0]
        else:
            queries = self.ctx.retriever.rewrite(hop_goal)
            query = queries[0]
        bundle = self.ctx.retriever.retrieve(query)
        record.cost["retrieval_calls"] += 1
        return bundle

    def _last_query_for_hop(self, record: HopRecord) -> RetrievalQuery:
        for s in reversed(record.steps):
            if s.evidence is not None:
                return s.evidence.query
        # fallback
        return self.ctx.retriever.rewrite(record.hop_goal)[0]


def _jsonify(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make a shallow JSON-friendly snapshot of an inputs dict."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x if isinstance(x, (str, int, float, bool, type(None))) else repr(x)
                for x in v
            ]
        elif isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool, type(None)))}
        else:
            out[k] = repr(v)
    return out

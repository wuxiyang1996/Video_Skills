"""Online serving loop — wires controller + harness + verifier + retriever.

Implements §2D from
``infra_plans/03_controller/actors_reasoning_model.md``:

::

    on_question(q):
      qa            = controller.analyze_question(q)
      trace         = new ReasoningTrace(qa)
      while not done(trace):
          hop_goal  = controller.next_hop(trace)
          skill     = controller.select_skill(hop_goal, bank)
          hop       = harness.run_hop(hop_goal, skill)
          vresult   = verifier.verify_hop(hop)
          trace.append(hop, vresult)
          # ... action dispatch ...

      final_v   = verifier.verify_final(trace)
      if final_v.passed:
          trace.answer = controller.compose_answer(trace)
      else:
          trace.abstain = verifier.decide_abstain(trace)

This module does not own any state — it is a thin orchestrator. Tests use
:func:`build_runtime` to construct a ready-to-run set of subsystems for a
:class:`Memory` instance, and :func:`run_question` to drive the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .contracts import (
    QuestionAnalysis,
    ReasoningTrace,
    new_id,
    now_ts,
)
from .controller import Controller, ControllerConfig
from .harness import Harness, HarnessConfig, HopExecutionContext
from .memory import Memory, MemoryProcedureRegistry
from .retriever import Retriever, RetrieverConfig
from .skills import ReasoningSkillBank, build_starter_bank
from .verifier import Verifier, VerifierConfig


@dataclass
class Runtime:
    """Bundle of all live subsystems for one Video_Skills instance."""

    memory: Memory
    memory_procedures: MemoryProcedureRegistry
    bank: ReasoningSkillBank
    retriever: Retriever
    verifier: Verifier
    controller: Controller
    harness: Harness


def build_runtime(
    *,
    memory: Optional[Memory] = None,
    bank: Optional[ReasoningSkillBank] = None,
    retriever_config: Optional[RetrieverConfig] = None,
    verifier_config: Optional[VerifierConfig] = None,
    harness_config: Optional[HarnessConfig] = None,
    controller_config: Optional[ControllerConfig] = None,
) -> Runtime:
    """Construct a fully-wired runtime."""
    mem = memory or Memory()
    procs = MemoryProcedureRegistry(mem)
    skill_bank = bank or build_starter_bank()
    retriever = Retriever(mem, retriever_config)
    verifier = Verifier(verifier_config)
    controller = Controller(skill_bank, controller_config)
    harness = Harness(
        HopExecutionContext(
            bank=skill_bank,
            memory=mem,
            memory_procedures=procs,
            retriever=retriever,
            verifier=verifier,
            config=harness_config or HarnessConfig(),
        )
    )
    return Runtime(
        memory=mem,
        memory_procedures=procs,
        bank=skill_bank,
        retriever=retriever,
        verifier=verifier,
        controller=controller,
        harness=harness,
    )


def run_question(
    runtime: Runtime,
    question_text: str,
    *,
    target_entities: Optional[List[str]] = None,
    perspective_anchor: Optional[str] = None,
    time_anchor: Optional[Any] = None,
) -> ReasoningTrace:
    """Run the §2D loop end-to-end on one question.

    Returns the populated :class:`ReasoningTrace`. The trace is the only
    thing GRPO / SFT / reflection downstream consume — see
    ``infra_plans/06_training/training_plan_sft_grpo.md``.
    """
    qa = runtime.controller.analyze_question(
        question_text,
        target_entities=target_entities,
        perspective_anchor=perspective_anchor,
        time_anchor=time_anchor,
    )
    trace = ReasoningTrace(
        trace_id=new_id("trc"),
        question_id=qa.question_id,
        question_analysis=qa,
        started_at=now_ts(),
    )

    abstain_break = False
    while True:
        hop_goal = runtime.controller.next_hop(trace)
        if hop_goal is None:
            break
        skill_id = runtime.controller.select_skill(hop_goal, runtime.bank, trace=trace)
        skill = runtime.bank.get(skill_id)
        seed_inputs = _carryover_inputs_from_trace(trace)
        hop = runtime.harness.run_hop(hop_goal, skill, seed_inputs=seed_inputs)
        trace.append_hop(hop)

        action = hop.hop_verification.next_action
        if action == "switch_skill":
            runtime.controller.blacklist(skill_id, trace=trace)
            continue
        if action == "abstain":
            abstain_break = True
            break
        # broaden / retry are already handled inside the harness loop;
        # at the trace level we just continue to the next hop.

    trace.final_verification = runtime.verifier.verify_final(trace)
    if trace.hops:
        trace.final_evidence = _last_nonempty_bundle(trace)

    if trace.final_verification.passed and not abstain_break:
        trace.answer = runtime.controller.compose_answer(trace)
    else:
        decision = runtime.verifier.decide_abstain(trace)
        trace.abstain = runtime.controller.maybe_override_abstain(trace, decision)

    trace.finished_at = now_ts()
    return trace


def _last_nonempty_bundle(trace: ReasoningTrace):
    for hop in reversed(trace.hops):
        for step in reversed(hop.steps):
            if step.evidence is not None and not step.evidence.is_empty():
                return step.evidence
    return None


def _carryover_inputs_from_trace(trace: ReasoningTrace) -> Optional[dict]:
    """Pull useful slot bindings from prior hops so the next hop's first
    atomic can consume them.

    This is the trace-level analog of the harness's per-hop input forwarding.
    It looks back through all prior hops (newest first) and collects the most
    recent ``span`` / ``entity_ref`` outputs.
    """
    seed: dict = {}
    event_ids: list = []
    entity_id = None
    mention = None
    for hop in reversed(trace.hops):
        for step in reversed(hop.steps):
            if step.output_type == "span" and not event_ids:
                ids = step.output.get("event_ids") or []
                event_ids = list(ids)
            if step.output_type == "entity_ref" and entity_id is None:
                entity_id = step.output.get("entity_id")
                mention = step.output.get("mention")
            if event_ids and entity_id is not None:
                break
        if event_ids and entity_id is not None:
            break
    if event_ids:
        seed["event_ids"] = event_ids
        if len(event_ids) >= 1:
            seed["event_a_id"] = event_ids[0]
        if len(event_ids) >= 2:
            seed["event_b_id"] = event_ids[1]
    if entity_id is not None:
        seed["entity_id"] = entity_id
        if mention is not None:
            seed["mention"] = mention
    return seed or None

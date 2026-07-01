"""L2 reasoning planner: gpt-oss plans question-conditioned reasoning skill programs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_skills import export_skill_ontology  # noqa: E402
from atomic_skills.common import stable_id  # noqa: E402
from atomic_skills.reasoning_graph_assembly import (  # noqa: E402
    assign_evidence_role,
    bridge_evidence_hops,
    commit_answer,
    compare_hypotheses,
    compose_evidence_chain,
    detect_missing_role,
    extract_claim,
    generate_answer_hypotheses,
    infer_causal_relation,
    infer_intention_or_motive,
    infer_social_contradiction,
    infer_state_change,
    infer_temporal_relation,
    localize_clue,
    parse_question_target,
    propose_evidence_roles,
    retrieve_by_entity,
    retrieve_evidence_for_hypothesis,
    retrieve_by_event,
    retrieve_by_relation,
    retrieve_by_time,
    search_counterevidence,
    score_hypothesis_support,
    verify_claim_support,
    verify_temporal_social_consistency,
)

from .clue_memory import make_reasoning_rollout_shell
from .graph_plan_validator import resolve_plan_value, _coerce_node_ref
from .openrouter_client import OpenRouterClient, load_openrouter_api_key
from .schemas import GraphComposerConfig

REASONING_SKILL_EXECUTORS = {
    "parse_question_target": parse_question_target,
    "propose_evidence_roles": propose_evidence_roles,
    "retrieve_by_event": retrieve_by_event,
    "retrieve_by_entity": retrieve_by_entity,
    "retrieve_by_time": retrieve_by_time,
    "retrieve_by_relation": retrieve_by_relation,
    "localize_clue": localize_clue,
    "extract_claim": extract_claim,
    "assign_evidence_role": assign_evidence_role,
    "generate_answer_hypotheses": generate_answer_hypotheses,
    "retrieve_evidence_for_hypothesis": retrieve_evidence_for_hypothesis,
    "score_hypothesis_support": score_hypothesis_support,
    "compare_hypotheses": compare_hypotheses,
    "bridge_evidence_hops": bridge_evidence_hops,
    "verify_temporal_social_consistency": verify_temporal_social_consistency,
    "compose_evidence_chain": compose_evidence_chain,
    "detect_missing_role": detect_missing_role,
    "search_counterevidence": search_counterevidence,
    "infer_temporal_relation": infer_temporal_relation,
    "infer_state_change": infer_state_change,
    "infer_causal_relation": infer_causal_relation,
    "infer_intention_or_motive": infer_intention_or_motive,
    "infer_social_contradiction": infer_social_contradiction,
    "verify_claim_support": verify_claim_support,
    "commit_answer": commit_answer,
}

REASONING_SKILL_IDS = sorted(REASONING_SKILL_EXECUTORS.keys())

_REASONING_PLAN_PROMPT = """You are an expert video-reasoning planner. Given a question and a Layer-1
clue-memory graph (perception evidence), plan which reasoning skills to execute and in what order.

This is a MULTIPLE-CHOICE skill selection task:
- skill_id MUST be exactly one value from allowed_skill_ids.
- Reference prior step outputs with $step.<step_id>.evidence_refs.N or $step.<step_id>.<output_field>.
- Use $bindings.question_text, $bindings.options, $bindings.graph for inputs.
- L1 node_ids from the clue graph can be used directly as string refs.

Return JSON only:
{
  "reasoning_plan": [
    {
      "step_id": "r1",
      "skill_id": "parse_question_target",
      "args": {"question_text": "$bindings.question_text", "options": "$bindings.options"},
      "depends_on": []
    },
    {
      "step_id": "r2",
      "skill_id": "retrieve_by_event",
      "args": {"evidence_graph": "$bindings.graph", "event_description": "$bindings.question_text"},
      "depends_on": ["r1"]
    }
  ],
  "notes": "short reasoning strategy summary",
  "expected_answer_format": "multiple_choice"
}

Skill execution rules:
1. Always start with parse_question_target and propose_evidence_roles.
2. For multiple-choice or complex social questions, prefer the option-level path:
   generate_answer_hypotheses -> retrieve_evidence_for_hypothesis ->
   score_hypothesis_support -> compare_hypotheses.
3. Use bridge_evidence_hops when the answer requires linking source evidence to
   an option through object/location/action/state evidence.
4. Use verify_temporal_social_consistency before final verification when social
   or temporal plausibility matters.
5. Use retrieve_by_event / retrieve_by_entity / retrieve_by_time to find relevant L1 evidence.
6. Use localize_clue and extract_claim to ground claims in evidence.
7. Use assign_evidence_role + compose_evidence_chain to build the support structure.
8. Use infer_* skills for temporal, causal, state-change, or social reasoning as needed.
9. Always end with verify_claim_support then commit_answer.
10. Keep plans between 8-18 steps. Do not over-plan.
11. Do not output chain-of-thought.
"""


def _build_reasoning_plan_schema(allowed_skill_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "reasoning_skill_plan",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "step_id": {"type": "string"},
                                "skill_id": {"type": "string", "enum": allowed_skill_ids},
                                "args": {"type": "object", "additionalProperties": True},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["step_id", "skill_id", "args", "depends_on"],
                        },
                    },
                    "notes": {"type": "string"},
                    "expected_answer_format": {"type": "string"},
                },
                "required": ["reasoning_plan", "notes"],
            },
        },
    }


def _summarize_clue_graph(clue_memory_graph: dict[str, Any], max_nodes: int = 20) -> dict[str, Any]:
    """Compact summary of L1 graph for the reasoning planner prompt."""
    nodes = clue_memory_graph.get("nodes") or []
    summary_nodes = []
    for node in nodes[:max_nodes]:
        summary_nodes.append({
            "node_id": node.get("node_id"),
            "node_type": node.get("node_type"),
            "text": (node.get("text") or node.get("event_description") or node.get("surface_form") or "")[:120],
            "time_span": node.get("time_span"),
        })
    return {
        "total_nodes": len(nodes),
        "shown_nodes": summary_nodes,
        "node_types": list({n.get("node_type") for n in nodes if n.get("node_type")}),
        "edge_count": len(clue_memory_graph.get("edges") or []),
    }


def plan_reasoning_skills(
    *,
    question: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    task_family: str,
    client: OpenRouterClient,
) -> dict[str, Any]:
    """Call gpt-oss to plan a question-conditioned reasoning skill program."""
    ontology = export_skill_ontology()["reasoning_graph_assembly"]
    graph_summary = _summarize_clue_graph(clue_memory_graph)

    payload = {
        "task": "plan_reasoning_over_clue_graph",
        "question": question,
        "task_family": task_family,
        "clue_graph_summary": graph_summary,
        "allowed_skill_ids": REASONING_SKILL_IDS,
        "ontology": ontology,
        "instructions": _REASONING_PLAN_PROMPT,
    }

    response = client.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert video-reasoning planner. "
                    "Choose reasoning skills from allowed_skill_ids (multiple choice). "
                    "Plan a skill program to answer the question using L1 evidence."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format=_build_reasoning_plan_schema(REASONING_SKILL_IDS),
    )
    response["model"] = client.model
    response["planner"] = "gpt_oss_reasoning_planner"
    return response


def execute_reasoning_plan(
    *,
    reasoning_plan: list[dict[str, Any]],
    clue_memory_graph: dict[str, Any],
    question: dict[str, Any],
    skill_executor: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute a reasoning skill plan over the clue graph, returning trace + step_outputs.

    If skill_executor is provided (a SkillExecutor instance), skills configured
    for LLM/VLM mode will be dispatched via API calls; otherwise pure rule-based.
    """
    from copy import deepcopy

    graph = {
        "schema_version": clue_memory_graph.get("schema_version"),
        "nodes": deepcopy(clue_memory_graph.get("nodes") or []),
        "edges": deepcopy(clue_memory_graph.get("edges") or []),
    }
    question_text = question.get("question_text") or ""
    options = question.get("options") or []

    bindings = {
        "question_text": question_text,
        "options": options,
        "graph": graph,
        "task_family": question.get("task_family") or "",
    }

    trace: list[dict[str, Any]] = []
    step_outputs: dict[str, Any] = {}

    def _one_hypothesis(value: Any) -> Any:
        if isinstance(value, list):
            return next((item for item in value if isinstance(item, dict)), value[0] if value else {"claim_text": question_text})
        return value or {"claim_text": question_text}

    for step in reasoning_plan:
        step_id = step.get("step_id")
        skill_id = step.get("skill_id")
        raw_args = dict(step.get("args") or {})

        if skill_id not in REASONING_SKILL_EXECUTORS:
            trace.append({"step_id": step_id, "skill_id": skill_id, "ok": False, "failure_code": "unknown_skill_id"})
            continue

        resolved_args = resolve_plan_value(raw_args, bindings, step_outputs)

        # --- LLM/VLM dispatch via SkillExecutor ---
        if skill_executor is not None:
            from atomic_skills.skill_backends import SkillBackendMode
            mode = skill_executor.config.mode_for(skill_id)
            if mode in (SkillBackendMode.LLM, SkillBackendMode.VLM):
                has_client = (
                    (mode == SkillBackendMode.LLM and skill_executor.llm_client)
                    or (mode == SkillBackendMode.VLM and skill_executor.vlm_client)
                )
                if has_client:
                    try:
                        result = skill_executor.execute(skill_id, args=resolved_args, graph=graph)
                        trace.append({
                            "step_id": step_id,
                            "skill_id": skill_id,
                            "ok": result.ok,
                            "failure_code": result.failure_code,
                            "evidence_refs": result.evidence_refs,
                            "confidence": result.confidence,
                            "backend": mode.value,
                        })
                        if step_id:
                            step_outputs[step_id] = {**result.outputs, "evidence_refs": result.evidence_refs}
                        continue
                    except Exception as exc:
                        trace.append({
                            "step_id": step_id,
                            "skill_id": skill_id,
                            "ok": False,
                            "failure_code": f"{mode.value}_backend_error",
                            "messages": [str(exc)],
                            "backend": mode.value,
                        })
                        continue

        executor = REASONING_SKILL_EXECUTORS[skill_id]

        try:
            if skill_id in ("retrieve_by_event", "retrieve_by_entity", "retrieve_by_time",
                            "retrieve_by_relation", "assign_evidence_role",
                            "search_counterevidence"):
                filtered = {k: v for k, v in resolved_args.items() if k != "evidence_graph"}
                if skill_id == "assign_evidence_role":
                    ev_ref = filtered.get("evidence_ref")
                    if isinstance(ev_ref, list):
                        filtered["evidence_ref"] = ev_ref[0] if ev_ref else "missing"
                    elif isinstance(ev_ref, dict):
                        filtered["evidence_ref"] = ev_ref.get("node_id") or ev_ref.get("evidence_ref") or "missing"
                result = executor(graph, **filtered)
            elif skill_id == "extract_claim":
                filtered = {k: v for k, v in resolved_args.items() if k != "evidence_graph"}
                ev_ref = filtered.get("evidence_ref")
                if isinstance(ev_ref, list):
                    filtered["evidence_ref"] = ev_ref[0] if ev_ref else "missing"
                elif isinstance(ev_ref, dict):
                    filtered["evidence_ref"] = ev_ref.get("node_id") or ev_ref.get("evidence_ref") or "missing"
                result = executor(graph, **filtered)
            elif skill_id == "infer_state_change":
                filtered = {k: v for k, v in resolved_args.items() if k != "evidence_graph"}
                ba_refs = filtered.get("before_after_refs") or []
                if isinstance(ba_refs, str):
                    ba_refs = [ba_refs]
                ba_refs = [r if isinstance(r, str) else (r.get("node_id") if isinstance(r, dict) else str(r)) for r in ba_refs]
                filtered["before_after_refs"] = ba_refs
                result = executor(graph, **filtered)
            elif skill_id == "infer_temporal_relation":
                event_refs = resolved_args.get("event_refs") or []
                if isinstance(event_refs, str):
                    event_refs = [event_refs]
                elif isinstance(event_refs, dict):
                    event_refs = [event_refs.get("node_id") or str(event_refs)]
                event_refs = [r if isinstance(r, str) else (r.get("node_id") if isinstance(r, dict) else str(r)) for r in event_refs]
                result = executor(event_refs, evidence_graph=graph)
            elif skill_id == "localize_clue":
                candidate_evidence = resolved_args.get("candidate_evidence") or [
                    n for n in graph.get("nodes", []) if n.get("node_type") in ("observation", "event")
                ]
                result = executor(
                    candidate_evidence,
                    role_constraint=resolved_args.get("role_constraint") or "supporting_evidence",
                    question_context=resolved_args.get("question_context") or question_text,
                )
            elif skill_id == "parse_question_target":
                result = executor(
                    resolved_args.get("question_text") or question_text,
                    options=resolved_args.get("options") or options or None,
                )
            elif skill_id == "propose_evidence_roles":
                parsed_target = resolved_args.get("parsed_target")
                if not isinstance(parsed_target, dict):
                    parsed_target = step_outputs.get(reasoning_plan[0].get("step_id"), {})
                    if not isinstance(parsed_target, dict):
                        parsed_target = {}
                result = executor(
                    resolved_args.get("question_text") or question_text,
                    parsed_target,
                    task_family=resolved_args.get("task_family") or "",
                )
            elif skill_id == "generate_answer_hypotheses":
                parsed_target = resolved_args.get("parsed_target")
                if not isinstance(parsed_target, dict):
                    parsed_target = {}
                result = executor(
                    resolved_args.get("question_text") or question_text,
                    options=resolved_args.get("options") or options or None,
                    parsed_target=parsed_target,
                )
            elif skill_id == "retrieve_evidence_for_hypothesis":
                result = executor(
                    graph,
                    hypothesis=_one_hypothesis(resolved_args.get("hypothesis")),
                    max_refs=int(resolved_args.get("max_refs") or 6),
                )
            elif skill_id == "score_hypothesis_support":
                counter = resolved_args.get("counterevidence") or []
                if isinstance(counter, dict):
                    counter = counter.get("counterevidence_refs") or counter.get("evidence_refs") or []
                result = executor(
                    _one_hypothesis(resolved_args.get("hypothesis")),
                    support_evidence=resolved_args.get("support_evidence") or [],
                    counterevidence=counter,
                )
            elif skill_id == "compare_hypotheses":
                scored = resolved_args.get("scored_hypotheses") or []
                if isinstance(scored, dict):
                    scored = [scored]
                scored = [item for item in scored if isinstance(item, dict)]
                result = executor(
                    scored,
                    decision_policy=resolved_args.get("decision_policy") if isinstance(resolved_args.get("decision_policy"), dict) else None,
                )
            elif skill_id == "bridge_evidence_hops":
                source = resolved_args.get("source_evidence") or []
                if isinstance(source, dict):
                    source = source.get("evidence_refs") or source.get("support_refs") or []
                result = executor(
                    graph,
                    source_evidence=source,
                    target_hypothesis=_one_hypothesis(resolved_args.get("target_hypothesis")),
                    allowed_hop_types=resolved_args.get("allowed_hop_types") if isinstance(resolved_args.get("allowed_hop_types"), list) else None,
                    max_hops=int(resolved_args.get("max_hops") or 2),
                )
            elif skill_id == "verify_temporal_social_consistency":
                result = executor(
                    resolved_args.get("evidence_chain") or {"evidence_refs": []},
                    hypothesis=_one_hypothesis(resolved_args.get("hypothesis")),
                    evidence_graph=graph,
                )
            elif skill_id == "compose_evidence_chain":
                labeled = resolved_args.get("role_labeled_evidence") or []
                labeled = [item for item in labeled if isinstance(item, dict)]
                result = executor(
                    labeled or [{"role": "supporting_evidence", "evidence_ref": "unknown", "text": "", "confidence": 0.0}],
                    dependency_template=resolved_args.get("dependency_template") or "support_chain",
                )
            elif skill_id == "detect_missing_role":
                result = executor(
                    resolved_args.get("evidence_chain") or {"items": [], "evidence_refs": []},
                    required_roles=resolved_args.get("required_roles") or [],
                )
            elif skill_id == "infer_causal_relation":
                result = executor(
                    resolved_args.get("candidate_cause") or "cause",
                    resolved_args.get("candidate_effect") or "effect",
                    evidence_chain=resolved_args.get("evidence_chain") or {"evidence_refs": []},
                )
            elif skill_id == "infer_intention_or_motive":
                result = executor(
                    resolved_args.get("agent") or "person",
                    resolved_args.get("actions") or ["action"],
                    context_evidence=resolved_args.get("context_evidence") or [],
                )
            elif skill_id == "infer_social_contradiction":
                result = executor(
                    resolved_args.get("claim_or_alibi") or {"claim_text": question_text},
                    evidence_chain=resolved_args.get("evidence_chain") or {"evidence_refs": []},
                    counterevidence=resolved_args.get("counterevidence") or [],
                )
            elif skill_id == "verify_claim_support":
                claim_arg = resolved_args.get("claim") or {"claim_text": question_text, "claim_status": "candidate"}
                if isinstance(claim_arg, str):
                    claim_arg = {"claim_text": claim_arg, "claim_status": "candidate"}
                support_policy = resolved_args.get("support_policy") or {"min_evidence_refs": 1}
                if isinstance(support_policy, str):
                    support_policy = {"min_evidence_refs": 1}
                result = executor(
                    claim_arg,
                    evidence_chain=resolved_args.get("evidence_chain") or {"evidence_refs": []},
                    support_policy=support_policy,
                )
            elif skill_id == "commit_answer":
                result = executor(
                    resolved_args.get("verified_claim") or {"claim_text": question_text, "claim_status": "verified"},
                    options=resolved_args.get("options") or options or None,
                    answer_format=resolved_args.get("answer_format") or ("multiple_choice" if options else "free_text"),
                    support_chain=resolved_args.get("support_chain") or {"evidence_refs": []},
                )
            else:
                result = executor(**resolved_args)
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            trace.append({
                "step_id": step_id,
                "skill_id": skill_id,
                "ok": False,
                "failure_code": "invalid_skill_args",
                "messages": [str(exc)],
            })
            continue

        trace.append({
            "step_id": step_id,
            "skill_id": skill_id,
            "ok": result.ok,
            "failure_code": result.failure_code,
            "evidence_refs": result.evidence_refs,
            "confidence": result.confidence,
        })
        if step_id:
            step_outputs[step_id] = {**result.outputs, "evidence_refs": result.evidence_refs}

    return trace, step_outputs


def build_llm_reasoning_rollout(
    example: dict[str, Any],
    clue_memory_graph: dict[str, Any],
    *,
    client: OpenRouterClient,
    skill_executor: Any | None = None,
) -> dict[str, Any]:
    """Full L2: plan with gpt-oss then execute reasoning skills. Falls back to deterministic.

    Args:
        skill_executor: Optional SkillExecutor for LLM-backed skill dispatch.
            If provided, skills configured for LLM mode will call the model API.
    """
    from .reasoning_rollout import build_reasoning_rollout

    question = example.get("question") or {}
    task_family = example.get("task_family") or ""

    try:
        plan_response = plan_reasoning_skills(
            question=question,
            clue_memory_graph=clue_memory_graph,
            task_family=task_family,
            client=client,
        )
        reasoning_plan = plan_response.get("reasoning_plan") or []
    except Exception as exc:
        plan_response = {"reasoning_plan": [], "notes": f"planner failed: {exc}", "planner_error": str(exc)}
        reasoning_plan = []

    if not reasoning_plan:
        rollout = build_reasoning_rollout(example, clue_memory_graph, rollout_source="deterministic_fallback_from_llm")
        rollout["metadata"]["llm_plan"] = plan_response
        return rollout

    trace, step_outputs = execute_reasoning_plan(
        reasoning_plan=reasoning_plan,
        clue_memory_graph=clue_memory_graph,
        question=question,
        skill_executor=skill_executor,
    )

    failed_steps = [t for t in trace if t.get("ok") is False]
    ok_steps = [t for t in trace if t.get("ok")]
    crash_steps = [t for t in failed_steps if t.get("failure_code") in ("unknown_skill_id", "invalid_skill_args")]

    # --- Fault localization + repair attempt ---
    repair_result = None
    if failed_steps and not crash_steps:
        from .fault_repair import attempt_repair
        repair_result = attempt_repair(
            trace, step_outputs, reasoning_plan, clue_memory_graph, question,
            skill_executor=skill_executor,
            max_repair_attempts=1,
        )
        if repair_result.get("attempted") and repair_result.get("repaired_count", 0) > 0:
            trace = trace + repair_result["repair_trace"]
            ok_steps = [t for t in trace if t.get("ok")]
            failed_steps = [t for t in trace if t.get("ok") is False]

    if crash_steps or (not ok_steps and len(failed_steps) > 3):
        rollout = build_reasoning_rollout(example, clue_memory_graph, rollout_source="deterministic_fallback_from_llm")
        rollout["metadata"]["llm_plan"] = plan_response
        rollout["metadata"]["llm_trace"] = trace
        rollout["metadata"]["fallback_reason"] = "too_many_failures"
        if repair_result:
            rollout["metadata"]["repair"] = repair_result
        return rollout

    rollout = make_reasoning_rollout_shell(example, clue_memory_graph, rollout_source="gpt_oss_reasoning_planner")
    rollout["rollout_id"] = f"skill_rollout:{example.get('example_id')}:llm_v1"

    executed_skills: list[str] = []
    prev: str | None = None
    for step_trace in trace:
        node_id = stable_id("skill", step_trace.get("skill_id"), len(rollout["nodes"]))
        rollout["nodes"].append({
            "node_id": node_id,
            "skill_id": step_trace.get("skill_id"),
            "step_id": step_trace.get("step_id"),
            "evidence_refs": step_trace.get("evidence_refs") or [],
            "status": "verified" if step_trace.get("ok") else "failed",
            "failure_code": step_trace.get("failure_code"),
            "confidence": step_trace.get("confidence", 0.0),
        })
        if prev:
            rollout["edges"].append({
                "edge_id": stable_id("edge", prev, node_id),
                "src": prev,
                "dst": node_id,
                "edge_type": "data",
            })
        prev = node_id
        if step_trace.get("skill_id"):
            executed_skills.append(step_trace["skill_id"])

    last_output = step_outputs.get(trace[-1].get("step_id", ""), {}) if trace else {}
    final_answer = last_output.get("final_answer")
    options = question.get("options") or []

    rollout["claims"] = [{
        "claim_id": stable_id("claim", str(final_answer)),
        "text": str(final_answer),
        "claim_status": "verified" if ok_steps else "insufficient",
        "supported_by_refs": last_output.get("evidence_refs") or [],
    }]
    rollout["final_answer"] = {
        "label": final_answer,
        "text": str(final_answer),
        "confidence": last_output.get("confidence", 0.0) if final_answer else 0.0,
    }
    rollout["acceptance_status"] = "accepted_weak" if final_answer else "rejected"
    rollout["failure_reasons"] = [] if final_answer else ["no_final_answer"]
    rollout["metadata"] = {
        "executed_skill_ids": list(dict.fromkeys(executed_skills)),
        "executed_skill_count": len(set(executed_skills)),
        "expected_reasoning_skill_count": len(REASONING_SKILL_IDS),
        "llm_plan": plan_response,
        "llm_trace_ok": len(ok_steps),
        "llm_trace_fail": len(failed_steps),
    }
    if repair_result:
        rollout["metadata"]["repair"] = repair_result
    return rollout

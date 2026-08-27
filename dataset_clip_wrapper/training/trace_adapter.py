#!/usr/bin/env python3
"""Adapt compact L1/L2 expert demos into controller-training traces.

This module is intentionally a schema bridge, not a reasoner. It preserves the
verified teacher trajectory produced by Qwen/GPT-OSS/verifier runs and maps it
into the canonical ``video_skills.contracts.ReasoningTrace`` shape plus a
compact chat-SFT view.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from video_skills.contracts import (
    AbstainDecision,
    AtomicStepResult,
    EvidenceBundle,
    EvidenceRef,
    HopGoal,
    HopRecord,
    QuestionAnalysis,
    ReasoningTrace,
    RetrievalQuery,
    VerificationCheck,
    VerificationResult,
)


ACCEPTED_STATUSES = {"accepted_strong", "accepted_bridge"}
TRAINING_SCHEMA_VERSION = "video-skills/l1l2-controller-training-v0.1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_text(demo: dict[str, Any]) -> str:
    question = ((demo.get("visible_demo_inputs") or {}).get("question") or {})
    for key in ("question_text", "text", "question", "prompt"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(question).strip()[:1000]


def _question_type(demo: dict[str, Any]) -> str:
    task_family = str(demo.get("task_family") or "").lower()
    if "order" in task_family or "temporal" in task_family:
        return "ordering"
    if "belief" in task_family or "social" in task_family:
        return "belief"
    if "cause" in task_family:
        return "causal"
    if "presence" in task_family:
        return "presence"
    if "state" in task_family:
        return "state"
    return "free"


def _compact_nodes(demo: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((demo.get("l1") or {}).get("compact_evidence_nodes") or []))


def _node_lookup(demo: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("ref")): node for node in _compact_nodes(demo) if node.get("ref")}


def _append_refs(refs: list[str], values: Any) -> None:
    if isinstance(values, str) and values:
        refs.append(values)
    elif isinstance(values, list):
        for value in values:
            _append_refs(refs, value)


def _collect_support_refs(demo: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    l2 = demo.get("l2") if isinstance(demo.get("l2"), dict) else {}
    status = l2.get("l2_status") if isinstance(l2.get("l2_status"), dict) else {}
    for key in ("support_refs", "answer_support_chain", "evidence_refs"):
        _append_refs(refs, status.get(key))

    trajectory = l2.get("trajectory") if isinstance(l2.get("trajectory"), dict) else {}
    for round_row in trajectory.get("rounds") or []:
        if not isinstance(round_row, dict):
            continue
        obs = round_row.get("observation_summary") if isinstance(round_row.get("observation_summary"), dict) else {}
        _append_refs(refs, obs.get("support_refs"))
        verifier = round_row.get("verifier_signal") if isinstance(round_row.get("verifier_signal"), dict) else {}
        pack = verifier.get("verified_evidence_pack") if isinstance(verifier.get("verified_evidence_pack"), dict) else {}
        _append_refs(refs, pack.get("support_refs"))

    repair = l2.get("repair_report") if isinstance(l2.get("repair_report"), dict) else {}
    selector = repair.get("option_evidence_selector") if isinstance(repair.get("option_evidence_selector"), dict) else {}
    for pack in selector.get("option_packs") or []:
        if isinstance(pack, dict):
            _append_refs(refs, pack.get("positive_refs"))

    return list(dict.fromkeys(refs))


def _evidence_ref(ref: str, lookup: dict[str, dict[str, Any]]) -> EvidenceRef:
    node = lookup.get(ref, {})
    time_span = node.get("time_span") if isinstance(node.get("time_span"), dict) else {}
    span: tuple[float, float] | None = None
    start = time_span.get("start_s") if isinstance(time_span, dict) else None
    end = time_span.get("end_s") if isinstance(time_span, dict) else None
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        span = (float(start), float(end))
    producer = str(node.get("producer") or "")
    inferred = "bridge" in producer or "repair" in producer or node.get("node_type") in {"l2_gap_diagnosis", "repair_plan"}
    return EvidenceRef(
        ref_id=ref,
        modality="memory_node",
        source_id=ref,
        time_span=span,
        provenance="inferred" if inferred else "observed",
        confidence=0.8 if inferred else 0.95,
        text=str(node.get("text") or "")[:600] or None,
        meta={
            "node_type": node.get("node_type"),
            "source_type": node.get("source_type"),
            "producer": node.get("producer"),
        },
    )


def _evidence_bundle(
    *,
    demo: dict[str, Any],
    refs: list[str],
    query_id: str,
    query_text: str,
    sufficiency: float,
) -> EvidenceBundle:
    lookup = _node_lookup(demo)
    return EvidenceBundle(
        bundle_id=f"bundle:{demo.get('demo_id')}:{query_id}",
        refs=[_evidence_ref(ref, lookup) for ref in refs],
        query=RetrievalQuery(
            query_id=query_id,
            text=query_text,
            k=max(1, len(refs)),
            mode="hybrid",
            meta={"source": "compact_l1_l2_expert_demo"},
        ),
        coverage={"support_ref_count": len(refs)},
        sufficiency_hint=sufficiency,
        confidence=sufficiency,
        inferred=False,
        meta={"graph_id": (demo.get("l1") or {}).get("graph_id")},
    )


def _round_refs(round_row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    obs = round_row.get("observation_summary") if isinstance(round_row.get("observation_summary"), dict) else {}
    _append_refs(refs, obs.get("support_refs"))
    verifier = round_row.get("verifier_signal") if isinstance(round_row.get("verifier_signal"), dict) else {}
    pack = verifier.get("verified_evidence_pack") if isinstance(verifier.get("verified_evidence_pack"), dict) else {}
    _append_refs(refs, pack.get("support_refs"))
    return list(dict.fromkeys(refs))


def _hop_from_round(demo: dict[str, Any], round_row: dict[str, Any], index: int) -> HopRecord:
    question_id = str(demo.get("example_id") or demo.get("demo_id"))
    round_type = str(round_row.get("round_type") or round_row.get("stage") or f"round_{index}")
    goal_text = str(round_row.get("goal") or round_row.get("repair_goal") or f"Resolve L2 evidence round {index}: {round_type}")
    refs = _round_refs(round_row)
    accepted = str((demo.get("l2") or {}).get("final_acceptance_status") or "") in ACCEPTED_STATUSES
    bundle = _evidence_bundle(
        demo=demo,
        refs=refs,
        query_id=f"q:{question_id}:hop:{index}",
        query_text=goal_text,
        sufficiency=0.9 if refs else 0.0,
    )
    passed = bool(refs) or accepted
    next_action = "continue" if passed else "abstain"
    verification = VerificationResult(
        passed=passed,
        checks=[
            VerificationCheck(
                name="teacher_verified_evidence_refs",
                passed=bool(refs),
                evidence_refs=refs,
                score=1.0 if refs else 0.0,
                notes="Mapped from verified L2/repair trajectory; no new reasoning performed.",
            )
        ],
        score=1.0 if passed else 0.0,
        reasons=[] if passed else ["no verified support refs in this round"],
        next_action=next_action,
        meta={"source_round": round_row},
    )
    step = AtomicStepResult(
        step_id=f"step:{question_id}:{index}:l2_controller",
        hop_id=f"hop:{question_id}:{index}",
        skill_id=f"l2_controller.{round_type}",
        inputs={"question": _question_text(demo), "round_type": round_type},
        output={
            "round_type": round_type,
            "support_refs": refs,
            "summary": round_row.get("observation_summary") or round_row.get("reason_short") or "",
        },
        output_type="claim" if refs else "abstain",
        evidence=bundle,
        verification=verification,
        confidence=0.9 if passed else 0.0,
        inferred=True,
        failure_mode=None if passed else "empty_evidence",
        meta={"teacher_round_index": index},
    )
    return HopRecord(
        hop_goal=HopGoal(
            hop_id=f"hop:{question_id}:{index}",
            parent_question_id=question_id,
            goal_text=goal_text,
            target_claim_type=_question_type(demo),
            retrieval_hints=[bundle.query],
            success_predicate="verified_support_refs_non_empty_or_abstain",
            meta={"round_type": round_type},
        ),
        steps=[step],
        hop_verification=verification,
        outcome="resolved" if passed else "abstain",
        cost={"atomic_steps": 1, "retrieval_calls": 1 if refs else 0, "broaden_levels": 0, "latency_ms": 0},
        meta={"source": "l1_l2_expert_demo_round"},
    )


def demo_to_reasoning_trace(demo: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical ``ReasoningTrace`` dictionary for one expert demo."""
    question_id = str(demo.get("example_id") or demo.get("demo_id"))
    l2 = demo.get("l2") if isinstance(demo.get("l2"), dict) else {}
    status = str(l2.get("final_acceptance_status") or "")
    accepted = status in ACCEPTED_STATUSES
    trajectory = l2.get("trajectory") if isinstance(l2.get("trajectory"), dict) else {}
    rounds = [row for row in trajectory.get("rounds") or [] if isinstance(row, dict)]
    support_refs = _collect_support_refs(demo)
    final_bundle = _evidence_bundle(
        demo=demo,
        refs=support_refs,
        query_id=f"q:{question_id}:final",
        query_text=_question_text(demo),
        sufficiency=0.95 if accepted and support_refs else 0.0,
    )
    final_check = VerificationCheck(
        name="final_acceptance_from_strict_verifier",
        passed=accepted and bool(support_refs or status == "accepted_bridge"),
        evidence_refs=support_refs,
        score=1.0 if accepted else 0.0,
        notes=str(l2.get("verifier_reason") or status),
    )
    final_verification = VerificationResult(
        passed=accepted,
        checks=[final_check],
        score=1.0 if accepted else 0.0,
        reasons=[] if accepted else [str(l2.get("verifier_reason") or "needs_more_evidence")],
        next_action="continue" if accepted else "abstain",
        meta={"final_acceptance_status": status, "final_repair_applied": bool(l2.get("final_repair_applied"))},
    )
    trace = ReasoningTrace(
        trace_id=f"trace:{demo.get('demo_id')}",
        question_id=question_id,
        question_analysis=QuestionAnalysis(
            question_id=question_id,
            question_text=_question_text(demo),
            question_type=_question_type(demo),
            expected_answer_type="multiple_choice" if ((demo.get("visible_demo_inputs") or {}).get("question") or {}).get("options") else "free_text",
            decomposition=[str(row.get("round_type") or row.get("stage") or f"round_{i}") for i, row in enumerate(rounds)],
            meta={
                "dataset": demo.get("dataset"),
                "video_regime": demo.get("video_regime"),
                "task_family": demo.get("task_family"),
            },
        ),
        hops=[_hop_from_round(demo, row, i) for i, row in enumerate(rounds)],
        final_claim=(l2.get("l2_status") or {}).get("final_answer") if accepted else None,
        final_evidence=final_bundle,
        final_verification=final_verification,
        abstain=None
        if accepted
        else AbstainDecision(
            abstain=True,
            reason="insufficient_evidence",
            blocking_checks=[status or "needs_more_evidence"],
            last_evidence=final_bundle,
            confidence_ceiling=0.0,
            meta={"verifier_reason": l2.get("verifier_reason")},
        ),
        answer=json.dumps((l2.get("l2_status") or {}).get("final_answer"), ensure_ascii=False) if accepted else None,
        cost={
            "hops": len(rounds),
            "atomic_steps": len(rounds),
            "retrieval_calls": sum(1 for row in rounds if _round_refs(row)),
            "tokens": 0,
            "latency_ms": 0,
            "large_model_calls": 0,
        },
        meta={
            "schema_version": TRAINING_SCHEMA_VERSION,
            "demo_id": demo.get("demo_id"),
            "demo_type": demo.get("demo_type"),
            "quality_flags": demo.get("quality_flags"),
            "l1": demo.get("l1"),
            "teacher_l2": l2,
        },
    )
    return asdict(trace)


def _sft_prompt(demo: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Generate an L2 reasoning/repair/verification trace from video-only L1 evidence.",
        "visible_demo_inputs": demo.get("visible_demo_inputs"),
        "l1": demo.get("l1"),
        "allowed_actions": [
            "select_verified_evidence",
            "propose_missing_evidence_repair",
            "add_commonsense_bridge_with_flag",
            "verify_support_chain",
            "commit_answer",
            "abstain_needs_more_evidence",
        ],
        "constraints": [
            "Do not use non-visible labels or gold answer fields.",
            "Cite only L1/repair evidence refs visible in the prompt.",
            "Common sense may bridge reasoning gaps but must not be labeled as visual evidence.",
            "Commit only when verifier evidence is sufficient; otherwise abstain.",
        ],
    }


def _sft_target(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": trace.get("trace_id"),
        "question_analysis": trace.get("question_analysis"),
        "hops": trace.get("hops"),
        "final_claim": trace.get("final_claim"),
        "final_evidence": trace.get("final_evidence"),
        "final_verification": trace.get("final_verification"),
        "abstain": trace.get("abstain"),
        "answer": trace.get("answer"),
    }


def demo_to_sft_chat(demo: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "demo_id": demo.get("demo_id"),
        "dataset": demo.get("dataset"),
        "messages": [
            {
                "role": "system",
                "content": "You are the Video_Skills L2 controller. Produce verifier-aware JSON traces grounded in provided L1 evidence.",
            },
            {
                "role": "user",
                "content": json.dumps(_sft_prompt(demo), ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": json.dumps(_sft_target(trace), ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "metadata": {
            "demo_type": demo.get("demo_type"),
            "quality_flags": demo.get("quality_flags"),
            "final_acceptance_status": (demo.get("l2") or {}).get("final_acceptance_status"),
        },
    }


def build_training_exports(demos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    chats: list[dict[str, Any]] = []
    for demo in demos:
        flags = demo.get("quality_flags") if isinstance(demo.get("quality_flags"), dict) else {}
        if not (flags.get("training_candidate") or flags.get("abstain_candidate")):
            continue
        trace = demo_to_reasoning_trace(demo)
        traces.append(trace)
        chats.append(demo_to_sft_chat(demo, trace))
    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "input_demos": len(demos),
        "exported_traces": len(traces),
        "exported_sft_chats": len(chats),
        "accepted_traces": sum(1 for trace in traces if trace.get("final_verification", {}).get("passed")),
        "abstain_traces": sum(1 for trace in traces if trace.get("abstain")),
        "support_ref_total": sum(len((((trace.get("final_evidence") or {}).get("refs")) or [])) for trace in traces),
        "gold_visible_policy": "visible_demo_inputs_only; hidden_supervision is never placed in SFT prompt",
    }
    return traces, chats, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export compact expert demos as ReasoningTrace and SFT chat JSONL.")
    parser.add_argument("--expert-demos", type=Path, required=True)
    parser.add_argument("--trace-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    demos = _read_jsonl(args.expert_demos)
    traces, chats, summary = build_training_exports(demos)
    _write_jsonl(args.trace_output_jsonl, traces)
    _write_jsonl(args.sft_output_jsonl, chats)
    _write_json(args.quality_report_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

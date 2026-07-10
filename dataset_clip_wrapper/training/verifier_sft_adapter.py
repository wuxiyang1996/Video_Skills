#!/usr/bin/env python3
"""Export option-level verifier decisions as MDP-style judge SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_json, read_jsonl, write_json, write_jsonl


def _source_graph_nodes(plan_path: Path, example_id: str) -> list[dict[str, Any]]:
    if not plan_path.exists():
        return []
    plan = read_json(plan_path)
    source_value = plan.get("source_path")
    if not source_value:
        return []
    source_path = Path(str(source_value))
    if not source_path.exists():
        return []
    for example in read_jsonl(source_path):
        if str(example.get("example_id")) != example_id:
            continue
        metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
        graph = metadata.get("clue_memory_graph") if isinstance(metadata.get("clue_memory_graph"), dict) else {}
        return [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    return []


def _evidence_catalog(
    patch: dict[str, Any],
    refs: list[str],
    source_nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    wanted = set(refs)
    rows = []
    for node in list(source_nodes or []) + list(patch.get("nodes", [])):
        if not isinstance(node, dict) or str(node.get("node_id")) not in wanted:
            continue
        rows.append({key: node[key] for key in ("node_id", "node_type", "text", "modality", "confidence", "clip_id", "time_span") if key in node})
    return rows


def _expert_evidence_catalog(demo: dict[str, Any], refs: list[str]) -> list[dict[str, Any]]:
    l1 = demo.get("l1") if isinstance(demo.get("l1"), dict) else {}
    wanted = set(refs)
    rows = []
    for node in l1.get("compact_evidence_nodes", []):
        if not isinstance(node, dict) or str(node.get("ref")) not in wanted:
            continue
        rows.append({key: node[key] for key in ("ref", "role", "node_type", "source_type", "time_span", "text") if key in node})
    return rows


def _append_expert_demo_transitions(transitions: list[dict[str, Any]], expert_demos_path: Path) -> None:
    # Weak acceptance is useful as a repair/abstention target, not as positive
    # cold-start verifier supervision. Otherwise a merely non-empty support list
    # teaches the learned verifier to rubber-stamp unsupported commits.
    positive_statuses = {"accepted_strong", "resolved_strong"}
    for demo in read_jsonl(expert_demos_path):
        l2 = demo.get("l2") if isinstance(demo.get("l2"), dict) else {}
        trajectory = l2.get("trajectory") if isinstance(l2.get("trajectory"), dict) else {}
        rounds = trajectory.get("rounds") if isinstance(trajectory.get("rounds"), list) else []
        for index, round_row in enumerate(rounds):
            if not isinstance(round_row, dict):
                continue
            signal = round_row.get("verifier_signal") if isinstance(round_row.get("verifier_signal"), dict) else {}
            pack = signal.get("verified_evidence_pack") if isinstance(signal.get("verified_evidence_pack"), dict) else {}
            claim_text = pack.get("claim_text")
            if not claim_text:
                continue
            status = str(signal.get("status") or "needs_more_evidence")
            decision = "supported" if status in positive_statuses else "insufficient"
            failure_code = None if decision == "supported" else (
                "weak_evidence_not_positive" if status == "accepted_weak" else status
            )
            refs = [str(value) for value in pack.get("support_refs", [])]
            state = {
                "schema_version": "video-skills/verifier-state-v0.1",
                "process_model": "mdp_style_auxiliary_verifier",
                "dataset": demo.get("dataset"),
                "example_id": demo.get("example_id"),
                "candidate_claim": {"claim_text": claim_text, "candidate_label": pack.get("final_label")},
                "proposed_evidence_pack": {
                    "support_refs": refs,
                    "counter_refs": [],
                    "evidence_catalog": _expert_evidence_catalog(demo, refs),
                },
                "verification_policy": {
                    "min_support_refs": pack.get("min_support_refs"),
                    "strong_min_refs": pack.get("strong_min_refs"),
                },
            }
            action = {
                "schema_version": "video-skills/verifier-action-v0.1",
                "tool_name": "emit_verifier_decision",
                "arguments": {
                    "decision": decision,
                    "failure_code": failure_code,
                    "confidence": None,
                    "support_score": None,
                    "target_alignment_score": None,
                    "missing_requirements": compact_visibility(pack.get("missing_requirements") or []),
                    "reason_short": pack.get("verifier_reason") or signal.get("reason"),
                },
            }
            transitions.append({
                "schema_version": "video-skills/verifier-transition-v0.1",
                "transition_id": f"{demo.get('demo_id')}::expert_verify::{index}",
                "controller": "auxiliary_verifier",
                "state_t": state,
                "action_t": action,
                "observation_t": {"runtime_gate_decision": decision, "runtime_acceptance_status": status},
                "state_t_plus_1_summary": {"candidate_status": decision},
                "reward_proxy_t": {"matches_verified_expert_demo": 1.0},
                "done": True,
                "source_expert_demos": str(expert_demos_path),
            })


def _balance_by_decision(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in transitions:
        decision = str(row["action_t"]["arguments"].get("decision") or "unknown")
        grouped.setdefault(decision, []).append(row)
    if len(grouped) < 2:
        return transitions
    quota = min(len(rows) for rows in grouped.values())
    selected: list[dict[str, Any]] = []
    for decision in sorted(grouped):
        rows = grouped[decision]
        failure_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            failure = str(row["action_t"]["arguments"].get("failure_code") or "none")
            failure_groups.setdefault(failure, []).append(row)
        group_names = sorted(failure_groups)
        offsets = {name: 0 for name in group_names}
        while sum(offsets.values()) < quota:
            made_progress = False
            for name in group_names:
                if sum(offsets.values()) >= quota:
                    break
                offset = offsets[name]
                if offset < len(failure_groups[name]):
                    selected.append(failure_groups[name][offset])
                    offsets[name] += 1
                    made_progress = True
            if not made_progress:
                break
    return sorted(selected, key=lambda row: str(row.get("transition_id")))


def build_verifier_exports(
    stage_root: Path | None,
    expert_demos_path: Path | None = None,
    *,
    balance_decisions: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    skipped_unresolved_refs = 0
    verifier_paths = sorted(stage_root.glob("*/repair_04_l2_verifier.json")) if stage_root else []
    for verifier_path in verifier_paths:
        verifier = read_json(verifier_path)
        patch_path = verifier_path.parent / "repair_03_l1_patch.json"
        patch = read_json(patch_path) if patch_path.exists() else {"nodes": []}
        source_nodes = _source_graph_nodes(
            verifier_path.parent / "repair_01_plan.json",
            str(verifier.get("example_id") or ""),
        )
        policy = verifier.get("option_verifier_policy") if isinstance(verifier.get("option_verifier_policy"), dict) else {}
        for index, option in enumerate(verifier.get("option_verifications", [])):
            if not isinstance(option, dict):
                continue
            support_refs = [str(value) for value in option.get("positive_refs", [])]
            counter_refs = [str(value) for value in option.get("negative_refs", [])]
            all_refs = list(dict.fromkeys(support_refs + counter_refs))
            evidence_catalog = _evidence_catalog(patch, all_refs, source_nodes)
            resolved_refs = {
                str(row.get("node_id") or row.get("ref"))
                for row in evidence_catalog
                if row.get("node_id") or row.get("ref")
            }
            if any(ref not in resolved_refs for ref in all_refs):
                skipped_unresolved_refs += 1
                continue
            state = {
                "schema_version": "video-skills/verifier-state-v0.1",
                "process_model": "mdp_style_auxiliary_verifier",
                "dataset": verifier.get("dataset"),
                "example_id": verifier.get("example_id"),
                "candidate_claim": {
                    "option_label": option.get("option_label"),
                    "option_text": option.get("option_text"),
                },
                "proposed_evidence_pack": {
                    "support_refs": support_refs,
                    "counter_refs": counter_refs,
                    "evidence_catalog": evidence_catalog,
                },
                "verification_policy": {key: policy[key] for key in ("min_verify_refs", "max_verify_refs", "min_verify_confidence") if key in policy},
            }
            action = {
                "schema_version": "video-skills/verifier-action-v0.1",
                "tool_name": "emit_verifier_decision",
                "arguments": {
                    "decision": option.get("verifier_decision") or ("supported" if option.get("ok") else "insufficient"),
                    "failure_code": option.get("failure_code"),
                    "confidence": option.get("confidence"),
                    "support_score": option.get("support_score"),
                    "target_alignment_score": option.get("target_alignment_score"),
                    "missing_requirements": compact_visibility(option.get("missing_requirements") or []),
                    "reason_short": option.get("reason_short"),
                },
            }
            decision = str(action["arguments"]["decision"] or "unknown")
            failure = str(action["arguments"].get("failure_code") or "none")
            transitions.append({
                "schema_version": "video-skills/verifier-transition-v0.1",
                "transition_id": f"{verifier.get('example_id')}::verify::{index}",
                "controller": "auxiliary_verifier",
                "state_t": state,
                "action_t": action,
                "observation_t": {"runtime_gate_decision": decision, "runtime_gate_ok": bool(option.get("ok"))},
                "state_t_plus_1_summary": {"candidate_status": decision},
                "reward_proxy_t": {"matches_runtime_verifier": 1.0},
                "done": True,
                "source_stage_dir": str(verifier_path.parent),
            })
    if expert_demos_path:
        _append_expert_demo_transitions(transitions, expert_demos_path)
    pre_balance_counts = Counter(
        str(row["action_t"]["arguments"].get("decision") or "unknown") for row in transitions
    )
    if balance_decisions:
        transitions = _balance_by_decision(transitions)
    for row in transitions:
        arguments = row["action_t"]["arguments"]
        decision_counts[str(arguments.get("decision") or "unknown")] += 1
        failure_counts[str(arguments.get("failure_code") or "none")] += 1
    chats: list[dict[str, Any]] = []
    forbidden_hits = 0
    for row in transitions:
        user_payload = {"task": "judge_candidate_support", "state_t": row["state_t"]}
        forbidden_hits += int(contains_forbidden_prompt_key(user_payload))
        chats.append({
            "schema_version": "video-skills/verifier-sft-chat-v0.1",
            "transition_id": row["transition_id"],
            "messages": [
                {"role": "system", "content": "You are an auxiliary Video_Skills evidence verifier. Judge only whether the candidate claim is supported by the supplied visible evidence. Return the verifier tool action as JSON. The deterministic runtime verifier remains the final gate."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(row["action_t"], ensure_ascii=False, separators=(",", ":"))},
            ],
            "metadata": {"controller": "auxiliary_verifier", "decision": row["action_t"]["arguments"]["decision"]},
        })
    report = {
        "schema_version": "video-skills/verifier-sft-report-v0.1",
        "source_stage_root": str(stage_root) if stage_root else "",
        "source_expert_demos": str(expert_demos_path) if expert_demos_path else "",
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "decision_counts": dict(decision_counts),
        "pre_balance_decision_counts": dict(pre_balance_counts),
        "balanced_decisions": balance_decisions,
        "failure_code_counts": dict(failure_counts),
        "prompt_forbidden_key_hits": forbidden_hits,
        "skipped_unresolved_evidence_refs": skipped_unresolved_refs,
        "intended_use": "cold-start learned verifier assistant; deterministic runtime verifier remains authoritative",
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--expert-demos", type=Path)
    parser.add_argument("--balance-decisions", action="store_true")
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.stage_root and not args.expert_demos:
        parser.error("at least one of --stage-root or --expert-demos is required")
    transitions, chats, report = build_verifier_exports(
        args.stage_root,
        args.expert_demos,
        balance_decisions=args.balance_decisions,
    )
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

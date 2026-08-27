#!/usr/bin/env python3
"""Export motif lifecycle decisions as MDP-style curation SFT."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_jsonl, write_json, write_jsonl


def _gated_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Apply the non-negotiable evidence gates before imitating a bank label."""
    source_status = str(row.get("status") or "shadow")
    refs = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), list) else []
    failures: list[str] = []
    if not refs:
        failures.append("no_evidence_refs")
    if not refs or not all(isinstance(ref, dict) and ref.get("verifier_passed") is True for ref in refs):
        failures.append("verifier_not_passed")
    if not refs or not all(isinstance(ref, dict) and ref.get("evidence_valid") is True for ref in refs):
        failures.append("evidence_not_valid")
    if not refs or not all(isinstance(ref, dict) and ref.get("no_hidden_leakage") is True for ref in refs):
        failures.append("hidden_leakage_or_unknown")
    if failures:
        return "rejected", failures

    support = row.get("support") if isinstance(row.get("support"), dict) else {}
    support_count = int(support.get("support_count") or len(refs))
    if support_count < 2:
        return "shadow", ["insufficient_support_for_candidate"]
    return source_status, []


def _candidate_state(row: dict[str, Any]) -> dict[str, Any]:
    support = row.get("support") if isinstance(row.get("support"), dict) else {}
    evidence_refs = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), list) else []
    return {
        "schema_version": "video-skills/motif-curation-state-v0.1",
        "process_model": "mdp_style_motif_lifecycle_controller",
        "candidate": {
            "motif_id": row.get("motif_id"),
            "name": row.get("name"),
            "description": row.get("description"),
            "motif_type": row.get("motif_type"),
            "trigger_signature": compact_visibility(row.get("trigger_signature") or {}),
            "l1_template": compact_visibility(row.get("l1_template") or {}),
            "l2_template": compact_visibility(row.get("l2_template") or {}),
            "expansion_constraints": compact_visibility(row.get("expansion_constraints") or []),
            "false_binding_patterns": compact_visibility(row.get("false_binding_patterns") or []),
        },
        "support_signals": {
            "support_count": support.get("support_count", len(evidence_refs)),
            "verified_task_families": support.get("verified_task_families") or [],
            "empirical_confidence": support.get("empirical_confidence"),
            "evidence_ref_count": len(evidence_refs),
            "all_verifier_passed": bool(evidence_refs) and all(bool(ref.get("verifier_passed")) for ref in evidence_refs if isinstance(ref, dict)),
            "all_evidence_valid": bool(evidence_refs) and all(bool(ref.get("evidence_valid")) for ref in evidence_refs if isinstance(ref, dict)),
            "all_no_hidden_leakage": bool(evidence_refs) and all(bool(ref.get("no_hidden_leakage")) for ref in evidence_refs if isinstance(ref, dict)),
        },
        "lifecycle_rule": {
            "candidate_and_shadow_are_non_executable": True,
            "must_expand_before_execution": True,
            "must_pass_l1_l2_verification_after_expansion": True,
        },
    }


def build_motif_exports(bank_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(bank_path)
    transitions: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    gate_failure_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        source_status = str(row.get("status") or "shadow")
        status, gate_failures = _gated_status(row)
        motif_type = str(row.get("motif_type") or "unknown")
        state = _candidate_state(row)
        action = {
            "schema_version": "video-skills/motif-curation-action-v0.1",
            "tool_name": "set_motif_lifecycle_status",
            "arguments": {
                "motif_id": row.get("motif_id"),
                "status": status,
                "require_expansion_before_use": True,
                "required_verifiers": ["l1_evidence_verifier", "l2_claim_support_verifier"],
                "do_not_answer_directly": True,
            },
        }
        status_counts[status] += 1
        source_status_counts[source_status] += 1
        gate_failure_counts.update(gate_failures)
        type_counts[motif_type] += 1
        transitions.append({
            "schema_version": "video-skills/motif-curation-transition-v0.1",
            "transition_id": f"{row.get('motif_id')}::curate::{index}",
            "controller": "motif_lifecycle",
            "state_t": state,
            "action_t": action,
            "observation_t": {
                "source_lifecycle_status": source_status,
                "gated_lifecycle_status": status,
                "gate_failures": gate_failures,
                "executable": False,
            },
            "state_t_plus_1_summary": {"registered_status": status},
            "reward_proxy_t": {"matches_mined_bank": 1.0, "boundary_preserved": 1.0},
            "done": True,
            "source_bank": str(bank_path),
        })
    chats: list[dict[str, Any]] = []
    forbidden_hits = 0
    for row in transitions:
        user_payload = {"task": "choose_motif_lifecycle_action", "state_t": row["state_t"]}
        forbidden_hits += int(contains_forbidden_prompt_key(user_payload))
        chats.append({
            "schema_version": "video-skills/motif-curation-sft-chat-v0.1",
            "transition_id": row["transition_id"],
            "messages": [
                {"role": "system", "content": "You are the Video_Skills motif lifecycle controller. Curate a reusable graph prior from support signals. Motifs are non-executable priors: they must expand into ordinary L1/L2 nodes and pass verification. Return JSON only."},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(row["action_t"], ensure_ascii=False, separators=(",", ":"))},
            ],
            "metadata": {"controller": "motif_lifecycle", "status": row["action_t"]["arguments"]["status"]},
        })
    report = {
        "schema_version": "video-skills/motif-curation-sft-report-v0.1",
        "source_bank": str(bank_path),
        "input_candidates": len(rows),
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "status_counts": dict(status_counts),
        "source_status_counts": dict(source_status_counts),
        "gate_failure_counts": dict(gate_failure_counts),
        "motif_type_counts": dict(type_counts),
        "prompt_forbidden_key_hits": forbidden_hits,
        "granularity": "one motif lifecycle action per mined candidate",
        "known_bias": "no positive promotion without transfer tests; invalid, unverifiable, or leakage-marked support is exported as rejected",
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-bank", type=Path, required=True)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_motif_exports(args.motif_bank)
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

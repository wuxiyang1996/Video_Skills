#!/usr/bin/env python3
"""Export evidence-level motif audit SFT from existing mined artifacts.

Outcome flags are targets only.  Prompts contain the motif hypothesis,
provenance, and node payloads resolved from the existing rollout graph; gold
answers and the audit flags themselves are never exposed to the student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_common import compact_visibility, contains_forbidden_prompt_key, read_jsonl, write_json, write_jsonl


OUTCOME_KEYS = {"final_answer_correct", "verifier_passed", "evidence_valid", "no_hidden_leakage"}


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: compact_visibility(node[key])
        for key in (
            "node_id", "node_type", "skill_id", "status", "failure_code",
            "confidence", "modality", "clip_id", "time_span", "visibility",
        )
        if key in node
    }
    if node.get("text"):
        result["text"] = str(node["text"])[:400]
    refs = node.get("evidence_refs") if isinstance(node.get("evidence_refs"), list) else []
    if refs:
        result["evidence_refs"] = [str(value) for value in refs[:12]]
        result["evidence_ref_count"] = len(refs)
    return result


def _node_index(rollout_paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for path in rollout_paths:
        for row in read_jsonl(path):
            example_id = str(row.get("example_id") or "")
            if not example_id:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            graph = metadata.get("clue_memory_graph") if isinstance(metadata.get("clue_memory_graph"), dict) else {}
            rollout = metadata.get("reasoning_rollout") if isinstance(metadata.get("reasoning_rollout"), dict) else {}
            nodes: dict[str, dict[str, Any]] = {}
            for node in list(graph.get("nodes") or []) + list(rollout.get("nodes") or []):
                if isinstance(node, dict) and node.get("node_id"):
                    nodes[str(node["node_id"])] = _compact_node(node)
            result[example_id] = nodes
    return result


def _failures(ref: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if ref.get("verifier_passed") is not True:
        failures.append("verifier_not_passed")
    if ref.get("evidence_valid") is not True:
        failures.append("evidence_not_valid")
    if ref.get("no_hidden_leakage") is not True:
        failures.append("hidden_leakage_or_unknown")
    return failures


def _select_refs(refs: list[dict[str, Any]], motif_id: str, cap: int) -> list[dict[str, Any]]:
    ordered = sorted(
        refs,
        key=lambda ref: hashlib.sha256(
            f"{motif_id}:{ref.get('example_id')}:{ref.get('l1_node_ids')}:{ref.get('l2_node_ids')}".encode()
        ).hexdigest(),
    )
    accepted = [ref for ref in ordered if not _failures(ref)]
    rejected = [ref for ref in ordered if _failures(ref)]
    selected: list[dict[str, Any]] = []
    half = max(1, cap // 2)
    selected.extend(accepted[:half])
    selected.extend(rejected[:half])
    selected_ids = {id(ref) for ref in selected}
    selected.extend(ref for ref in ordered if id(ref) not in selected_ids)
    return selected[:cap]


def build_motif_evidence_exports(
    motif_bank_path: Path,
    rollout_paths: list[Path],
    *,
    max_refs_per_motif: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    motifs = read_jsonl(motif_bank_path)
    nodes_by_example = _node_index(rollout_paths)
    transitions: list[dict[str, Any]] = []
    skipped = Counter()
    verdict_counts = Counter()
    failure_counts = Counter()
    for motif in motifs:
        motif_id = str(motif.get("motif_id") or "")
        refs = [ref for ref in motif.get("evidence_refs") or [] if isinstance(ref, dict)]
        for ordinal, ref in enumerate(_select_refs(refs, motif_id, max_refs_per_motif)):
            example_id = str(ref.get("example_id") or "")
            node_ids = [str(value) for value in list(ref.get("l1_node_ids") or []) + list(ref.get("l2_node_ids") or [])]
            node_index = nodes_by_example.get(example_id, {})
            resolved_nodes = [node_index[node_id] for node_id in node_ids if node_id in node_index]
            if not resolved_nodes:
                skipped["no_resolved_visible_nodes"] += 1
                continue
            failures = _failures(ref)
            verdict = "accept_ref" if not failures else "reject_ref"
            visible_ref = {
                key: compact_visibility(value)
                for key, value in ref.items()
                if key not in OUTCOME_KEYS and key not in {"source_path", "l1_node_ids", "l2_node_ids"}
            }
            visible_ref["l1_node_ids"] = [str(value) for value in (ref.get("l1_node_ids") or [])[:12]]
            visible_ref["l2_node_ids"] = [str(value) for value in (ref.get("l2_node_ids") or [])[:12]]
            visible_ref["l1_node_count"] = len(ref.get("l1_node_ids") or [])
            visible_ref["l2_node_count"] = len(ref.get("l2_node_ids") or [])
            state = {
                "schema_version": "video-skills/motif-evidence-audit-state-v0.1",
                "process_model": "motif_evidence_ref_auditor",
                "dataset": ref.get("dataset"),
                "example_id": example_id,
                "motif_candidate": compact_visibility({
                    "motif_id": motif_id,
                    "name": motif.get("name"),
                    "description": motif.get("description"),
                    "motif_type": motif.get("motif_type"),
                    "trigger_signature": motif.get("trigger_signature") or {},
                    "l1_template": motif.get("l1_template") or {},
                    "l2_template": motif.get("l2_template") or {},
                }),
                "evidence_ref": visible_ref,
                "resolved_visible_nodes": resolved_nodes[:12],
                "resolved_visible_node_count": len(resolved_nodes),
                "audit_observations": {
                    "runtime_verifier_passed": ref.get("verifier_passed") is True,
                    "evidence_validation_passed": ref.get("evidence_valid") is True,
                    "leakage_scan_passed": ref.get("no_hidden_leakage") is True,
                },
                "audit_policy": {
                    "require_verifier_pass": True,
                    "require_valid_evidence": True,
                    "require_no_hidden_leakage": True,
                },
            }
            action = {
                "schema_version": "video-skills/motif-evidence-audit-action-v0.1",
                "tool_name": "set_motif_evidence_ref_audit",
                "arguments": {
                    "motif_id": motif_id,
                    "example_id": example_id,
                    "verdict": verdict,
                    "failure_codes": failures,
                },
            }
            transition_id = f"{motif_id}::evidence_audit::{example_id}:{ordinal}"
            transitions.append({
                "schema_version": "video-skills/motif-evidence-audit-transition-v0.1",
                "transition_id": transition_id,
                "controller": "motif_evidence_auditor",
                "state_t": state,
                "action_t": action,
                "done": True,
            })
            verdict_counts[verdict] += 1
            failure_counts.update(failures)

    transitions.sort(key=lambda row: str(row["transition_id"]))
    chats: list[dict[str, Any]] = []
    forbidden_hits = 0
    for row in transitions:
        payload = {"task": "audit_motif_evidence_ref", "state_t": row["state_t"]}
        forbidden_hits += int(contains_forbidden_prompt_key(payload))
        chats.append({
            "schema_version": "video-skills/motif-evidence-audit-sft-chat-v0.1",
            "transition_id": row["transition_id"],
            "messages": [
                {"role": "system", "content": "You are the Video_Skills motif evidence gate controller. Apply the visible runtime verifier, evidence-validation, and leakage-scan observations to decide whether one evidence reference may support a reusable, non-executable motif prior. Reject if any required gate failed. Return JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(row["action_t"], ensure_ascii=False, separators=(",", ":"))},
            ],
            "metadata": {
                "controller": "motif_evidence_auditor",
                "verdict": row["action_t"]["arguments"]["verdict"],
                "dataset": row["state_t"].get("dataset"),
            },
        })
    report = {
        "schema_version": "video-skills/motif-evidence-audit-sft-report-v0.1",
        "input_motifs": len(motifs),
        "max_refs_per_motif": max_refs_per_motif,
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(chats),
        "verdict_counts": dict(verdict_counts),
        "failure_counts": dict(failure_counts),
        "skipped_counts": dict(skipped),
        "prompt_forbidden_key_hits": forbidden_hits,
    }
    return transitions, chats, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-bank", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--max-refs-per-motif", type=int, default=4)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    transitions, chats, report = build_motif_evidence_exports(
        args.motif_bank, args.rollout_jsonl, max_refs_per_motif=args.max_refs_per_motif
    )
    write_jsonl(args.transition_output_jsonl, transitions)
    write_jsonl(args.sft_output_jsonl, chats)
    write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

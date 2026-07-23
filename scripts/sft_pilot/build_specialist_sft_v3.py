#!/usr/bin/env python3
"""Build five independent, leakage-aware specialist LoRA datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.training.sft_common import (
    contains_forbidden_prompt_key,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


EVAL_ONLY = {"ovo_bench", "videomme", "vrbench"}
KNOWN_DATASETS = EVAL_ONLY | {"cg_bench", "video_holmes", "siv_bench"}
CLIP_ID_PATTERN = re.compile(r"^clip:[A-Za-z0-9_-]+:(?:fine|coarse|whole):\d{4}$")


def _stable(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def _user_state(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("state_t"), dict):
            return payload["state_t"]
    return {}


def _example_id(row: dict[str, Any]) -> str:
    state = _user_state(row)
    value = row.get("example_id") or state.get("example_id")
    if value:
        return str(value)
    return str(row.get("transition_id") or row.get("demo_id") or "").split("::", 1)[0]


def _dataset(row: dict[str, Any]) -> str:
    state = _user_state(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("dataset") or row.get("dataset") or state.get("dataset")
    if value:
        return str(value).lower().replace("-", "_")
    example_id = _example_id(row)
    candidate = example_id.split(":", 1)[0].lower().replace("-", "_") if ":" in example_id else "unknown"
    return candidate if candidate in KNOWN_DATASETS else "unknown"


def _motif_id(row: dict[str, Any]) -> str:
    state = _user_state(row)
    candidate = state.get("candidate") if isinstance(state.get("candidate"), dict) else {}
    motif_candidate = state.get("motif_candidate") if isinstance(state.get("motif_candidate"), dict) else {}
    return str(candidate.get("motif_id") or motif_candidate.get("motif_id") or "")


def _augment_video_map(path: Path, rollout_path: Path) -> dict[str, str]:
    mapping = read_json(path) if path.exists() else {}
    for row in read_jsonl(rollout_path):
        example_id = str(row.get("example_id") or "")
        video = row.get("video") if isinstance(row.get("video"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        graph = metadata.get("clue_memory_graph") if isinstance(metadata.get("clue_memory_graph"), dict) else {}
        video_id = video.get("video_id") or graph.get("video_id")
        dataset = str(row.get("dataset") or "unknown")
        if example_id and video_id:
            mapping[example_id] = f"{dataset}:video:{video_id}"
    return {str(key): str(value) for key, value in mapping.items()}


def _group(row: dict[str, Any], video_map: dict[str, str], specialist: str) -> str:
    example_id = _example_id(row)
    if example_id in video_map:
        return video_map[example_id]
    motif_id = _motif_id(row)
    if specialist == "motif" and motif_id:
        return f"motif:artifact:{motif_id}"
    return f"{_dataset(row)}:example:{example_id or row.get('transition_id')}"


def _character_count(row: dict[str, Any]) -> int:
    return sum(len(str(message.get("content") or "")) for message in row.get("messages") or [] if isinstance(message, dict))


def _valid_row(row: dict[str, Any], max_characters: int) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or [message.get("role") for message in messages if isinstance(message, dict)] != ["system", "user", "assistant"]:
        return False
    try:
        json.loads(str(messages[-1].get("content") or ""))
    except json.JSONDecodeError:
        return False
    if _character_count(row) > max_characters:
        return False
    return True


def _sample_by_key(rows: list[dict[str, Any]], key_fn, quotas: dict[str, int], salt: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    result: list[dict[str, Any]] = []
    for key, quota in quotas.items():
        candidates = sorted(grouped.get(key, []), key=lambda row: _stable(str(row.get("transition_id")), f"{salt}:{key}"))
        result.extend(candidates[:quota])
    return result


def _l1_rows(snapshot: Path, salt: str) -> list[dict[str, Any]]:
    builder = read_jsonl(snapshot / "l1_builder_sft.jsonl")
    patch = read_jsonl(snapshot / "l1_patch_sft.jsonl")
    quotas = {
        "segment_video_or_select_clip": 200,
        "neighbor_vlm_l1_create_node": 600,
        "neighbor_vlm_l1_create_schema_anchor": 300,
        "neighbor_vlm_l1_create_edge": 600,
        "neighbor_vlm_l1_skip_edge": 100,
        "short_video_recurrence_create_clue": 100,
        "short_video_recurrence_link": 100,
    }
    sampled = _sample_by_key(builder, lambda row: (row.get("metadata") or {}).get("skill_id"), quotas, salt)
    return sampled + patch


def _l1_quality_reason(row: dict[str, Any]) -> str | None:
    try:
        user = next(message["content"] for message in row["messages"] if message.get("role") == "user")
        action = json.loads(next(message["content"] for message in row["messages"] if message.get("role") == "assistant"))
    except (KeyError, StopIteration, TypeError, json.JSONDecodeError):
        return "l1_unreadable_action"
    tool = str(action.get("tool_name") or "")
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    if tool in {"neighbor_vlm_l1_create_edge", "short_video_recurrence_link"}:
        edge = arguments.get("edge") if isinstance(arguments.get("edge"), dict) else {}
        if not edge.get("src") or not edge.get("dst"):
            return "l1_missing_edge_endpoint"
        if str(edge["src"]) not in user or str(edge["dst"]) not in user:
            return "l1_invisible_edge_endpoint"
    if tool == "neighbor_vlm_l1_skip_edge":
        for key in ("src_clip_id", "dst_clip_id"):
            value = str(arguments.get(key) or "")
            if not CLIP_ID_PATTERN.fullmatch(value):
                return "l1_malformed_skip_clip_id"
            if value not in user:
                return "l1_invisible_skip_clip_id"
    if tool == "short_video_recurrence_create_clue":
        # Existing rows expose neither recurrence endpoint nor their evidence
        # text, so the standalone clue target cannot be inferred.
        return "l1_invisible_recurrence_support"
    if tool == "apply_l1_evidence_patch":
        state = _user_state(row)
        expected = str((state.get("clip_schema") or {}).get("clip_id") or "")
        if not expected or str(arguments.get("clip_id") or "") != expected:
            return "l1_patch_clip_mismatch"
    return None


def _sanitize_repair_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(row)
    for message in payload.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        try:
            user_payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        state = user_payload.get("state_t") if isinstance(user_payload.get("state_t"), dict) else {}
        visible = state.get("visible_demo_inputs") if isinstance(state.get("visible_demo_inputs"), dict) else {}
        for leaked_decision in ("repair_mode", "strategy", "selection_mode"):
            visible.pop(leaked_decision, None)
        video = visible.get("video") if isinstance(visible.get("video"), dict) else None
        if video is not None:
            visible["video"] = {
                key: video[key]
                for key in ("video_id", "duration_s", "language")
                if key in video
            }
            visible["video"]["derived_clip_count"] = len(video.get("derived_clips") or [])
            visible["video"]["segment_count"] = len(video.get("segments") or [])
        state["visible_demo_inputs"] = visible
        l1 = state.get("l1_compact") if isinstance(state.get("l1_compact"), dict) else None
        if l1 is not None:
            state["l1_compact"] = {
                key: copy.deepcopy(l1[key])
                for key in ("graph_id", "training_view", "quality", "counts", "used_ref_count", "compact_policy")
                if key in l1
            }
            state["l1_compact"]["compact_evidence_nodes"] = copy.deepcopy((l1.get("compact_evidence_nodes") or [])[:20])
        user_payload["state_t"] = state
        message["content"] = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    return payload


def _repair_rows(snapshot: Path, salt: str) -> list[dict[str, Any]]:
    report_rows = read_jsonl(snapshot / "l2_repair_from_reports_sft.jsonl")
    compact_rows = read_jsonl(snapshot / "l2_repair_rounds_batch3_p5_video_only_expert_demos_compact_sft.jsonl")
    positive = []
    negative = []
    for row in report_rows:
        status = str((row.get("metadata") or {}).get("terminal_status") or "")
        (positive if status in {"resolved_strong", "accepted_bridge"} else negative).append(row)
    negative_cap = max(1, 3 * len(positive))
    negative = sorted(negative, key=lambda row: _stable(str(row.get("transition_id")), salt))[:negative_cap]
    return [_sanitize_repair_row(row) for row in positive + negative + compact_rows]


def _verifier_rows(snapshot: Path, salt: str) -> list[dict[str, Any]]:
    rows = read_jsonl(snapshot / "verifier_sft.jsonl")
    supported: list[dict[str, Any]] = []
    hard_negative: list[dict[str, Any]] = []
    empty_negative: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("decision") == "supported":
            supported.append(row)
            continue
        state = _user_state(row)
        pack = state.get("proposed_evidence_pack") if isinstance(state.get("proposed_evidence_pack"), dict) else {}
        (hard_negative if pack.get("support_refs") else empty_negative).append(row)
    # Empty-support negatives are useful for format warm-up but should not
    # dominate the actual evidence-judgment cases.
    empty_negative.sort(key=lambda row: _stable(str(row.get("transition_id")), f"{salt}:verifier-empty"))
    empty_negative = empty_negative[: max(1, len(hard_negative))]
    return supported + hard_negative + empty_negative


def _motif_rows(raw_root: Path, finalized: Path, salt: str) -> list[dict[str, Any]]:
    audit = read_jsonl(raw_root / "motif/evidence_sft.jsonl")
    accepted = [row for row in audit if (row.get("metadata") or {}).get("verdict") == "accept_ref"]
    rejected = [row for row in audit if (row.get("metadata") or {}).get("verdict") != "accept_ref"]
    rejected = sorted(rejected, key=lambda row: _stable(str(row.get("transition_id")), salt))[: 3 * len(accepted)]
    lifecycle = read_jsonl(finalized / "motif_lifecycle_sft.jsonl")
    non_rejected = [row for row in lifecycle if (row.get("metadata") or {}).get("status") != "rejected"]
    lifecycle_rejected = [row for row in lifecycle if (row.get("metadata") or {}).get("status") == "rejected"]
    lifecycle_rejected = sorted(lifecycle_rejected, key=lambda row: _stable(str(row.get("transition_id")), salt))[: 3 * len(non_rejected)]
    return accepted + rejected + non_rejected + lifecycle_rejected


def _specialist_rows(snapshot: Path, finalized: Path, raw_root: Path, salt: str) -> dict[str, list[dict[str, Any]]]:
    l2 = read_jsonl(raw_root / "l2/sft.jsonl")
    return {
        "l1": _l1_rows(snapshot, salt),
        "l2": l2,
        "verifier": _verifier_rows(snapshot, salt),
        "repair": _repair_rows(snapshot, salt),
        "motif": _motif_rows(raw_root, finalized, salt),
    }


def _audit(rows: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("transition_id") or row.get("demo_id") or "") for row in rows]
    task_counts = Counter(str((row.get("metadata") or {}).get("task") or (row.get("metadata") or {}).get("controller") or "unknown") for row in rows)
    label_counts = Counter(
        str((row.get("metadata") or {}).get("verdict") or (row.get("metadata") or {}).get("status") or (row.get("metadata") or {}).get("decision") or "none")
        for row in rows
    )
    train_groups = {str(row["split_group_id"]) for row in train}
    dev_groups = {str(row["split_group_id"]) for row in dev}
    forbidden = 0
    for row in rows:
        for message in row.get("messages") or []:
            if message.get("role") != "user":
                continue
            try:
                forbidden += int(contains_forbidden_prompt_key(json.loads(str(message.get("content") or ""))))
            except json.JSONDecodeError:
                forbidden += 1
    return {
        "rows": len(rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "unique_record_ids": len(set(ids)),
        "duplicate_record_ids": len(ids) - len(set(ids)),
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "group_overlap_count": len(train_groups & dev_groups),
        "prompt_forbidden_key_hits": forbidden,
        "max_characters": max((_character_count(row) for row in rows), default=0),
        "dataset_counts": dict(Counter(_dataset(row) for row in rows)),
        "task_counts": dict(task_counts),
        "label_counts": dict(label_counts),
    }


def _l2_length_shortcut(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        task = str(metadata.get("task") or "")
        if task not in {"rank_coarse_candidates", "rank_coarse_candidates_listwise"}:
            continue
        state = _user_state(row)
        candidates = state.get("candidate_coarse_summaries") if isinstance(state.get("candidate_coarse_summaries"), list) else []
        try:
            action = json.loads(str(row["messages"][2]["content"]))
            gold = int(action["arguments"]["coarse_index"])
            lengths = {
                int(candidate["coarse_index"]): len(json.dumps(candidate, ensure_ascii=False, sort_keys=True))
                for candidate in candidates
                if isinstance(candidate, dict)
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not lengths:
            continue
        family = "pairwise" if task == "rank_coarse_candidates" else "listwise"
        counts[f"{family}_rows"] += 1
        counts[f"{family}_longest_correct"] += int(max(lengths, key=lengths.get) == gold)
    result: dict[str, Any] = dict(counts)
    for family in ("pairwise", "listwise"):
        total = counts[f"{family}_rows"]
        result[f"{family}_longest_accuracy"] = counts[f"{family}_longest_correct"] / max(1, total)
    result["passed"] = (
        result["pairwise_longest_accuracy"] < 0.7
        and result["listwise_longest_accuracy"] < 0.7
    )
    return result


def _repair_prompt_copy_count(rows: list[dict[str, Any]]) -> int:
    copied = 0
    for row in rows:
        state = _user_state(row)
        visible = state.get("visible_demo_inputs") if isinstance(state.get("visible_demo_inputs"), dict) else {}
        try:
            action = json.loads(str(row["messages"][2]["content"])).get("action") or {}
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        copied += int(any(
            action.get(key) is not None and action.get(key) == visible.get(key)
            for key in ("repair_mode", "selection_mode")
        ))
    return copied


def _motif_gate_inconsistency_count(rows: list[dict[str, Any]]) -> int:
    inconsistent = 0
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("controller") != "motif_evidence_auditor":
            continue
        observations = _user_state(row).get("audit_observations") or {}
        try:
            verdict = json.loads(str(row["messages"][2]["content"]))["arguments"]["verdict"]
        except (KeyError, TypeError, json.JSONDecodeError):
            inconsistent += 1
            continue
        expected = "accept_ref" if observations and all(observations.values()) else "reject_ref"
        inconsistent += int(verdict != expected)
    return inconsistent


def _verifier_evidence_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        state = _user_state(row)
        pack = state.get("proposed_evidence_pack") if isinstance(state.get("proposed_evidence_pack"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        decision = str(metadata.get("decision") or "unknown")
        counts[f"{decision}_{'nonempty' if pack.get('support_refs') else 'empty'}_support"] += 1
    return dict(counts)


def build(args: argparse.Namespace) -> dict[str, Any]:
    video_map = _augment_video_map(args.example_video_map, args.finalized / "deduplicated_rollouts.jsonl")
    source = _specialist_rows(args.snapshot, args.finalized, args.raw_root, args.salt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_groups: set[str] = set()
    prepared: dict[str, list[dict[str, Any]]] = {}
    excluded: dict[str, Counter[str]] = {}
    for specialist, rows in source.items():
        counts = Counter()
        unique: dict[str, dict[str, Any]] = {}
        seen_fingerprints: set[str] = set()
        for row in rows:
            if specialist == "l1":
                quality_reason = _l1_quality_reason(row)
                if quality_reason:
                    counts[quality_reason] += 1
                    continue
            if not _valid_row(row, args.max_characters):
                counts["invalid_schema_or_too_long"] += 1
                continue
            dataset = _dataset(row)
            if dataset in EVAL_ONLY:
                counts[f"evaluation_only:{dataset}"] += 1
                continue
            record_id = str(row.get("transition_id") or row.get("demo_id") or "")
            if not record_id:
                counts["missing_record_id"] += 1
                continue
            fingerprint = hashlib.sha256(json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen_fingerprints:
                counts["exact_duplicate_dropped"] += 1
                continue
            seen_fingerprints.add(fingerprint)
            payload = dict(row)
            payload["specialist"] = specialist
            payload["split_group_id"] = _group(payload, video_map, specialist)
            if record_id in unique:
                field = "transition_id" if payload.get("transition_id") else "demo_id"
                metadata = dict(payload.get("metadata") or {})
                metadata["source_record_id"] = record_id
                metadata["record_id_rewritten"] = True
                payload["metadata"] = metadata
                payload[field] = f"{record_id}::variant:{fingerprint[:16]}"
                record_id = str(payload[field])
                counts["record_id_collision_rewritten"] += 1
            unique[record_id] = payload
        prepared[specialist] = list(unique.values())
        excluded[specialist] = counts
        all_groups.update(str(row["split_group_id"]) for row in prepared[specialist])

    group_to_split = {
        group: ("dev" if int(_stable(group, args.salt)[:8], 16) % 100 < args.dev_percent else "train")
        for group in sorted(all_groups)
    }
    reports: dict[str, Any] = {}
    for specialist, rows in prepared.items():
        rows.sort(key=lambda row: str(row.get("transition_id") or row.get("demo_id")))
        train = [row for row in rows if group_to_split[row["split_group_id"]] == "train"]
        dev = [row for row in rows if group_to_split[row["split_group_id"]] == "dev"]
        target = args.output_dir / specialist
        write_jsonl(target / "all_sft.jsonl", rows)
        write_jsonl(target / "train.jsonl", train)
        write_jsonl(target / "dev.jsonl", dev)
        report = _audit(rows, train, dev)
        if specialist == "l2":
            core = [row for row in rows if (row.get("metadata") or {}).get("is_core")]
            derived = [row for row in rows if not (row.get("metadata") or {}).get("is_core")]
            write_jsonl(target / "core_sft.jsonl", core)
            write_jsonl(target / "derived_sft.jsonl", derived)
            write_jsonl(target / "core_train.jsonl", [row for row in train if (row.get("metadata") or {}).get("is_core")])
            write_jsonl(target / "core_dev.jsonl", [row for row in dev if (row.get("metadata") or {}).get("is_core")])
            write_jsonl(target / "derived_train.jsonl", [row for row in train if not (row.get("metadata") or {}).get("is_core")])
            write_jsonl(target / "derived_dev.jsonl", [row for row in dev if not (row.get("metadata") or {}).get("is_core")])
            weight_sums: dict[tuple[str, str], float] = defaultdict(float)
            for row in rows:
                metadata = row.get("metadata") or {}
                weight_sums[(
                    str(metadata.get("source_example_id") or ""),
                    str(metadata.get("augmentation_family") or "unknown"),
                )] += float(metadata.get("source_family_weight") or 0.0)
            report["core_rows"] = len(core)
            report["derived_rows"] = len(derived)
            report["augmentation_family_counts"] = dict(Counter(
                str((row.get("metadata") or {}).get("augmentation_family") or "unknown")
                for row in rows
            ))
            report["source_family_weight_sum_min"] = min(weight_sums.values(), default=0.0)
            report["source_family_weight_sum_max"] = max(weight_sums.values(), default=0.0)
            report["length_shortcut_audit"] = _l2_length_shortcut(rows)
        if specialist == "repair":
            report["prompt_target_decision_copy_count"] = _repair_prompt_copy_count(rows)
        if specialist == "motif":
            report["motif_gate_inconsistency_count"] = _motif_gate_inconsistency_count(rows)
        if specialist == "verifier":
            report["evidence_mix"] = _verifier_evidence_mix(rows)
        report["excluded_counts"] = dict(excluded[specialist])
        report["hard_gates_passed"] = (
            report["duplicate_record_ids"] == 0
            and report["group_overlap_count"] == 0
            and report["prompt_forbidden_key_hits"] == 0
            and report["train_rows"] > 0
            and report["dev_rows"] > 0
            and (specialist != "l2" or report["length_shortcut_audit"]["passed"])
            and (specialist != "repair" or report["prompt_target_decision_copy_count"] == 0)
            and (specialist != "motif" or report["motif_gate_inconsistency_count"] == 0)
        )
        write_json(target / "report.json", report)
        reports[specialist] = report
    manifest = {
        "schema_version": "video-skills/five-specialist-sft-manifest-v0.1",
        "salt": args.salt,
        "dev_percent": args.dev_percent,
        "split_unit": "shared source-video group when resolvable; motif artifact otherwise",
        "group_count": len(group_to_split),
        "specialists": reports,
        "all_hard_gates_passed": all(report["hard_gates_passed"] for report in reports.values()),
    }
    write_json(args.output_dir / "training_manifest.json", manifest)
    write_json(args.output_dir / "group_split_manifest.json", group_to_split)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--finalized", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--example-video-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--salt", default="video-skills-five-specialists-v3")
    parser.add_argument("--dev-percent", type=int, default=15)
    parser.add_argument("--max-characters", type=int, default=48000)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["all_hard_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

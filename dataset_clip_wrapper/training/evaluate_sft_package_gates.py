#!/usr/bin/env python3
"""Preflight quality gates for a five-specialist SFT package before training.

This gate answers: is the package safe and balanced enough to start SFT warm-up?
It does not require GPU and does not measure downstream task success.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_split_manifest import EVALUATION_ONLY_DATASETS, DEFAULT_DATASET_ROOT, role_lookup
from .sft_common import contains_forbidden_prompt_key, read_json, read_jsonl, write_json


SPECIALISTS = ("l1", "l2", "repair", "verifier", "motif")
ALLOWED_SFT_ROLES = {"sft_seed", "dev_tune"}


def build_example_to_video_lookup(dataset_root: Path) -> dict[str, str]:
    """Map example_id / qid keys to split-manifest video keys."""
    mapping: dict[str, str] = {}
    cg_path = dataset_root / "CG-Bench" / "cgbench.json"
    if cg_path.exists():
        for row in json.loads(cg_path.read_text(encoding="utf-8")):
            qid = str(row.get("qid") or "")
            video_id = str(row.get("video_uid") or "")
            if qid and video_id:
                mapping[f"cg_bench:{qid}"] = f"cg_bench:{video_id}"
    vh_benchmark = dataset_root / "Video-Holmes" / "Benchmark"
    for path in (vh_benchmark / "train_Video-Holmes.json", vh_benchmark / "test_Video-Holmes.json"):
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            video_id = str(row.get("video ID") or "")
            qid = str(row.get("Question ID") or "")
            if video_id:
                mapping[f"video_holmes:{video_id}"] = f"video_holmes:{video_id}"
            if qid and video_id:
                # Common example_id shapes in exports.
                mapping[f"video_holmes:train:{video_id}:{qid}"] = f"video_holmes:{video_id}"
                mapping[f"video_holmes:test:{video_id}:{qid}"] = f"video_holmes:{video_id}"
                mapping[f"video_holmes:{video_id}:{qid}"] = f"video_holmes:{video_id}"
    return mapping


def _assistant_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _user_state(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("state_t"), dict):
            return payload["state_t"]
    return {}


def _dataset(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    state = _user_state(row)
    value = row.get("dataset") or metadata.get("dataset") or state.get("dataset") or "unknown"
    return str(value).lower().replace("-", "_")


def _video_key(
    row: dict[str, Any],
    *,
    example_to_video: dict[str, str] | None = None,
) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    state = _user_state(row)
    video_state = state.get("video_state") if isinstance(state.get("video_state"), dict) else {}
    video_id = (
        metadata.get("video_id")
        or state.get("video_id")
        or video_state.get("video_id")
        or row.get("video_id")
    )
    dataset = _dataset(row)
    if video_id and dataset not in {"unknown", ""}:
        # CG rows sometimes stash qid in video_id by mistake; prefer lookup below.
        candidate = f"{dataset}:{video_id}"
        if example_to_video and candidate in example_to_video:
            return example_to_video[candidate]
        if dataset == "cg_bench" and example_to_video and candidate.startswith("cg_bench:"):
            # If video_id is actually a qid, map it.
            mapped = example_to_video.get(candidate)
            if mapped:
                return mapped
        return candidate

    example_id = str(
        row.get("example_id")
        or state.get("example_id")
        or metadata.get("source_example_id")
        or metadata.get("example_id")
        or ""
    )
    transition_id = str(row.get("transition_id") or "")
    candidates = [example_id]
    if "::" in transition_id:
        candidates.append(transition_id.split("::", 1)[0])
    if example_to_video:
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in example_to_video:
                return example_to_video[candidate]
            # video_holmes:...:qN or expert_demo embeds.
            parts = candidate.split(":")
            if len(parts) >= 3 and parts[0] == "video_holmes":
                # shapes: video_holmes:train:<vid>:q... or video_holmes:<vid>:q...
                if parts[1] in {"train", "test"} and len(parts) >= 4:
                    key = f"video_holmes:{parts[2]}"
                    if key in example_to_video:
                        return example_to_video[key]
                key = f"video_holmes:{parts[1]}"
                if key in example_to_video:
                    return example_to_video[key]
            if parts and parts[0] == "cg_bench" and len(parts) >= 2:
                key = f"cg_bench:{parts[1]}"
                if key in example_to_video:
                    return example_to_video[key]
    return None


def _action_family(payload: dict[str, Any] | None, metadata: dict[str, Any]) -> str:
    if not payload:
        return "parse_fail"
    if metadata.get("skill_id"):
        return str(metadata["skill_id"])
    if metadata.get("task"):
        return str(metadata["task"])
    if payload.get("tool_name"):
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        if "decision" in args:
            return f"{payload['tool_name']}::{args.get('decision')}"
        if "status" in args:
            return f"{payload['tool_name']}::{args.get('status')}"
        if "verdict" in args:
            return f"{payload['tool_name']}::{args.get('verdict')}"
        return str(payload["tool_name"])
    action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
    return f"{payload.get('round_type') or '?'}::{action.get('action_type') or '?'}"


def _chat_roles_ok(row: dict[str, Any]) -> bool:
    roles = [message.get("role") for message in row.get("messages") or [] if isinstance(message, dict)]
    return roles == ["system", "user", "assistant"]


def evaluate_split_file(
    path: Path,
    *,
    specialist: str,
    split_name: str,
    role_map: dict[str, str] | None = None,
    example_to_video: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    families: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    role_hits: Counter[str] = Counter()
    forbidden = 0
    parse_ok = 0
    roles_ok = 0
    eval_only = 0
    bad_role = 0
    unknown_video = 0
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        payload = _assistant_payload(row)
        if payload is not None:
            parse_ok += 1
        if _chat_roles_ok(row):
            roles_ok += 1
        family = _action_family(payload, metadata)
        families[family] += 1
        dataset = _dataset(row)
        datasets[dataset] += 1
        if dataset in EVALUATION_ONLY_DATASETS:
            eval_only += 1
        # Check prompt-visible leakage only (system/user). Assistant may mention
        # gold fields only as bookkeeping in rare exports; those are not prompt.
        leaked = False
        for message in row.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None and contains_forbidden_prompt_key(parsed):
                    leaked = True
                    break
                # raw string fallback for accidental gold dumps
                lowered = content.lower()
                if any(f'"{key}"' in lowered or f"'{key}'" in lowered for key in ("gold_answer", "official_answer", "hidden_supervision")):
                    leaked = True
                    break
            elif contains_forbidden_prompt_key(content):
                leaked = True
                break
        if leaked:
            forbidden += 1
        if role_map is not None:
            key = _video_key(row, example_to_video=example_to_video)
            if key is None:
                unknown_video += 1
            else:
                role = role_map.get(key)
                if role is None:
                    unknown_video += 1
                else:
                    role_hits[role] += 1
                    if role not in ALLOWED_SFT_ROLES:
                        bad_role += 1

    n = len(rows)
    top1 = families.most_common(1)[0][1] if families else 0
    return {
        "specialist": specialist,
        "split": split_name,
        "path": str(path),
        "n_rows": n,
        "assistant_json_parse_rate": parse_ok / max(1, n),
        "chat_roles_ok_rate": roles_ok / max(1, n),
        "prompt_forbidden_key_hits": forbidden,
        "eval_only_rows": eval_only,
        "dataset_counts": dict(datasets),
        "action_family_counts": dict(families),
        "n_action_families": len(families),
        "top1_family_share": top1 / max(1, n),
        "majority_action_family": families.most_common(1)[0][0] if families else None,
        "split_role_counts": dict(role_hits),
        "bad_role_rows": bad_role,
        "unknown_video_rows": unknown_video,
    }


def decide_package_gates(
    reports: list[dict[str, Any]],
    *,
    require_split_manifest: bool,
    max_top1_share: dict[str, float] | None = None,
) -> dict[str, Any]:
    max_top1_share = max_top1_share or {
        "l1": 0.65,
        "l2": 0.45,
        "repair": 0.75,
        "verifier": 0.75,
        "motif": 0.70,
    }
    failures: list[str] = []
    warnings: list[str] = []
    for report in reports:
        tag = f"{report['specialist']}/{report['split']}"
        if report["n_rows"] <= 0:
            failures.append(f"{tag}: empty split")
            continue
        if report["prompt_forbidden_key_hits"] != 0:
            failures.append(f"{tag}: prompt_forbidden_key_hits={report['prompt_forbidden_key_hits']}")
        if report["eval_only_rows"] != 0:
            failures.append(f"{tag}: eval_only_rows={report['eval_only_rows']}")
        if report["assistant_json_parse_rate"] < 1.0:
            failures.append(f"{tag}: assistant_json_parse_rate={report['assistant_json_parse_rate']:.3f}")
        if report["chat_roles_ok_rate"] < 1.0:
            failures.append(f"{tag}: chat_roles_ok_rate={report['chat_roles_ok_rate']:.3f}")
        limit = max_top1_share.get(report["specialist"], 0.80)
        if report["top1_family_share"] > limit:
            warnings.append(
                f"{tag}: top1_family_share={report['top1_family_share']:.3f} > {limit:.3f}; "
                "require family-weighted sampling at train time"
            )
        if require_split_manifest:
            if report["bad_role_rows"] != 0:
                failures.append(f"{tag}: bad_role_rows={report['bad_role_rows']} (non sft_seed/dev_tune)")
            # Motif lifecycle rows may be video-agnostic; warn rather than fail.
            if report["specialist"] != "motif" and report["unknown_video_rows"] > 0:
                warnings.append(f"{tag}: unknown_video_rows={report['unknown_video_rows']}")
            if report["specialist"] == "motif" and report["unknown_video_rows"] == report["n_rows"]:
                warnings.append(f"{tag}: all motif rows lack video keys; treat as bank-level artifacts")
    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def evaluate_five_lora_package(
    package_root: Path,
    *,
    split_manifest_path: Path | None = None,
    require_split_manifest: bool = True,
    dataset_root: Path | None = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    role_map = None
    manifest_hash = None
    if split_manifest_path is not None:
        manifest = read_json(split_manifest_path)
        role_map = role_lookup(manifest)
        manifest_hash = manifest.get("manifest_hash")
    elif require_split_manifest:
        raise ValueError("split_manifest_path is required when require_split_manifest=True")

    example_to_video = build_example_to_video_lookup(dataset_root) if dataset_root is not None else None
    reports = []
    for specialist in SPECIALISTS:
        for split_name in ("train", "dev"):
            path = package_root / specialist / f"{split_name}.jsonl"
            if not path.exists():
                reports.append(
                    {
                        "specialist": specialist,
                        "split": split_name,
                        "path": str(path),
                        "n_rows": 0,
                        "prompt_forbidden_key_hits": 0,
                        "eval_only_rows": 0,
                        "assistant_json_parse_rate": 0.0,
                        "chat_roles_ok_rate": 0.0,
                        "top1_family_share": 1.0,
                        "bad_role_rows": 0,
                        "unknown_video_rows": 0,
                        "action_family_counts": {},
                        "dataset_counts": {},
                    }
                )
                continue
            reports.append(
                evaluate_split_file(
                    path,
                    specialist=specialist,
                    split_name=split_name,
                    role_map=role_map,
                    example_to_video=example_to_video,
                )
            )
    decision = decide_package_gates(reports, require_split_manifest=require_split_manifest)
    return {
        "schema_version": "video-skills/sft-package-gates-v1",
        "package_root": str(package_root),
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
        "split_manifest_hash": manifest_hash,
        "specialist_reports": reports,
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--allow-missing-split-manifest", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_five_lora_package(
        args.package_root,
        split_manifest_path=args.split_manifest,
        require_split_manifest=not args.allow_missing_split_manifest,
        dataset_root=args.dataset_root,
    )
    write_json(args.output, report)
    print(json.dumps({"passed": report["decision"]["passed"], "output": str(args.output), **report["decision"]}, indent=2))
    return 0 if report["decision"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build train/dev JSONL splits from a collected SFT snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .sft_common import contains_forbidden_prompt_key, read_jsonl, write_json, write_jsonl


DEFAULT_PATTERNS = [
    "l1_builder_sft.jsonl",
    "l1_patch_sft.jsonl",
    "l2_retrieval_sft.jsonl",
    "l2_repair_rounds_*_sft.jsonl",
    "l2_repair_from_reports_sft.jsonl",
    "verifier_sft.jsonl",
    "motif_lifecycle_sft.jsonl",
]


DEFAULT_MIXTURE = {
    "l1": 35,
    "l2": 35,
    "verifier": 20,
    "motif": 10,
}


EVALUATION_ONLY_DATASETS = {"ovo_bench", "videomme", "vrbench"}
KNOWN_DATASETS = EVALUATION_ONLY_DATASETS | {"cg_bench", "siv_bench", "video_holmes"}


def _stable_bucket(key: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10000


def _message_state(row: dict[str, Any]) -> dict[str, Any]:
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


def _example_id(row: dict[str, Any]) -> str:
    state = _message_state(row)
    value = row.get("example_id") or state.get("example_id")
    if value:
        return str(value)
    transition_id = str(row.get("transition_id") or row.get("demo_id") or "")
    return transition_id.split("::", 1)[0]


def _dataset(row: dict[str, Any]) -> str:
    """Resolve dataset provenance even for adapters that omit top-level metadata."""
    state = _message_state(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = row.get("dataset") or metadata.get("dataset") or state.get("dataset")
    if value:
        return str(value).lower().replace("-", "_")
    group = str(row.get("split_group_id") or "")
    if ":" in group and not group.startswith("unknown:"):
        candidate = group.split(":", 1)[0].lower().replace("-", "_")
        if candidate in KNOWN_DATASETS:
            return candidate
    example_id = _example_id(row)
    if ":" in example_id:
        candidate = example_id.split(":", 1)[0].lower().replace("-", "_")
        if candidate in KNOWN_DATASETS:
            return candidate
    return "unknown"


def _message_character_count(row: dict[str, Any]) -> int:
    return sum(
        len(str(message.get("content") or ""))
        for message in row.get("messages") or []
        if isinstance(message, dict)
    )


def _split_group(row: dict[str, Any], example_video_map: dict[str, str] | None = None) -> str:
    """Return a source-video group so no trajectory leaks across train/dev."""
    state = _message_state(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    dataset = str(metadata.get("dataset") or state.get("dataset") or row.get("dataset") or "unknown")
    video_state = state.get("video_state") if isinstance(state.get("video_state"), dict) else {}
    video_id = metadata.get("video_id") or video_state.get("video_id") or state.get("video_id")
    if video_id:
        return f"{dataset}:video:{video_id}"

    example_id = _example_id(row)
    if example_video_map and example_id in example_video_map:
        return str(example_video_map[example_id])
    parts = example_id.split(":")
    if dataset == "video_holmes" and len(parts) >= 4:
        return f"{dataset}:video:{parts[2]}"

    # Some controllers only retain a benchmark example id. Keeping all actions
    # for that example together is still safer than transition-level splitting.
    if example_id:
        return f"{dataset}:example:{example_id}"
    return f"{dataset}:artifact:{row.get('transition_id') or row.get('demo_id') or 'unknown'}"


def _controller(row: dict[str, Any], source: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("controller"):
        return str(metadata["controller"])
    name = source.name
    if name.endswith("_sft.jsonl"):
        return name[: -len("_sft.jsonl")]
    return name


def _controller_family(controller: str) -> str:
    if controller in {"l1_builder", "l1_patch"}:
        return "l1"
    if controller.startswith("l2_") or controller == "l2_repair":
        return "l2"
    if controller in {"verifier", "auxiliary_verifier"}:
        return "verifier"
    if controller.startswith("motif"):
        return "motif"
    return "other"


def _read_snapshot_rows(
    snapshot_dir: Path,
    patterns: list[str],
    example_video_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(snapshot_dir.glob(pattern)):
            for row in read_jsonl(path):
                payload = dict(row)
                payload.setdefault("source_sft_path", str(path))
                payload.setdefault("controller", _controller(payload, path))
                payload.setdefault("split_group_id", _split_group(payload, example_video_map))
                rows.append(payload)
    return rows


def _record_id(row: dict[str, Any]) -> str:
    return str(row.get("transition_id") or row.get("demo_id") or "")


def _row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "controller": row.get("controller"),
        "split_group_id": row.get("split_group_id"),
        "messages": row.get("messages"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_record_ids(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop exact duplicates and make semantically distinct colliding IDs unique."""
    deduplicated: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    exact_duplicates_dropped = 0
    for row in rows:
        fingerprint = _row_fingerprint(row)
        if fingerprint in seen_fingerprints:
            exact_duplicates_dropped += 1
            continue
        seen_fingerprints.add(fingerprint)
        deduplicated.append(row)

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated:
        by_id[_record_id(row)].append(row)

    collision_groups = 0
    rewritten_rows = 0
    missing_ids = 0
    for record_id, variants in by_id.items():
        if not record_id:
            missing_ids += len(variants)
            for row in variants:
                digest = _row_fingerprint(row)[:16]
                row["transition_id"] = f"generated::{digest}"
            rewritten_rows += len(variants)
            continue
        if len(variants) == 1:
            continue
        collision_groups += 1
        for row in variants:
            digest = _row_fingerprint(row)[:16]
            field = "transition_id" if row.get("transition_id") else "demo_id"
            metadata = dict(row.get("metadata") or {})
            metadata["source_record_id"] = record_id
            metadata["record_id_rewritten"] = True
            row["metadata"] = metadata
            row[field] = f"{record_id}::variant:{digest}"
            rewritten_rows += 1

    return deduplicated, {
        "exact_duplicates_dropped": exact_duplicates_dropped,
        "record_id_collision_groups": collision_groups,
        "record_ids_rewritten": rewritten_rows,
        "missing_record_ids_repaired": missing_ids,
    }


def _balanced_sample(
    rows: list[dict[str, Any]],
    *,
    target_total: int,
    mixture: dict[str, int],
    salt: str,
    controller_minimums: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if target_total <= 0:
        return rows, dict(Counter(_controller_family(str(row.get("controller") or "")) for row in rows))

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[_controller_family(str(row.get("controller") or ""))].append(row)
    for family_rows in by_family.values():
        family_rows.sort(
            key=lambda row: _stable_bucket(
                str(row.get("transition_id") or row.get("demo_id") or json.dumps(row, sort_keys=True)[:512]),
                f"{salt}:sample",
            )
        )

    weight_total = sum(mixture.values())
    quotas = {
        family: min(len(by_family.get(family, [])), target_total * weight // weight_total)
        for family, weight in mixture.items()
    }
    remaining = min(target_total, sum(len(rows) for rows in by_family.values())) - sum(quotas.values())
    while remaining > 0:
        candidates = [
            family for family in mixture
            if quotas.get(family, 0) < len(by_family.get(family, []))
        ]
        if not candidates:
            break
        for family in candidates:
            if remaining <= 0:
                break
            quotas[family] = quotas.get(family, 0) + 1
            remaining -= 1

    controller_minimums = controller_minimums or {}
    selected: list[dict[str, Any]] = []
    for family, quota in quotas.items():
        family_rows = by_family.get(family, [])
        family_selected: list[dict[str, Any]] = []
        selected_fingerprints: set[str] = set()
        for controller, minimum in controller_minimums.items():
            if _controller_family(controller) != family or minimum <= 0:
                continue
            candidates = [row for row in family_rows if str(row.get("controller") or "") == controller]
            for row in candidates[: min(minimum, len(candidates), quota - len(family_selected))]:
                family_selected.append(row)
                selected_fingerprints.add(_row_fingerprint(row))
        for row in family_rows:
            if len(family_selected) >= quota:
                break
            if _row_fingerprint(row) in selected_fingerprints:
                continue
            family_selected.append(row)
        selected.extend(family_selected)
    selected.sort(key=lambda row: str(row.get("transition_id") or row.get("demo_id") or ""))
    return selected, quotas


def _audit_rows(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    rows = train + dev
    ids = [_record_id(row) for row in rows]
    invalid_message_schema = 0
    empty_message_content = 0
    assistant_json_errors = 0
    prompt_forbidden_key_hits = 0
    char_lengths: list[int] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages if isinstance(item, dict)] != [
            "system",
            "user",
            "assistant",
        ]:
            invalid_message_schema += 1
            continue
        contents = [str(item.get("content") or "") for item in messages]
        empty_message_content += int(any(not content.strip() for content in contents))
        char_lengths.append(sum(len(content) for content in contents))
        try:
            json.loads(contents[-1])
        except (json.JSONDecodeError, TypeError):
            assistant_json_errors += 1
        try:
            user_payload = json.loads(contents[1])
        except (json.JSONDecodeError, TypeError):
            user_payload = {}
        prompt_forbidden_key_hits += int(contains_forbidden_prompt_key(user_payload))

    sorted_lengths = sorted(char_lengths)
    p95_index = int(0.95 * (len(sorted_lengths) - 1)) if sorted_lengths else 0
    train_ids = {_record_id(row) for row in train}
    dev_ids = {_record_id(row) for row in dev}
    return {
        "rows_checked": len(rows),
        "missing_record_ids": sum(not value for value in ids),
        "duplicate_record_ids": len(ids) - len(set(ids)),
        "invalid_message_schema": invalid_message_schema,
        "empty_message_content": empty_message_content,
        "assistant_json_errors": assistant_json_errors,
        "prompt_forbidden_key_hits": prompt_forbidden_key_hits,
        "train_dev_record_id_overlap": len(train_ids & dev_ids),
        "character_lengths": {
            "min": min(sorted_lengths) if sorted_lengths else 0,
            "p50": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0,
            "p95": sorted_lengths[p95_index] if sorted_lengths else 0,
            "max": max(sorted_lengths) if sorted_lengths else 0,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_splits(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    dev_percent: int,
    salt: str,
    patterns: list[str],
    target_total: int = 0,
    mixture: dict[str, int] | None = None,
    example_video_map: dict[str, str] | None = None,
    exclude_datasets: set[str] | None = None,
    controller_minimums: dict[str, int] | None = None,
    max_characters: int = 0,
    strict: bool = False,
) -> dict[str, Any]:
    all_rows = _read_snapshot_rows(snapshot_dir, patterns, example_video_map)
    normalized_exclusions = {value.lower().replace("-", "_") for value in (exclude_datasets or set())}
    excluded_dataset_counts = Counter(_dataset(row) for row in all_rows if _dataset(row) in normalized_exclusions)
    eligible_rows = [row for row in all_rows if _dataset(row) not in normalized_exclusions]
    rows_excluded_too_long = 0
    if max_characters > 0:
        rows_excluded_too_long = sum(_message_character_count(row) > max_characters for row in eligible_rows)
        eligible_rows = [row for row in eligible_rows if _message_character_count(row) <= max_characters]
    eligible_rows, id_normalization = _normalize_record_ids(eligible_rows)
    rows, mixture_counts = _balanced_sample(
        eligible_rows,
        target_total=target_total,
        mixture=mixture or DEFAULT_MIXTURE,
        salt=salt,
        controller_minimums=controller_minimums,
    )
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    threshold = max(0, min(10000, dev_percent * 100))
    for row in rows:
        # Motifs are mined from accepted training rollouts and are priors for
        # training, not independent held-out evaluation examples.
        if _controller_family(str(row.get("controller") or "")) == "motif":
            train.append(row)
        elif _stable_bucket(str(row["split_group_id"]), f"{salt}:split") < threshold:
            dev.append(row)
        else:
            train.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_sft.jsonl"
    dev_path = output_dir / "dev_sft.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    audit = _audit_rows(train, dev)
    selected_dataset_counts = Counter(_dataset(row) for row in rows)
    selected_controller_counts = Counter(str(row.get("controller") or "unknown") for row in rows)
    controller_minimum_shortfalls = {
        controller: minimum - selected_controller_counts.get(controller, 0)
        for controller, minimum in (controller_minimums or {}).items()
        if selected_controller_counts.get(controller, 0) < minimum
    }
    group_overlap_count = len(
        {str(row["split_group_id"]) for row in train}
        & {str(row["split_group_id"]) for row in dev}
    )
    hard_gate_failures = {
        "target_row_count": bool(target_total > 0 and len(rows) != target_total),
        "excluded_dataset_present": any(dataset in selected_dataset_counts for dataset in normalized_exclusions),
        "group_overlap": bool(group_overlap_count),
        "duplicate_record_ids": bool(audit["duplicate_record_ids"]),
        "missing_record_ids": bool(audit["missing_record_ids"]),
        "invalid_message_schema": bool(audit["invalid_message_schema"]),
        "empty_message_content": bool(audit["empty_message_content"]),
        "assistant_json_errors": bool(audit["assistant_json_errors"]),
        "prompt_forbidden_key_hits": bool(audit["prompt_forbidden_key_hits"]),
        "train_dev_record_id_overlap": bool(audit["train_dev_record_id_overlap"]),
        "controller_minimum_shortfall": bool(controller_minimum_shortfalls),
    }
    summary = {
        "schema_version": "video-skills/sft-split-report-v0.2",
        "snapshot_dir": str(snapshot_dir),
        "output_dir": str(output_dir),
        "dev_percent": dev_percent,
        "salt": salt,
        "split_unit": "source_video_or_example_fallback",
        "rows_available": len(all_rows),
        "rows_eligible": len(eligible_rows),
        "rows_total": len(rows),
        "rows_train": len(train),
        "rows_dev": len(dev),
        "controller_counts_total": dict(selected_controller_counts),
        "controller_counts_train": dict(Counter(str(row.get("controller") or "unknown") for row in train)),
        "controller_counts_dev": dict(Counter(str(row.get("controller") or "unknown") for row in dev)),
        "controller_family_counts_total": dict(Counter(_controller_family(str(row.get("controller") or "")) for row in rows)),
        "dataset_counts_total": dict(selected_dataset_counts),
        "excluded_datasets": sorted(normalized_exclusions),
        "excluded_dataset_counts": dict(excluded_dataset_counts),
        "id_normalization": id_normalization,
        "controller_minimums": controller_minimums or {},
        "controller_minimum_shortfalls": controller_minimum_shortfalls,
        "max_characters": max_characters,
        "rows_excluded_too_long": rows_excluded_too_long,
        "target_total": target_total,
        "target_mixture_percent": mixture or DEFAULT_MIXTURE,
        "selected_mixture_counts": mixture_counts,
        "group_counts": {
            "total": len({str(row["split_group_id"]) for row in rows}),
            "train": len({str(row["split_group_id"]) for row in train}),
            "dev": len({str(row["split_group_id"]) for row in dev}),
        },
        "group_overlap_count": group_overlap_count,
        "audit": audit,
        "hard_gate_failures": hard_gate_failures,
        "hard_gates_passed": not any(hard_gate_failures.values()),
        "train_path": str(train_path),
        "dev_path": str(dev_path),
    }
    write_json(output_dir / "split_report.json", summary)
    manifest = {
        "schema_version": "video-skills/sft-training-manifest-v0.1",
        "split_report": str(output_dir / "split_report.json"),
        "source_snapshot": str(snapshot_dir),
        "files": {
            "train": {"path": str(train_path), "rows": len(train), "sha256": _sha256(train_path)},
            "dev": {"path": str(dev_path), "rows": len(dev), "sha256": _sha256(dev_path)},
        },
        "hard_gates_passed": summary["hard_gates_passed"],
    }
    write_json(output_dir / "training_manifest.json", manifest)
    if strict and not summary["hard_gates_passed"]:
        failed = [name for name, value in hard_gate_failures.items() if value]
        raise ValueError(f"SFT hard gates failed: {', '.join(failed)}")
    return summary


def _parse_controller_minimums(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONTROLLER=COUNT, got {value!r}")
        controller, raw_count = value.split("=", 1)
        result[controller] = int(raw_count)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-percent", type=int, default=5)
    parser.add_argument("--salt", default="video-skills-sft-v1")
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--target-total", type=int, default=0, help="Downsample to a controller-balanced pilot; 0 keeps all rows.")
    parser.add_argument("--l1-percent", type=int, default=35)
    parser.add_argument("--l2-percent", type=int, default=35)
    parser.add_argument("--verifier-percent", type=int, default=20)
    parser.add_argument("--motif-percent", type=int, default=10)
    parser.add_argument("--example-video-map", type=Path, help="JSON map from example_id to dataset:video:<video_id> split group.")
    parser.add_argument("--exclude-dataset", action="append", default=[], help="Dataset to keep out of both train and dev.")
    parser.add_argument(
        "--controller-minimum",
        action="append",
        default=[],
        metavar="CONTROLLER=COUNT",
        help="Reserve at least COUNT rows for a controller inside its family quota.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any training-data hard gate is non-zero.")
    parser.add_argument(
        "--max-characters",
        type=int,
        default=0,
        help="Exclude chats longer than this many characters before sampling; 0 disables the cap.",
    )
    args = parser.parse_args(argv)
    example_video_map = None
    if args.example_video_map:
        example_video_map = json.loads(args.example_video_map.read_text(encoding="utf-8"))
        if not isinstance(example_video_map, dict):
            raise ValueError("--example-video-map must contain a JSON object")
    summary = build_splits(
        args.snapshot_dir,
        args.output_dir,
        dev_percent=args.dev_percent,
        salt=args.salt,
        patterns=args.pattern or DEFAULT_PATTERNS,
        target_total=args.target_total,
        mixture={
            "l1": args.l1_percent,
            "l2": args.l2_percent,
            "verifier": args.verifier_percent,
            "motif": args.motif_percent,
        },
        example_video_map=example_video_map,
        exclude_datasets=set(args.exclude_dataset),
        controller_minimums=_parse_controller_minimums(args.controller_minimum),
        max_characters=args.max_characters,
        strict=args.strict,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed coverage/leakage audit for Video-Holmes heldout L1 catalogs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


FORBIDDEN_VISIBLE_SOURCE_TYPES = {
    "segment_description",
    "inference_shot",
    "key_relationship",
}
EXPECTED_L1_PERCEPTION_PROTOCOL = "no-redundant-covered-tail-v1"


def _manifest_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    videos = manifest.get("videos") or []
    if isinstance(videos, Mapping):
        return [row for row in videos.values() if isinstance(row, Mapping)]
    return [row for row in videos if isinstance(row, Mapping)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    manifest: Mapping[str, Any],
    l1_rows: Iterable[tuple[Path, Mapping[str, Any]]],
    *,
    expected_count: int = 270,
    expected_clip_model: str = "Qwen/Qwen3.5-9B",
    expected_sampled_frames: int = 4,
    expected_anchor_repass_frames: int = 6,
    expected_max_tokens: int = 1600,
) -> dict[str, Any]:
    expected_ids = {
        str(row.get("video_id") or "")
        for row in _manifest_rows(manifest)
        if row.get("dataset") == "video_holmes"
        and row.get("role") == "heldout_test"
        and row.get("official_split") == "test"
    }
    observed: dict[str, Path] = {}
    duplicates: list[str] = []
    invalid_contract: list[str] = []
    forbidden_visible: list[str] = []
    empty_catalogs: list[str] = []
    files_without_caption_spans: list[str] = []
    incomplete_perception: list[str] = []
    invalid_perception_schemas: list[str] = []
    inconsistent_perception_configs: list[str] = []
    inconsistent_perception_protocols: list[str] = []
    source_types: Counter[str] = Counter()
    file_records: list[tuple[str, str]] = []
    for path, row in l1_rows:
        video = row.get("video") or {}
        video_id = str(video.get("video_id") or "") if isinstance(video, Mapping) else ""
        example_id = str(row.get("example_id") or "")
        hidden = row.get("hidden_supervision") or {}
        metadata = row.get("metadata") or {}
        graph = metadata.get("clue_memory_graph") or {} if isinstance(metadata, Mapping) else {}
        stats = graph.get("index_stats") or {} if isinstance(graph, Mapping) else {}
        perception = graph.get("perception") or {} if isinstance(graph, Mapping) else {}
        clip_schemas = metadata.get("clip_schemas") or [] if isinstance(metadata, Mapping) else []
        if (
            row.get("dataset") != "video_holmes"
            or row.get("split") != "test"
            or not example_id.startswith("video_holmes:test:")
            or not video_id
            or not isinstance(hidden, Mapping)
            or hidden.get("available_for_inference") is not False
        ):
            invalid_contract.append(str(path))
        if (
            not isinstance(stats, Mapping)
            or int(stats.get("fine_clip_count") or 0) <= 0
            or int(stats.get("perception_clip_count") or 0)
            != int(stats.get("fine_clip_count") or 0)
            or not isinstance(perception, Mapping)
            or not perception.get("clip_schema_model")
        ):
            incomplete_perception.append(video_id or str(path))
        fine_clip_count = int(stats.get("fine_clip_count") or 0) if isinstance(stats, Mapping) else 0
        schema_clip_ids = [
            str(schema.get("clip_id") or "")
            for schema in clip_schemas
            if isinstance(schema, Mapping)
        ] if isinstance(clip_schemas, list) else []
        if (
            not isinstance(clip_schemas, list)
            or len(clip_schemas) != fine_clip_count
            or len(schema_clip_ids) != len(clip_schemas)
            or not all(schema_clip_ids)
            or len(set(schema_clip_ids)) != len(schema_clip_ids)
            or any(
                schema.get("model_error")
                or schema.get("producer") != "qwen_clip_schema"
                or not schema.get("model")
                for schema in clip_schemas
                if isinstance(schema, Mapping)
            )
        ):
            invalid_perception_schemas.append(video_id or str(path))
        if isinstance(clip_schemas, list) and any(
            not isinstance(schema.get("llm_usage"), Mapping)
            or schema.get("model") != expected_clip_model
            or (
                schema.get("schema_attempt_context") == "query_time_anchor_repass"
                and (
                    int(schema.get("request_frames") or 0) != expected_anchor_repass_frames
                    or int((schema.get("llm_usage") or {}).get("sampled_frame_count") or 0)
                    != expected_anchor_repass_frames
                )
            )
            or (
                schema.get("schema_attempt_context") != "query_time_anchor_repass"
                and not 1
                <= int((schema.get("llm_usage") or {}).get("sampled_frame_count") or 0)
                <= expected_sampled_frames
            )
            or int((schema.get("llm_usage") or {}).get("max_tokens") or 0)
            != expected_max_tokens
            for schema in clip_schemas
            if isinstance(schema, Mapping) and not schema.get("model_error")
        ):
            inconsistent_perception_configs.append(video_id or str(path))
        if metadata.get("l1_perception_protocol") != EXPECTED_L1_PERCEPTION_PROTOCOL:
            inconsistent_perception_protocols.append(video_id or str(path))
        if video_id in observed:
            duplicates.append(video_id)
        else:
            observed[video_id] = path
        candidates = row.get("evidence_candidates") or []
        if not isinstance(candidates, list) or not candidates:
            empty_catalogs.append(video_id or str(path))
            candidates = []
        row_caption_spans = 0
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                invalid_contract.append(str(path))
                continue
            source_type = str(candidate.get("source_type") or "")
            source_types[source_type] += 1
            row_caption_spans += int(source_type == "caption_span")
            if (
                source_type in FORBIDDEN_VISIBLE_SOURCE_TYPES
                or candidate.get("trust_level") == "gold"
            ):
                forbidden_visible.append(f"{video_id}:{source_type}")
        if row_caption_spans == 0:
            files_without_caption_spans.append(video_id or str(path))
        file_records.append((video_id or str(path), _sha256(path)))
    observed_ids = set(observed)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    l1_set_digest = hashlib.sha256()
    for identity, digest in sorted(file_records):
        l1_set_digest.update(identity.encode("utf-8"))
        l1_set_digest.update(b"\0")
        l1_set_digest.update(digest.encode("ascii"))
        l1_set_digest.update(b"\n")
    checks = {
        "manifest_has_expected_official_test_videos": len(expected_ids) == expected_count,
        "one_l1_per_expected_video": len(file_records) == len(expected_ids),
        "exact_video_coverage": not missing and not unexpected,
        "no_duplicate_videos": not duplicates,
        "test_video_only_contract": not invalid_contract,
        "nonempty_visible_catalogs": not empty_catalogs,
        "every_video_has_model_caption_spans": not files_without_caption_spans,
        "every_fine_clip_has_perception_schema": not incomplete_perception,
        "all_perception_schemas_are_valid_qwen_outputs": not invalid_perception_schemas,
        "uniform_frozen_perception_config": not inconsistent_perception_configs,
        "uniform_frozen_perception_protocol": not inconsistent_perception_protocols,
        "no_hidden_gold_in_visible_candidates": not forbidden_visible,
    }
    return {
        "schema_version": "video-skills/vh-heldout-l1-audit-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_videos": len(expected_ids),
        "observed_files": len(file_records),
        "observed_videos": len(observed_ids),
        "missing_video_ids": missing,
        "unexpected_video_ids": unexpected,
        "duplicate_video_ids": sorted(set(duplicates)),
        "invalid_contract_files": sorted(set(invalid_contract)),
        "empty_catalogs": sorted(set(empty_catalogs)),
        "files_without_caption_spans": sorted(set(files_without_caption_spans)),
        "incomplete_perception": sorted(set(incomplete_perception)),
        "invalid_perception_schemas": sorted(set(invalid_perception_schemas)),
        "inconsistent_perception_configs": sorted(set(inconsistent_perception_configs)),
        "inconsistent_perception_protocols": sorted(set(inconsistent_perception_protocols)),
        "forbidden_visible_candidates": sorted(set(forbidden_visible)),
        "visible_source_type_counts": dict(sorted(source_types.items())),
        "l1_set_sha256": l1_set_digest.hexdigest(),
        "split_manifest_content_hash": manifest.get("manifest_hash"),
        "expected_perception_config": {
            "model": expected_clip_model,
            "producer": "qwen_clip_schema",
            "base_sampled_frame_count_range": [1, expected_sampled_frames],
            "anchor_repass": {
                "schema_attempt_context": "query_time_anchor_repass",
                "request_frames": expected_anchor_repass_frames,
                "sampled_frame_count": expected_anchor_repass_frames,
            },
            "max_tokens": expected_max_tokens,
            "l1_perception_protocol": EXPECTED_L1_PERCEPTION_PROTOCOL,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--frozen-l1-glob", action="append", required=True)
    parser.add_argument("--expected-count", type=int, default=270)
    parser.add_argument("--expected-clip-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--expected-sampled-frames", type=int, default=4)
    parser.add_argument("--expected-anchor-repass-frames", type=int, default=6)
    parser.add_argument("--expected-max-tokens", type=int, default=1600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        {Path(value) for pattern in args.frozen_l1_glob for value in glob.glob(pattern, recursive=True)}
    )
    rows = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    report = audit(
        json.loads(args.split_manifest.read_text(encoding="utf-8")),
        rows,
        expected_count=args.expected_count,
        expected_clip_model=args.expected_clip_model,
        expected_sampled_frames=args.expected_sampled_frames,
        expected_anchor_repass_frames=args.expected_anchor_repass_frames,
        expected_max_tokens=args.expected_max_tokens,
    )
    report["split_manifest_sha256"] = _sha256(args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

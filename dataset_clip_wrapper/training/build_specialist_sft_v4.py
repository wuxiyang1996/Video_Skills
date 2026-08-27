#!/usr/bin/env python3
"""Rebuild five_lora SFT package from v3, keeping only sft_seed / dev_tune videos.

Does not mutate the v3 artifact. Re-buckets rows by split-manifest role:
  train.jsonl <- role == sft_seed
  dev.jsonl   <- role == dev_tune

Motif rows without a resolvable video key are kept in train (bank-level).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .build_split_manifest import DEFAULT_DATASET_ROOT, EVALUATION_ONLY_DATASETS, role_lookup
from .evaluate_sft_package_gates import (
    SPECIALISTS,
    _dataset,
    _video_key,
    build_example_to_video_lookup,
)
from .sft_common import read_json, read_jsonl, write_json


def _stamp_row(row: dict[str, Any], *, video_key: str | None, role: str | None) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata") or {}) if isinstance(out.get("metadata"), dict) else {}
    if video_key:
        metadata["video_key"] = video_key
        if ":" in video_key:
            metadata["video_id"] = video_key.split(":", 1)[1]
    if role:
        metadata["split_role"] = role
    out["metadata"] = metadata
    if role:
        out["split_role"] = role
    if video_key:
        out["video_key"] = video_key
    return out


def filter_specialist(
    source_dir: Path,
    *,
    specialist: str,
    role_map: dict[str, str],
    example_to_video: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "dev"):
        path = source_dir / specialist / f"{split_name}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))

    kept_train: list[dict[str, Any]] = []
    kept_dev: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()

    for row in rows:
        dataset = _dataset(row)
        if dataset in EVALUATION_ONLY_DATASETS:
            dropped["eval_only"] += 1
            continue
        video_key = _video_key(row, example_to_video=example_to_video)
        if video_key is None:
            if specialist == "motif":
                kept_train.append(_stamp_row(row, video_key=None, role="sft_seed"))
                dropped["motif_unknown_kept_train"] += 1
            else:
                dropped["unknown_video"] += 1
            continue
        role = role_map.get(video_key)
        if role is None:
            dropped["video_not_in_manifest"] += 1
            continue
        if role == "sft_seed":
            kept_train.append(_stamp_row(row, video_key=video_key, role=role))
        elif role == "dev_tune":
            kept_dev.append(_stamp_row(row, video_key=video_key, role=role))
        else:
            dropped[f"role:{role}"] += 1

    return {
        "specialist": specialist,
        "source_rows": len(rows),
        "train_rows": len(kept_train),
        "dev_rows": len(kept_dev),
        "dropped": dict(dropped),
        "train": kept_train,
        "dev": kept_dev,
    }


def build_specialist_sft_v4(
    *,
    source_root: Path,
    output_root: Path,
    split_manifest_path: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    manifest = read_json(split_manifest_path)
    role_map = role_lookup(manifest)
    example_to_video = build_example_to_video_lookup(dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)

    specialist_summaries = []
    for specialist in SPECIALISTS:
        result = filter_specialist(
            source_root,
            specialist=specialist,
            role_map=role_map,
            example_to_video=example_to_video,
        )
        dest = output_root / specialist
        dest.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "dev"):
            path = dest / f"{split_name}.jsonl"
            rows = result[split_name]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
        summary = {
            "specialist": specialist,
            "source_rows": result["source_rows"],
            "train_rows": result["train_rows"],
            "dev_rows": result["dev_rows"],
            "dropped": result["dropped"],
        }
        write_json(dest / "filter_summary.json", summary)
        specialist_summaries.append(summary)

    package = {
        "schema_version": "video-skills/specialist-sft-v4",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_hash": manifest.get("manifest_hash"),
        "specialists": specialist_summaries,
    }
    write_json(output_root / "package_build_report.json", package)
    return package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v3_20260722/five_lora"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v4/five_lora"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json"),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args(argv)

    report = build_specialist_sft_v4(
        source_root=args.source_root,
        output_root=args.output_root,
        split_manifest_path=args.split_manifest,
        dataset_root=args.dataset_root,
    )
    print(json.dumps({"ok": True, "specialists": report["specialists"]}, indent=2))
    empty = [s["specialist"] for s in report["specialists"] if s["train_rows"] == 0 or s["dev_rows"] == 0]
    if empty:
        print(json.dumps({"warning_empty_splits": empty}, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze source-video roles for SFT / OPD / GRPO / held-out evaluation.

Roles are assigned at the source-video level so no QA, L1 cache, trajectory, or
transition from the same video can leak across post-training stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .sft_common import write_json


ROLES = ("sft_seed", "opd_pool", "grpo_pool", "dev_tune", "heldout_test")
EVALUATION_ONLY_DATASETS = ("vrbench", "videomme", "ovo_bench", "streaming_bench")
DEFAULT_SALT = "video-skills-split-manifest-v1"
DEFAULT_DATASET_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets")


def _stable_bucket(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _role_from_bucket(
    bucket: float,
    *,
    heldout_fraction: float,
    sft_fraction: float,
    opd_fraction: float,
    grpo_fraction: float,
    force_heldout: bool = False,
) -> str:
    if force_heldout:
        return "heldout_test"
    if bucket < heldout_fraction:
        return "heldout_test"
    rem = 1.0 - heldout_fraction
    if rem <= 0:
        return "heldout_test"
    local = (bucket - heldout_fraction) / rem
    cut_sft = sft_fraction
    cut_opd = cut_sft + opd_fraction
    cut_grpo = cut_opd + grpo_fraction
    if local < cut_sft:
        return "sft_seed"
    if local < cut_opd:
        return "opd_pool"
    if local < cut_grpo:
        return "grpo_pool"
    return "dev_tune"


def _load_video_holmes(dataset_root: Path) -> dict[str, dict[str, Any]]:
    benchmark = dataset_root / "Video-Holmes" / "Benchmark"
    videos: dict[str, dict[str, Any]] = {}
    for split_name, path in (
        ("train", benchmark / "train_Video-Holmes.json"),
        ("test", benchmark / "test_Video-Holmes.json"),
    ):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            video_id = str(row["video ID"])
            key = f"video_holmes:{video_id}"
            entry = videos.setdefault(
                key,
                {
                    "dataset": "video_holmes",
                    "video_id": video_id,
                    "official_split": split_name,
                    "n_questions": 0,
                    "question_ids": [],
                },
            )
            # Official test always wins if a video appears in both (should not).
            if split_name == "test":
                entry["official_split"] = "test"
            entry["n_questions"] += 1
            qid = str(row.get("Question ID") or "")
            if qid:
                entry["question_ids"].append(qid)
    return videos


def _load_cg_bench(dataset_root: Path, *, use_full: bool = True) -> dict[str, dict[str, Any]]:
    bench = dataset_root / "CG-Bench"
    path = bench / ("cgbench.json" if use_full else "cgbench_mini.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    videos: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row["video_uid"])
        key = f"cg_bench:{video_id}"
        entry = videos.setdefault(
            key,
            {
                "dataset": "cg_bench",
                "video_id": video_id,
                "official_split": None,
                "n_questions": 0,
                "question_ids": [],
            },
        )
        entry["n_questions"] += 1
        qid = str(row.get("qid") or "")
        if qid:
            entry["question_ids"].append(qid)
    return videos


def build_split_manifest(
    dataset_root: Path,
    *,
    salt: str = DEFAULT_SALT,
    cg_heldout_fraction: float = 0.18,
    train_sft_fraction: float = 0.50,
    train_opd_fraction: float = 0.20,
    train_grpo_fraction: float = 0.20,
    use_full_cg: bool = True,
) -> dict[str, Any]:
    if abs((train_sft_fraction + train_opd_fraction + train_grpo_fraction) - 0.90) > 1e-6:
        # Remaining 10% of non-heldout videos become dev_tune by construction.
        raise ValueError(
            "train_sft_fraction + train_opd_fraction + train_grpo_fraction must equal 0.90; "
            "the residual 0.10 is reserved for dev_tune"
        )

    videos = {}
    videos.update(_load_video_holmes(dataset_root))
    videos.update(_load_cg_bench(dataset_root, use_full=use_full_cg))

    records: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    dataset_role_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for key in sorted(videos):
        meta = videos[key]
        dataset = meta["dataset"]
        video_id = meta["video_id"]
        force_heldout = dataset == "video_holmes" and meta.get("official_split") == "test"
        heldout_fraction = 0.0 if dataset == "video_holmes" else cg_heldout_fraction
        # Video-Holmes train videos never enter heldout_test via hash; test is official-only.
        bucket = _stable_bucket(key, salt)
        role = _role_from_bucket(
            bucket,
            heldout_fraction=heldout_fraction,
            sft_fraction=train_sft_fraction,
            opd_fraction=train_opd_fraction,
            grpo_fraction=train_grpo_fraction,
            force_heldout=force_heldout,
        )
        if dataset == "video_holmes" and meta.get("official_split") == "train" and role == "heldout_test":
            # Defensive: VH train must stay in train-side roles.
            role = "sft_seed"
        record = {
            "key": key,
            "dataset": dataset,
            "video_id": video_id,
            "role": role,
            "official_split": meta.get("official_split"),
            "n_questions": meta["n_questions"],
            "question_ids": sorted(set(meta["question_ids"])),
        }
        records.append(record)
        role_counts[role] += 1
        dataset_role_counts[dataset][role] += 1

    payload = {
        "schema_version": "video-skills/split-manifest-v1",
        "salt": salt,
        "dataset_root": str(dataset_root),
        "roles": list(ROLES),
        "evaluation_only_datasets": list(EVALUATION_ONLY_DATASETS),
        "assignment": {
            "cg_heldout_fraction": cg_heldout_fraction,
            "train_side_fractions": {
                "sft_seed": train_sft_fraction,
                "opd_pool": train_opd_fraction,
                "grpo_pool": train_grpo_fraction,
                "dev_tune": round(1.0 - train_sft_fraction - train_opd_fraction - train_grpo_fraction, 4),
            },
            "video_holmes_test": "always heldout_test",
            "video_holmes_train": "hash into sft_seed/opd_pool/grpo_pool/dev_tune only",
            "cg_bench": "hash into heldout_test + train-side roles",
        },
        "summary": {
            "n_videos": len(records),
            "role_video_counts": dict(role_counts),
            "dataset_role_video_counts": {
                dataset: dict(counts) for dataset, counts in sorted(dataset_role_counts.items())
            },
            "n_questions": sum(row["n_questions"] for row in records),
        },
        "videos": records,
    }
    digest = hashlib.sha256(
        json.dumps(
            {
                "salt": salt,
                "assignment": payload["assignment"],
                "videos": [(row["key"], row["role"], row["n_questions"]) for row in records],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["manifest_hash"] = digest

    # Hard invariants.
    by_key = {row["key"]: row["role"] for row in records}
    assert len(by_key) == len(records)
    vh_test = [row for row in records if row["dataset"] == "video_holmes" and row["official_split"] == "test"]
    if any(row["role"] != "heldout_test" for row in vh_test):
        raise RuntimeError("Video-Holmes official test videos must all be heldout_test")
    vh_train = [row for row in records if row["dataset"] == "video_holmes" and row["official_split"] == "train"]
    if any(row["role"] == "heldout_test" for row in vh_train):
        raise RuntimeError("Video-Holmes official train videos must not be heldout_test")
    return payload


def role_lookup(manifest: dict[str, Any]) -> dict[str, str]:
    return {row["key"]: row["role"] for row in manifest.get("videos") or []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json"),
    )
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--cg-heldout-fraction", type=float, default=0.18)
    parser.add_argument("--train-sft-fraction", type=float, default=0.50)
    parser.add_argument("--train-opd-fraction", type=float, default=0.20)
    parser.add_argument("--train-grpo-fraction", type=float, default=0.20)
    parser.add_argument("--use-cg-mini", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_split_manifest(
        args.dataset_root,
        salt=args.salt,
        cg_heldout_fraction=args.cg_heldout_fraction,
        train_sft_fraction=args.train_sft_fraction,
        train_opd_fraction=args.train_opd_fraction,
        train_grpo_fraction=args.train_grpo_fraction,
        use_full_cg=not args.use_cg_mini,
    )
    write_json(args.output, manifest)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "manifest_hash": manifest["manifest_hash"],
            "output": str(args.output),
            "summary": manifest["summary"],
            "assignment": manifest["assignment"],
        },
    )
    print(json.dumps({"ok": True, "output": str(args.output), "summary": manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

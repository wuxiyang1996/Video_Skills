"""Freeze post-training run contracts for OPD / GRPO reproducibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class PosttrainingRunManifest:
    schema_version: str = "video-skills/posttraining-run-manifest-v1"
    stage: str = "grpo"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    split_manifest_path: str = ""
    split_manifest_hash: str | None = None
    reward_spec_version: str = "video-skills/verified-reward-v2"
    judge_model: str | None = None
    judge_rubric_version: str | None = None
    teacher_model: str | None = None
    policy_checkpoint: str | None = None
    reference_checkpoint: str | None = None
    motif_bank_path: str | None = None
    grpo_mode: str = "l2_repair"
    update_modules: list[str] = field(default_factory=lambda: ["l2", "repair"])
    k_samples: int = 4
    candidate_order_seeds: list[int] = field(default_factory=lambda: [7, 99, 13, 42])
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("created_at", None)
        return _sha256_json(payload)


def build_posttraining_manifest(
    *,
    stage: str,
    split_manifest_path: str | Path,
    reward_spec_version: str,
    grpo_mode: str = "l2_repair",
    update_modules: list[str] | None = None,
    judge_model: str | None = None,
    judge_rubric_version: str | None = None,
    teacher_model: str | None = None,
    policy_checkpoint: str | None = None,
    reference_checkpoint: str | None = None,
    motif_bank_path: str | None = None,
    k_samples: int = 4,
    candidate_order_seeds: list[int] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> PosttrainingRunManifest:
    split_path = Path(split_manifest_path)
    return PosttrainingRunManifest(
        stage=stage,
        split_manifest_path=str(split_path),
        split_manifest_hash=_sha256_file(split_path),
        reward_spec_version=reward_spec_version,
        judge_model=judge_model,
        judge_rubric_version=judge_rubric_version,
        teacher_model=teacher_model,
        policy_checkpoint=policy_checkpoint,
        reference_checkpoint=reference_checkpoint,
        motif_bank_path=str(motif_bank_path) if motif_bank_path else None,
        grpo_mode=grpo_mode,
        update_modules=list(update_modules or ["l2", "repair"]),
        k_samples=int(k_samples),
        candidate_order_seeds=list(candidate_order_seeds or [7, 99, 13, 42]),
        extras=dict(extras or {}),
    )


def save_posttraining_manifest(path: str | Path, manifest: PosttrainingRunManifest) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    payload["content_hash"] = manifest.content_hash()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out

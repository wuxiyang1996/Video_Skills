#!/usr/bin/env python3
"""Interpolate two compatible LoRA adapters with provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


WEIGHT_NAME = "adapter_model.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate(base_dir: Path, tuned_dir: Path, output_dir: Path, alpha: float) -> dict:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    base_weight = base_dir / WEIGHT_NAME
    tuned_weight = tuned_dir / WEIGHT_NAME
    if output_dir.resolve() in {base_dir.resolve(), tuned_dir.resolve()}:
        raise ValueError("output_dir must differ from both input adapters")

    base = load_file(str(base_weight), device="cpu")
    tuned = load_file(str(tuned_weight), device="cpu")
    if base.keys() != tuned.keys():
        missing = sorted(base.keys() - tuned.keys())
        extra = sorted(tuned.keys() - base.keys())
        raise ValueError(f"adapter keys differ: missing={missing[:5]}, extra={extra[:5]}")

    mixed: dict[str, torch.Tensor] = {}
    for key in base:
        if base[key].shape != tuned[key].shape:
            raise ValueError(
                f"shape mismatch for {key}: {tuple(base[key].shape)} != {tuple(tuned[key].shape)}"
            )
        mixed[key] = base[key].lerp(tuned[key], alpha)

    output_dir.mkdir(parents=True, exist_ok=False)
    for source in sorted(base_dir.iterdir()):
        if source.is_file() and source.name != WEIGHT_NAME:
            shutil.copy2(source, output_dir / source.name)
    output_weight = output_dir / WEIGHT_NAME
    save_file(mixed, str(output_weight), metadata={"format": "pt"})
    report = {
        "schema_version": "video-skills/lora-trust-region-interpolation-v1",
        "base_adapter": str(base_dir.resolve()),
        "base_weight_sha256": _sha256(base_weight),
        "tuned_adapter": str(tuned_dir.resolve()),
        "tuned_weight_sha256": _sha256(tuned_weight),
        "alpha_tuned": alpha,
        "output_adapter": str(output_dir.resolve()),
        "output_weight_sha256": _sha256(output_weight),
        "tensor_count": len(mixed),
    }
    (output_dir / "interpolation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--tuned-adapter", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--alpha-tuned", type=float, required=True)
    args = parser.parse_args()
    report = interpolate(
        args.base_adapter, args.tuned_adapter, args.output_adapter, args.alpha_tuned
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

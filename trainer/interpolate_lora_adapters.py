"""Interpolate an OPD-updated LoRA back toward its SFT initialization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from trainer.artifact_hash import adapter_weight_sha256


WEIGHTS = "adapter_model.safetensors"


def interpolate_adapters(
    base_adapter: Path, tuned_adapter: Path, output_dir: Path, *, alpha: float
) -> dict[str, object]:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = load_file(base_adapter / WEIGHTS, device="cpu")
    tuned = load_file(tuned_adapter / WEIGHTS, device="cpu")
    if base.keys() != tuned.keys():
        missing = sorted(base.keys() - tuned.keys())
        extra = sorted(tuned.keys() - base.keys())
        raise ValueError(f"adapter tensor keys differ: missing={missing[:3]} extra={extra[:3]}")
    output: dict[str, torch.Tensor] = {}
    squared_delta = 0.0
    for key in base:
        if base[key].shape != tuned[key].shape:
            raise ValueError(f"shape mismatch for {key}: {base[key].shape} != {tuned[key].shape}")
        base_float = base[key].float()
        delta = tuned[key].float() - base_float
        squared_delta += float(torch.sum(delta * delta))
        output[key] = (base_float + float(alpha) * delta).to(base[key].dtype).contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(output, output_dir / WEIGHTS, metadata={"format": "pt"})
    for source_name in (
        "adapter_config.json", "README.md", "chat_template.jinja",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    ):
        source = tuned_adapter / source_name
        if source.is_file():
            shutil.copy2(source, output_dir / source_name)
    report = {
        "schema_version": "video-skills/lora-parameter-interpolation-v0.1",
        "base_adapter": str(base_adapter),
        "base_adapter_weight_sha256": adapter_weight_sha256(base_adapter),
        "tuned_adapter": str(tuned_adapter),
        "tuned_adapter_weight_sha256": adapter_weight_sha256(tuned_adapter),
        "output_adapter": str(output_dir),
        "output_adapter_weight_sha256": adapter_weight_sha256(output_dir),
        "alpha": float(alpha),
        "tensor_count": len(output),
        "parameter_delta_l2": squared_delta ** 0.5,
    }
    (output_dir / "interpolation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--tuned-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    report = interpolate_adapters(
        args.base_adapter, args.tuned_adapter, args.output_dir, alpha=args.alpha
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.posttraining.interpolate_lora_adapters import interpolate


def _adapter(path: Path, value: float) -> Path:
    path.mkdir()
    save_file({"layer.lora_A.weight": torch.tensor([value, value + 1])}, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text('{"r": 1}\n', encoding="utf-8")
    return path


def test_interpolate_adapter_records_provenance_and_copies_config(tmp_path: Path) -> None:
    base = _adapter(tmp_path / "base", 0.0)
    tuned = _adapter(tmp_path / "tuned", 2.0)
    output = tmp_path / "output"
    report = interpolate(base, tuned, output, 0.25)
    tensor = load_file(str(output / "adapter_model.safetensors"))["layer.lora_A.weight"]
    assert torch.allclose(tensor, torch.tensor([0.5, 1.5]))
    assert json.loads((output / "adapter_config.json").read_text())["r"] == 1
    assert report["alpha_tuned"] == 0.25
    assert report["tensor_count"] == 1
    assert report["output_weight_sha256"]


def test_interpolate_adapter_rejects_unsafe_output_and_alpha(tmp_path: Path) -> None:
    base = _adapter(tmp_path / "base", 0.0)
    tuned = _adapter(tmp_path / "tuned", 2.0)
    with pytest.raises(ValueError, match="alpha"):
        interpolate(base, tuned, tmp_path / "bad", 1.1)
    with pytest.raises(ValueError, match="output_dir"):
        interpolate(base, tuned, base, 0.5)

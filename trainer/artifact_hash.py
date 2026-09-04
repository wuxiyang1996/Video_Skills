"""Stable hashes for adapter provenance in paper reports."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_weight_sha256(adapter_dir: str | Path) -> str | None:
    root = Path(adapter_dir)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = root / name
        if path.is_file():
            return sha256_file(path)
    return None

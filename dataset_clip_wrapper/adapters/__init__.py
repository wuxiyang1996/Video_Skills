"""Dataset adapter registry."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter
from .cg_bench import CGBenchAdapter
from .siv_bench import SIVBenchAdapter
from .video_holmes import VideoHolmesAdapter
from .vrbench import VRBenchAdapter

from ..schemas import DatasetName


def get_adapter(dataset: DatasetName, dataset_root: Path, split: str = "train") -> DatasetAdapter:
    if dataset == "video_holmes":
        return VideoHolmesAdapter(dataset_root, split=split)
    if dataset == "cg_bench":
        return CGBenchAdapter(dataset_root, split=split)
    if dataset == "vrbench":
        return VRBenchAdapter(dataset_root, split=split)
    if dataset == "siv_bench":
        return SIVBenchAdapter(dataset_root, split=split)
    raise ValueError(f"unsupported dataset: {dataset}")

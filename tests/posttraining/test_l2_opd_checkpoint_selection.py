from __future__ import annotations

import json
from pathlib import Path

from scripts.eval.select_l2_opd_checkpoint import main


def _report(path: Path, *, dataset: str, recall: float, hit: float, segment: float = 0, inference: float = 0, relationship: float = 0) -> None:
    topk = "pointwise_top2" if dataset == "cg_bench" else "pointwise_top4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": {
        topk: {"mean_recall": recall, "hit_rate": hit},
        "dataset_metrics": {dataset: {"process_metrics": {
            "segment_recall": segment,
            "inference_shot_recall": inference,
            "relationship_support": relationship,
        }}},
    }}))


def test_selector_chooses_smallest_alpha_passing_all_frozen_gates(tmp_path: Path) -> None:
    sft_cg, sft_vh = tmp_path / "sft_cg.json", tmp_path / "sft_vh.json"
    _report(sft_cg, dataset="cg_bench", recall=0.60, hit=0.64)
    _report(sft_vh, dataset="video_holmes", recall=0.04, hit=0.50, segment=0.50, inference=0.02, relationship=0.39)
    candidates = []
    for name, alpha, gains in (
        ("a25", 0.25, (0.51, 0.03, 0.40)),
        ("a50", 0.50, (0.55, 0.04, 0.42)),
        ("a75", 0.75, (0.58, 0.05, 0.44)),
    ):
        adapter = tmp_path / name / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(name.encode())
        cg, vh = tmp_path / name / "cg.json", tmp_path / name / "vh.json"
        _report(cg, dataset="cg_bench", recall=0.60, hit=0.64)
        _report(vh, dataset="video_holmes", recall=0.05, hit=0.50, segment=gains[0], inference=gains[1], relationship=gains[2])
        candidates.extend(["--candidate", f"{name}|{alpha}|{adapter}|{cg}|{vh}"])
    output = tmp_path / "selection.json"
    assert main(["--sft-cg-report", str(sft_cg), "--sft-vh-report", str(sft_vh), *candidates, "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["selected"]["name"] == "a25"

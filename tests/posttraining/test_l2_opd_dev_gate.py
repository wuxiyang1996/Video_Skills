from __future__ import annotations

from scripts.eval.gate_l2_opd_dev import gate_reports


def _report(dataset: str, recall: float, hit: float, process: dict[str, float], top_k: int) -> dict:
    return {
        "top_k": top_k,
        "metrics": {
            f"pointwise_top{top_k}": {"mean_recall": recall, "hit_rate": hit},
            "dataset_metrics": {dataset: {"process_metrics": process}},
        },
    }


def test_opd_gate_requires_both_dataset_gain_and_cg_threshold() -> None:
    sft_cg = _report("cg_bench", 0.60, 0.60, {"clue_recall": 0.60}, 2)
    opd_cg = _report("cg_bench", 0.65, 0.65, {"clue_recall": 0.65}, 2)
    sft_vh = _report("video_holmes", 0.20, 0.60, {"inference_shot_recall": 0.10, "relationship_support": 0.30}, 4)
    opd_vh = _report("video_holmes", 0.25, 0.70, {"inference_shot_recall": 0.15, "relationship_support": 0.35}, 4)
    assert gate_reports(sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh)["passed"]
    regressed = _report("video_holmes", 0.10, 0.50, {"inference_shot_recall": 0.05, "relationship_support": 0.20}, 4)
    assert not gate_reports(sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=regressed)["passed"]


def test_opd_gate_accepts_legacy_cg_report_without_process_metrics() -> None:
    sft_cg = _report("cg_bench", 0.60, 0.60, {}, 2)
    opd_cg = _report("cg_bench", 0.65, 0.65, {"clue_recall": 0.65}, 2)
    sft_vh = _report("video_holmes", 0.20, 0.60, {"inference_shot_recall": 0.10, "relationship_support": 0.30}, 4)
    opd_vh = _report("video_holmes", 0.25, 0.70, {"inference_shot_recall": 0.15, "relationship_support": 0.35}, 4)
    result = gate_reports(sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh)
    assert result["passed"]
    assert "cg_clue_recall" not in result["gains"]
    assert sorted(result["compared_process_metrics"]) == [
        "vh_inference_shot_recall", "vh_relationship_support"
    ]


def test_low_sample_opd_gate_counts_vh_segment_recall_gain() -> None:
    sft_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    opd_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    sft_process = {
        "segment_recall": 0.40,
        "inference_shot_recall": 0.10,
        "relationship_support": 0.30,
    }
    opd_process = {**sft_process, "segment_recall": 0.42}
    sft_vh = _report("video_holmes", 0.20, 0.60, sft_process, 4)
    opd_vh = _report("video_holmes", 0.20, 0.60, opd_process, 4)
    result = gate_reports(
        sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh
    )
    assert result["passed"]
    assert result["gains"]["vh_segment_recall"] > 0.0


def test_opd_gate_rejects_mismatched_topk_and_boundary_contracts() -> None:
    sft_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    opd_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    sft_vh = _report("video_holmes", 0.20, 0.60, {"segment_recall": 0.4}, 4)
    opd_vh = _report("video_holmes", 0.21, 0.61, {"segment_recall": 0.5}, 2)
    sft_cg["boundary_anchor_index0"] = True
    opd_cg["boundary_anchor_index0"] = False
    result = gate_reports(
        sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh
    )
    assert not result["passed"]
    assert not result["checks"]["cg_same_boundary_anchor_contract"]
    assert not result["checks"]["vh_same_top_k_contract"]


def test_opd_gate_can_use_separate_cg_process_dev_report() -> None:
    sft_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    opd_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    sft_process = _report("cg_bench", 0.50, 0.50, {"clue_recall": 0.50, "clue_mean_best_iou": 0.12}, 2)
    opd_process = _report("cg_bench", 0.50, 0.50, {"clue_recall": 0.50, "clue_mean_best_iou": 0.14}, 2)
    sft_vh = _report("video_holmes", 0.20, 0.60, {"inference_shot_recall": 0.10, "relationship_support": 0.30}, 4)
    opd_vh = _report("video_holmes", 0.21, 0.60, {"inference_shot_recall": 0.10, "relationship_support": 0.31}, 4)
    result = gate_reports(
        sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh,
        sft_cg_process=sft_process, opd_cg_process=opd_process,
    )
    assert result["passed"]
    assert result["gains"]["cg_clue_mean_best_iou"] > 0


def test_low_sample_opd_gate_allows_cg_preservation_but_optional_strict_mode_does_not() -> None:
    sft_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    opd_cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    sft_vh = _report("video_holmes", 0.20, 0.60, {"inference_shot_recall": 0.10, "relationship_support": 0.30}, 4)
    opd_vh = _report("video_holmes", 0.20, 0.60, {"inference_shot_recall": 0.12, "relationship_support": 0.31}, 4)
    assert gate_reports(sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh)["passed"]
    assert not gate_reports(
        sft_cg=sft_cg, opd_cg=opd_cg, sft_vh=sft_vh, opd_vh=opd_vh,
        require_cg_strict_gain=True,
    )["passed"]


def test_post_opd_gate_can_require_preservation_without_another_vh_gain() -> None:
    cg = _report("cg_bench", 0.61, 0.64, {"clue_recall": 0.60}, 2)
    vh = _report(
        "video_holmes", 0.20, 0.60,
        {"segment_recall": 0.40, "inference_shot_recall": 0.10, "relationship_support": 0.30},
        4,
    )
    strict = gate_reports(sft_cg=cg, opd_cg=cg, sft_vh=vh, opd_vh=vh)
    preservation = gate_reports(
        sft_cg=cg, opd_cg=cg, sft_vh=vh, opd_vh=vh,
        require_vh_strict_gain=False,
    )
    assert not strict["passed"]
    assert preservation["passed"]
    assert preservation["thresholds"]["require_vh_strict_gain"] is False
    assert "vh_strict_gain_exists_for_low_sample_opd" not in preservation["checks"]


def test_post_opd_gate_still_rejects_any_vh_regression() -> None:
    cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    vh = _report("video_holmes", 0.20, 0.60, {"segment_recall": 0.40}, 4)
    regressed = _report("video_holmes", 0.20, 0.60, {"segment_recall": 0.39}, 4)
    result = gate_reports(
        sft_cg=cg, opd_cg=cg, sft_vh=vh, opd_vh=regressed,
        require_vh_strict_gain=False,
    )
    assert not result["passed"]
    assert not result["checks"]["vh_segment_recall_not_below_sft"]


def test_vh_process_primary_gate_reports_headline_drop_but_uses_evidence_contract() -> None:
    cg = _report("cg_bench", 0.61, 0.64, {}, 2)
    sft_vh = _report(
        "video_holmes", 0.20, 0.60,
        {"segment_recall": 0.40, "inference_shot_recall": 0.10, "relationship_support": 0.30},
        4,
    )
    opd_vh = _report(
        "video_holmes", 0.19, 0.55,
        {"segment_recall": 0.45, "inference_shot_recall": 0.11, "relationship_support": 0.31},
        4,
    )
    strict = gate_reports(sft_cg=cg, opd_cg=cg, sft_vh=sft_vh, opd_vh=opd_vh)
    process = gate_reports(
        sft_cg=cg, opd_cg=cg, sft_vh=sft_vh, opd_vh=opd_vh,
        vh_process_primary=True,
    )
    assert not strict["passed"]
    assert process["passed"]
    assert process["gains"]["vh_mean_recall"] < 0
    assert process["thresholds"]["vh_process_primary"] is True
    assert "vh_recall_not_below_sft" not in process["checks"]

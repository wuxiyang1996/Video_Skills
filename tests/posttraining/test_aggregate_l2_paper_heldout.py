from scripts.eval.aggregate_l2_paper_heldout import (
    EXPECTED_MODELS,
    aggregate_heldout,
    expected_hashes_from_references,
)


def _report(model: str, dataset: str) -> dict:
    top_k = 2 if dataset == "cg_bench" else 4
    process = (
        {"clue_recall": 0.6, "evidence_precision": 0.3, "clue_mean_best_iou": 0.1}
        if dataset == "cg_bench"
        else {"segment_recall": 0.5, "segment_precision": 0.9,
              "inference_shot_recall": 0.1, "relationship_support": 0.4}
    )
    return {
        "adapter_weight_sha256": f"hash-{model}",
        "top_k": top_k,
        "boundary_anchor_index0": dataset == "cg_bench",
        "candidate_rows": 100,
        "input_rows": 100,
        "input_examples": 20,
        "input_split_roles": ["heldout_test"],
        "input_datasets": [dataset],
        "evaluation_jsonl_sha256": f"input-{dataset}",
        "metrics": {
            f"pointwise_top{top_k}": {"mean_recall": 0.6, "hit_rate": 0.7},
            "dataset_metrics": {dataset: {"process_metrics": process}},
        },
    }


def _matrix() -> tuple[dict, dict]:
    reports = {
        model: {dataset: _report(model, dataset) for dataset in ("cg_bench", "video_holmes")}
        for model in EXPECTED_MODELS
    }
    expected = {model: f"hash-{model}" for model in EXPECTED_MODELS}
    return reports, expected


def test_heldout_aggregate_accepts_complete_frozen_matrix() -> None:
    reports, expected = _matrix()
    result = aggregate_heldout(reports, expected)
    assert result["passed"] is True
    assert result["performance_is_not_a_release_gate"] is True
    assert result["grpo_three_seed"]["cg_bench"]["mean_recall"]["mean"] == 0.6


def test_heldout_aggregate_rejects_dev_or_mismatched_input() -> None:
    reports, expected = _matrix()
    reports["grpo_seed44"]["video_holmes"]["input_split_roles"] = ["dev_tune"]
    reports["grpo_seed43"]["cg_bench"]["evaluation_jsonl_sha256"] = "other"
    result = aggregate_heldout(reports, expected)
    assert result["passed"] is False
    assert result["integrity_checks"]["same_cg_bench_input_hash"] is False
    assert result["models"][-1]["checks"]["heldout_split_role"] is False


def test_expected_hashes_come_from_training_chain() -> None:
    hashes = expected_hashes_from_references(
        {"adapter_weight_sha256": "sft"},
        {"selected": {"adapter_weight_sha256": "opd"}},
        {"seeds": [
            {"seed": seed, "trained_adapter": {"adapter_weight_sha256": f"g{seed}"}}
            for seed in (42, 43, 44)
        ]},
    )
    assert hashes == {
        "sft": "sft", "opd_alpha075": "opd",
        "grpo_seed42": "g42", "grpo_seed43": "g43", "grpo_seed44": "g44",
    }

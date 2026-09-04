import json

from scripts.eval.analyze_l2_paper_heldout_bootstrap import MODELS, analyze


def _report(dataset, model_index):
    value = 0.0 if model_index == 0 else 0.5 if model_index == 1 else 1.0
    process = (
        {
            "clue_recall": value,
            "evidence_precision": value,
            "clue_mean_best_iou": value,
        }
        if dataset == "cg_bench"
        else {
            "segment_recall": value,
            "segment_precision": value,
            "inference_shot_recall": value,
            "relationship_support": value,
        }
    )
    return {
        "evaluation_jsonl_sha256": f"frozen-{dataset}",
        "input_datasets": [dataset],
        "adapter_weight_sha256": f"adapter-{model_index}",
        "results": [
            {
                "example_id": f"{dataset}:{example}",
                "metrics": {"recall": value, "hit": bool(value)},
                "process_metrics": process,
            }
            for example in range(2)
        ],
    }


def test_paired_bootstrap_uses_matching_frozen_examples(tmp_path):
    for dataset in ("cg_bench", "video_holmes"):
        for model_index, model in enumerate(MODELS):
            path = tmp_path / "results" / model / dataset / "eval_report.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_report(dataset, model_index)), encoding="utf-8")

    result = analyze(tmp_path, samples=20, seed=7)

    assert result["passed"] is True
    for dataset in ("cg_bench", "video_holmes"):
        assert all(result["datasets"][dataset]["checks"].values())
        assert result["datasets"][dataset]["comparisons"]["opd_vs_sft"]["mean_recall"][
            "gain"
        ] == 0.5
        assert result["datasets"][dataset]["comparisons"]["grpo_mean_vs_opd"][
            "mean_recall"
        ]["ci95"] == [0.5, 0.5]

import json

from scripts.eval.finalize_l2_paper_artifacts import (
    MODEL_ORDER,
    SOURCE_NAMES,
    build_paper_artifacts,
)


def _sources(tmp_path):
    payloads = {name: {"schema_version": f"test/{name}", "passed": True} for name in SOURCE_NAMES}
    payloads["grpo_aggregate"].update({
        "seed_count": 3,
        "same_training_contracts": True,
        "seeds": [{"seed": seed, "passed": True} for seed in (42, 43, 44)],
        "metrics": {"cg_terminal_success_rate": {"mean": 0.2, "std": 0.0}},
    })
    payloads["reward_normalization"]["normalization_contract"] = (
        "dataset-homogeneous-group-mean-std-v1"
    )
    models = []
    for index, model in enumerate(MODEL_ORDER):
        score = 0.1 + index / 100
        models.append({
            "model": model,
            "adapter_weight_sha256": str(index + 1) * 64,
            "metrics": {
                "cg_bench": {"mean_recall": score, "hit_rate": score},
                "video_holmes": {
                    "segment_recall": score,
                    "inference_shot_recall": score,
                    "relationship_support": score,
                },
            },
        })
    payloads["heldout_aggregate"].update({
        "models": models,
        "integrity_checks": {
            "same_cg_bench_input_hash": True,
            "same_video_holmes_input_hash": True,
        },
        "grpo_three_seed": {
            dataset: {"mean_recall": {"mean": 0.2, "std": 0.01}}
            for dataset in ("cg_bench", "video_holmes")
        },
        "grpo_gains": {"cg_bench": {"mean_recall": {"vs_sft": 0.1}}},
    })
    sources = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sources[name] = (path, payload)
    return sources


def test_build_paper_artifacts_is_hash_pinned_and_three_seeded(tmp_path):
    report, rows, manifest = build_paper_artifacts(_sources(tmp_path))

    assert report["passed"] is True
    assert len(rows) == 7
    assert rows[-2]["model"] == "grpo_three_seed_mean"
    assert manifest["grpo_seeds"] == [42, 43, 44]
    assert report["interpretation"]["reward_normalization_scope"] == (
        "dataset-homogeneous-group-mean-std-v1"
    )
    assert all(len(source["sha256"]) == 64 for source in manifest["sources"].values())


def test_build_paper_artifacts_fails_closed_on_bad_source(tmp_path):
    sources = _sources(tmp_path)
    path, payload = sources["vh_l1_audit"]
    payload["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    report, _, _ = build_paper_artifacts(sources)

    assert report["passed"] is False
    assert report["integrity_checks"]["vh_l1_audit_passed"] is False

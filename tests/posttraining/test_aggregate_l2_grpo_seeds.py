from scripts.eval.aggregate_l2_grpo_seeds import aggregate_seed_runs


def _row(seed: int, rate: float = 0.30, groups_per_dataset: int = 70) -> dict:
    adapter_hash = f"hash-{seed}"
    dataset_metrics = {
        name: {"groups_seen": groups_per_dataset, "groups_trained": 30, "trainable_group_rate": rate}
        for name in ("cg_bench", "video_holmes")
    }
    return {
        "seed": seed,
        "train_report": f"train-{seed}",
        "terminal_report": f"terminal-{seed}",
        "train": {
            "artifact_status": "trained", "groups_seen": 2 * groups_per_dataset,
            "split_role": "grpo_pool", "split_manifest_sha256": "split-hash",
            "pool_filters": {
                "example_id_allowlist_sha256": "allowlist-hash",
                "exact_mined_group_allowlist": True,
                "preserve_allowlist_order": True,
                "dataset_balanced_sampling": True,
            },
            "trained_adapter_outputs": {"default": {"adapter_weight_sha256": adapter_hash}},
            "dataset_metrics": dataset_metrics,
            "controller_action_contract": "routed", "sampling_protocol": {"t": 0.9},
            "relationship_support_contract": "relv2", "reference_runtime_contract": "shared",
            "terminal_reward_contract": "dataset-aware-terminal-reward:repair-v1",
        },
        "cg_dev": {
            "adapter_weight_sha256": adapter_hash,
            "metrics": {"pointwise_top2": {"mean_recall": 0.61, "hit_rate": 0.64},
                        "dataset_metrics": {"cg_bench": {"process_metrics": {}}}},
        },
        "vh_dev": {
            "adapter_weight_sha256": adapter_hash,
            "metrics": {"pointwise_top4": {"mean_recall": 0.1, "hit_rate": 0.5},
                        "dataset_metrics": {"video_holmes": {"process_metrics": {
                            "segment_recall": 0.55, "inference_shot_recall": 0.05,
                            "relationship_support": 0.42}}}},
        },
        "terminal": {
            "source_adapter_weight_sha256": adapter_hash,
            "dataset_metrics": {
                "cg_bench": {"terminal_success_rate": 0.25},
                "video_holmes": {"terminal_success_rate": 0.10},
            },
        },
        "pointwise_gate": {"passed": True},
        "cg_gate": {"passed": True},
        "vh_gate": {"passed": True},
    }


def test_three_seed_aggregate_requires_all_paper_gates() -> None:
    report = aggregate_seed_runs([_row(42), _row(43), _row(44)])
    assert report["passed"] is True
    assert report["metrics"]["cg_pointwise_recall_at_2"]["mean"] == 0.61


def test_three_seed_aggregate_rejects_low_trainable_rate() -> None:
    rows = [_row(42), _row(43), _row(44, rate=0.24)]
    report = aggregate_seed_runs(rows)
    assert report["passed"] is False
    assert report["seeds"][2]["checks"]["cg_trainable_group_rate_at_least_25pct"] is False


def test_three_seed_aggregate_missing_artifacts_fail_hash_and_contract_checks() -> None:
    rows = [_row(42), _row(43), _row(44)]
    rows[0]["train"] = {"artifact_status": "missing", "trained_adapter_outputs": {}}
    rows[0]["cg_dev"] = {}
    rows[0]["vh_dev"] = {}
    rows[0]["terminal"] = {}
    report = aggregate_seed_runs(rows)
    checks = report["seeds"][0]["checks"]
    assert checks["cg_dev_hash_matches_trained"] is False
    assert checks["vh_dev_hash_matches_trained"] is False
    assert checks["terminal_hash_matches_trained"] is False
    assert report["same_training_contracts"] is False
    assert report["passed"] is False


def test_three_seed_aggregate_rejects_different_allowlists_across_seeds() -> None:
    rows = [_row(42), _row(43), _row(44)]
    rows[2]["train"]["pool_filters"]["example_id_allowlist_sha256"] = "different"
    report = aggregate_seed_runs(rows)
    assert report["same_training_contracts"] is False
    assert report["passed"] is False


def test_three_seed_aggregate_rejects_different_terminal_reward_contracts() -> None:
    rows = [_row(42), _row(43), _row(44)]
    rows[1]["train"]["terminal_reward_contract"] = "dataset-aware-terminal-reward:stale"
    report = aggregate_seed_runs(rows)
    assert report["same_training_contracts"] is False
    assert report["passed"] is False

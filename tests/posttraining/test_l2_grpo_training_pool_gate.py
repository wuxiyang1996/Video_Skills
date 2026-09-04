from scripts.eval.audit_l2_grpo_training_pool import (
    EXPECTED_TERMINAL_REWARD_CONTRACT,
    audit_training_pool,
)


def report(*, cg_rate: float = 0.25, vh_rate: float = 0.30, groups: int = 50) -> dict:
    return {
        "artifact_status": "trained",
        "split_role": "grpo_pool",
        "split_manifest_sha256": "split-hash",
        "terminal_reward_contract": EXPECTED_TERMINAL_REWARD_CONTRACT,
        "pool_filters": {
            "exact_mined_group_allowlist": True,
            "preserve_allowlist_order": True,
            "dataset_balanced_sampling": True,
            "example_id_allowlist_sha256": "allowlist-hash",
        },
        "trained_adapter_outputs": {"default": {"adapter_weight_sha256": "abc"}},
        "groups_seen": 2 * groups,
        "dataset_metrics": {
            "cg_bench": {
                "groups_seen": groups,
                "groups_trainable": round(groups * cg_rate),
                "groups_trained": round(groups * cg_rate),
                "trainable_group_rate": cg_rate,
            },
            "video_holmes": {
                "groups_seen": groups,
                "groups_trainable": round(groups * vh_rate),
                "groups_trained": round(groups * vh_rate),
                "trainable_group_rate": vh_rate,
            },
        },
    }


def test_training_pool_gate_passes_complete_balanced_run() -> None:
    assert audit_training_pool(report())["passed"] is True


def test_training_pool_gate_rejects_low_cg_rate() -> None:
    gate = audit_training_pool(report(cg_rate=0.24))
    assert gate["passed"] is False
    assert gate["checks"]["cg_trainable_group_rate_at_least_25pct"] is False


def test_training_pool_gate_rejects_too_few_groups() -> None:
    gate = audit_training_pool(report(groups=49))
    assert gate["passed"] is False
    assert gate["checks"]["balanced_50_to_100_groups_per_dataset"] is False


def test_training_pool_gate_rejects_missing_dataset_update() -> None:
    value = report()
    value["dataset_metrics"]["video_holmes"]["groups_trained"] = 0
    gate = audit_training_pool(value)
    assert gate["passed"] is False
    assert gate["checks"]["optimizer_updated_both_datasets"] is False


def test_training_pool_gate_rejects_unhashed_or_nonexact_pool() -> None:
    value = report()
    value["pool_filters"]["example_id_allowlist_sha256"] = None
    gate = audit_training_pool(value)
    assert gate["passed"] is False
    assert gate["checks"]["frozen_exact_balanced_training_pool"] is False


def test_training_pool_gate_rejects_stale_terminal_reward_contract() -> None:
    value = report()
    value["terminal_reward_contract"] = "dataset-aware-terminal-reward:stale"
    gate = audit_training_pool(value)
    assert gate["passed"] is False
    assert gate["checks"]["current_terminal_reward_contract"] is False

from __future__ import annotations

from pathlib import Path

from scripts.eval.validate_l2_mining_report import validate_report
from trainer.artifact_hash import adapter_weight_sha256


def test_mining_gate_requires_matching_policy_contract(tmp_path: Path) -> None:
    source = tmp_path / "sft"
    route = tmp_path / "opd"
    source.mkdir()
    route.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"sft")
    (route / "adapter_model.safetensors").write_bytes(b"opd")
    report = {
        "split_role": "grpo_pool",
        "source_adapter_weight_sha256": adapter_weight_sha256(source),
        "dataset_adapter_backends": {
            "video_holmes": {"adapter_weight_sha256": adapter_weight_sha256(route)}
        },
        "controller_action_contract": "dataset-routed-cg-set-vh-pointwise-v1",
        "relationship_support_contract": "structured-concept-overlap-v2",
        "sampling_protocol": {"generation_temperature": 0.9, "pointwise_temperature": 0.9},
        "pointwise_action_datasets": ["video_holmes"],
        "dataset_metrics": {
            "cg_bench": {"groups_eligible": 50, "eligible_group_rate": 0.25},
            "video_holmes": {"groups_eligible": 51, "eligible_group_rate": 0.30},
        },
        "allowlist_selection": {
            "max_groups_per_dataset": 100,
            "balanced_datasets": True,
            "balanced_target_per_dataset": 50,
            "ordering_contract": "dataset-round-robin-v1",
            "groups_by_dataset": {"cg_bench": 50, "video_holmes": 50},
            "groups": 100,
        },
    }
    result = validate_report(
        report,
        source_adapter=source,
        dataset_adapters={"video_holmes": route},
        controller_action_contract="dataset-routed-cg-set-vh-pointwise-v1",
        relationship_support_contract="structured-concept-overlap-v2",
        generation_temperature=0.9,
        pointwise_temperature=0.9,
        pointwise_datasets=["video_holmes"],
        min_eligible_per_dataset=50,
        min_eligible_group_rate=0.25,
    )
    assert result["passed"]
    report["sampling_protocol"]["pointwise_temperature"] = 1.1
    failed = validate_report(
        report,
        source_adapter=source,
        dataset_adapters={"video_holmes": route},
        controller_action_contract="dataset-routed-cg-set-vh-pointwise-v1",
        relationship_support_contract="structured-concept-overlap-v2",
        generation_temperature=0.9,
        pointwise_temperature=0.9,
        pointwise_datasets=["video_holmes"],
        min_eligible_per_dataset=50,
        min_eligible_group_rate=0.25,
    )
    assert not failed["passed"]
    assert "pointwise_temperature" in failed["failed_checks"]


def test_mining_gate_rejects_unbalanced_selected_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    report = {
        "split_role": "grpo_pool",
        "source_adapter_weight_sha256": adapter_weight_sha256(source),
        "controller_action_contract": "dataset-routed-cg-set-vh-pointwise-v1",
        "relationship_support_contract": "structured-concept-overlap-v2",
        "sampling_protocol": {"generation_temperature": 0.9, "pointwise_temperature": 0.9},
        "pointwise_action_datasets": ["video_holmes"],
        "dataset_metrics": {
            "cg_bench": {"groups_eligible": 60, "eligible_group_rate": 0.30},
            "video_holmes": {"groups_eligible": 100, "eligible_group_rate": 0.50},
        },
        "allowlist_selection": {
            "max_groups_per_dataset": 100,
            "balanced_datasets": False,
            "ordering_contract": "eligible-log-order-v1",
            "groups_by_dataset": {"cg_bench": 60, "video_holmes": 100},
            "groups": 160,
        },
    }
    result = validate_report(
        report, source_adapter=source, dataset_adapters={},
        controller_action_contract="dataset-routed-cg-set-vh-pointwise-v1",
        relationship_support_contract="structured-concept-overlap-v2",
        generation_temperature=0.9, pointwise_temperature=0.9,
        pointwise_datasets=["video_holmes"], min_eligible_per_dataset=50,
        min_eligible_group_rate=0.25,
    )
    assert result["passed"] is False
    assert "allowlist_balanced_contract" in result["failed_checks"]
    assert "allowlist_balanced_counts" in result["failed_checks"]

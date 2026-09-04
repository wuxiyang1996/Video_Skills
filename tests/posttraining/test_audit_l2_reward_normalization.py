from __future__ import annotations

from scripts.eval.audit_l2_reward_normalization import audit_seed
from trainer.grpo.objective import centered_group_advantages


def _report() -> dict:
    return {
        "dataset_metrics": {
            "cg_bench": {
                "groups_seen": 1,
                "groups_trained": 1,
                "trainable_group_rate": 0.5,
                "mean_reward_components": {
                    "clue_recall": 0.5,
                    "clue_mean_best_iou": 0.4,
                    "evidence_precision": 0.3,
                },
            },
            "video_holmes": {
                "groups_seen": 1,
                "groups_trained": 1,
                "trainable_group_rate": 0.25,
                "mean_reward_components": {
                    "segment_recall": 0.5,
                    "segment_precision": 0.4,
                    "inference_shot_recall": 0.3,
                    "relationship_support": 0.2,
                },
            },
        }
    }


def _row(example_id: str, rewards: list[float]) -> dict:
    return {
        "example_id": example_id,
        "rewards": rewards,
        "advantages": centered_group_advantages(rewards),
    }


def test_reward_normalization_audit_accepts_balanced_dataset_local_groups() -> None:
    result = audit_seed(
        "seed42",
        _report(),
        [_row("cg_bench:q1", [0.0, 1.0]), _row("video_holmes:test:v:q1", [0.1, 0.9])],
    )
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_reward_normalization_audit_rejects_wrong_advantages() -> None:
    rows = [_row("cg_bench:q1", [0.0, 1.0]), _row("video_holmes:test:v:q1", [0.1, 0.9])]
    rows[1]["advantages"] = [0.0, 0.0]
    result = audit_seed("seed42", _report(), rows)
    assert result["passed"] is False
    assert result["checks"]["stored_advantages_match_mean_std_normalization"] is False


def test_reward_normalization_audit_rejects_missing_dataset_components() -> None:
    report = _report()
    del report["dataset_metrics"]["video_holmes"]["mean_reward_components"]["relationship_support"]
    result = audit_seed(
        "seed42",
        report,
        [_row("cg_bench:q1", [0.0, 1.0]), _row("video_holmes:test:v:q1", [0.1, 0.9])],
    )
    assert result["passed"] is False
    assert result["checks"]["dataset_specific_components_reported"] is False

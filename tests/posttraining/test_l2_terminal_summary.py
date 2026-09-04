from __future__ import annotations

from scripts.eval.summarize_l2_terminal_samples import minimum_verified_terminal_success, summarize


def test_terminal_summary_reports_trainable_variance_and_rates() -> None:
    samples = [
        {"dataset": "video_holmes", "group": 0, "reward": 1.0, "terminal_success": True, "answer_correct": True, "verifier_passed": True, "acceptance_status": "accepted_strong", "reward_components": {"inference_shot_recall": 1.0}},
        {"dataset": "video_holmes", "group": 0, "reward": -0.25, "terminal_success": False, "answer_correct": False, "verifier_passed": False, "acceptance_status": "rejected", "reward_components": {"inference_shot_recall": 0.0}},
        {"dataset": "video_holmes", "group": 1, "reward": 0.0, "terminal_success": False, "answer_correct": False, "verifier_passed": False, "acceptance_status": "invalid_retrieval_action"},
    ]
    report = summarize(samples)
    metrics = report["dataset_metrics"]["video_holmes"]
    assert metrics["groups_seen"] == 2
    assert metrics["groups_trainable"] == 1
    assert metrics["trainable_group_rate"] == 0.5
    assert metrics["terminal_success_rate"] == 1 / 3
    assert metrics["valid_retrieval_action_rate"] == 2 / 3


def test_minimum_verified_reclassification_keeps_process_requirement() -> None:
    row = {
        "acceptance_status": "accepted_weak",
        "answer_correct": True,
        "verifier_passed": True,
        "process_supported": True,
        "format_budget_compliant": True,
        "rollout_diagnostic": {
            "support_ref_count": 3,
            "min_support_refs": 2,
            "trace_fail": 0,
        },
    }
    assert minimum_verified_terminal_success(row)
    row["process_supported"] = False
    assert not minimum_verified_terminal_success(row)

from scripts.eval.audit_l2_terminal_reward_separation import audit


def test_reward_separation_requires_three_empirical_outcomes() -> None:
    rows = [
        {"dataset": "cg_bench", "terminal_success": True, "process_supported": True, "answer_correct": True, "reward": 1.0},
        {"dataset": "cg_bench", "terminal_success": False, "process_supported": False, "reward": 0.0},
        {"dataset": "video_holmes", "terminal_success": True, "process_supported": True, "answer_correct": True, "reward": 1.0},
        {"dataset": "video_holmes", "terminal_success": False, "process_supported": True, "answer_correct": False, "reward": -0.25},
        {"dataset": "video_holmes", "terminal_success": False, "process_supported": False, "reward": 0.1},
    ]
    report = audit(rows, min_samples=2)
    assert report["passed"]
    assert report["checks"]["all_three_outcomes_observed"]


def test_reward_separation_records_consistent_terminal_reward_contract() -> None:
    rows = [
        {"dataset": "cg_bench", "terminal_success": True, "process_supported": True, "answer_correct": True, "reward": 1.0},
        {"dataset": "cg_bench", "terminal_success": False, "process_supported": False, "reward": 0.0},
        {"dataset": "video_holmes", "terminal_success": True, "process_supported": True, "answer_correct": True, "reward": 1.0},
        {"dataset": "video_holmes", "terminal_success": False, "process_supported": True, "answer_correct": False, "reward": -0.25},
        {"dataset": "video_holmes", "terminal_success": False, "process_supported": False, "reward": 0.1},
    ]
    report = audit(rows, min_samples=2, terminal_reward_contracts=["repair-v1", "repair-v1"])
    assert report["passed"] is True
    assert report["terminal_reward_contract"] == "repair-v1"


def test_reward_separation_rejects_mixed_terminal_reward_contracts() -> None:
    report = audit([], min_samples=0, terminal_reward_contracts=["old", "new"])
    assert report["passed"] is False
    assert report["checks"]["terminal_reward_contract_consistent"] is False


def test_reward_separation_distinguishes_correct_but_uncommitted() -> None:
    rows = [
        {"dataset": "cg_bench", "terminal_success": True, "process_supported": True, "answer_correct": True, "reward": 1.0},
        {"dataset": "cg_bench", "terminal_success": False, "process_supported": True, "answer_correct": True, "reward": 0.8},
        {"dataset": "cg_bench", "terminal_success": False, "process_supported": False, "answer_correct": False, "reward": 0.0},
        {"dataset": "video_holmes", "terminal_success": True, "process_supported": True, "answer_correct": True, "reward": 1.0},
        {"dataset": "video_holmes", "terminal_success": False, "process_supported": True, "answer_correct": False, "reward": -0.25},
        {"dataset": "video_holmes", "terminal_success": False, "process_supported": False, "answer_correct": False, "reward": 0.1},
    ]
    report = audit(rows, min_samples=2)
    bucket = report["dataset_metrics"]["cg_bench"]["categories"]
    assert "correct_uncommitted_or_rejected" in bucket
    assert "incorrect_or_rejected" not in bucket
    assert report["checks"]["cg_bench:uncommitted_below_success"] is True

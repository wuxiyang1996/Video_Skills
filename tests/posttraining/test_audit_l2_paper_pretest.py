from scripts.eval.audit_l2_paper_pretest import audit_pretest_release


def _artifacts() -> dict:
    reward_contract = "dataset-aware-terminal-reward:repair-v1"
    passed = {"passed": True}
    return {
        "split_audit": passed,
        "reward_separation": {"passed": True, "terminal_reward_contract": reward_contract},
        "opd_selection": {"passed": True, "selected": {"alpha": 0.75}},
        "opd_terminal_selection": {"passed": True, "selected": {
            "alpha": 0.75, "terminal_reward_contract": reward_contract,
        }},
        "mining": {"allowlist_selection": {
            "balanced_datasets": True,
            "groups_by_dataset": {"cg_bench": 91, "video_holmes": 91},
            "groups": 182,
            "ordering_contract": "dataset-round-robin-v1",
        }},
        "pilot_pointwise_gate": passed,
        "pilot_cg_gate": passed,
        "pilot_vh_gate": passed,
        "three_seed_aggregate": {
            "passed": True,
            "seed_count": 3,
            "same_training_contracts": True,
            "seeds": [
                {"passed": True, "contracts": {"terminal_reward_contract": reward_contract}},
                {"passed": True, "contracts": {"terminal_reward_contract": reward_contract}},
                {"passed": True, "contracts": {"terminal_reward_contract": reward_contract}},
            ],
        },
    }


def test_pretest_release_accepts_only_complete_canonical_chain() -> None:
    report = audit_pretest_release(_artifacts())
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_pretest_release_fails_closed_on_missing_aggregate() -> None:
    artifacts = _artifacts()
    artifacts["three_seed_aggregate"] = {"_missing": "aggregate.json"}
    report = audit_pretest_release(artifacts)
    assert report["passed"] is False
    assert report["checks"]["three_seed_aggregate"] is False
    assert report["checks"]["exactly_three_seeds"] is False


def test_pretest_release_fails_closed_on_missing_selection_report() -> None:
    artifacts = _artifacts()
    artifacts["mining"] = {"_missing": "selection.json"}
    report = audit_pretest_release(artifacts)
    assert report["passed"] is False
    assert report["checks"]["selection_report_passed"] is False


def test_pretest_release_rejects_unbalanced_mining() -> None:
    artifacts = _artifacts()
    artifacts["mining"]["allowlist_selection"]["groups_by_dataset"]["video_holmes"] = 49
    report = audit_pretest_release(artifacts)
    assert report["passed"] is False
    assert report["checks"]["mining_balanced"] is False
    assert report["checks"]["mining_50_to_100_groups_per_dataset"] is False


def test_pretest_release_accepts_passed_terminal_consensus_selection() -> None:
    artifacts = _artifacts()
    artifacts["mining"] = {
        "schema_version": "video-skills/l2-terminal-consensus-group-selection-v1",
        "passed": True,
        "target_per_dataset": 50,
        "ordering_contract": "dataset-round-robin-v1",
        "selection_uses_training_pool_terminal_outcomes_only": True,
        "dataset_metrics": {
            "cg_bench": {"selected": 50},
            "video_holmes": {"selected": 50},
        },
    }
    report = audit_pretest_release(artifacts)
    assert report["passed"] is True
    assert report["checks"]["selection_training_pool_only"] is True


def test_pretest_release_rejects_failed_terminal_consensus_selection() -> None:
    artifacts = _artifacts()
    artifacts["mining"] = {
        "schema_version": "video-skills/l2-terminal-consensus-group-selection-v1",
        "passed": False,
        "target_per_dataset": 50,
        "ordering_contract": "dataset-round-robin-v1",
        "selection_uses_training_pool_terminal_outcomes_only": True,
        "dataset_metrics": {
            "cg_bench": {"selected": 50},
            "video_holmes": {"selected": 50},
        },
    }
    report = audit_pretest_release(artifacts)
    assert report["passed"] is False
    assert report["checks"]["selection_report_passed"] is False


def test_pretest_release_rejects_mixed_terminal_reward_contracts() -> None:
    artifacts = _artifacts()
    artifacts["reward_separation"]["terminal_reward_contract"] = "stale"
    report = audit_pretest_release(artifacts)
    assert report["passed"] is False
    assert report["checks"]["same_terminal_reward_contract"] is False

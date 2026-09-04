from scripts.eval.gate_l2_terminal import gate_terminal_reports


def _report(dataset: str, success: float, verifier: float, fmt: float) -> dict:
    return {
        "split_role": "dev_tune",
        "pool_filters": {
            "datasets": None,
            "example_id_allowlist": "/frozen/dev.txt",
            "preserve_allowlist_order": False,
            "dataset_balanced_sampling": True,
        },
        "groups_seen": 10,
        "unique_pool_examples_before_repeats": 10,
        "repeats_per_example": 1,
        "repeat_start_index": 0,
        "boundary_anchor_index0": False,
        "eval_only": True,
        "retrieval_only": False,
        "terminal_on_process_hit": True,
        "mock_semantic_judge": False,
        "remote_rollout_policy": False,
        "fixed_remote_environment_executor": True,
        "terminal_reward_contract": "dataset-aware-terminal-reward:test-v1",
        "sampling_protocol": {
            "generation_temperature": 0.9,
            "pointwise_temperature": 0.9,
            "pointwise_sampler": "gumbel-top-k-without-replacement-v1",
        },
        "dataset_metrics": {dataset: {
            "samples": 20,
            "terminal_success_rate": success,
            "valid_retrieval_action_rate": 1.0,
            "verifier_pass_rate": verifier,
            "format_compliance_rate": fmt,
        }},
    }


def test_terminal_gate_requires_ten_percent_and_no_regression() -> None:
    base = _report("video_holmes", 0.05, 0.80, 0.90)
    good = _report("video_holmes", 0.10, 0.78, 0.90)
    assert gate_terminal_reports(base, good, dataset="video_holmes")["passed"]
    zero = _report("video_holmes", 0.0, 0.80, 0.90)
    assert not gate_terminal_reports(zero, zero, dataset="video_holmes")["passed"]
    bad_format = _report("video_holmes", 0.10, 0.80, 0.70)
    assert not gate_terminal_reports(base, bad_format, dataset="video_holmes")["passed"]


def test_terminal_gate_rejects_mixed_action_contracts() -> None:
    base = _report("cg_bench", 0.20, 0.20, 1.0)
    opd = _report("cg_bench", 0.20, 0.20, 1.0)
    base["controller_action_contract"] = "legacy"
    opd["controller_action_contract"] = "select-coarse-clips-exact-v1"
    report = gate_terminal_reports(base, opd, dataset="cg_bench")
    assert report["checks"]["same_controller_action_contract"] is False
    assert report["passed"] is False


def test_terminal_gate_rejects_mixed_executor_isolation_contracts() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    base["executor_isolation_contract"] = "legacy-full-index"
    opd["executor_isolation_contract"] = "selected-window-closure-v1"
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_executor_isolation_contract"] is False
    assert report["passed"] is False


def test_terminal_gate_rejects_mixed_terminal_reward_contracts() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    opd["terminal_reward_contract"] = "dataset-aware-terminal-reward:changed"
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_terminal_reward_contract"] is False
    assert report["passed"] is False


def test_terminal_gate_rejects_mismatched_frozen_dev_protocol() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    opd["pool_filters"] = {**opd["pool_filters"], "example_id_allowlist": "/other/dev.txt"}
    opd["groups_seen"] = 11
    opd["boundary_anchor_index0"] = True
    opd["dataset_metrics"]["video_holmes"]["samples"] = 24
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_pool_filters"] is False
    assert report["checks"]["same_group_protocol"] is False
    assert report["checks"]["same_boundary_anchor_contract"] is False
    assert report["checks"]["same_dataset_sample_count"] is False
    assert report["passed"] is False


def test_terminal_gate_accepts_explicit_false_for_legacy_pool_default() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    opd["pool_filters"]["exact_mined_group_allowlist"] = False
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_pool_filters"] is True
    assert report["passed"] is True


def test_terminal_gate_ignores_training_only_sampling_metadata_in_eval() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    opd["sampling_protocol"].update({
        "pointwise_gradient_contract": None,
        "pointwise_train_batch_size": 1,
    })
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_sampling_protocol"] is True
    assert report["passed"] is True


def test_terminal_gate_rejects_changed_eval_sampling_protocol() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    opd["sampling_protocol"]["generation_temperature"] = 1.1
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_sampling_protocol"] is False
    assert report["passed"] is False


def test_terminal_gate_rejects_missing_sampling_protocol() -> None:
    base = _report("video_holmes", 0.20, 0.20, 1.0)
    opd = _report("video_holmes", 0.20, 0.20, 1.0)
    del opd["sampling_protocol"]
    report = gate_terminal_reports(base, opd, dataset="video_holmes")
    assert report["checks"]["same_sampling_protocol"] is False
    assert report["passed"] is False


def test_terminal_gate_rejects_missing_protocol_metadata() -> None:
    base = _report("cg_bench", 0.20, 0.20, 1.0)
    opd = _report("cg_bench", 0.20, 0.20, 1.0)
    del opd["split_role"]
    del opd["eval_only"]
    report = gate_terminal_reports(base, opd, dataset="cg_bench")
    assert report["checks"]["same_split_role"] is False
    assert report["checks"]["same_eval_mode"] is False
    assert report["passed"] is False

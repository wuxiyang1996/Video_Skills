from scripts.eval.audit_l2_atomic_skill_execution import audit_atomic_skill_execution


def _rows():
    samples = []
    traces = []
    for dataset in ("cg_bench", "video_holmes"):
        for sample in range(10):
            success = sample == 0
            samples.append({
                "dataset": dataset, "group": sample, "sample": 0,
                "terminal_success": success, "process_supported": success,
                "executor_cache_hit": sample % 2 == 1,
                "rollout_diagnostic": {"trace_ok": 2 if success else 1, "trace_fail": 0 if success else 1},
            })
            if sample % 2 == 0:
                traces.append({
                    "dataset": dataset, "group": sample, "sample": 0,
                    "llm_plan": {"reasoning_plan": [{"step_id": "r1"}, {"step_id": "r2"}]},
                    "skill_trace": [
                        {"step_id": "r1", "skill_id": "retrieve_by_event", "status": "verified", "failure_code": None},
                        {"step_id": "r2", "skill_id": "commit_answer", "status": "verified" if success else "failed",
                         "failure_code": None if success else "claim_not_verified"},
                    ],
                })
    return samples, traces


def test_atomic_skill_audit_reports_cache_and_per_skill_health() -> None:
    samples, traces = _rows()
    report = audit_atomic_skill_execution(samples, traces)
    assert report["passed"] is True
    cg = report["datasets"]["cg_bench"]
    assert cg["rollout_metrics"]["executor_cache_hits"] == 5
    assert cg["fresh_execution_metrics"]["planned_step_completion_rate"] == 1.0
    assert cg["skills"]["commit_answer"]["failure_codes"] == {"claim_not_verified": 4}
    assert cg["outcome_skill_health"]["terminal_success"]["mean_trace_fail"] == 0


def test_atomic_skill_audit_fails_on_unmatched_trace() -> None:
    samples, traces = _rows()
    traces[0]["group"] = 999
    report = audit_atomic_skill_execution(samples, traces)
    assert report["passed"] is False
    assert report["checks"]["all_traces_match_samples"] is False

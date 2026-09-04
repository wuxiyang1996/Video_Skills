from scripts.eval.select_l2_opd_terminal_checkpoint import select_terminal_checkpoint


def _pointwise() -> dict:
    return {
        "passed": True,
        "candidates": [
            {"name": "a50", "alpha": 0.5, "adapter": "/a50", "adapter_weight_sha256": "h50", "passed": True},
            {"name": "a75", "alpha": 0.75, "adapter": "/a75", "adapter_weight_sha256": "h75", "passed": True},
        ],
    }


def _candidate(name: str, alpha: float, passed: bool) -> dict:
    adapter = f"/{name}"
    return {
        "name": name,
        "alpha": alpha,
        "adapter": adapter,
        "terminal_report": f"/{name}/terminal.json",
        "terminal": {
            "source_adapter": adapter,
            "source_adapter_weight_sha256": "h50" if alpha == 0.5 else "h75",
            "terminal_success_rate": 0.1,
            "terminal_reward_contract": "reward-repair-v1",
            "dataset_metrics": {},
        },
        "cg_gate_report": f"/{name}/cg.json",
        "cg_gate": {"dataset": "cg_bench", "passed": True},
        "vh_gate_report": f"/{name}/vh.json",
        "vh_gate": {"dataset": "video_holmes", "passed": passed},
    }


def test_selects_smallest_candidate_passing_pointwise_and_terminal() -> None:
    report = select_terminal_checkpoint(
        _pointwise(), [_candidate("a75", 0.75, True), _candidate("a50", 0.5, False)]
    )
    assert report["passed"] is True
    assert report["selected"]["name"] == "a75"
    assert report["selected"]["terminal_reward_contract"] == "reward-repair-v1"


def test_rejects_terminal_report_from_wrong_adapter() -> None:
    candidate = _candidate("a50", 0.5, True)
    candidate["terminal"]["source_adapter_weight_sha256"] = "wrong"
    report = select_terminal_checkpoint(_pointwise(), [candidate])
    assert report["passed"] is False
    assert report["candidates"][0]["checks"]["terminal_source_hash_matches"] is False

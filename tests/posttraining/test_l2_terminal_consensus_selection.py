import pytest

from scripts.eval.select_l2_terminal_consensus_groups import (
    failure_report,
    select_terminal_consensus_groups,
)


def _seed_rows(seed: int, groups_per_dataset: int = 60) -> list[dict]:
    rows = []
    for dataset in ("cg_bench", "video_holmes"):
        for group in range(groups_per_dataset):
            # The first 25 groups are terminal-capable with staggered seed coverage.
            success_sample = (group + seed) % 8 if group < 25 else -1
            for sample in range(8):
                success = sample == success_sample
                rows.append({
                    "dataset": dataset,
                    "example_id": f"{dataset}:{group}",
                    "repeat_index": 0,
                    "group": 2 * group + (dataset == "video_holmes"),
                    "reward": 1.0 if success else (0.1 if sample % 2 else 0.0),
                    "terminal_success": success,
                    "process_supported": group < 55,
                    "format_budget_compliant": True,
                })
    return rows


def test_selects_balanced_terminal_capable_groups_with_auditable_prediction() -> None:
    rows = {seed: _seed_rows(seed) for seed in (42, 43, 44)}
    allowlist, report = select_terminal_consensus_groups(rows, target_per_dataset=50)
    assert report["passed"] is True
    assert len(allowlist) == 100
    assert report["dataset_metrics"]["cg_bench"]["terminal_consensus2"] == 25
    assert report["dataset_metrics"]["cg_bench"]["mean_predicted_trainable_rate"] >= 0.25
    assert allowlist[0][0].startswith("cg_bench:")
    assert allowlist[1][0].startswith("video_holmes:")


def test_rejects_partial_or_mismatched_seed_probe() -> None:
    rows = {seed: _seed_rows(seed) for seed in (42, 43, 44)}
    rows[44].pop()
    with pytest.raises(ValueError, match="incomplete group|same complete group"):
        select_terminal_consensus_groups(rows, target_per_dataset=50)


def test_fails_closed_when_fewer_than_50_groups_exist() -> None:
    rows = {seed: _seed_rows(seed, groups_per_dataset=40) for seed in (42, 43, 44)}
    allowlist, report = select_terminal_consensus_groups(rows, target_per_dataset=50)
    assert report["passed"] is False
    assert report["checks"]["enough_candidates"] is False
    assert len(allowlist) == 80


def test_rejects_selection_that_starves_one_seed_despite_good_pooled_prediction() -> None:
    rows = {seed: _seed_rows(seed) for seed in (42, 43, 44)}
    for row in rows[44]:
        row["terminal_success"] = False
    _, report = select_terminal_consensus_groups(rows, target_per_dataset=50)
    assert report["checks"]["predicted_trainable_rate_at_least_threshold"] is True
    assert report["checks"]["observed_trainable_rate_each_seed_at_least_threshold"] is False
    assert report["passed"] is False


def test_input_error_has_auditable_failure_report() -> None:
    report = failure_report(ValueError("seed probes differ"))
    assert report["passed"] is False
    assert report["checks"]["inputs_valid_and_complete"] is False
    assert report["error"] == {"type": "ValueError", "message": "seed probes differ"}

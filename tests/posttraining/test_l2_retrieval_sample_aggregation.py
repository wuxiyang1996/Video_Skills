from scripts.eval.aggregate_l2_retrieval_samples import (
    aggregate,
    process_supported,
    select_allowlist_groups,
)


def _sample(example_id: str, reward: float, *, inference: float, relationship: float) -> dict:
    return {
        "event": "terminal_sample",
        "dataset": "video_holmes",
        "example_id": example_id,
        "repeat_index": 0,
        "question_type": "SR",
        "reward": reward,
        "reward_components": {
            "inference_shot_recall": inference,
            "relationship_support": relationship,
        },
        "format_budget_compliant": True,
    }


def test_sr_requires_inference_and_relationship() -> None:
    assert process_supported(_sample("a", 0.1, inference=1.0, relationship=0.0)) is False
    assert process_supported(_sample("a", 0.1, inference=0.0, relationship=1.0)) is False
    assert process_supported(_sample("a", 0.1, inference=1.0, relationship=0.25)) is True


def test_aggregate_selects_only_variance_dual_evidence_groups() -> None:
    rows = [
        _sample("good", 0.1, inference=0.0, relationship=0.0),
        _sample("good", 0.4, inference=1.0, relationship=0.25),
        _sample("relationship_only", 0.1, inference=0.0, relationship=1.0),
        _sample("relationship_only", 0.3, inference=0.0, relationship=1.0),
        _sample("equal", 0.2, inference=1.0, relationship=1.0),
        _sample("equal", 0.2, inference=1.0, relationship=1.0),
    ]
    selected, report = aggregate(rows)
    assert selected == ["good"]
    assert report["groups_seen"] == 3
    assert report["groups_eligible"] == 1


def test_balanced_allowlist_uses_smaller_bucket_and_round_robin_order() -> None:
    eligible = [
        {"dataset": "cg_bench", "example_id": f"cg{i}"} for i in range(3)
    ] + [{"dataset": "video_holmes", "example_id": f"vh{i}"} for i in range(2)]
    selected, counts, target = select_allowlist_groups(
        eligible, max_groups_per_dataset=100, balanced_datasets=True
    )
    assert target == 2
    assert counts == {"cg_bench": 2, "video_holmes": 2}
    assert [row["dataset"] for row in selected] == [
        "cg_bench", "video_holmes", "cg_bench", "video_holmes"
    ]

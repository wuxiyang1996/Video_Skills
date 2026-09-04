from dataset_clip_wrapper.training.build_l2_mixed_v9 import mix_rows


def row(example_id: str, weight: float) -> dict:
    return {
        "transition_id": f"{example_id}:{weight}",
        "metadata": {"source_example_id": example_id, "source_family_weight": weight},
    }


def test_mix_rows_balances_two_action_interfaces() -> None:
    mixed, report = mix_rows(
        [row("a", 0.7), row("a", 0.3)],
        [row("a", 0.5), row("a", 0.5)],
    )
    assert len(mixed) == 4
    assert sum(item["metadata"]["source_family_weight"] for item in mixed) == 1.0
    assert report["lane_rows"] == {"selection_ranking": 2, "pointwise": 2}
    assert report["source_weight_min"] == report["source_weight_max"] == 1.0

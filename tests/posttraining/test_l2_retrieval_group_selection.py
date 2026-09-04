from scripts.eval.select_l2_retrieval_groups import select_groups


def test_selects_only_process_hit_reward_variance_format_groups() -> None:
    rows = [
        {"example_id": "a", "dataset": "cg_bench", "reward_variance": True, "process_supported_samples": 2, "format_compliant_samples": 4},
        {"example_id": "a", "dataset": "cg_bench", "reward_variance": True, "process_supported_samples": 1, "format_compliant_samples": 4},
        {"example_id": "b", "dataset": "video_holmes", "reward_variance": False, "process_supported_samples": 1, "format_compliant_samples": 4},
        {"example_id": "c", "dataset": "video_holmes", "reward_variance": True, "process_supported_samples": 1, "format_compliant_samples": 0},
    ]
    selected, report = select_groups(rows)
    assert selected == ["a"]
    assert report["groups_eligible"] == 2
    assert report["unique_examples_selected"] == 1

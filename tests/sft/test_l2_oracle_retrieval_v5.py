from dataset_clip_wrapper.training.l2_oracle_retrieval_v5 import (
    policy_catalog,
    build_oracle_package,
    select_oracle_windows,
)


def test_oracle_window_selection_prefers_overlap_and_reports_coverage() -> None:
    coarse = [
        {"time_span": {"start_s": 0, "end_s": 30}},
        {"time_span": {"start_s": 28, "end_s": 58}},
        {"time_span": {"start_s": 56, "end_s": 86}},
    ]
    selected, coverage = select_oracle_windows(coarse, [[31, 40]], topk=2)
    assert selected == [1]
    assert coverage == 1.0


def test_policy_catalog_is_label_independent_and_bounded() -> None:
    coarse = [{
        "time_span": {"start_s": 0, "end_s": 30},
        "scene_description": "x" * 200,
        "observable_facts": ["y" * 100, "unused"],
        "events": ["z" * 100],
        "searchable_phrases": ["p" * 100],
    }]
    row = policy_catalog(coarse)[0]
    assert set(row) == {"coarse_index", "time_span", "scene_description"}
    assert len(row["scene_description"]) == 96


def test_oracle_package_hides_labels_and_normalizes_source_weight() -> None:
    row = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:7",
        "video": {"video_id": "v1"},
        "question": {
            "question_text": "What happened?",
            "options": [{"label": "A", "text": "left"}],
            "answer": {"label": "A", "text": "left"},
        },
        "metadata": {
            "coarse_clip_schemas": [
                {"time_span": {"start_s": 0, "end_s": 30}, "scene_description": "intro"},
                {"time_span": {"start_s": 28, "end_s": 58}, "scene_description": "person leaves"},
            ]
        },
    }
    manifest = {"videos": [{"dataset": "cg_bench", "video_id": "v1", "role": "sft_seed"}]}
    package, report = build_oracle_package([row], manifest, [{"qid": 7, "clue_intervals": [[35, 40]]}])
    train = package["sft_seed"]
    assert report["exported_core_by_role"]["sft_seed"] == 1
    assert report["prompt_forbidden_key_hits"] == 0
    assert abs(sum(item["metadata"]["source_family_weight"] for item in train) - 1.0) < 1e-9
    user_text = "\n".join(item["messages"][1]["content"] for item in train)
    assert "clue_intervals" not in user_text
    assert '"answer"' not in user_text
    core = next(item for item in train if item["metadata"]["is_core"])
    assert '"selected_coarse_indices":[1]' in core["messages"][2]["content"]

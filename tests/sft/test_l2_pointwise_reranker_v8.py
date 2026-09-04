import json

from dataset_clip_wrapper.training.l2_pointwise_reranker_v8 import (
    build_label_independent_eval_split,
    build_label_independent_train_split,
    build_split,
    relevance_action,
)


def _source() -> dict:
    state = {
        "dataset": "cg_bench",
        "example_id": "cg_bench:1",
        "question": {"question_text": "What happens?"},
        "candidate_retrieval": {"method": "visual"},
        "l1_coarse_summary_catalog": [
            {"coarse_index": 7, "retrieval_rank": 1, "scene_description": "wrong"},
            {"coarse_index": 3, "retrieval_rank": 2, "scene_description": "right"},
            {"coarse_index": 9, "retrieval_rank": 3, "scene_description": "wrong too"},
        ],
    }
    return {
        "split_group_id": "video-1",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": json.dumps({"task": "select_coarse_set", "state_t": state})},
            {"role": "assistant", "content": json.dumps({"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": [3]}})},
        ],
        "metadata": {"task": "select_coarse_set", "is_core": True, "source_example_id": "cg_bench:1", "dataset": "cg_bench"},
    }


def test_relevance_action_is_minimal_boolean_tool() -> None:
    assert relevance_action(True)["arguments"] == {"relevant": True}
    assert relevance_action(False)["arguments"] == {"relevant": False}


def test_build_split_balances_positive_and_negative_family_weight() -> None:
    rows, report = build_split([_source()], split_role="sft_seed", hard_negatives=2)
    assert len(rows) == 3
    positives = [row for row in rows if row["metadata"]["candidate_relevant"]]
    negatives = [row for row in rows if not row["metadata"]["candidate_relevant"]]
    assert len(positives) == 1 and len(negatives) == 2
    assert sum(row["metadata"]["source_family_weight"] for row in positives) == 0.5
    assert sum(row["metadata"]["source_family_weight"] for row in negatives) == 0.5
    assert report["source_weight_sum"] == 1.0
    prompt = json.loads(rows[0]["messages"][1]["content"])
    assert "selected_coarse_indices" not in json.dumps(prompt)


def test_label_independent_eval_does_not_force_gold_into_candidate_pool() -> None:
    source = _source()
    report = {
        "label_independent": True,
        "results": [{"example_id": "cg_bench:1", "top32": [7, 9]}],
    }
    rows, summary = build_label_independent_eval_split([source], report)
    assert [row["metadata"]["candidate_index"] for row in rows] == [7, 9]
    assert all(row["metadata"]["gold_indices"] == [3] for row in rows)
    assert summary["gold_outside_candidate_pool_examples"] == 1


def test_label_independent_train_uses_pool_without_forcing_gold() -> None:
    source = _source()
    report = {
        "label_independent": True,
        "model": "retriever",
        "results": [{"example_id": "cg_bench:1", "top32": [7, 3, 9]}],
    }
    rows, summary = build_label_independent_train_split([source], report)
    assert [row["metadata"]["candidate_index"] for row in rows] == [3, 7, 9]
    positives = [row for row in rows if row["metadata"]["candidate_relevant"]]
    negatives = [row for row in rows if not row["metadata"]["candidate_relevant"]]
    assert [row["metadata"]["candidate_index"] for row in positives] == [3]
    assert sum(row["metadata"]["source_family_weight"] for row in positives) == 0.5
    assert sum(row["metadata"]["source_family_weight"] for row in negatives) == 0.5
    assert summary["candidate_selection_label_independent"] is True


def test_label_independent_train_excludes_gold_outside_pool() -> None:
    source = _source()
    report = {
        "label_independent": True,
        "results": [{"example_id": "cg_bench:1", "top32": [7, 9]}],
    }
    rows, summary = build_label_independent_train_split([source], report)
    assert rows == []
    assert summary["excluded"] == {"gold_outside_candidate_pool": 1}


def test_visual_teacher_features_and_teacher_hard_negatives_are_used() -> None:
    source = _source()
    candidate_report = {
        "label_independent": True,
        "model": "retriever",
        "results": [{"example_id": "cg_bench:1", "top32": [7, 3, 9]}],
    }
    teacher_report = {
        "label_independent": True,
        "model": "teacher-8b",
        "results": [{
            "example_id": "cg_bench:1",
            "ranking": [
                {"candidate_index": 9, "score": 0.9},
                {"candidate_index": 3, "score": 0.6},
                {"candidate_index": 7, "score": 0.1},
            ],
        }],
    }
    rows, summary = build_label_independent_train_split(
        [source],
        candidate_report,
        visual_teacher_report=teacher_report,
        teacher_hard_negatives=1,
    )
    assert [row["metadata"]["candidate_index"] for row in rows] == [3, 9]
    assert summary["visual_teacher"] == "teacher-8b"
    assert summary["teacher_hard_negatives"] == 1
    assert summary["missing_teacher_features"] == 0
    teacher_negative = rows[1]
    assert teacher_negative["metadata"]["visual_teacher_rank"] == 1
    prompt = json.loads(teacher_negative["messages"][1]["content"])
    candidate = prompt["state_t"]["candidate_coarse_summary"]
    assert candidate["visual_teacher_reranker"]["rank"] == 1

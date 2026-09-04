import json

from dataset_clip_wrapper.training.l2_candidate_reranker_v7 import (
    FAMILY_BUDGETS,
    candidate_specs,
    core_transitions,
    expand_core,
)


def _chat(selected=(1,)):
    state = {"dataset": "cg_bench", "example_id": "e", "question": {"question_text": "find object"}}
    action = {"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": list(selected), "rationale_short": "leak"}}
    return {
        "metadata": {"task": "select_coarse_set", "is_core": True, "source_example_id": "e"},
        "messages": [{}, {"content": json.dumps({"state_t": state})}, {"content": json.dumps(action)}],
    }


def _source():
    return {
        "example_id": "e",
        "metadata": {"coarse_clip_schemas": [
            {
                "coarse_index": i,
                "time_span": {"start_s": i * 10, "end_s": i * 10 + 10},
                "scene_description": f"scene {i}",
                "observable_facts": [{"text": f"fact {i}"}],
                "events": [{"description": f"event {i}"}],
                "searchable_phrases": [f"phrase {i}"],
            }
            for i in range(5)
        ]},
    }


def test_v7_exposes_rank_and_preserves_rich_evidence():
    report = {"results": [{"example_id": "e", "catalog_size": 5, "top32": [2, 1, 3, 4, 0]}]}
    specs = candidate_specs(report)
    core, excluded = core_transitions([_chat()], specs, {"e": _source()})
    assert not excluded
    catalog = core[0]["state_t"]["l1_coarse_summary_catalog"]
    assert [row["coarse_index"] for row in catalog] == [2, 1, 3, 4, 0]
    assert catalog[0]["retrieval_rank"] == 1
    assert catalog[0]["observable_facts"] == ["fact 2"]
    assert core[0]["state_t"]["candidate_retrieval"]["rank_visible_to_policy"] is True
    assert "rationale_short" not in core[0]["action_t"]["arguments"]


def test_v7_full_view_gets_majority_weight_and_fail_closes():
    report = {"results": [{"example_id": "e", "catalog_size": 5, "top32": [2, 1, 3, 4, 0]}]}
    core, _ = core_transitions([_chat()], candidate_specs(report), {"e": _source()})
    expanded = expand_core(core[0], hard_negatives=2)
    assert sum(row["augmentation_family"] == "full_select" for row in expanded) == 1
    assert FAMILY_BUDGETS["full_select"] == 0.65
    assert sum(FAMILY_BUDGETS.values()) == 1.0

    core, excluded = core_transitions([_chat(selected=(9,))], candidate_specs(report), {"e": _source()})
    assert core == []
    assert excluded["gold_outside_candidates"] == 1


def test_v7_ranking_does_not_put_teacher_candidate_always_first():
    report = {"results": [{"example_id": "e", "catalog_size": 5, "top32": [2, 1, 3, 4, 0]}]}
    core, _ = core_transitions([_chat()], candidate_specs(report), {"e": _source()})
    ranking = [row for row in expand_core(core[0], hard_negatives=4) if row["task"] == "rank_coarse_candidates"]
    positive_positions = [
        [candidate["coarse_index"] for candidate in row["state_t"]["candidate_coarse_summaries"]].index(1)
        for row in ranking
    ]
    assert set(positive_positions) == {0, 1}

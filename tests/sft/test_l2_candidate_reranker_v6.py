import json

from dataset_clip_wrapper.training.l2_candidate_reranker_v6 import candidate_index, core_transitions


def _chat(selected=(1,)):
    state = {
        "example_id": "e",
        "question": {"question_text": "q"},
        "l1_coarse_summary_catalog": [
            {"coarse_index": i, "time_span": {"start_s": i}, "scene_description": str(i)} for i in range(4)
        ],
    }
    action = {"tool_name": "select_coarse_clips", "arguments": {"selected_coarse_indices": list(selected)}}
    return {
        "metadata": {"task": "select_coarse_set", "is_core": True, "source_example_id": "e"},
        "messages": [{}, {"content": json.dumps({"state_t": state})}, {"content": json.dumps(action)}],
    }


def test_candidate_index_and_fail_closed_core_transition():
    report = {"results": [{"example_id": "e", "catalog_size": 4, "top32": [2, 1, 3, 0]}]}
    candidates = candidate_index(report)
    core, excluded = core_transitions([_chat()], candidates)
    assert not excluded
    assert [row["coarse_index"] for row in core[0]["state_t"]["l1_coarse_summary_catalog"]] == [0, 1, 2, 3]
    assert core[0]["state_t"]["candidate_retrieval"]["rank_hidden_from_policy"] is True

    core, excluded = core_transitions([_chat(selected=(9,))], candidates)
    assert core == []
    assert excluded["gold_outside_candidates"] == 1

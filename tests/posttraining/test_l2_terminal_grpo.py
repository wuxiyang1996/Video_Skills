from __future__ import annotations

from trainer.grpo.train_l2_terminal_on_policy import (
    ACTION_CONTRACT_VERSION,
    DATASET_BALANCING_VERSION,
    EXECUTOR_ISOLATION_VERSION,
    EXECUTOR_FALLBACK_VERSION,
    EXECUTOR_CACHE_VERSION,
    PROCESS_WARMUP_REWARD_VERSION,
    POINTWISE_ACTION_CONTRACT_VERSION,
    POINTWISE_GRADIENT_CONTRACT_VERSION,
    RESUME_CHECKPOINT_VERSION,
    DATASET_ROUTED_ACTION_CONTRACT_VERSION,
    SYSTEM,
    action_budget_compliant,
    aligned_process_warmup_reward,
    balanced_repeat_grpo_pool,
    build_retrieval_state,
    cached_executor_rollout,
    compact_executor_trace,
    compact_rollout_diagnostic,
    dataset_balanced_order,
    expand_temporal_neighbors,
    executor_backend_for_dataset,
    executor_cache_key,
    filter_example_for_retrieval,
    filtered_grpo_pool,
    is_trainable_reward_group,
    retry_exact_backward_after_oom,
    process_supervised_pool,
    parse_dataset_adapters,
    exact_group_pool,
    read_exact_group_allowlist,
    repeat_grpo_pool,
    retrieval_catalog,
    selected_indices,
    sample_pointwise_set,
    stable_rollout_seed,
    terminal_reward,
    _load_resume_checkpoint,
    _truncate_to_checkpoint,
)


def test_controller_prompt_declares_one_exact_action_contract() -> None:
    assert ACTION_CONTRACT_VERSION == "select-coarse-clips-exact-v1"
    assert EXECUTOR_ISOLATION_VERSION == "selected-window-closure-v1"
    assert EXECUTOR_FALLBACK_VERSION == "dataset-routed-cg-rule-vh-relative-mcq-typed-plan-v5"
    assert EXECUTOR_CACHE_VERSION == "shared-locked-rollout-typed-plan-v2"
    assert PROCESS_WARMUP_REWARD_VERSION == "dataset-routed-process-hit-aligned-v2"
    assert POINTWISE_ACTION_CONTRACT_VERSION == "pointwise-logodds-set-sampling-v1"
    assert POINTWISE_GRADIENT_CONTRACT_VERSION == "score-space-vjp-candidate-recompute-v1"
    assert RESUME_CHECKPOINT_VERSION == "group-boundary-optimizer-resume-v1"
    assert DATASET_ROUTED_ACTION_CONTRACT_VERSION == "dataset-routed-cg-set-vh-pointwise-v1"
    assert 'tool_name="select_coarse_clips"' in SYSTEM
    assert "arguments.coarse_indices" in SYSTEM
    assert "budget_state.topk" in SYSTEM
    assert executor_backend_for_dataset("cg_bench", "openai/gpt-oss-120b") == "deterministic-rule-assembly-v1"
    assert executor_backend_for_dataset("video_holmes", "openai/gpt-oss-120b") == "openai/gpt-oss-120b"


def test_exact_backward_oom_retry_discards_partial_gradients() -> None:
    class SyntheticOOM(RuntimeError):
        pass

    class Optimizer:
        def __init__(self) -> None:
            self.zero_calls = 0

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            self.zero_calls += 1

    events: list[str] = []
    optimizer = Optimizer()

    def normal_backward() -> list[float]:
        events.append("partial-normal-backward")
        raise SyntheticOOM("synthetic CUDA OOM")

    def offloaded_backward() -> list[float]:
        import sys

        assert sys.exc_info() == (None, None, None)
        events.append("exact-offloaded-recompute")
        return [0.1, 0.2]

    result, retried = retry_exact_backward_after_oom(
        normal_backward,
        offloaded_backward,
        optimizer=optimizer,
        empty_cache=lambda: events.append("empty-cache"),
        oom_type=SyntheticOOM,
        prepare_retry=lambda: events.append("prepare-retry"),
    )
    assert result == [0.1, 0.2]
    assert retried is True
    assert optimizer.zero_calls == 1
    assert events == [
        "partial-normal-backward",
        "prepare-retry",
        "empty-cache",
        "exact-offloaded-recompute",
    ]


def test_exact_backward_does_not_offload_without_oom() -> None:
    class Optimizer:
        def zero_grad(self, *, set_to_none: bool) -> None:
            raise AssertionError("zero_grad must not run on the success path")

    result, retried = retry_exact_backward_after_oom(
        lambda: [0.3],
        lambda: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
        optimizer=Optimizer(),
        empty_cache=lambda: (_ for _ in ()).throw(AssertionError("unexpected cache clear")),
        oom_type=RuntimeError,
    )
    assert result == [0.3]
    assert retried is False


def test_resume_checkpoint_is_complete_signature_checked_and_log_truncation_is_exact(
    tmp_path,
) -> None:
    import json
    import pytest

    checkpoint = tmp_path / "resume_checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "state.json").write_text(
        json.dumps(
            {
                "schema_version": RESUME_CHECKPOINT_VERSION,
                "run_signature": "frozen-run",
                "next_group_index": 10,
            }
        ),
        encoding="utf-8",
    )
    loaded_path, state = _load_resume_checkpoint(
        tmp_path, expected_signature="frozen-run"
    )
    assert loaded_path == checkpoint
    assert state["next_group_index"] == 10
    with pytest.raises(RuntimeError, match="does not match"):
        _load_resume_checkpoint(tmp_path, expected_signature="different-run")

    journal = tmp_path / "samples.jsonl"
    journal.write_bytes(b"committed\npartial\n")
    _truncate_to_checkpoint(journal, len(b"committed\n"))
    assert journal.read_bytes() == b"committed\n"


def test_pointwise_set_sampling_is_bounded_reproducible_and_can_anchor() -> None:
    scores = [0.1, 3.0, 2.0, -1.0]
    first = sample_pointwise_set(scores, topk=2, seed=7, temperature=0.9)
    assert first == sample_pointwise_set(scores, topk=2, seed=7, temperature=0.9)
    assert len(first) == len(set(first)) == 2
    assert all(0 <= index < len(scores) for index in first)
    anchored = sample_pointwise_set(
        scores, topk=2, seed=7, temperature=0.9, boundary_anchor_index0=True
    )
    assert len(anchored) == 2
    assert 0 in anchored


def test_dataset_adapter_routes_are_explicit_and_unique() -> None:
    routes = parse_dataset_adapters(["video_holmes=/tmp/opd", "cg_bench=/tmp/sft"])
    assert str(routes["video_holmes"]) == "/tmp/opd"
    assert str(routes["cg_bench"]) == "/tmp/sft"
    import pytest
    with pytest.raises(ValueError, match="DATASET=PATH"):
        parse_dataset_adapters(["video_holmes"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_dataset_adapters(["video_holmes=/a", "video_holmes=/b"])


def test_exact_mined_group_allowlist_preserves_repeat_seed(tmp_path) -> None:
    path = tmp_path / "groups.tsv"
    path.write_text("cg:1\t2\nvh:1\t7\n", encoding="utf-8")
    groups = read_exact_group_allowlist(path)
    assert groups == [("cg:1", 2), ("vh:1", 7)]
    pool = exact_group_pool(
        [{"example_id": "cg:1", "dataset": "cg_bench"}, {"example_id": "vh:1", "dataset": "video_holmes"}],
        groups,
    )
    assert [(row["example_id"], row["_grpo_repeat_index"]) for row in pool] == groups


def test_exact_group_allowlist_rejects_mixed_rows(tmp_path) -> None:
    import pytest

    path = tmp_path / "groups.tsv"
    path.write_text("cg:1\t2\nvh:1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mix"):
        read_exact_group_allowlist(path)


def test_process_warmup_trains_only_varying_groups_with_a_process_hit() -> None:
    varying = [
        {"reward": 0.0, "process_supported": False, "terminal_success": False},
        {"reward": 0.3, "process_supported": True, "terminal_success": False},
    ]
    assert not is_trainable_reward_group(varying)
    assert is_trainable_reward_group(varying, process_reward_warmup=True)
    assert not is_trainable_reward_group(
        [{"reward": 0.3, "process_supported": True}] * 2,
        process_reward_warmup=True,
    )
    assert not is_trainable_reward_group(
        [{"reward": 0.0, "process_supported": False}, {"reward": 0.2, "process_supported": False}],
        process_reward_warmup=True,
    )


def test_vh_process_warmup_hit_outranks_incidental_segment_overlap() -> None:
    weak_overlap = {
        "process_supported": False,
        "reward_components": {
            "segment_recall": 1.0,
            "inference_shot_recall": 0.0,
            "relationship_support": 0.5,
        },
    }
    inference_hit = {
        "process_supported": True,
        "reward_components": {
            "segment_recall": 0.0,
            "inference_shot_recall": 0.1,
            "relationship_support": 0.25,
        },
    }
    assert aligned_process_warmup_reward(
        inference_hit, dataset="video_holmes"
    ) > aligned_process_warmup_reward(weak_overlap, dataset="video_holmes")


def test_compact_executor_trace_keeps_plan_and_failed_steps_without_graph_nodes() -> None:
    trace = compact_executor_trace({
        "final_answer": {},
        "acceptance_status": "rejected",
        "failure_reasons": ["no_final_answer"],
        "nodes": [
            {"node_id": "evidence:1", "node_type": "observation", "text": "large"},
            {
                "node_id": "skill:1", "skill_id": "commit_answer", "step_id": "s2",
                "status": "failed", "failure_code": "missing_support", "evidence_refs": [],
            },
        ],
        "metadata": {
            "llm_plan": {"reasoning_plan": [{"skill_id": "commit_answer"}]},
            "failed_skill_ids": ["commit_answer"],
            "failed_skill_codes": [
                {"skill_id": "commit_answer", "failure_code": "missing_support"},
            ],
        },
    })
    assert trace["failed_skill_ids"] == ["commit_answer"]
    assert trace["skill_trace"][0]["failure_code"] == "missing_support"
    assert len(trace["skill_trace"]) == 1


def test_executor_cache_is_stable_and_reuses_one_rollout(tmp_path) -> None:
    key = executor_cache_key(
        example={"example_id": "vh:q1", "question": {"question_text": "why"}},
        indices=[2, 3],
        graph={"nodes": [{"node_id": "n1"}], "edges": []},
        planner_model="planner",
        skill_model="skill",
    )
    calls = []

    def build() -> dict:
        calls.append(1)
        return {"final_answer": {"label": "A"}}

    first, first_hit = cached_executor_rollout(cache_dir=tmp_path, key=key, build=build)
    second, second_hit = cached_executor_rollout(cache_dir=tmp_path, key=key, build=build)
    assert first == second
    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1


def test_selected_indices_are_bounded_and_deduplicated() -> None:
    payload = {
        "tool_name": "select_coarse_clips",
        "arguments": {"selected_coarse_indices": [2, 2, 99, 1]},
    }
    assert selected_indices(payload, catalog_size=4, topk=2) == [2, 1]
    assert selected_indices({"tool_name": "commit_answer"}, catalog_size=4) == []


def test_selected_indices_accept_pointwise_candidate_action_shape() -> None:
    payload = {
        "tool_name": "score_coarse_candidate",
        "arguments": {"candidate_index": 3, "score": 0.8},
    }
    assert selected_indices(payload, catalog_size=5, topk=2, boundary_anchor_index0=True) == [3, 0]
    assert selected_indices({"candidate_index": 2}, catalog_size=5, topk=2) == [2]


def test_boundary_anchor_does_not_overwrite_native_set_action() -> None:
    payload = {
        "tool_name": "select_coarse_clips",
        "arguments": {"coarse_indices": [3, 4, 5]},
    }
    assert selected_indices(
        payload, catalog_size=8, topk=2, boundary_anchor_index0=True
    ) == [3, 4]
    assert action_budget_compliant(payload, catalog_size=8, topk=2) is False
    payload["arguments"]["coarse_indices"] = [3, 4]
    assert action_budget_compliant(payload, catalog_size=8, topk=2) is True


def test_video_holmes_point_prediction_expands_to_temporal_evidence_chain() -> None:
    assert expand_temporal_neighbors([5], catalog_size=10, topk=4) == [5, 4, 6, 3]
    assert expand_temporal_neighbors([0], catalog_size=3, topk=4) == [0, 1, 2]


def test_filter_graph_keeps_only_selected_span_and_requirement() -> None:
    example = {
        "evidence_index": {
            "nodes": [
                {"node_id": "ei_a", "time_span": {"start_s": 1, "end_s": 2}},
                {"node_id": "ei_b", "time_span": {"start_s": 11, "end_s": 12}},
            ],
            "edges": [],
        },
        "metadata": {
            "coarse_clip_schemas": [
                {"time_span": {"start_s": 0, "end_s": 10}},
                {"time_span": {"start_s": 10, "end_s": 20}},
            ],
            "clue_memory_graph": {
                "nodes": [
                    {"node_id": "a", "node_type": "observation", "time_span": {"start_s": 1, "end_s": 2}},
                    {"node_id": "b", "node_type": "observation", "time_span": {"start_s": 11, "end_s": 12}},
                    {"node_id": "q", "node_type": "question_requirement"},
                ],
                "edges": [
                    {"src": "a", "dst": "q"},
                    {"src": "b", "dst": "q"},
                ],
            },
            "coarse_fine_graph": {
                "fine_graph": {
                    "nodes": [
                        {"node_id": "fine_a", "time_span": {"start_s": 1, "end_s": 2}},
                        {"node_id": "fine_b", "time_span": {"start_s": 11, "end_s": 12}},
                    ],
                    "edges": [],
                },
            },
            "reasoning_rollout": {"final_answer": {"label": "LEAK"}},
        }
    }
    isolated, graph = filter_example_for_retrieval(example, [1])
    assert {node["node_id"] for node in graph["nodes"]} == {"b", "q"}
    assert graph["edges"] == [{"src": "b", "dst": "q"}]
    assert {node["node_id"] for node in isolated["evidence_index"]["nodes"]} == {"ei_b"}
    assert {
        node["node_id"]
        for node in isolated["metadata"]["coarse_fine_graph"]["fine_graph"]["nodes"]
    } == {"fine_b"}
    assert "reasoning_rollout" not in isolated["metadata"]


def test_video_holmes_uses_fine_catalog_when_coarse_is_empty() -> None:
    example = {
        "dataset": "video_holmes",
        "metadata": {
            "coarse_clip_schemas": [],
            "clip_schemas": [
                {"clip_id": "f0", "time_span": {"start_s": 0, "end_s": 4}},
                {"clip_id": "f1", "time_span": {"start_s": 3, "end_s": 7}},
            ],
            "clue_memory_graph": {"nodes": [], "edges": []},
        },
    }
    catalog, source = retrieval_catalog(example)
    assert source == "clip_schemas"
    assert len(catalog) == 2
    state = build_retrieval_state(example)
    assert state["retrieval_catalog_source"] == "clip_schemas"
    assert len(state["l1_coarse_summary_catalog"]) == 2
    isolated, graph = filter_example_for_retrieval(example, [1])
    assert [row["clip_id"] for row in isolated["metadata"]["clip_schemas"]] == ["f1"]
    assert graph["retrieval"]["catalog_source"] == "clip_schemas"


def test_terminal_reward_requires_correct_verified_strong_answer() -> None:
    gold = {"label": "B"}
    success = terminal_reward(
        {
            "final_answer": {"label": "B"},
            "acceptance_status": "accepted_strong",
            "metadata": {"runtime_verifier": {"passed": True}},
        },
        gold,
    )
    assert success["terminal_success"] is True
    wrong = terminal_reward(
        {
            "final_answer": {"label": "A"},
            "acceptance_status": "accepted_strong",
            "metadata": {"runtime_verifier": {"passed": True}},
        },
        gold,
    )
    assert wrong["terminal_success"] is False
    abstain = terminal_reward(
        {
            "final_answer": {},
            "acceptance_status": "rejected",
            "metadata": {"runtime_verifier": {"passed": False}},
        },
        gold,
    )
    assert wrong["reward"] < abstain["reward"] < success["reward"]
    assert abstain["reward_components"]["verifier"] == 0.0


def test_compact_rollout_diagnostic_keeps_failure_evidence() -> None:
    diagnostic = compact_rollout_diagnostic({
        "failure_reasons": ["insufficient_support_refs"],
        "verified_evidence_pack": {"support_ref_count": 1, "min_support_refs": 2},
        "metadata": {
            "llm_trace_ok": 3,
            "llm_trace_fail": 2,
            "llm_plan": {
                "query_memory_finalizer": {
                    "attempted": True,
                    "qa_answerability": {"grade": "insufficient"},
                    "selected_option": None,
                    "verified": False,
                }
            },
            "failed_skill_ids": ["commit_answer"],
            "failed_skill_codes": [
                {"skill_id": "commit_answer", "failure_code": "missing_support"},
            ],
        },
    })
    assert diagnostic == {
        "failure_reasons": ["insufficient_support_refs"],
        "failed_skill_ids": ["commit_answer"],
        "failed_skill_codes": [
            {"skill_id": "commit_answer", "failure_code": "missing_support"},
        ],
        "support_ref_count": 1,
        "min_support_refs": 2,
        "trace_ok": 3,
        "trace_fail": 2,
        "query_memory_finalizer": {
            "attempted": True,
            "qa_answerability": {"grade": "insufficient"},
            "selected_option": None,
            "verified": False,
        },
    }


def test_dataset_rewards_expose_distinct_process_components() -> None:
    rollout = {
        "final_answer": {"label": "B"},
        "acceptance_status": "accepted_weak",
        "metadata": {"runtime_verifier": {"passed": True}},
    }
    entries = [{
        "time_span": {"start_s": 9, "end_s": 13},
        "scene_description": "The woman in the dress reveals glowing eyes and attacks.",
    }]
    cg = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=entries,
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert cg["reward_components"]["clue_recall"] == 1.0
    vh = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="video_holmes",
        selected_entries=entries,
        supervision={
            "segment_spans": [{"start_s": 0, "end_s": 31}],
            "inference_spans": [{"start_s": 10, "end_s": 11}],
            "relationship_texts": ["woman in dress attacks the woman"],
        },
    )
    assert vh["reward_components"]["segment_recall"] == 1.0
    assert vh["reward_components"]["inference_shot_recall"] == 1.0
    assert vh["reward_components"]["relationship_support"] > 0


def test_video_holmes_process_support_is_question_type_aware() -> None:
    rollout = {
        "final_answer": {},
        "acceptance_status": "rejected",
        "metadata": {"runtime_verifier": {"passed": False}},
    }
    supervision = {
        "segment_spans": [{"start_s": 0, "end_s": 20}],
        "inference_spans": [{"start_s": 10, "end_s": 12}],
        "relationship_texts": ["father and son"],
    }
    entries = [{"time_span": {"start_s": 10, "end_s": 14}, "scene_description": "a car explodes"}]
    sr = terminal_reward(
        rollout, {}, dataset="video_holmes", selected_entries=entries,
        supervision=supervision, question_type="SR",
    )
    assert sr["reward_components"]["inference_shot_recall"] == 1.0
    assert sr["process_supported"] is False
    relationship_only = terminal_reward(
        rollout, {}, dataset="video_holmes",
        selected_entries=[{
            "time_span": {"start_s": 30, "end_s": 34},
            "scene_description": "The father helps his son",
        }],
        supervision=supervision, question_type="SR",
    )
    assert relationship_only["reward_components"]["relationship_support"] >= 0.25
    assert relationship_only["process_supported"] is False
    both = terminal_reward(
        rollout, {}, dataset="video_holmes",
        selected_entries=[{
            "time_span": {"start_s": 10, "end_s": 14},
            "scene_description": "The father helps his son",
        }],
        supervision=supervision, question_type="SR",
    )
    assert both["process_supported"] is True
    mhr = terminal_reward(
        rollout, {}, dataset="video_holmes", selected_entries=entries,
        supervision=supervision, question_type="MHR",
    )
    assert mhr["process_supported"] is True


def test_dataset_terminal_success_requires_process_support() -> None:
    rollout = {
        "final_answer": {"label": "B"},
        "acceptance_status": "accepted_strong",
        "metadata": {"runtime_verifier": {"passed": True}},
    }
    miss = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=[{"time_span": {"start_s": 0, "end_s": 4}}],
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert miss["answer_terminal_success"] is True
    assert miss["process_supported"] is False
    assert miss["terminal_success"] is False
    assert miss["reward"] <= 0.35
    hit = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=[{"time_span": {"start_s": 10, "end_s": 14}}],
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert hit["terminal_success"] is True


def test_correct_verified_weak_acceptance_requires_minimum_evidence_and_clean_trace() -> None:
    rollout = {
        "final_answer": {"label": "B"},
        "acceptance_status": "accepted_weak",
        "verified_evidence_pack": {
            "support_ref_count": 3,
            "min_support_refs": 2,
            "trace_fail": 0,
        },
        "metadata": {"runtime_verifier": {"passed": True}},
    }
    success = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=[{"time_span": {"start_s": 10, "end_s": 14}}],
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert success["minimum_verified_acceptance"] is True
    assert success["terminal_success"] is True
    rollout["verified_evidence_pack"]["trace_fail"] = 1
    failed = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=[{"time_span": {"start_s": 10, "end_s": 14}}],
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert failed["terminal_success"] is False


def test_verified_query_memory_repair_can_resolve_a_failed_exploratory_trace() -> None:
    rollout = {
        "final_answer": {"label": "B"},
        "acceptance_status": "accepted_weak",
        "verified_evidence_pack": {
            "support_ref_count": 3,
            "min_support_refs": 2,
            "trace_fail": 4,
        },
        "nodes": [{
            "step_id": "query_memory_commit_final",
            "skill_id": "commit_answer",
            "status": "verified",
        }],
        "metadata": {
            "runtime_verifier": {"passed": True},
            "llm_plan": {
                "query_memory_finalizer": {
                    "attempted": True,
                    "verified": True,
                    "selected_option": {"label": "B"},
                }
            },
        },
    }
    resolved = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=[{"time_span": {"start_s": 10, "end_s": 14}}],
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert resolved["repaired_minimum_verified_acceptance"] is True
    assert resolved["minimum_verified_acceptance"] is True
    assert resolved["terminal_success"] is True

    rollout["nodes"][0]["status"] = "failed"
    uncommitted = terminal_reward(
        rollout,
        {"label": "B"},
        dataset="cg_bench",
        selected_entries=[{"time_span": {"start_s": 10, "end_s": 14}}],
        supervision={"clue_spans": [{"start_s": 10, "end_s": 12}]},
    )
    assert uncommitted["repaired_minimum_verified_acceptance"] is False
    assert uncommitted["terminal_success"] is False


def test_dataset_balanced_order_round_robins_before_truncation() -> None:
    rows = [
        *({"example_id": f"cg{i}", "dataset": "cg_bench"} for i in range(5)),
        *({"example_id": f"vh{i}", "dataset": "video_holmes"} for i in range(2)),
    ]
    ordered = dataset_balanced_order(rows, seed=7)
    assert [row["dataset"] for row in ordered[:4]] == [
        "cg_bench", "video_holmes", "cg_bench", "video_holmes"
    ]


def test_process_supervised_pool_requires_usable_dataset_specific_labels() -> None:
    rows = [
        {"example_id": "cg_bench:1", "dataset": "cg_bench", "question": {"question_id": "1"}},
        {"example_id": "video_holmes:train:v1:q1", "dataset": "video_holmes", "question": {"question_type": "SR"}},
        {"example_id": "video_holmes:train:v2:q2", "dataset": "video_holmes", "question": {"question_type": "MHR"}},
        {"example_id": "video_holmes:train:v3:q3", "dataset": "video_holmes", "question": {"question_type": "SR"}},
    ]
    supervision = {
        "cg_bench:1": {"clue_spans": [{"start_s": 1, "end_s": 2}]},
        "video_holmes:v1": {"inference_spans": [{"start_s": 1, "end_s": 2}], "relationship_texts": ["same person"]},
        "video_holmes:v2": {"inference_spans": [{"start_s": 1, "end_s": 2}], "relationship_texts": []},
        "video_holmes:v3": {"inference_spans": [], "relationship_texts": ["same person"]},
    }
    kept = process_supervised_pool(rows, supervision)
    assert [row["example_id"] for row in kept] == [
        "cg_bench:1", "video_holmes:train:v1:q1", "video_holmes:train:v2:q2"
    ]


def test_repeat_pool_preserves_balanced_order_and_tracks_repeat() -> None:
    rows = [
        {"example_id": "cg0", "dataset": "cg_bench"},
        {"example_id": "vh0", "dataset": "video_holmes"},
    ]
    repeated = repeat_grpo_pool(rows, repeats_per_example=3)
    assert [row["dataset"] for row in repeated] == [
        "cg_bench", "video_holmes", "cg_bench", "video_holmes", "cg_bench", "video_holmes"
    ]
    assert [row["_grpo_repeat_index"] for row in repeated] == [0, 0, 1, 1, 2, 2]
    supplemental = repeat_grpo_pool(rows, repeats_per_example=2, repeat_start_index=3)
    assert [row["_grpo_repeat_index"] for row in supplemental] == [3, 3, 4, 4]


def test_balanced_repeat_pool_equalizes_counts_without_duplicate_seeds() -> None:
    assert DATASET_BALANCING_VERSION == "equal-groups-cyclic-repeats-v1"
    rows = [
        *({"example_id": f"cg{i}", "dataset": "cg_bench"} for i in range(4)),
        *({"example_id": f"vh{i}", "dataset": "video_holmes"} for i in range(2)),
    ]
    ordered = dataset_balanced_order(rows, seed=7)
    repeated = balanced_repeat_grpo_pool(ordered, repeats_per_example=2)
    assert [row["dataset"] for row in repeated[:8]] == [
        "cg_bench", "video_holmes", "cg_bench", "video_holmes",
        "cg_bench", "video_holmes", "cg_bench", "video_holmes",
    ]
    assert sum(row["dataset"] == "cg_bench" for row in repeated) == 8
    assert sum(row["dataset"] == "video_holmes" for row in repeated) == 8
    vh_pairs = [
        (row["example_id"], row["_grpo_repeat_index"])
        for row in repeated if row["dataset"] == "video_holmes"
    ]
    assert len(vh_pairs) == len(set(vh_pairs))


def test_rollout_seed_is_subset_stable_and_repeat_specific() -> None:
    first = stable_rollout_seed(42, example_id="vh:q1", repeat_index=0, sample_index=3)
    assert first == stable_rollout_seed(42, example_id="vh:q1", repeat_index=0, sample_index=3)
    assert first != stable_rollout_seed(42, example_id="vh:q1", repeat_index=1, sample_index=3)
    assert first != stable_rollout_seed(42, example_id="vh:q2", repeat_index=0, sample_index=3)


def test_filtered_grpo_pool_applies_dataset_allowlist_and_catalog_bounds() -> None:
    def schemas(count: int) -> list[dict]:
        return [{"time_span": {"start_s": i, "end_s": i + 1}} for i in range(count)]

    rows = [
        {"example_id": "a", "dataset": "cg_bench", "metadata": {"coarse_clip_schemas": schemas(2)}},
        {"example_id": "b", "dataset": "video_holmes", "metadata": {"coarse_clip_schemas": [], "clip_schemas": schemas(2)}},
        {"example_id": "c", "dataset": "cg_bench", "metadata": {"coarse_clip_schemas": schemas(4)}},
    ]
    kept = filtered_grpo_pool(
        rows,
        datasets={"cg_bench"},
        example_id_allowlist={"a", "b", "c"},
        min_catalog_size=2,
        max_catalog_size=3,
    )
    assert [row["example_id"] for row in kept] == ["a"]


def test_process_supervision_filter_excludes_unlabeled_video() -> None:
    rows = [
        {"example_id": "video_holmes:train:labeled:q1", "dataset": "video_holmes", "question": {"question_type": "SR"}},
        {"example_id": "video_holmes:train:missing:q2", "dataset": "video_holmes", "question": {"question_type": "SR"}},
    ]
    kept = process_supervised_pool(rows, {
        "video_holmes:labeled": {
            "inference_spans": [{"start_s": 1, "end_s": 2}],
            "relationship_texts": ["same person"],
        }
    })
    assert [row["example_id"] for row in kept] == ["video_holmes:train:labeled:q1"]

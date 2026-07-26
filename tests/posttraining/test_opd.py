from __future__ import annotations

from pathlib import Path

from trainer.candidate_action_builder import build_l2_candidate_actions, gate_candidate_set
from trainer.closed_loop_harness import ClosedLoopHarness
from trainer.exact_request_cache import ExactRequestCache
from trainer.opd_action_distill_adapter import OpdDistillRow, load_opd_rows, save_opd_rows
from trainer.teacher_action_query import (
    aggregate_rankings_to_action_probs,
    average_action_probs,
    borda_scores_from_rankings,
    map_candidates_to_letters,
    mock_teacher_preferring_oracle,
    mock_teacher_ranking_preferring_oracle,
    parse_ranked_letters,
    query_teacher_action_distribution,
    query_teacher_averaged,
    query_teacher_ranking_averaged,
    query_teacher_ranking_distribution,
    soft_calibration_gates,
    soft_calibration_passed,
)
from trainer.train_opd_kl import run_opd_smoke


def test_candidate_builder_coverage_and_oracle_recall() -> None:
    oracle = {
        "schema_version": "video-skills/l2-specialist-action-v0.1",
        "tool_name": "choose_best_coarse_candidate",
        "arguments": {"coarse_index": 3},
    }
    action_set = build_l2_candidate_actions(
        state_id="s1",
        oracle_action=oracle,
        coarse_indices=[1, 2, 3],
    )
    gate = gate_candidate_set(action_set)
    assert gate["passed"], gate
    assert action_set.candidate_recall == 1.0
    assert action_set.coverage["has_stop"]
    assert action_set.coverage["has_abstain"]
    assert action_set.coverage["has_hard_negative"]


def test_teacher_order_permutation_maps_back(tmp_path: Path) -> None:
    oracle = {
        "schema_version": "video-skills/l2-specialist-action-v0.1",
        "tool_name": "choose_best_coarse_candidate",
        "arguments": {"coarse_index": 2},
    }
    action_set = build_l2_candidate_actions(state_id="s2", oracle_action=oracle)
    state = {"example_id": "e", "task_family": "causal", "question": {"question_text": "q"}}
    cache = ExactRequestCache(tmp_path / "cache.json", {"model": "mock"})

    d0 = query_teacher_action_distribution(
        action_set,
        state=state,
        teacher_fn=mock_teacher_preferring_oracle,
        order_seed=0,
        cache=cache,
    )
    d1 = query_teacher_action_distribution(
        action_set,
        state=state,
        teacher_fn=mock_teacher_preferring_oracle,
        order_seed=99,
        cache=cache,
    )
    # Prefer oracle action id regardless of letter order.
    assert max(d0.action_probs, key=d0.action_probs.get) == "oracle"
    assert max(d1.action_probs, key=d1.action_probs.get) == "oracle"

    ordered_a, map_a = map_candidates_to_letters(action_set.candidates, order_seed=1)
    ordered_b, map_b = map_candidates_to_letters(action_set.candidates, order_seed=2)
    assert set(map_a.values()) == set(map_b.values())
    assert [c.action_id for c in ordered_a] != [c.action_id for c in ordered_b] or len(ordered_a) <= 1

    avg, dists = query_teacher_averaged(
        action_set,
        state=state,
        teacher_fn=mock_teacher_preferring_oracle,
        order_seeds=[0, 7, 99],
        cache=cache,
    )
    assert len(dists) == 3
    assert max(avg, key=avg.get) == "oracle"
    assert abs(sum(avg.values()) - 1.0) < 1e-6
    assert average_action_probs(dists)["oracle"] == avg["oracle"]


def test_harness_requires_motif_attempt() -> None:
    def rollout_fn(example, clue):
        del clue
        return {
            "acceptance_status": "accepted_strong",
            "metadata": {
                "runtime_verifier": {"passed": True},
                "llm_plan": {"reasoning_plan": [{"skill_id": "commit_answer", "args": {}}]},
                "motif_online": {
                    "motif_retrieval_attempted": True,
                    "candidate_ids": ["m1"],
                    "selected_motif_id": "m1",
                    "expansion_valid": False,
                    "fallback_reason": "expansion_invalid",
                },
            },
        }

    harness = ClosedLoopHarness(rollout_fn=rollout_fn, motif_enabled=True)
    state = harness.run_example(
        {
            "example_id": "x",
            "dataset": "cg_bench",
            "task_family": "causal",
            "question": {"question_text": "q"},
            "metadata": {"clue_memory_graph": {"nodes": [], "edges": []}},
        }
    )
    assert state.motif_online["motif_retrieval_attempted"] is True
    assert state.student_action is not None


def test_opd_distill_and_kl_smoke(tmp_path: Path) -> None:
    oracle = {
        "schema_version": "video-skills/l2-specialist-action-v0.1",
        "tool_name": "choose_best_coarse_candidate",
        "arguments": {"coarse_index": 1},
    }
    action_set = build_l2_candidate_actions(state_id="s3", oracle_action=oracle)
    precheck = gate_candidate_set(action_set)
    teacher = query_teacher_action_distribution(
        action_set,
        state={"example_id": "e"},
        teacher_fn=mock_teacher_preferring_oracle,
        order_seed=3,
    )
    from trainer.closed_loop_harness import HarnessState

    state = HarnessState(
        state_id="s3",
        example_id="e",
        dataset="cg_bench",
        task_family="causal",
        question={"question_text": "q"},
        l1_graph_summary={"node_count": 0, "edge_count": 0},
        motif_online={"motif_retrieval_attempted": True},
    )
    row = OpdDistillRow.from_parts(
        state=state,
        action_set=action_set,
        teacher=teacher,
        precheck=precheck,
    )
    path = tmp_path / "opd.jsonl"
    save_opd_rows(path, [row])
    assert len(load_opd_rows(path)) == 1
    report = run_opd_smoke(path, output_path=tmp_path / "smoke.json")
    assert report["n_rows"] == 1
    assert report["mean_kl"] is not None


def test_soft_calibration_gates() -> None:
    ok = soft_calibration_gates(top1_match_rate=0.95, mean_l1=0.10, n_rows=8)
    assert soft_calibration_passed(ok)
    bad = soft_calibration_gates(top1_match_rate=0.125, mean_l1=1.34, n_rows=8)
    assert not soft_calibration_passed(bad)
    assert bad["top1_match_ge_0_90"] is False
    assert bad["mean_l1_le_0_15"] is False


def test_borda_and_bt_ranking_aggregation() -> None:
    rankings = [
        ["oracle", "student", "stop"],
        ["oracle", "stop", "student"],
        ["student", "oracle", "stop"],
    ]
    ids = ["oracle", "student", "stop"]
    borda = borda_scores_from_rankings(rankings, action_ids=ids)
    assert borda["oracle"] > borda["student"] >= borda["stop"]
    probs_borda = aggregate_rankings_to_action_probs(
        rankings, action_ids=ids, method="borda", temperature=1.0
    )
    assert abs(sum(probs_borda.values()) - 1.0) < 1e-6
    assert max(probs_borda, key=probs_borda.get) == "oracle"
    probs_bt = aggregate_rankings_to_action_probs(
        rankings, action_ids=ids, method="bt", temperature=1.0
    )
    assert abs(sum(probs_bt.values()) - 1.0) < 1e-6
    assert max(probs_bt, key=probs_bt.get) == "oracle"


def test_parse_ranked_letters_from_content_json() -> None:
    letter_map = {"A": "student", "B": "oracle", "C": "stop"}
    ranked = parse_ranked_letters(
        {"content": '```json\n{"ranked_letters": ["B", "A", "C"]}\n```'},
        letter_map,
    )
    assert ranked == ["B", "A", "C"]


def test_teacher_ranking_order_permutation_maps_back(tmp_path: Path) -> None:
    oracle = {
        "schema_version": "video-skills/l2-specialist-action-v0.1",
        "tool_name": "choose_best_coarse_candidate",
        "arguments": {"coarse_index": 2},
    }
    action_set = build_l2_candidate_actions(state_id="s_rank", oracle_action=oracle)
    state = {"example_id": "e", "task_family": "causal", "question": {"question_text": "q"}}
    cache = ExactRequestCache(tmp_path / "rank_cache.json", {"model": "mock_ranking"})

    d0 = query_teacher_ranking_distribution(
        action_set,
        state=state,
        teacher_fn=mock_teacher_ranking_preferring_oracle,
        order_seed=0,
        cache=cache,
        method="borda",
    )
    d1 = query_teacher_ranking_distribution(
        action_set,
        state=state,
        teacher_fn=mock_teacher_ranking_preferring_oracle,
        order_seed=99,
        cache=cache,
        method="borda",
    )
    assert d0.teacher_mode == "ranking_borda"
    assert d0.ranked_action_ids[0] == "oracle"
    assert d1.ranked_action_ids[0] == "oracle"
    assert max(d0.action_probs, key=d0.action_probs.get) == "oracle"
    assert max(d1.action_probs, key=d1.action_probs.get) == "oracle"

    avg, dists = query_teacher_ranking_averaged(
        action_set,
        state=state,
        teacher_fn=mock_teacher_ranking_preferring_oracle,
        order_seeds=[0, 7, 99],
        cache=cache,
        method="borda",
    )
    assert len(dists) == 3
    assert max(avg, key=avg.get) == "oracle"
    assert abs(sum(avg.values()) - 1.0) < 1e-6

    avg_bt, _ = query_teacher_ranking_averaged(
        action_set,
        state=state,
        teacher_fn=mock_teacher_ranking_preferring_oracle,
        order_seeds=[0, 7, 99],
        cache=cache,
        method="bt",
    )
    assert max(avg_bt, key=avg_bt.get) == "oracle"


def test_opd_distill_from_ranking_teacher(tmp_path: Path) -> None:
    oracle = {
        "schema_version": "video-skills/l2-specialist-action-v0.1",
        "tool_name": "choose_best_coarse_candidate",
        "arguments": {"coarse_index": 1},
    }
    action_set = build_l2_candidate_actions(state_id="s_rank_row", oracle_action=oracle)
    precheck = gate_candidate_set(action_set)
    avg, dists = query_teacher_ranking_averaged(
        action_set,
        state={"example_id": "e"},
        teacher_fn=mock_teacher_ranking_preferring_oracle,
        order_seeds=[1, 2],
        method="borda",
    )
    teacher = dists[0]
    teacher.action_probs = dict(avg)
    teacher.teacher_mode = "ranking_borda"
    from trainer.closed_loop_harness import HarnessState

    state = HarnessState(
        state_id="s_rank_row",
        example_id="e",
        dataset="cg_bench",
        task_family="causal",
        question={"question_text": "q"},
        l1_graph_summary={"node_count": 0, "edge_count": 0},
        motif_online={"motif_retrieval_attempted": True},
    )
    row = OpdDistillRow.from_parts(
        state=state,
        action_set=action_set,
        teacher=teacher,
        precheck={**precheck, "teacher_mode": "ranking_borda"},
    )
    path = tmp_path / "opd_rank.jsonl"
    save_opd_rows(path, [row])
    loaded = load_opd_rows(path)
    assert loaded[0]["teacher"]["teacher_mode"] == "ranking_borda"
    report = run_opd_smoke(path, output_path=tmp_path / "smoke_rank.json")
    assert report["n_rows"] == 1
    assert report["mean_kl"] is not None

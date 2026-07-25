from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.candidate_action_builder import build_l2_candidate_actions
from trainer.grpo import MODE_JOINT_L1, MODE_L2_REPAIR, GrpoTrainConfig, grpo_surrogate_loss
from trainer.grpo.collect_rollouts import collect_grpo_group, save_grpo_groups
from trainer.grpo.train_verified import run_grpo_smoke, train_step_on_group
from trainer.reward import policy_safe_rollout_view, score_rollout_trace
from trainer.split_filter import example_video_key, filter_examples_by_role
from trainer.teacher_action_query import (
    mock_teacher_preferring_oracle,
    query_teacher_action_distribution,
)


def _minimal_clue(video_id: str = "vid") -> dict:
    return {
        "graph_id": f"clue:{video_id}",
        "layer": "clue_memory",
        "video_id": video_id,
        "nodes": [],
        "edges": [],
    }


def _smoke_rollout(example: dict, clue: dict) -> dict:
    seed = int((example.get("metadata") or {}).get("grpo_seed") or 0)
    gold = (example.get("question") or {}).get("answer") or {"label": "A"}
    success = seed % 2 == 0
    graph = clue if clue.get("graph_id") else _minimal_clue(str(example.get("video_id") or "vid"))
    return {
        "layer": "reasoning",
        "rollout_id": f"rollout-{seed}",
        "clue_memory_ref": {"graph_id": graph["graph_id"]},
        "acceptance_status": "accepted_strong" if success else "accepted_weak",
        "final_answer": gold if success else {"label": "Z"},
        "answer_correct": success,  # will be stripped from policy view
        "gold_answer": gold,
        "metadata": {
            "motif_online": {
                "motif_retrieval_attempted": True,
                "motif_phase": "accelerate",
                "expansion_valid": True,
                "candidate_mined": False,
            },
            "executed_skill_ids": ["retrieve_by_event", "commit_answer"],
            "costs": {"clip_reads": 1, "tool_calls": 2, "tokens": 10, "repair_rounds": 0},
            "milestone_events": [
                {"kind": "retrieval", "key": f"r{seed}", "step_index": 0, "grounded": True}
            ]
            if success
            else [],
            "final_used_milestone_keys": [f"retrieval:r{seed}"] if success else [],
            "clue_memory_graph": graph,
        },
    }


def test_collect_group_assigns_advantages_and_strips_hidden() -> None:
    example = {
        "example_id": "cg_bench:demo",
        "dataset": "cg_bench",
        "video_id": "vid1",
        "question": {"question_text": "q", "answer": {"label": "A", "text": "a"}},
        "metadata": {"clue_memory_graph": _minimal_clue("vid1")},
    }
    group = collect_grpo_group(
        example,
        rollout_fn=_smoke_rollout,
        k=4,
        base_seed=10,
        mode=MODE_L2_REPAIR,
    )
    assert len(group.rollouts) == 4
    assert group.rollouts[0].update_modules == ("l2", "repair")
    advs = [r.advantage for r in group.rollouts]
    assert max(advs) > min(advs)
    for r in group.rollouts:
        assert "answer_correct" not in r.policy_view
        assert "gold_answer" not in r.policy_view
        assert r.motif_online.get("motif_retrieval_attempted") is True


def test_joint_l1_requires_stable_gate() -> None:
    with pytest.raises(RuntimeError):
        GrpoTrainConfig(mode=MODE_JOINT_L1, l2_stable_flag=False).update_modules()
    modules = GrpoTrainConfig(mode=MODE_JOINT_L1, l2_stable_flag=True).update_modules()
    assert modules == ("l1", "l2", "repair")


def test_train_smoke_and_persist(tmp_path: Path) -> None:
    example = {
        "example_id": "e2",
        "dataset": "cg_bench",
        "video_id": "v2",
        "question": {"answer": {"label": "B"}},
        "metadata": {"clue_memory_graph": _minimal_clue("v2")},
    }
    group = collect_grpo_group(example, rollout_fn=_smoke_rollout, k=4, base_seed=0)
    path = save_grpo_groups(tmp_path / "groups.jsonl", [group])
    summary = run_grpo_smoke(
        path,
        config=GrpoTrainConfig(mode=MODE_L2_REPAIR),
        output_path=tmp_path / "train.json",
    )
    assert summary["n_groups"] == 1
    assert "l2" in summary["update_modules"]
    assert "l1" not in summary["update_modules"]
    step = train_step_on_group(
        group.to_dict(),
        config=GrpoTrainConfig(mode=MODE_JOINT_L1, l2_stable_flag=True),
    )
    assert step["update_modules"] == ["l1", "l2", "repair"]
    assert step["l1_lr_scale"] == 0.1


def test_grpo_surrogate_finite() -> None:
    stats = grpo_surrogate_loss(
        advantages=[-1.0, 1.0],
        logprobs=[-2.0, -1.0],
        ref_logprobs=[-2.1, -1.1],
        kl_coef=0.05,
    )
    assert stats["n"] == 2.0
    assert abs(stats["loss"]) < 100


def test_split_filter_by_role() -> None:
    manifest = {
        "videos": [
            {"key": "cg_bench:v1", "dataset": "cg_bench", "video_id": "v1", "role": "grpo_pool"},
            {"key": "cg_bench:v2", "dataset": "cg_bench", "video_id": "v2", "role": "opd_pool"},
        ]
    }
    examples = [
        {"dataset": "cg_bench", "video_id": "v1", "example_id": "a"},
        {"dataset": "cg_bench", "video_id": "v2", "example_id": "b"},
    ]
    kept = filter_examples_by_role(examples, manifest=manifest, role="grpo_pool")
    assert [e["example_id"] for e in kept] == ["a"]
    assert example_video_key(examples[0]) == "cg_bench:v1"


def test_teacher_fail_closed_without_logprobs() -> None:
    action_set = build_l2_candidate_actions(
        state_id="s",
        oracle_action={
            "schema_version": "video-skills/l2-specialist-action-v0.1",
            "tool_name": "choose_best_coarse_candidate",
            "arguments": {"coarse_index": 1},
        },
    )

    def empty_teacher(request):
        return {"letter": "A", "letter_logprobs": {}, "teacher": "empty"}

    with pytest.raises(RuntimeError, match="missing letter_logprobs"):
        query_teacher_action_distribution(
            action_set,
            state={"example_id": "e", "question": {}},
            teacher_fn=empty_teacher,
            require_logprobs=True,
        )

    # Mock teacher still works (provides logprobs).
    dist = query_teacher_action_distribution(
        action_set,
        state={"example_id": "e", "question": {}},
        teacher_fn=mock_teacher_preferring_oracle,
        require_logprobs=True,
    )
    assert dist.action_probs


def test_policy_safe_view_and_score_trace() -> None:
    clue = _minimal_clue("v3")
    rollout = {
        "layer": "reasoning",
        "rollout_id": "r1",
        "clue_memory_ref": {"graph_id": clue["graph_id"]},
        "acceptance_status": "accepted_strong",
        "final_answer": {"label": "A"},
        "answer_correct": True,
        "gold_answer": {"label": "A"},
        "metadata": {
            "clue_memory_graph": clue,
            "costs": {"clip_reads": 1, "tool_calls": 1, "tokens": 1, "repair_rounds": 0},
            "milestone_events": [],
        },
    }
    safe = policy_safe_rollout_view(rollout)
    assert "gold_answer" not in safe
    assert "answer_correct" not in safe
    scored = score_rollout_trace(rollout, clue_graph=clue, gold_answer={"label": "A"})
    assert scored.hard_feasible
    assert scored.spec_version.startswith("video-skills/verified-reward")
